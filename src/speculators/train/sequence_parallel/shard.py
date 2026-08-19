"""Sequence sharding for sequence-parallel Eagle3 training."""

from __future__ import annotations

from typing import Any

import torch

# Keys that are laid out as ``[1, seq_len, ...]`` and must be sliced on dim 1.
_SEQUENCE_KEYS = (
    "hidden_states",
    "input_ids",
    "verifier_last_hidden_states",
    "loss_mask",
    "position_ids",
    "document_ids",
)


def pad_seq_to_sp_multiple(batch: dict[str, Any], sp_size: int) -> dict[str, Any]:
    """Pad packed batch seq dim so ``seq_len % sp_size == 0``.

    Pads with zeros / ``document_ids=-1`` / ``loss_mask=False`` so padding is
    ignored by attention and loss. Absolute ``position_ids`` continue from the
    last real token.
    """
    if sp_size <= 1:
        return batch

    seq_len = None
    for key in _SEQUENCE_KEYS:
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[0] == 1:
            seq_len = value.shape[1]
            break
    if seq_len is None:
        return batch

    remainder = seq_len % sp_size
    if remainder == 0:
        return batch
    pad_len = sp_size - remainder

    out = dict(batch)
    for key in _SEQUENCE_KEYS:
        value = out.get(key)
        if not isinstance(value, torch.Tensor) or value.dim() < 2 or value.shape[0] != 1:
            continue
        pad_shape = list(value.shape)
        pad_shape[1] = pad_len
        if key == "document_ids":
            pad = value.new_full(pad_shape, -1)
        elif key == "loss_mask":
            pad = value.new_zeros(pad_shape)
        elif key == "position_ids":
            last = int(value[0, -1].item()) if value.numel() else 0
            pad = (
                torch.arange(
                    last + 1,
                    last + 1 + pad_len,
                    device=value.device,
                    dtype=value.dtype,
                )
                .unsqueeze(0)
            )
        else:
            pad = value.new_zeros(pad_shape)
        out[key] = torch.cat([value, pad], dim=1)

    if "lengths" in out and isinstance(out["lengths"], torch.Tensor):
        # Keep lengths unchanged: padding is not a real document.
        pass
    return out


def shard_batch_for_sp(
    batch: dict[str, Any],
    sp_rank: int,
    sp_size: int,
) -> dict[str, Any]:
    """Slice a packed ``[1, S, ...]`` batch to this rank's contiguous shard.

    Assumes ``S % sp_size == 0`` (call :func:`pad_seq_to_sp_multiple` first).
    Keeps absolute ``position_ids`` so RoPE stays globally consistent.
    """
    if sp_size <= 1:
        return batch

    seq_len = None
    for key in _SEQUENCE_KEYS:
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[0] == 1:
            seq_len = value.shape[1]
            break
    if seq_len is None:
        return batch
    if seq_len % sp_size != 0:
        raise ValueError(
            f"Sequence length {seq_len} is not divisible by sp_size={sp_size}. "
            "Call pad_seq_to_sp_multiple first."
        )

    local_len = seq_len // sp_size
    start = sp_rank * local_len
    end = start + local_len

    out: dict[str, Any] = {}
    for key, value in batch.items():
        if (
            key in _SEQUENCE_KEYS
            and isinstance(value, torch.Tensor)
            and value.dim() >= 2
            and value.shape[0] == 1
        ):
            out[key] = value[:, start:end]
        else:
            out[key] = value

    # Stash global metadata for loss/metrics alignment under SP.
    out["sp_global_seq_len"] = seq_len
    out["sp_global_start"] = start
    return out


def sync_and_shard_sp_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Broadcast the packed batch from SP rank 0, then pad and shard.

    Collate runs independently per rank (and in dataloader workers, so it cannot
    use NCCL). Rank 0 holds the real packed sequence; other SP ranks may have
    empty placeholders. Ulysses requires every rank to shard the *same* sequence.
    """
    from speculators.train.distributed import (  # noqa: PLC0415
        get_local_rank,
        get_rank,
        get_sp_group,
        get_sp_rank,
        get_sp_size,
    )

    import torch.distributed as dist  # noqa: PLC0415

    sp_size = get_sp_size()
    sp_rank = get_sp_rank()
    if sp_size <= 1:
        return batch

    src = get_rank() - sp_rank
    payload: list[Any] = [batch if sp_rank == 0 else None]
    group = get_sp_group()
    try:
        acc = torch.accelerator.current_accelerator()
        device = (
            torch.device(acc.type, get_local_rank()) if acc is not None else None
        )
        dist.broadcast_object_list(payload, src=src, group=group, device=device)
    except TypeError:
        dist.broadcast_object_list(payload, src=src, group=group)

    synced = payload[0]
    if not isinstance(synced, dict):
        raise RuntimeError(
            "SP batch broadcast failed (got no packed batch from rank 0)"
        )
    synced = pad_seq_to_sp_multiple(synced, sp_size)
    return shard_batch_for_sp(synced, sp_rank, sp_size)


def validate_sp_head_divisibility(
    num_attention_heads: int,
    num_key_value_heads: int,
    sp_size: int,
) -> None:
    """Raise if Ulysses SP cannot evenly shard attention heads."""
    if sp_size <= 1:
        return
    if num_attention_heads % sp_size != 0:
        raise ValueError(
            f"Ulysses SP requires num_attention_heads ({num_attention_heads}) "
            f"divisible by sp_size ({sp_size})."
        )
    if num_key_value_heads % sp_size != 0:
        raise ValueError(
            f"Ulysses SP requires num_key_value_heads ({num_key_value_heads}) "
            f"divisible by sp_size ({sp_size})."
        )
