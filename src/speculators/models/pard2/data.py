"""PARD-2 data utilities: parallel warp collator and preprocessing."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

from speculators.models.pard2.loss_ops import (
    DEFAULT_VOCAB_CHUNK_SIZE,
    apply_rms_norm,
    gather_label_log_prob_chunked,
)

IGNORE_LABEL = -100
BatchType = dict[str, Any]

__all__ = [
    "IGNORE_LABEL",
    "BatchType",
    "apply_end_token_label_mask",
    "apply_parallel_warp",
    "build_pard2_prev_prob_batch",
    "build_prev_prob",
    "build_shifted_target_feat",
    "compute_pard2_warped_length",
    "compute_teacher_gold_prob",
    "create_pard2_collate_fn",
    "create_pard2_collate_fn_from_draft_model",
    "preprocess_pard2_sample",
]


def slice_and_pad_to_length(
    tensor: torch.Tensor,
    length: int,
    value: float = 0,
) -> torch.Tensor:
    sliced_tensor = tensor[:length]
    padding = [0, 0] * sliced_tensor.dim()
    padding[-1] = length - sliced_tensor.shape[0]
    return F.pad(sliced_tensor, padding, value=value)


def compute_pard2_warped_length(
    source_len: int,
    para_num: int,
    down_sample_ratio: float,
    down_sample_ratio_min: float,
) -> int:
    """Maximum sequence length after PARD-2 parallel warp and COD.

    PARD applies ``max_seq_length`` before the parallel warp. The warped training
    sequence is longer, so speculators must budget for the post-warp length
    instead of truncating it back to the pre-warp token length.
    """
    if para_num <= 1:
        return source_len
    if down_sample_ratio == 1:
        return source_len * para_num

    total = source_len
    prev_count = source_len
    for i in range(1, para_num):
        count = int(source_len * max(down_sample_ratio**i, down_sample_ratio_min))
        if count <= 0:
            break
        count = min(count, prev_count)
        total += count
        prev_count = count
    return total


def align_attention_mask(attn: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resize a 4D mask to [B, 1, target_len, target_len]."""
    cur_len = attn.shape[-1]
    if cur_len == target_len:
        return attn
    if cur_len > target_len:
        return attn[:, :, :target_len, :target_len]
    min_val = torch.finfo(attn.dtype).min
    pad = target_len - cur_len
    return F.pad(attn, (0, pad, 0, pad), value=min_val)


def create_empty_sample(
    hidden_size: int,
    num_target_layers: int = 3,
    dtype: torch.dtype = torch.bfloat16,
) -> BatchType:
    return {
        "hidden_states": torch.empty(0, num_target_layers * hidden_size, dtype=dtype),
        "input_ids": torch.empty(0, dtype=torch.long),
        "verifier_last_hidden_states": torch.empty(0, hidden_size, dtype=dtype),
        "loss_mask": torch.empty(0, dtype=torch.bool),
        "lengths": torch.tensor([0], dtype=torch.long),
        "position_ids": torch.arange(0, dtype=torch.long),
    }


def build_prev_prob(teacher_gold_prob: torch.Tensor, para_num: int) -> torch.Tensor:
    """CAT cumulative gold-token probability weights across parallel blocks."""
    teacher_gold_prob = teacher_gold_prob.float()
    ones = torch.ones_like(teacher_gold_prob, dtype=torch.float32)
    blocks = []

    for i in range(para_num):
        block = ones.clone()
        for s in range(1, i + 1):
            shifted = torch.cat(
                [ones[:, :s], teacher_gold_prob[:, :-s]],
                dim=1,
            )
            block = block * shifted
        blocks.append(block)

    return torch.cat(blocks, dim=1)


def compute_teacher_gold_prob(
    input_ids: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None = None,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
) -> torch.Tensor:
    """Gold-token probability from verifier hidden states and lm_head.

    For each position t>0, computes P_teacher(x_t | h_{t-1}) via the verifier
    final RMSNorm + lm_head (equivalent to official PARD ``outputs.logits``).
    Used to build CAT ``prev_prob`` weights for CE/KD loss.
    """
    teacher_gold_prob = torch.ones(
        input_ids.shape,
        dtype=torch.float32,
        device=input_ids.device,
    )
    if input_ids.shape[1] <= 1:
        return teacher_gold_prob

    if norm_weight is not None:
        teacher_hidden = apply_rms_norm(teacher_hidden, norm_weight, eps=norm_eps)

    shift_hidden = teacher_hidden[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    gold_log_prob = gather_label_log_prob_chunked(
        shift_hidden,
        shift_labels,
        lm_head_weight,
        lm_head_bias,
        vocab_chunk_size=vocab_chunk_size,
    )
    teacher_gold_prob[:, 1:] = gold_log_prob.exp()
    return teacher_gold_prob


def _build_warp_block_labels(labels: torch.Tensor, para_num: int) -> torch.Tensor:
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    return torch.cat(
        [
            torch.cat(
                [labels[:, :i] * 0 + IGNORE_LABEL, labels[:, i:]],
                dim=1,
            )
            for i in range(para_num)
        ],
        dim=1,
    )


def _apply_cod_indices_to_prev_prob(
    prev_prob: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return torch.roll(
        torch.roll(prev_prob, shifts=-1, dims=1)[:, indices],
        shifts=1,
        dims=1,
    )


def _pad_warp_indices_batch(
    indices_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not indices_list:
        return torch.empty(0, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    lengths = torch.tensor([idx.numel() for idx in indices_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.full((len(indices_list), max_len), -1, dtype=torch.long)
    for row, indices in enumerate(indices_list):
        padded[row, : indices.numel()] = indices.cpu()
    return padded, lengths


def build_pard2_prev_prob_batch(
    gold_input_ids: torch.Tensor,
    gold_teacher_hidden: torch.Tensor,
    gold_labels: torch.Tensor,
    gold_seq_len: torch.Tensor,
    warp_indices: torch.Tensor,
    warp_indices_len: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    para_num: int,
    target_len: int,
    *,
    device: torch.device,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    gold_teacher_gold_prob: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build warped ``prev_prob`` on ``device`` (deferred from CPU collate)."""
    batch_size = gold_input_ids.shape[0]
    prev_prob = torch.zeros(
        batch_size,
        target_len,
        dtype=torch.float32,
        device=device,
    )

    if para_num == 1:
        for batch_idx in range(batch_size):
            seq_len = int(gold_seq_len[batch_idx].item())
            if seq_len > 0:
                copy_len = min(seq_len, target_len)
                prev_prob[batch_idx, :copy_len] = 1.0
        return prev_prob

    for batch_idx in range(batch_size):
        seq_len = int(gold_seq_len[batch_idx].item())
        if seq_len <= 0:
            continue

        ids = gold_input_ids[batch_idx : batch_idx + 1, :seq_len].to(device)
        hidden = gold_teacher_hidden[batch_idx : batch_idx + 1, :seq_len].to(device)
        labels = gold_labels[batch_idx : batch_idx + 1, :seq_len].to(device)

        if gold_teacher_gold_prob is not None:
            gold_prob = gold_teacher_gold_prob[
                batch_idx : batch_idx + 1, :seq_len
            ].to(device=device, dtype=torch.float32)
        else:
            gold_prob = compute_teacher_gold_prob(
                ids,
                hidden,
                lm_head_weight,
                lm_head_bias,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )
        prev = build_prev_prob(gold_prob, para_num)
        warp_labels = _build_warp_block_labels(labels, para_num)
        prev = prev.masked_fill(warp_labels == IGNORE_LABEL, 0.0)

        idx_len = int(warp_indices_len[batch_idx].item())
        indices = warp_indices[batch_idx, :idx_len].to(device)
        prev = _apply_cod_indices_to_prev_prob(prev, indices)
        # Match collate: slice_and_pad_to_length truncates warped seq to max_len.
        copy_len = min(prev.shape[1], target_len)
        prev_prob[batch_idx, :copy_len] = prev.squeeze(0)[:copy_len]

    return prev_prob


def build_shifted_target_feat(multi_layer_hidden: torch.Tensor) -> torch.Tensor:
    """Shift concatenated target layers: [zeros@t0, feat@t1..T-1]."""
    if multi_layer_hidden.dim() == 2:
        multi_layer_hidden = multi_layer_hidden.unsqueeze(0)
    zeros = torch.zeros_like(multi_layer_hidden[:, :1])
    return torch.cat([zeros, multi_layer_hidden[:, :-1]], dim=1)


def apply_end_token_label_mask(
    labels: torch.Tensor,
    end_token_id: int | None,
) -> torch.Tensor:
    if end_token_id is None:
        return labels
    labels = labels.clone()
    batch_size, _ = labels.shape
    for i in range(batch_size):
        pos = (labels[i] == end_token_id).nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            continue
        labels[i, : pos[-1].item()] = IGNORE_LABEL
    return labels


def apply_parallel_warp(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    target_feat: torch.Tensor | None,
    teacher_hidden: torch.Tensor | None,
    teacher_gold_prob: torch.Tensor | None,
    para_num: int,
    unused_tokenids: list[int],
    down_sample_ratio: float = 1.0,
    down_sample_ratio_min: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Replicate sequence into parallel blocks with optional COD downsampling."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)

    single_length = input_ids.shape[1]

    if para_num == 1:
        out: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "prev_prob": torch.ones_like(labels, dtype=torch.float32),
            "position_ids": torch.arange(single_length, dtype=torch.long).unsqueeze(0),
        }
        if target_feat is not None:
            out["target_feat"] = target_feat
        if teacher_hidden is not None:
            out["teacher_hidden"] = teacher_hidden
        return out

    if len(unused_tokenids) < para_num - 1:
        raise ValueError(
            f"mask_token_ids length ({len(unused_tokenids)}) < para_num-1 ({para_num - 1})"
        )

    tgt_len = single_length * para_num
    mask = torch.full((tgt_len, tgt_len), torch.finfo(torch.float32).min)
    mask_cond = torch.arange(mask.size(-1), device=input_ids.device)

    tmp_mask = mask_cond == mask_cond.view(mask.size(-1), 1)
    for i in range(para_num):
        tmp_mask = tmp_mask | (
            mask_cond == (mask_cond - single_length * i - i).view(mask.size(-1), 1)
        )
        tmp_mask = tmp_mask | (
            (mask_cond < (mask_cond - i * single_length - (i - 1)).view(mask.size(-1), 1))
            & (mask_cond < (i + 1) * single_length).view(-1, 1)
        )

    mask.masked_fill_(tmp_mask, 0)
    mask = mask[None, None, :, :]

    bs = input_ids.shape[0]
    new_labels = torch.cat(
        [
            torch.cat(
                [labels[:, :i] * 0 + IGNORE_LABEL, labels[:, i:]],
                dim=1,
            )
            for i in range(para_num)
        ],
        dim=1,
    )

    prev_prob = (
        build_prev_prob(teacher_gold_prob, para_num)
        if teacher_gold_prob is not None
        else torch.ones(bs, tgt_len, dtype=torch.float32, device=input_ids.device)
    )
    prev_prob = prev_prob.masked_fill(new_labels == IGNORE_LABEL, 0.0)

    new_data: dict[str, torch.Tensor] = {
        "input_ids": torch.cat(
            [
                input_ids,
                torch.cat(
                    [
                        input_ids * 0 + unused_tokenids[i]
                        for i in range(para_num - 1)
                    ],
                    dim=1,
                ),
            ],
            dim=1,
        ),
        "attention_mask": torch.cat([mask for _ in range(bs)], dim=0),
        "position_ids": torch.cat(
            [torch.arange(single_length, dtype=torch.long, device=input_ids.device) for _ in range(para_num)],
            dim=0,
        ).unsqueeze(0).repeat(bs, 1),
        "labels": new_labels,
        "prev_prob": prev_prob,
    }

    if target_feat is not None:
        new_data["target_feat"] = torch.cat(
            [
                torch.cat(
                    [
                        target_feat[:, :i] * 0,
                        target_feat[:, : target_feat.shape[1] - i],
                    ],
                    dim=1,
                )
                for i in range(para_num)
            ],
            dim=1,
        )

    if teacher_hidden is not None:
        new_data["teacher_hidden"] = torch.cat(
            [teacher_hidden for _ in range(para_num)],
            dim=1,
        )

    use_cod = down_sample_ratio != 1 and para_num > 1
    if para_num > 1:
        new_data["gold_input_ids"] = input_ids
        new_data["gold_labels"] = labels
        new_data["warp_single_length"] = torch.tensor(single_length, dtype=torch.long)
        if teacher_hidden is not None:
            new_data["gold_teacher_hidden"] = teacher_hidden
        if (
            teacher_gold_prob is not None
            and teacher_gold_prob.dim() == 2
            and teacher_gold_prob.shape == input_ids.shape
        ):
            new_data["gold_teacher_gold_prob"] = teacher_gold_prob
        if use_cod:
            index_mask = torch.zeros(
                para_num, single_length, dtype=torch.bool, device=input_ids.device
            )
            index_mask[0, :] = True
            prev_indices = torch.arange(single_length, device=input_ids.device)

            for i in range(1, para_num):
                num_ones = int(
                    single_length * max(down_sample_ratio**i, down_sample_ratio_min)
                )
                if num_ones <= 0:
                    break

                num_ones = min(num_ones, len(prev_indices))
                selected_indices = prev_indices[
                    torch.randperm(len(prev_indices), device=input_ids.device)[:num_ones]
                ]
                index_mask[i, selected_indices] = True
                prev_indices = (selected_indices + 1) % single_length

            indices = index_mask.reshape(-1).nonzero(as_tuple=True)[0]
        else:
            indices = torch.arange(tgt_len, device=input_ids.device)

        new_data["warp_indices"] = indices

        if use_cod:
            filtered: dict[str, torch.Tensor] = {
                "input_ids": new_data["input_ids"][:, indices].contiguous(),
                "position_ids": new_data["position_ids"][:, indices].contiguous(),
                "labels": torch.roll(
                    torch.roll(new_data["labels"], shifts=-1, dims=1)[:, indices],
                    shifts=1,
                    dims=1,
                ).contiguous(),
                "prev_prob": torch.roll(
                    torch.roll(new_data["prev_prob"], shifts=-1, dims=1)[:, indices],
                    shifts=1,
                    dims=1,
                ).contiguous(),
                "attention_mask": new_data["attention_mask"][:, :, indices, :][
                    :, :, :, indices
                ].contiguous(),
                "gold_input_ids": new_data["gold_input_ids"],
                "gold_labels": new_data["gold_labels"],
                "gold_teacher_hidden": new_data.get("gold_teacher_hidden"),
                "gold_teacher_gold_prob": new_data.get("gold_teacher_gold_prob"),
                "warp_single_length": new_data["warp_single_length"],
                "warp_indices": new_data["warp_indices"],
            }
            if "target_feat" in new_data:
                filtered["target_feat"] = new_data["target_feat"][:, indices].contiguous()
            if "teacher_hidden" in new_data:
                filtered["teacher_hidden"] = new_data["teacher_hidden"][
                    :, indices
                ].contiguous()
            new_data = filtered

    return new_data


def preprocess_pard2_sample(
    batch: BatchType,
    end_token_id: int | None = None,
) -> BatchType:
    """Prepare a single sample: labels, shifted target features."""
    input_ids = batch["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)

    loss_mask = batch.get("loss_mask")
    if loss_mask is None:
        loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    elif not isinstance(loss_mask, torch.Tensor):
        loss_mask = torch.tensor(loss_mask, dtype=torch.bool)
    elif loss_mask.dim() == 2:
        loss_mask = loss_mask.squeeze(0)

    labels = input_ids.clone()
    labels[loss_mask == 0] = IGNORE_LABEL
    labels = apply_end_token_label_mask(labels.unsqueeze(0), end_token_id).squeeze(0)

    multi_layer = batch.get("multi_layer_hidden_states")
    if multi_layer is None:
        multi_layer = batch.get("hidden_states")
    if multi_layer is not None and multi_layer.dim() == 2:
        target_feat = build_shifted_target_feat(multi_layer).squeeze(0)
    else:
        target_feat = None

    teacher_hidden = batch.get("verifier_last_hidden_states")
    if batch.get("verifier_kd_hidden") is not None:
        teacher_hidden = batch["verifier_kd_hidden"]
    if teacher_hidden is not None and teacher_hidden.dim() == 2:
        teacher_hidden = teacher_hidden.squeeze(0)

    teacher_gold_prob = batch.get("teacher_gold_prob")
    if teacher_gold_prob is not None and teacher_gold_prob.dim() == 1:
        teacher_gold_prob = teacher_gold_prob.unsqueeze(0)

    seq_len = input_ids.shape[0]
    lengths = batch.get("lengths", torch.tensor([seq_len], dtype=torch.long))
    if isinstance(lengths, torch.Tensor) and lengths.numel() == 1:
        lengths = lengths.clone()
    else:
        lengths = torch.tensor([seq_len], dtype=torch.long)

    position_ids = batch.get("position_ids")
    if position_ids is None:
        position_ids = torch.arange(seq_len, dtype=torch.long)
    elif position_ids.dim() == 2:
        position_ids = position_ids.squeeze(0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "target_feat": target_feat,
        "teacher_hidden": teacher_hidden,
        "teacher_gold_prob": teacher_gold_prob,
        "lengths": lengths,
        "position_ids": position_ids,
    }


def _warp_and_pad_sample(
    sample: BatchType,
    *,
    max_len: int,
    gold_max_len: int,
    para_num: int,
    unused_tokenids: list[int],
    down_sample_ratio: float,
    down_sample_ratio_min: float,
) -> BatchType:
    """Apply parallel warp to one sample and pad tensors to max_len."""
    input_ids = sample["input_ids"].unsqueeze(0)
    labels = sample["labels"].unsqueeze(0)

    target_feat = sample.get("target_feat")
    if target_feat is not None:
        target_feat = target_feat.unsqueeze(0)

    teacher_hidden = sample.get("teacher_hidden")
    if teacher_hidden is not None:
        teacher_hidden = teacher_hidden.unsqueeze(0)

    teacher_gold_prob = sample.get("teacher_gold_prob")
    if teacher_gold_prob is not None and teacher_gold_prob.dim() == 1:
        teacher_gold_prob = teacher_gold_prob.unsqueeze(0)

    warped = apply_parallel_warp(
        input_ids=input_ids,
        labels=labels,
        target_feat=target_feat,
        teacher_hidden=teacher_hidden,
        teacher_gold_prob=teacher_gold_prob,
        para_num=para_num,
        unused_tokenids=unused_tokenids,
        down_sample_ratio=down_sample_ratio,
        down_sample_ratio_min=down_sample_ratio_min,
    )

    gold_seq_len = torch.tensor([input_ids.shape[1]], dtype=torch.long)
    if "gold_input_ids" in warped:
        warped["gold_input_ids"] = slice_and_pad_to_length(
            warped["gold_input_ids"].squeeze(0), gold_max_len
        ).unsqueeze(0)
        warped["gold_labels"] = slice_and_pad_to_length(
            warped["gold_labels"].squeeze(0), gold_max_len, value=IGNORE_LABEL
        ).unsqueeze(0)
        if "gold_teacher_hidden" in warped and warped["gold_teacher_hidden"] is not None:
            warped["gold_teacher_hidden"] = slice_and_pad_to_length(
                warped["gold_teacher_hidden"].squeeze(0), gold_max_len
            ).unsqueeze(0)
        if "gold_teacher_gold_prob" in warped and warped["gold_teacher_gold_prob"] is not None:
            warped["gold_teacher_gold_prob"] = slice_and_pad_to_length(
                warped["gold_teacher_gold_prob"].squeeze(0), gold_max_len, value=1.0
            ).unsqueeze(0)
        warped["gold_seq_len"] = gold_seq_len

    skip_pad = {
        "attention_mask",
        "warp_single_length",
        "warp_indices",
        "gold_input_ids",
        "gold_labels",
        "gold_teacher_hidden",
        "gold_teacher_gold_prob",
        "gold_seq_len",
    }
    for key, tensor in list(warped.items()):
        if key in skip_pad or tensor is None:
            continue
        pad_value = IGNORE_LABEL if key == "labels" else 0
        if tensor.dim() == 2:
            warped[key] = slice_and_pad_to_length(
                tensor.squeeze(0), max_len, value=pad_value
            ).unsqueeze(0)
        elif tensor.dim() == 3:
            warped[key] = slice_and_pad_to_length(
                tensor.squeeze(0), max_len, value=pad_value
            ).unsqueeze(0)

    seq_len = warped["input_ids"].shape[1]
    if "attention_mask" in warped and warped["attention_mask"] is not None:
        warped["attention_mask"] = align_attention_mask(
            warped["attention_mask"], seq_len
        )

    warped["lengths"] = torch.tensor([seq_len], dtype=torch.long)
    return warped


def _merge_warped_samples(warped_samples: list[BatchType]) -> BatchType:
    """Concatenate per-sample warped batches along the batch dimension."""
    merged: BatchType = {}
    skip_merge = {
        "warp_single_length",
        "warp_indices",
        "gold_seq_len",
    }
    for key in warped_samples[0]:
        if key in skip_merge:
            continue
        values = [sample[key] for sample in warped_samples if sample.get(key) is not None]
        if not values:
            continue
        if key == "lengths":
            merged[key] = torch.cat(values, dim=0)
        else:
            merged[key] = torch.cat(values, dim=0)

    if "warp_indices" in warped_samples[0]:
        indices_list = [sample["warp_indices"].reshape(-1) for sample in warped_samples]
        merged["warp_indices"], merged["warp_indices_len"] = _pad_warp_indices_batch(
            indices_list
        )
        merged["warp_single_length"] = torch.stack(
            [sample["warp_single_length"].reshape(()) for sample in warped_samples],
            dim=0,
        )
    if "gold_seq_len" in warped_samples[0]:
        merged["gold_seq_len"] = torch.stack(
            [
                sample["gold_seq_len"].reshape(-1)[0]
                for sample in warped_samples
            ],
            dim=0,
        )
    return merged


def create_pard2_collate_fn_from_draft_model(
    draft_model: Any,
    *,
    max_len: int,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
):
    """Build collate_fn from a trained/initialized Pard2DraftModel config."""
    cfg = draft_model.config
    warped_max_len = compute_pard2_warped_length(
        max_len,
        cfg.para_num,
        cfg.down_sample_ratio,
        cfg.down_sample_ratio_min,
    )
    preprocess = partial(
        preprocess_pard2_sample,
        end_token_id=cfg.end_token_id,
    )
    return create_pard2_collate_fn(
        max_len=warped_max_len,
        gold_max_len=max_len,
        hidden_size=hidden_size,
        num_target_layers=len(cfg.target_layer_ids),
        para_num=cfg.para_num,
        unused_tokenids=list(cfg.mask_token_ids),
        down_sample_ratio=cfg.down_sample_ratio,
        down_sample_ratio_min=cfg.down_sample_ratio_min,
        dtype=dtype,
        preprocess=preprocess,
    )


def create_pard2_collate_fn(
    max_len: int,
    hidden_size: int,
    num_target_layers: int,
    para_num: int,
    unused_tokenids: list[int],
    down_sample_ratio: float,
    down_sample_ratio_min: float,
    dtype: torch.dtype = torch.bfloat16,
    preprocess: Callable[[BatchType], BatchType] | None = None,
    gold_max_len: int | None = None,
):
    """Collate one or more samples per step and apply parallel warp."""
    gold_max_len = max_len if gold_max_len is None else gold_max_len

    def collate_fn(batch: list[BatchType | None]) -> BatchType:
        batch = [preprocess(b) if preprocess else b for b in batch if b is not None]
        if not batch:
            empty = create_empty_sample(hidden_size, num_target_layers, dtype=dtype)
            if preprocess:
                empty = preprocess(empty)
            batch = [empty]

        warped_samples = [
            _warp_and_pad_sample(
                sample,
                max_len=max_len,
                gold_max_len=gold_max_len,
                para_num=para_num,
                unused_tokenids=unused_tokenids,
                down_sample_ratio=down_sample_ratio,
                down_sample_ratio_min=down_sample_ratio_min,
            )
            for sample in batch
        ]
        return _merge_warped_samples(warped_samples)

    return collate_fn
