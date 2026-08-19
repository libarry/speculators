"""Utility functions for DFlash draft model."""

import torch


def get_base_indices_for_anchored_blocks(
    anchor_positions: torch.Tensor,  # shape: [1, num_anchors]
    block_size: int,
) -> torch.Tensor:  # shape: [num_anchors*block_size]
    anchor_positions = anchor_positions.to(dtype=torch.long).view(-1)
    # dtype: long, shape: [num_anchors]

    offsets = torch.arange(block_size, device=anchor_positions.device, dtype=torch.long)
    idx = (
        anchor_positions[:, None] + offsets[None, :]
    )  # shape: [num_anchors, block_size]

    return idx.reshape(-1)


def select_anchors(
    loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
    num_anchors: int,
    block_size: int,
    *,
    global_start: int = 0,
    global_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly select anchor positions from valid tokens in sequence.

    Args:
        loss_mask: Binary mask indicating valid positions [1, local_seq_len]
        num_anchors: Number of anchors to select
        block_size: Block size. A block must fit inside the *global* sequence
            (not just this SP shard): ``global_pos + block_size <= global_seq_len``.
        global_start: Global index of local position 0 (SP shard offset).
        global_seq_len: Full packed length before sharding. Defaults to local length.

    Returns:
        tuple: (anchors, anchor_valid)
            - anchors: Selected *local* anchor indices [num_anchors]
            - anchor_valid: Boolean mask for valid anchors [num_anchors]
    """
    if loss_mask.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"Expected [B, T], got {loss_mask.shape}")

    if block_size <= 0:
        raise ValueError(f"Expected block size > 0, got {block_size}")

    seq_len = loss_mask.shape[1]
    if global_seq_len is None:
        global_seq_len = seq_len

    valid_mask = loss_mask.bool().clone()
    local_idx = torch.arange(seq_len, device=loss_mask.device)
    global_idx = local_idx + int(global_start)
    # Block [p, p+block_size) must fit in this shard *and* in the global sequence.
    valid_mask &= (local_idx + block_size) <= seq_len
    valid_mask &= (global_idx + block_size) <= global_seq_len

    valid_indices = torch.nonzero(valid_mask.squeeze(0), as_tuple=False).squeeze(
        -1
    )  # shape: [num_non_zero]

    device = loss_mask.device
    anchors = torch.zeros(num_anchors, dtype=torch.long, device=device)
    anchor_valid = torch.zeros(num_anchors, dtype=torch.bool, device=device)

    k = min(num_anchors, valid_indices.numel())

    # Constrain value of k for torch dynamo
    torch._check(k <= valid_indices.numel())  # noqa: SLF001
    torch._check(k >= 0)  # noqa: SLF001

    perm = torch.randperm(valid_indices.numel(), device=loss_mask.device)
    # Contiguous anchors let flex attention use dense (fast) blocks instead of
    # scattered all-partial (slow) ones; the order never affects the loss.
    anchors[:k] = torch.sort(torch.gather(valid_indices, 0, perm[:k])).values
    anchor_valid[:k] = True

    return anchors, anchor_valid
    # shape: [num_anchors], [num_anchors]
