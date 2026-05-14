# ruff: noqa: ERA001
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    or_masks,
)
from transformers.modeling_utils import AttentionInterface


def create_combined_mask_mod(lengths: torch.Tensor, total_seq_len: int):
    document_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=lengths.device, dtype=torch.long), lengths
    )
    # Pad ids with -1 to indicate padding
    document_ids = torch.cat(
        [
            document_ids,
            -1
            * torch.ones(
                total_seq_len - document_ids.shape[0],
                device=lengths.device,
                dtype=torch.long,
            ),
        ]
    ).contiguous()

    def causal_mask_mod(_b, _h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def document_mask_mod(_b, _h, q_idx, kv_idx):
        # Exclude padding tokens in attention mask
        return torch.logical_and(
            document_ids[q_idx] != -1,
            document_ids[q_idx] == document_ids[kv_idx % total_seq_len],
        )

    def diagonal_draft_mask_mod(_b, _h, q_idx, kv_idx):
        return kv_idx % total_seq_len == q_idx

    return or_masks(
        and_masks(causal_mask_mod, document_mask_mod), diagonal_draft_mask_mod
    )


def extend_mask_for_draft_tokens(block_mask):
    """
    Extend the block mask to include new draft tokens. Concatenates a diagonal mask for
    the new draft tokens.

    Assumptions:
    - block_mask BLOCK_SIZE := KV_BLOCK_SIZE == Q_BLOCK_SIZE
    - The number of query values is the original total_seq_len (or equivalently the
    number of query blocks is the original total_seq_len // BLOCK_SIZE)

    i.e. if block_mask is:
    [
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
        ]
    ]
    the result will be:
    [
        [
            [1, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
        ]
    ]
    and then calling again will give:
    [
        [
            [1, 0, 0, 1, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
        ]
    ]

    """
    kv_num_blocks = block_mask.kv_num_blocks
    # shape: [B, H, Q_LEN // BLOCK_SIZE]

    kv_indices = block_mask.kv_indices
    # shape: [B, H, Q_LEN // BLOCK_SIZE, KV_LEN // BLOCK_SIZE]
    b, h, q_blocks, kv_blocks = kv_indices.shape

    # extend kv indices if needed
    kv_indices = torch.cat(
        [kv_indices, kv_indices.new_zeros((b, h, q_blocks, q_blocks))], dim=-1
    )
    new_block_indices = torch.arange(
        kv_blocks,
        kv_blocks + q_blocks,
        dtype=kv_indices.dtype,
        device=kv_indices.device,
    ).reshape(1, 1, q_blocks, 1)
    kv_indices.scatter_(
        dim=-1, index=kv_num_blocks.unsqueeze(-1), src=new_block_indices
    )

    kv_num_blocks = kv_num_blocks + 1
    if block_mask.full_kv_indices is not None:
        extended_full_kv_indices = torch.cat(
            [
                block_mask.full_kv_indices,
                block_mask.full_kv_indices.new_zeros((b, h, q_blocks, q_blocks)),
            ],
            dim=-1,
        )
    else:
        extended_full_kv_indices = None
    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        block_mask.full_kv_num_blocks,
        extended_full_kv_indices,
        mask_mod=block_mask.mask_mod,
    )


def block_mask_to_dense_attention_mask(
    block_mask: BlockMask, device: torch.device, dtype: torch.dtype, cp_size: int = 1
):
    if cp_size < 1:
        raise ValueError(f"`cp_size` must be >= 1, got {cp_size}.")

    batch_size, num_heads, q_len, kv_len_local = block_mask.shape
    kv_len = kv_len_local * cp_size
    attention_mask = torch.ones(
        (batch_size, num_heads, q_len, kv_len),
        device=device,
        dtype=dtype,
    )
    kv_idx = torch.arange(kv_len, device=device, dtype=torch.long)
    kv_idx_for_mask = kv_idx % kv_len_local

    for batch_idx in range(batch_size):
        b = torch.tensor([batch_idx], device=device, dtype=torch.long)
        for head_idx in range(num_heads):
            h = torch.tensor([head_idx], device=device, dtype=torch.long)
            for q_idx in range(q_len):
                q = torch.tensor([q_idx], device=device, dtype=torch.long)
                attention_mask[batch_idx, head_idx, q_idx, :] = block_mask.mask_mod(
                    b,
                    h,
                    q,
                    kv_idx_for_mask,
                )
    return attention_mask


def flex_attention_forward(
    module: torch.nn.Module,  # noqa: ARG001
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    num_query_heads = query.shape[1]
    num_key_value_heads = key.shape[1]
    enable_gqa = num_query_heads != num_key_value_heads

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    cp_size = int(_kwargs.get("_cp_size", 1))
    sdpa_attention_mask = attention_mask
    if isinstance(attention_mask, BlockMask):
        sdpa_attention_mask = block_mask_to_dense_attention_mask(
            attention_mask,
            device=query.device,
            dtype=torch.bool,
            cp_size=cp_size,
        )

    attention_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=sdpa_attention_mask,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=enable_gqa,
        scale=scaling,
    )
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, None


ALL_ATTENTION_FUNCTIONS = AttentionInterface()  # Singleton class used for registry
ALL_ATTENTION_FUNCTIONS.register("simple_flex_attention", flex_attention_forward)
