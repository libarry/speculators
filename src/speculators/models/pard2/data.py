"""PARD-2 data utilities: parallel warp collator and preprocessing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

IGNORE_LABEL = -100
BatchType = dict[str, Any]

__all__ = [
    "IGNORE_LABEL",
    "BatchType",
    "apply_end_token_label_mask",
    "apply_parallel_warp",
    "build_prev_prob",
    "build_shifted_target_feat",
    "compute_teacher_gold_prob",
    "create_pard2_collate_fn",
    "preprocess_pard2_sample",
]


def slice_and_pad_to_length(tensor: torch.Tensor, length: int) -> torch.Tensor:
    sliced_tensor = tensor[:length]
    padding = [0, 0] * sliced_tensor.dim()
    padding[-1] = length - sliced_tensor.shape[0]
    return F.pad(sliced_tensor, padding)


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
) -> torch.Tensor:
    """Gold-token probability from verifier hidden states and lm_head."""
    logits = F.linear(
        teacher_hidden.float(),
        lm_head_weight,
        lm_head_bias,
    )
    teacher_gold_prob = torch.ones(
        input_ids.shape,
        dtype=torch.float32,
        device=input_ids.device,
    )
    if input_ids.shape[1] <= 1:
        return teacher_gold_prob

    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:].unsqueeze(-1)
    gold_log_prob = F.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=shift_labels,
    ).squeeze(-1)
    teacher_gold_prob[:, 1:] = gold_log_prob.exp()
    return teacher_gold_prob


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
    teacher_gold_prob: torch.Tensor,
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

    prev_prob = build_prev_prob(teacher_gold_prob, para_num)
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

    if down_sample_ratio != 1 and para_num > 1:
        index_mask = torch.zeros(para_num, single_length, dtype=torch.bool, device=input_ids.device)
        index_mask[0, :] = True
        prev_indices = torch.arange(single_length, device=input_ids.device)

        for i in range(1, para_num):
            num_ones = int(
                single_length * max(down_sample_ratio**i, down_sample_ratio_min)
            )
            if num_ones <= 0:
                break

            num_ones = min(num_ones, len(prev_indices))
            selected_indices = prev_indices[torch.randperm(len(prev_indices), device=input_ids.device)[:num_ones]]
            index_mask[i, selected_indices] = True
            prev_indices = (selected_indices + 1) % single_length

        index_mask = index_mask.reshape(-1)
        indices = index_mask.nonzero(as_tuple=True)[0]

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
            "attention_mask": new_data["attention_mask"][:, :, indices, :][:, :, :, indices].contiguous(),
        }
        if "target_feat" in new_data:
            filtered["target_feat"] = new_data["target_feat"][:, indices].contiguous()
        if "teacher_hidden" in new_data:
            filtered["teacher_hidden"] = new_data["teacher_hidden"][:, indices].contiguous()
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
    if teacher_hidden is not None and teacher_hidden.dim() == 2:
        teacher_hidden = teacher_hidden.squeeze(0)

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
        "lengths": lengths,
        "position_ids": position_ids,
    }


def create_pard2_collate_fn(
    max_len: int,
    hidden_size: int,
    num_target_layers: int,
    para_num: int,
    unused_tokenids: list[int],
    down_sample_ratio: float,
    down_sample_ratio_min: float,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    dtype: torch.dtype = torch.bfloat16,
    preprocess: Callable[[BatchType], BatchType] | None = None,
):
    """Collate exactly one sample per step and apply parallel warp."""

    def collate_fn(batch: list[BatchType | None]) -> BatchType:
        batch = [preprocess(b) if preprocess else b for b in batch if b is not None]
        if not batch:
            empty = create_empty_sample(hidden_size, num_target_layers, dtype=dtype)
            if preprocess:
                empty = preprocess(empty)
            batch = [empty]

        sample = batch[0]
        input_ids = sample["input_ids"].unsqueeze(0)
        labels = sample["labels"].unsqueeze(0)

        target_feat = sample.get("target_feat")
        if target_feat is not None:
            target_feat = target_feat.unsqueeze(0)

        teacher_hidden = sample.get("teacher_hidden")
        if teacher_hidden is not None:
            teacher_hidden = teacher_hidden.unsqueeze(0)
            teacher_gold_prob = compute_teacher_gold_prob(
                input_ids,
                teacher_hidden,
                lm_head_weight.to(teacher_hidden.device, dtype=torch.float32),
                lm_head_bias.to(teacher_hidden.device, dtype=torch.float32) if lm_head_bias is not None else None,
            )
        else:
            teacher_gold_prob = torch.ones_like(input_ids, dtype=torch.float32)

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

        for key, tensor in list(warped.items()):
            if key == "attention_mask" or tensor is None:
                continue
            if tensor.dim() == 2:
                warped[key] = slice_and_pad_to_length(tensor.squeeze(0), max_len).unsqueeze(0)
            elif tensor.dim() == 3:
                warped[key] = slice_and_pad_to_length(tensor.squeeze(0), max_len).unsqueeze(0)

        seq_len = warped["input_ids"].shape[1]
        if "attention_mask" in warped and warped["attention_mask"] is not None:
            warped["attention_mask"] = align_attention_mask(
                warped["attention_mask"], seq_len
            )

        warped["lengths"] = torch.tensor([seq_len], dtype=torch.long)
        return warped

    return collate_fn
