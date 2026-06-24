"""Numerical parity tests against PARD reference implementations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch")

_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "speculators"
    / "models"
    / "pard2"
    / "data.py"
)
_spec = importlib.util.spec_from_file_location("pard2_data_under_test", _DATA_PATH)
assert _spec and _spec.loader
_data_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data_mod)

build_prev_prob = _data_mod.build_prev_prob
apply_parallel_warp = _data_mod.apply_parallel_warp
compute_teacher_gold_prob = _data_mod.compute_teacher_gold_prob

def _pard_reference_build_prev_prob(
    teacher_gold_prob: torch.Tensor, para_num: int
) -> torch.Tensor:
    """Reference copy of PARD WarpDataCollator._build_prev_prob."""
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


def test_build_prev_prob_matches_pard():
    teacher_gold_prob = torch.tensor(
        [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]],
        dtype=torch.float32,
    )
    para_num = 4
    ours = build_prev_prob(teacher_gold_prob, para_num)
    theirs = _pard_reference_build_prev_prob(teacher_gold_prob, para_num)
    assert torch.allclose(ours, theirs)


def test_cod_indices_match_pard():
    torch.manual_seed(0)
    single_length = 16
    para_num = 4
    down_sample_ration = 0.7
    down_sample_ration_min = 0.1

    input_ids = torch.arange(single_length).unsqueeze(0)
    labels = input_ids.clone()
    teacher_gold_prob = torch.ones_like(input_ids, dtype=torch.float32)

    ours = apply_parallel_warp(
        input_ids=input_ids,
        labels=labels,
        target_feat=None,
        teacher_hidden=None,
        teacher_gold_prob=teacher_gold_prob,
        para_num=para_num,
        unused_tokenids=[99, 100, 101],
        down_sample_ratio=down_sample_ration,
        down_sample_ratio_min=down_sample_ration_min,
    )

    index_mask = torch.zeros(para_num, single_length, dtype=torch.bool)
    index_mask[0, :] = True
    prev_indices = torch.arange(single_length)
    for i in range(1, para_num):
        num_ones = int(
            single_length * max(down_sample_ration**i, down_sample_ration_min)
        )
        if num_ones <= 0:
            break
        num_ones = min(num_ones, len(prev_indices))
        selected_indices = prev_indices[torch.randperm(len(prev_indices))[:num_ones]]
        index_mask[i, selected_indices] = True
        prev_indices = (selected_indices + 1) % single_length

    expected_len = index_mask.reshape(-1).nonzero(as_tuple=True)[0].numel()
    assert ours["input_ids"].shape[1] == expected_len


def test_teacher_gold_prob_shape():
    batch, seq, hidden, vocab = 1, 8, 16, 32
    input_ids = torch.randint(0, vocab, (batch, seq))
    teacher_hidden = torch.randn(batch, seq, hidden)
    weight = torch.randn(vocab, hidden)
    probs = compute_teacher_gold_prob(input_ids, teacher_hidden, weight)
    assert probs.shape == input_ids.shape
    assert torch.all(probs[:, 0] == 1.0)


def test_parallel_attention_mask_blocks():
    warped = apply_parallel_warp(
        input_ids=torch.zeros(1, 4, dtype=torch.long),
        labels=torch.zeros(1, 4, dtype=torch.long),
        target_feat=None,
        teacher_hidden=None,
        teacher_gold_prob=torch.ones(1, 4),
        para_num=2,
        unused_tokenids=[7],
        down_sample_ratio=1.0,
    )
    mask = warped["attention_mask"]
    min_val = torch.finfo(torch.float32).min
    assert mask.shape == (1, 1, 8, 8)
    # Block-diagonal parallel mask: position attends only to itself within each block.
    assert mask[0, 0, 0, 0] == 0
    assert mask[0, 0, 4, 4] == 0
    assert mask[0, 0, 0, 4] == min_val
    assert mask[0, 0, 0, 1] == min_val


def test_align_attention_mask_pads_after_cod_downsample():
    torch.manual_seed(0)
    single_length = 26
    para_num = 16
    max_len = 128

    warped = apply_parallel_warp(
        input_ids=torch.arange(single_length).unsqueeze(0),
        labels=torch.arange(single_length).unsqueeze(0),
        target_feat=None,
        teacher_hidden=None,
        teacher_gold_prob=torch.ones(1, single_length),
        para_num=para_num,
        unused_tokenids=list(range(100, 115)),
        down_sample_ratio=0.7,
        down_sample_ratio_min=0.1,
    )
    warped_len = warped["input_ids"].shape[1]
    assert warped_len == 95
    assert warped_len < max_len

    aligned = _data_mod.align_attention_mask(warped["attention_mask"], max_len)
    assert aligned.shape == (1, 1, max_len, max_len)
    min_val = torch.finfo(aligned.dtype).min
    assert aligned[0, 0, warped_len:, :].eq(min_val).all()
    assert aligned[0, 0, :, warped_len:].eq(min_val).all()
    assert torch.equal(
        aligned[0, 0, :warped_len, :warped_len],
        warped["attention_mask"][0, 0, :warped_len, :warped_len],
    )


def test_collate_pads_prev_prob_to_max_len():
    torch.manual_seed(0)
    max_len = 128
    hidden_size = 32
    collate = _data_mod.create_pard2_collate_fn(
        max_len=max_len,
        hidden_size=hidden_size,
        num_target_layers=4,
        para_num=16,
        unused_tokenids=list(range(100, 115)),
        down_sample_ratio=0.7,
        down_sample_ratio_min=0.1,
        lm_head_weight=torch.randn(64, hidden_size),
        lm_head_bias=None,
    )
    seq_len = 26
    sample = {
        "input_ids": torch.arange(seq_len, dtype=torch.long),
        "labels": torch.arange(seq_len, dtype=torch.long),
        "teacher_hidden": torch.randn(seq_len, hidden_size),
        "target_feat": torch.randn(seq_len, hidden_size * 4),
    }
    batch = collate([sample])
    assert batch["input_ids"].shape[1] == max_len
    assert batch["labels"].shape[1] == max_len
    assert batch["prev_prob"].shape[1] == max_len
    assert batch["attention_mask"].shape[-1] == max_len
