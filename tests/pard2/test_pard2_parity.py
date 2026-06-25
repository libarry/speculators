"""Numerical parity tests against PARD reference implementations."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
compute_pard2_warped_length = _data_mod.compute_pard2_warped_length

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


def test_compute_pard2_warped_length_matches_cod_retention_budget():
    assert compute_pard2_warped_length(26, 16, 0.7, 0.1) == 95
    assert compute_pard2_warped_length(128, 2, 1.0, 0.0) == 256


def test_collate_multi_sample_batch_has_batch_dim():
    torch.manual_seed(0)
    max_len = 128
    hidden_size = 32
    collate = _data_mod.create_pard2_collate_fn(
        max_len=max_len,
        hidden_size=hidden_size,
        num_target_layers=4,
        para_num=4,
        unused_tokenids=[99, 100, 101],
        down_sample_ratio=0.7,
        down_sample_ratio_min=0.1,
    )
    samples = []
    for seq_len in (20, 24):
        samples.append(
            {
                "input_ids": torch.arange(seq_len, dtype=torch.long),
                "labels": torch.arange(seq_len, dtype=torch.long),
                "teacher_hidden": torch.randn(seq_len, hidden_size),
                "target_feat": torch.randn(seq_len, hidden_size * 4),
            }
        )
    batch = collate(samples)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] == max_len
    assert batch["attention_mask"].shape[0] == 2


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
    assert (batch["labels"][:, 95:] == _data_mod.IGNORE_LABEL).all()
    assert batch["gold_input_ids"].shape[0] == 1
    assert batch["warp_indices"].shape[0] == 1


def test_collate_from_draft_model_uses_post_warp_length_budget():
    torch.manual_seed(0)
    hidden_size = 32
    seq_len = 26
    draft_model = SimpleNamespace(
        config=SimpleNamespace(
            end_token_id=None,
            target_layer_ids=[-1, -8, -16, -24],
            para_num=16,
            mask_token_ids=list(range(100, 115)),
            down_sample_ratio=0.7,
            down_sample_ratio_min=0.1,
        )
    )
    collate = _data_mod.create_pard2_collate_fn_from_draft_model(
        draft_model,
        max_len=seq_len,
        hidden_size=hidden_size,
    )
    batch = collate(
        [
            {
                "input_ids": torch.arange(seq_len, dtype=torch.long),
                "loss_mask": torch.ones(seq_len, dtype=torch.bool),
                "multi_layer_hidden_states": torch.randn(seq_len, hidden_size * 4),
                "verifier_last_hidden_states": torch.randn(seq_len, hidden_size),
            }
        ]
    )
    assert batch["input_ids"].shape[1] == 95
    assert batch["gold_input_ids"].shape[1] == seq_len


def test_deferred_prev_prob_matches_inline_compute():
    torch.manual_seed(0)
    hidden_size = 32
    vocab = 64
    seq_len = 26
    para_num = 16
    lm_weight = torch.randn(vocab, hidden_size)

    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()
    teacher_hidden = torch.randn(1, seq_len, hidden_size)
    teacher_gold_prob = compute_teacher_gold_prob(
        input_ids, teacher_hidden, lm_weight
    )

    inline = apply_parallel_warp(
        input_ids=input_ids,
        labels=labels,
        target_feat=None,
        teacher_hidden=teacher_hidden,
        teacher_gold_prob=teacher_gold_prob,
        para_num=para_num,
        unused_tokenids=list(range(100, 115)),
        down_sample_ratio=0.7,
        down_sample_ratio_min=0.1,
    )

    deferred = _data_mod.build_pard2_prev_prob_batch(
        inline["gold_input_ids"],
        inline["gold_teacher_hidden"],
        inline["gold_labels"],
        torch.tensor([seq_len], dtype=torch.long),
        inline["warp_indices"].unsqueeze(0),
        torch.tensor([inline["warp_indices"].numel()], dtype=torch.long),
        lm_weight,
        None,
        para_num,
        inline["prev_prob"].shape[1],
        device=torch.device("cpu"),
    )

    assert torch.allclose(
        deferred[:, : inline["prev_prob"].shape[1]],
        inline["prev_prob"],
        atol=1e-5,
    )


def test_deferred_prev_prob_truncates_to_max_len():
    """Collate truncates warped seq to max_len; device prev_prob must match."""
    torch.manual_seed(0)
    hidden_size = 32
    vocab = 64
    seq_len = 180
    max_len = 128
    para_num = 16
    lm_weight = torch.randn(vocab, hidden_size)

    input_ids = (torch.arange(seq_len, dtype=torch.long) % vocab).unsqueeze(0)
    labels = input_ids.clone()
    teacher_hidden = torch.randn(1, seq_len, hidden_size)
    teacher_gold_prob = compute_teacher_gold_prob(
        input_ids, teacher_hidden, lm_weight
    )

    inline = apply_parallel_warp(
        input_ids=input_ids,
        labels=labels,
        target_feat=None,
        teacher_hidden=teacher_hidden,
        teacher_gold_prob=teacher_gold_prob,
        para_num=para_num,
        unused_tokenids=list(range(100, 115)),
        down_sample_ratio=0.7,
        down_sample_ratio_min=0.1,
    )
    assert inline["prev_prob"].shape[1] > max_len

    deferred = _data_mod.build_pard2_prev_prob_batch(
        inline["gold_input_ids"],
        inline["gold_teacher_hidden"],
        inline["gold_labels"],
        torch.tensor([seq_len], dtype=torch.long),
        inline["warp_indices"].unsqueeze(0),
        torch.tensor([inline["warp_indices"].numel()], dtype=torch.long),
        lm_weight,
        None,
        para_num,
        max_len,
        device=torch.device("cpu"),
    )

    assert deferred.shape == (1, max_len)
    assert torch.allclose(
        deferred,
        inline["prev_prob"][:, :max_len],
        atol=1e-5,
    )
