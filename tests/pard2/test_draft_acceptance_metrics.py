"""Tests for draft acceptance probability metrics."""

import torch

from speculators.models.pard2.core import (
    compute_draft_acceptance_prob,
    compute_step_acceptance_metrics,
)


def test_compute_draft_acceptance_prob_clamped_to_one():
    vocab = 8
    student_logits = torch.zeros(1, 3, vocab)
    student_logits[..., 2] = 5.0
    teacher_hidden = torch.zeros(1, 3, 4)
    weight = torch.eye(vocab, 4)

    accept = compute_draft_acceptance_prob(
        student_logits,
        teacher_hidden,
        weight,
        None,
        torch.ones(4),
        1e-6,
        vocab_chunk_size=vocab,
    )
    assert accept.shape == (1, 2)
    assert (accept <= 1.0).all()
    assert (accept >= 0.0).all()


def test_compute_draft_acceptance_prob_high_when_teacher_prefers_draft_token():
    vocab = 16
    hidden_size = 8
    token_id = 3
    student_logits = torch.full((1, 3, vocab), -5.0)
    student_logits[..., token_id] = 10.0

    weight = torch.randn(vocab, hidden_size) * 0.01
    weight[token_id] = 10.0
    teacher_hidden = torch.zeros(1, 3, hidden_size)
    teacher_hidden[..., :] = weight[token_id]

    accept = compute_draft_acceptance_prob(
        student_logits,
        teacher_hidden,
        weight,
        None,
        torch.ones(hidden_size),
        1e-6,
        vocab_chunk_size=vocab,
    )
    assert accept.mean().item() > 0.5


def _set_shift_greedy_match(
    student_logits: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    shift_idx: int,
    token_id: int,
) -> None:
    """Make draft and target argmax both equal ``token_id`` at ``shift_idx``."""
    hidden_size = teacher_hidden.shape[-1]
    dim = token_id % hidden_size
    lm_head_weight[token_id, :] = 0.0
    lm_head_weight[token_id, dim] = 1.0
    student_logits[0, shift_idx, :] = -10.0
    student_logits[0, shift_idx, token_id] = 10.0
    teacher_hidden[0, shift_idx, :] = 0.0
    teacher_hidden[0, shift_idx, dim] = 1.0


def test_compute_step_acceptance_metrics_conditional_chain():
    """acc_d matches PARD-style conditional greedy verify along anchor chains."""
    vocab_size = 16
    hidden_size = 8
    single_length = 4
    para_num = 3
    seq_len = single_length * para_num

    labels = torch.full((1, seq_len), -100, dtype=torch.long)
    for depth in range(para_num):
        block_labels = torch.full((single_length,), -100, dtype=torch.long)
        block_labels[depth:] = torch.arange(depth, single_length)
        labels[0, depth * single_length : (depth + 1) * single_length] = block_labels

    student_logits = torch.full((1, seq_len, vocab_size), -10.0)
    teacher_hidden = torch.zeros(1, seq_len, hidden_size)
    lm_head_weight = torch.zeros(vocab_size, hidden_size)

    # Anchor 0 chain: depths 0/1/2 at shift idx 0/5/10.
    _set_shift_greedy_match(student_logits, teacher_hidden, lm_head_weight, 0, 1)
    _set_shift_greedy_match(student_logits, teacher_hidden, lm_head_weight, 5, 2)
    _set_shift_greedy_match(student_logits, teacher_hidden, lm_head_weight, 10, 3)
    # Anchor 1 chain: depths 0/1 at shift idx 1/6 (depth 2 invalid for T=4).
    _set_shift_greedy_match(student_logits, teacher_hidden, lm_head_weight, 1, 2)
    _set_shift_greedy_match(student_logits, teacher_hidden, lm_head_weight, 6, 3)
    # Anchor 2 chain: depth 0 only at shift idx 2.
    _set_shift_greedy_match(student_logits, teacher_hidden, lm_head_weight, 2, 3)

    metrics = compute_step_acceptance_metrics(
        student_logits,
        teacher_hidden,
        labels,
        para_num=para_num,
        lm_head_weight=lm_head_weight,
        lm_head_bias=None,
        norm_weight=torch.ones(hidden_size),
        norm_eps=1e-6,
        warp_single_length=torch.tensor([single_length]),
    )

    assert metrics["acc_0_total"].item() == 3.0
    assert metrics["acc_0_sum"].item() == 3.0
    assert metrics["acc_1_total"].item() == 2.0
    assert metrics["acc_1_sum"].item() == 2.0
    assert metrics["acc_2_total"].item() == 1.0
    assert metrics["acc_2_sum"].item() == 1.0


def test_compute_step_acceptance_metrics_chain_breaks_at_depth_zero():
    vocab_size = 16
    hidden_size = 8
    single_length = 4
    para_num = 2
    seq_len = single_length * para_num

    labels = torch.full((1, seq_len), -100, dtype=torch.long)
    for depth in range(para_num):
        block_labels = torch.full((single_length,), -100, dtype=torch.long)
        block_labels[depth:] = torch.arange(depth, single_length)
        labels[0, depth * single_length : (depth + 1) * single_length] = block_labels

    student_logits = torch.full((1, seq_len, vocab_size), -10.0)
    teacher_hidden = torch.zeros(1, seq_len, hidden_size)
    lm_head_weight = torch.zeros(vocab_size, hidden_size)

    _set_shift_greedy_match(student_logits, teacher_hidden, lm_head_weight, 1, 2)
    _set_shift_greedy_match(student_logits, teacher_hidden, lm_head_weight, 2, 3)
    _set_shift_greedy_match(student_logits, teacher_hidden, lm_head_weight, 5, 3)
    _set_shift_greedy_match(student_logits, teacher_hidden, lm_head_weight, 6, 3)
    # Anchor 0 depth 0 mismatch at shift idx 0 (draft=9, target=1).
    _set_shift_greedy_match(student_logits, teacher_hidden, lm_head_weight, 0, 1)
    student_logits[0, 0, 9] = 20.0

    metrics = compute_step_acceptance_metrics(
        student_logits,
        teacher_hidden,
        labels,
        para_num=para_num,
        lm_head_weight=lm_head_weight,
        lm_head_bias=None,
        norm_weight=torch.ones(hidden_size),
        norm_eps=1e-6,
        warp_single_length=torch.tensor([single_length]),
    )

    assert metrics["acc_0_sum"].item() == 2.0
    assert metrics["acc_0_total"].item() == 3.0
    assert metrics["acc_1_total"].item() == 1.0
    assert metrics["acc_1_sum"].item() == 1.0
