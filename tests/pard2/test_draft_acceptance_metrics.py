"""Tests for draft acceptance probability metrics."""

import torch

from speculators.models.pard2.core import compute_draft_acceptance_prob


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
