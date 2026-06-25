"""Tests for memory-efficient PARD-2 KD loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from speculators.models.pard2.core import weighted_kd_loss_chunked


def _naive_weighted_kd_loss(
    shift_student_logits: torch.Tensor,
    shift_teacher_hidden: torch.Tensor,
    lm_head: nn.Linear,
    temperature: float,
    token_weight: torch.Tensor | None,
) -> torch.Tensor:
    teacher_logits = lm_head(shift_teacher_hidden.float())
    shift_teacher_logits = teacher_logits.float()
    shift_student_logits = shift_student_logits.float()
    teacher_prob = F.softmax(shift_teacher_logits / temperature, dim=-1)
    student_log_prob = F.log_softmax(shift_student_logits / temperature, dim=-1)
    token_kd = (
        F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
        * (temperature**2)
    )
    if token_weight is not None:
        weight = token_weight.to(token_kd.dtype)
        return (token_kd * weight).sum() / weight.sum().clamp_min(1e-6)
    return token_kd.mean()


def test_weighted_kd_loss_chunked_matches_naive():
    torch.manual_seed(0)
    batch, seq_len, hidden, vocab = 2, 37, 16, 64
    lm_head = nn.Linear(hidden, vocab, bias=False)
    shift_student_logits = torch.randn(batch, seq_len, vocab)
    shift_teacher_hidden = torch.randn(batch, seq_len, hidden)
    token_weight = torch.rand(batch, seq_len)
    temperature = 1.5

    naive = _naive_weighted_kd_loss(
        shift_student_logits,
        shift_teacher_hidden,
        lm_head,
        temperature,
        token_weight,
    )
    chunked = weighted_kd_loss_chunked(
        shift_student_logits,
        shift_teacher_hidden,
        lm_head,
        temperature,
        token_weight,
        chunk_size=8,
    )
    assert torch.allclose(naive, chunked, rtol=1e-5, atol=1e-6)


def test_weighted_kd_loss_chunked_unweighted():
    torch.manual_seed(1)
    batch, seq_len, hidden, vocab = 1, 25, 8, 32
    lm_head = nn.Linear(hidden, vocab, bias=False)
    shift_student_logits = torch.randn(batch, seq_len, vocab)
    shift_teacher_hidden = torch.randn(batch, seq_len, hidden)
    temperature = 1.0

    naive = _naive_weighted_kd_loss(
        shift_student_logits,
        shift_teacher_hidden,
        lm_head,
        temperature,
        None,
    )
    chunked = weighted_kd_loss_chunked(
        shift_student_logits,
        shift_teacher_hidden,
        lm_head,
        temperature,
        None,
        chunk_size=5,
    )
    assert torch.allclose(naive, chunked, rtol=1e-5, atol=1e-6)
