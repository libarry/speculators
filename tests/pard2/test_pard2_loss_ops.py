"""Tests for memory-efficient PARD-2 loss helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from speculators.models.pard2.loss_ops import (
    gather_label_log_prob_chunked,
    logsumexp_last_dim_chunked,
    token_kd_sum_chunked,
)


def test_logsumexp_last_dim_chunked_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(2, 11, 97)
    expected = torch.logsumexp(x, dim=-1)
    chunked = logsumexp_last_dim_chunked(x, chunk_size=13).squeeze(-1)
    assert torch.allclose(expected, chunked, rtol=1e-5, atol=1e-6)


def test_token_kd_sum_chunked_matches_kl_div():
    torch.manual_seed(1)
    batch, seq_len, vocab = 2, 9, 80
    temperature = 1.25
    student_logits = torch.randn(batch, seq_len, vocab)
    teacher_logits = torch.randn(batch, seq_len, vocab)

    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_prob = F.log_softmax(teacher_logits / temperature, dim=-1)
    expected = (
        F.kl_div(
            student_log_prob,
            teacher_log_prob,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        * (temperature**2)
    )
    chunked = token_kd_sum_chunked(
        student_logits,
        teacher_logits,
        temperature,
        vocab_chunk_size=17,
    )
    assert torch.allclose(expected, chunked, rtol=1e-4, atol=1e-5)


def test_gather_label_log_prob_chunked_matches_log_softmax_gather():
    torch.manual_seed(2)
    batch, seq_len, hidden, vocab = 2, 7, 12, 55
    hidden_states = torch.randn(batch, seq_len, hidden)
    labels = torch.randint(0, vocab, (batch, seq_len))
    lm_head = nn.Linear(hidden, vocab, bias=False)

    logits = lm_head(hidden_states)
    expected = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    chunked = gather_label_log_prob_chunked(
        hidden_states,
        labels,
        lm_head.weight,
        vocab_chunk_size=11,
    )
    assert torch.allclose(expected, chunked, rtol=1e-4, atol=1e-5)
