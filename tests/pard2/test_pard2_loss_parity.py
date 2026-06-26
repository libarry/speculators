"""Parity tests: optimized PARD-2 loss paths vs pre-optimization reference."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from speculators.models.pard2.core import (
    weighted_ce_kd_loss_chunked,
    weighted_ce_loss_chunked,
    weighted_kd_loss_chunked,
)
from speculators.models.pard2.data import compute_teacher_gold_prob
from speculators.models.pard2.loss_ops import token_kd_sum_chunked


# ---------------------------------------------------------------------------
# Reference implementations (exact pre-optimization logic)
# ---------------------------------------------------------------------------


def _old_weighted_kd_loss_chunked(
    shift_student_logits: torch.Tensor,
    shift_teacher_hidden: torch.Tensor,
    lm_head: nn.Linear,
    temperature: float,
    token_weight: torch.Tensor | None = None,
    *,
    chunk_size: int = 32,
) -> torch.Tensor:
    _, seq_len, _ = shift_student_logits.shape
    temp_scale = temperature**2
    weighted_sum = shift_student_logits.new_zeros(())
    weight_sum = shift_student_logits.new_zeros(())

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        student_chunk = shift_student_logits[:, start:end].float()
        teacher_chunk = shift_teacher_hidden[:, start:end].to(dtype=lm_head.weight.dtype)
        teacher_logits = lm_head(teacher_chunk).float()

        teacher_log_prob = F.log_softmax(teacher_logits / temperature, dim=-1)
        student_log_prob = F.log_softmax(student_chunk / temperature, dim=-1)
        token_kd = (
            F.kl_div(
                student_log_prob,
                teacher_log_prob,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
            * temp_scale
        )

        if token_weight is not None:
            chunk_weight = token_weight[:, start:end].to(token_kd.dtype)
            weighted_sum = weighted_sum + (token_kd * chunk_weight).sum()
            weight_sum = weight_sum + chunk_weight.sum()
        else:
            weighted_sum = weighted_sum + token_kd.sum()
            weight_sum = weight_sum + token_kd.numel()

    return weighted_sum / weight_sum.clamp_min(1e-6)


def _old_weighted_ce_loss_chunked(
    shift_student_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    token_weight: torch.Tensor,
    *,
    chunk_size: int = 32,
) -> torch.Tensor:
    _, seq_len, vocab_size = shift_student_logits.shape
    weighted_sum = shift_student_logits.new_zeros(())
    weight_sum = shift_student_logits.new_zeros(())

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        student_chunk = shift_student_logits[:, start:end].float()
        labels_chunk = shift_labels[:, start:end]
        weight_chunk = token_weight[:, start:end].to(student_chunk.dtype)
        ce_chunk = F.cross_entropy(
            student_chunk.reshape(-1, vocab_size),
            labels_chunk.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(labels_chunk)
        weighted_sum = weighted_sum + (ce_chunk * weight_chunk).sum()
        weight_sum = weight_sum + weight_chunk.sum()

    return weighted_sum / weight_sum.clamp_min(1e-6)


def _old_compute_teacher_gold_prob(
    input_ids: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None = None,
) -> torch.Tensor:
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


def _reference_token_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_prob = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log_prob = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    return (
        F.kl_div(
            student_log_prob,
            teacher_log_prob,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        * (temperature**2)
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_token_kd_sum_chunked_matches_kl_div_log_target():
    torch.manual_seed(10)
    for vocab in (64, 97, 4097):  # non-multiple of chunk size
        student_logits = torch.randn(2, 11, vocab)
        teacher_logits = torch.randn(2, 11, vocab)
        for vocab_chunk in (1, 7, 4096):
            ref = _reference_token_kd(student_logits, teacher_logits, 1.5)
            got = token_kd_sum_chunked(
                student_logits,
                teacher_logits,
                1.5,
                vocab_chunk_size=vocab_chunk,
            )
            assert torch.allclose(ref, got, rtol=1e-4, atol=1e-5), (
                f"vocab={vocab}, vocab_chunk={vocab_chunk}"
            )


def test_weighted_kd_loss_matches_old_impl():
    torch.manual_seed(11)
    batch, seq_len, hidden, vocab = 2, 37, 64, 4097
    lm_head = nn.Linear(hidden, vocab, bias=False)
    shift_student_logits = torch.randn(batch, seq_len, vocab)
    shift_teacher_hidden = torch.randn(batch, seq_len, hidden)
    token_weight = torch.rand(batch, seq_len)
    temperature = 1.5

    old = _old_weighted_kd_loss_chunked(
        shift_student_logits,
        shift_teacher_hidden,
        lm_head,
        temperature,
        token_weight,
        chunk_size=32,
    )
    new = weighted_kd_loss_chunked(
        shift_student_logits,
        shift_teacher_hidden,
        lm_head,
        temperature,
        token_weight,
        chunk_size=8,
        vocab_chunk_size=512,
    )
    assert torch.allclose(old, new, rtol=1e-4, atol=1e-5)


def test_weighted_kd_loss_unweighted_matches_old_impl():
    torch.manual_seed(12)
    batch, seq_len, hidden, vocab = 1, 25, 32, 97
    lm_head = nn.Linear(hidden, vocab, bias=False)
    shift_student_logits = torch.randn(batch, seq_len, vocab)
    shift_teacher_hidden = torch.randn(batch, seq_len, hidden)

    old = _old_weighted_kd_loss_chunked(
        shift_student_logits,
        shift_teacher_hidden,
        lm_head,
        1.0,
        None,
        chunk_size=32,
    )
    new = weighted_kd_loss_chunked(
        shift_student_logits,
        shift_teacher_hidden,
        lm_head,
        1.0,
        None,
        chunk_size=3,
        vocab_chunk_size=11,
    )
    assert torch.allclose(old, new, rtol=1e-4, atol=1e-5)


def test_weighted_ce_loss_matches_old_impl():
    torch.manual_seed(13)
    batch, seq_len, vocab = 2, 41, 256
    shift_student_logits = torch.randn(batch, seq_len, vocab)
    shift_labels = torch.randint(0, vocab, (batch, seq_len))
    shift_labels[:, ::5] = -100
    token_weight = torch.rand(batch, seq_len)

    old = _old_weighted_ce_loss_chunked(
        shift_student_logits, shift_labels, token_weight, chunk_size=32
    )
    new = weighted_ce_loss_chunked(
        shift_student_logits, shift_labels, token_weight, chunk_size=7
    )
    assert torch.allclose(old, new, rtol=1e-5, atol=1e-6)


def test_compute_teacher_gold_prob_matches_old_impl():
    torch.manual_seed(14)
    batch, seq_len, hidden, vocab = 2, 19, 48, 4097
    lm_head = nn.Linear(hidden, vocab, bias=True)
    input_ids = torch.randint(0, vocab, (batch, seq_len))
    teacher_hidden = torch.randn(batch, seq_len, hidden)

    old = _old_compute_teacher_gold_prob(
        input_ids, teacher_hidden, lm_head.weight, lm_head.bias
    )
    new = compute_teacher_gold_prob(
        input_ids, teacher_hidden, lm_head.weight, lm_head.bias, vocab_chunk_size=512
    )
    assert torch.allclose(old, new, rtol=1e-4, atol=1e-5)


def test_fused_ce_kd_matches_separate():
    torch.manual_seed(15)
    batch, seq_len, hidden, vocab = 2, 33, 64, 1024
    lm_head = nn.Linear(hidden, vocab, bias=False)
    shift_student_logits = torch.randn(batch, seq_len, vocab)
    shift_teacher_hidden = torch.randn(batch, seq_len, hidden)
    shift_labels = torch.randint(0, vocab, (batch, seq_len))
    shift_labels[:, ::7] = -100
    token_weight = torch.rand(batch, seq_len)
    ce_alpha, kd_alpha, temperature = 0.1, 1.0, 1.25

    fused_final, fused_ce, fused_kd = weighted_ce_kd_loss_chunked(
        shift_student_logits,
        shift_labels,
        shift_teacher_hidden,
        lm_head,
        token_weight,
        temperature,
        ce_alpha,
        kd_alpha,
        chunk_size=5,
        vocab_chunk_size=127,
    )
    sep_ce = weighted_ce_loss_chunked(
        shift_student_logits, shift_labels, token_weight, chunk_size=5
    )
    sep_kd = weighted_kd_loss_chunked(
        shift_student_logits,
        shift_teacher_hidden,
        lm_head,
        temperature,
        token_weight,
        chunk_size=5,
        vocab_chunk_size=127,
    )
    sep_final = sep_ce * ce_alpha + sep_kd * kd_alpha

    assert torch.allclose(fused_ce, sep_ce, rtol=1e-4, atol=1e-5)
    assert torch.allclose(fused_kd, sep_kd, rtol=1e-4, atol=1e-5)
    assert torch.allclose(fused_final, sep_final, rtol=1e-4, atol=1e-5)


def test_bf16_inputs_parity():
    """Training uses bf16 logits; loss should match float reference."""
    torch.manual_seed(16)
    batch, seq_len, hidden, vocab = 1, 17, 32, 512
    lm_head = nn.Linear(hidden, vocab, bias=False)
    shift_student_logits = torch.randn(batch, seq_len, vocab, dtype=torch.bfloat16)
    shift_teacher_hidden = torch.randn(batch, seq_len, hidden, dtype=torch.bfloat16)
    token_weight = torch.rand(batch, seq_len, dtype=torch.bfloat16)

    old = _old_weighted_kd_loss_chunked(
        shift_student_logits.float(),
        shift_teacher_hidden.float(),
        lm_head,
        1.0,
        token_weight.float(),
        chunk_size=32,
    )
    new = weighted_kd_loss_chunked(
        shift_student_logits,
        shift_teacher_hidden,
        lm_head,
        1.0,
        token_weight,
        chunk_size=4,
        vocab_chunk_size=64,
    )
    assert torch.allclose(old, new, rtol=1e-3, atol=1e-4)
