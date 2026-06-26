"""Memory-efficient loss helpers for PARD-2 (vocab/sequence chunking)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_SEQ_CHUNK_SIZE = 8
DEFAULT_VOCAB_CHUNK_SIZE = 4096

__all__ = [
    "DEFAULT_SEQ_CHUNK_SIZE",
    "DEFAULT_VOCAB_CHUNK_SIZE",
    "apply_rms_norm",
    "gather_label_log_prob_chunked",
    "gather_log_prob_chunked",
    "logsumexp_last_dim_chunked",
    "token_kd_sum_chunked",
]


def apply_rms_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Verifier final RMSNorm (matches HF ``model.norm`` before ``lm_head``)."""
    input_dtype = hidden.dtype
    hidden = hidden.to(torch.float32)
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps)
    return (weight * hidden).to(input_dtype)


def logsumexp_last_dim_chunked(
    x: torch.Tensor,
    chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
) -> torch.Tensor:
    """``logsumexp`` along the last dimension without full-vocab materialization."""
    vocab = x.shape[-1]
    max_val = x.new_full(x.shape[:-1] + (1,), float("-inf"))
    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        chunk_max = x[..., start:end].amax(dim=-1, keepdim=True)
        max_val = torch.maximum(max_val, chunk_max)

    sum_exp = x.new_zeros(x.shape[:-1] + (1,))
    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        sum_exp = sum_exp + (x[..., start:end] - max_val).exp().sum(dim=-1, keepdim=True)

    return max_val + sum_exp.log()


def token_kd_sum_chunked(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    *,
    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
) -> torch.Tensor:
    """Per-token KL(teacher || student) with ``log_target`` semantics.

    Returns a tensor with shape ``student_logits.shape[:-1]``.
    """
    temp = float(temperature)
    s_scaled = student_logits.float() / temp
    t_scaled = teacher_logits.float() / temp

    log_z_s = logsumexp_last_dim_chunked(s_scaled, vocab_chunk_size)
    log_z_t = logsumexp_last_dim_chunked(t_scaled, vocab_chunk_size)

    kl = student_logits.new_zeros(student_logits.shape[:-1])
    vocab = s_scaled.shape[-1]
    for start in range(0, vocab, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab)
        lp = t_scaled[..., start:end] - log_z_t
        lq = s_scaled[..., start:end] - log_z_s
        kl = kl + (lp.exp() * (lp - lq)).sum(dim=-1)

    return kl * (temp**2)


def gather_label_log_prob_chunked(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None = None,
    *,
    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
) -> torch.Tensor:
    """Log-probability of ``labels`` from ``lm_head(hidden)`` without full logits."""
    vocab = lm_head_weight.shape[0]
    running_max = hidden.new_full(labels.shape, float("-inf"))
    running_sum_exp = hidden.new_zeros(labels.shape)
    label_logit = hidden.new_zeros(labels.shape)

    for start in range(0, vocab, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab)
        weight = lm_head_weight[start:end]
        bias = lm_head_bias[start:end] if lm_head_bias is not None else None
        logits_chunk = F.linear(hidden, weight, bias).float()
        chunk_max = logits_chunk.amax(dim=-1)
        chunk_sum_exp = (logits_chunk - chunk_max.unsqueeze(-1)).exp().sum(dim=-1)

        new_max = torch.maximum(running_max, chunk_max)
        running_sum_exp = (running_max - new_max).exp() * running_sum_exp + (
            chunk_max - new_max
        ).exp() * chunk_sum_exp
        running_max = new_max

        in_chunk = (labels >= start) & (labels < end)
        if in_chunk.any():
            local_idx = (labels - start).clamp(min=0, max=end - start - 1)
            gathered = logits_chunk.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)
            label_logit = torch.where(in_chunk, gathered, label_logit)

    log_z = running_max + running_sum_exp.log()
    return label_logit - log_z


def gather_log_prob_chunked(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
) -> torch.Tensor:
    """Log-probability of ``token_ids`` under ``logits`` without full softmax."""
    vocab = logits.shape[-1]
    running_max = logits.new_full(token_ids.shape, float("-inf"))
    running_sum_exp = logits.new_zeros(token_ids.shape)
    token_logit = logits.new_zeros(token_ids.shape)

    for start in range(0, vocab, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab)
        logits_chunk = logits[..., start:end].float()
        chunk_max = logits_chunk.amax(dim=-1)
        chunk_sum_exp = (logits_chunk - chunk_max.unsqueeze(-1)).exp().sum(dim=-1)

        new_max = torch.maximum(running_max, chunk_max)
        running_sum_exp = (running_max - new_max).exp() * running_sum_exp + (
            chunk_max - new_max
        ).exp() * chunk_sum_exp
        running_max = new_max

        in_chunk = (token_ids >= start) & (token_ids < end)
        if in_chunk.any():
            local_idx = (token_ids - start).clamp(min=0, max=end - start - 1)
            gathered = logits_chunk.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)
            token_logit = torch.where(in_chunk, gathered, token_logit)

    log_z = running_max + running_sum_exp.log()
    return token_logit - log_z
