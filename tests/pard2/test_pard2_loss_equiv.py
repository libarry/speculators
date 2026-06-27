"""The memory-efficient hidden-state loss must match the logits-based loss.

``weighted_ce_kd_loss_from_hidden_chunked`` avoids materializing the full
``[batch, seq, vocab]`` student logits by projecting per chunk inside activation
checkpointing. It must be numerically equivalent (forward value AND gradients) to
``weighted_ce_kd_loss_chunked`` evaluated on ``draft_lm_head(hidden)`` so that the
optimization is a pure performance/memory change with no effect on training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from speculators.models.pard2.core import (
    weighted_ce_kd_loss_chunked,
    weighted_ce_kd_loss_from_hidden_chunked,
)


def _make_inputs(seed: int = 0):
    torch.manual_seed(seed)
    batch, seq, hidden_d, hidden_v, vocab = 2, 37, 24, 24, 50
    draft_hidden = torch.randn(batch, seq, hidden_d, requires_grad=True)
    draft_lm = nn.Linear(hidden_d, vocab, bias=False)
    teacher_hidden = torch.randn(batch, seq, hidden_v)
    verifier_lm = nn.Linear(hidden_v, vocab, bias=False)
    for p in verifier_lm.parameters():
        p.requires_grad_(False)
    labels = torch.randint(0, vocab, (batch, seq))
    labels[0, -3:] = -100  # exercise the ignore mask
    token_weight = torch.rand(batch, seq)
    return draft_hidden, draft_lm, teacher_hidden, verifier_lm, labels, token_weight


def test_hidden_loss_matches_logits_loss_forward_and_backward():
    (
        draft_hidden,
        draft_lm,
        teacher_hidden,
        verifier_lm,
        labels,
        token_weight,
    ) = _make_inputs()

    shift_labels = labels[..., 1:].contiguous()
    valid = (shift_labels != -100).float()
    weight = valid * token_weight[..., 1:]

    temperature, ce_alpha, kd_alpha = 1.3, 0.7, 0.9

    # Reference: full logits then the existing chunked loss.
    hidden_ref = draft_hidden.detach().clone().requires_grad_(True)
    draft_lm_ref = nn.Linear(*draft_lm.weight.shape[::-1], bias=False)
    draft_lm_ref.weight.data.copy_(draft_lm.weight.data)
    logits = draft_lm_ref(hidden_ref)
    ref_final, ref_ce, ref_kd = weighted_ce_kd_loss_chunked(
        logits[..., :-1, :],
        shift_labels,
        teacher_hidden[..., :-1, :],
        verifier_lm,
        weight,
        temperature,
        ce_alpha,
        kd_alpha,
    )
    ref_final.backward()

    # New: directly from hidden, with checkpointing.
    new_final, new_ce, new_kd = weighted_ce_kd_loss_from_hidden_chunked(
        draft_hidden[..., :-1, :],
        draft_lm.weight,
        shift_labels,
        teacher_hidden[..., :-1, :],
        verifier_lm.weight,
        weight,
        temperature,
        ce_alpha,
        kd_alpha,
        chunk_size=8,
    )
    new_final.backward()

    assert torch.allclose(new_final, ref_final, atol=1e-5, rtol=1e-4)
    assert torch.allclose(new_ce, ref_ce, atol=1e-5, rtol=1e-4)
    assert torch.allclose(new_kd, ref_kd, atol=1e-5, rtol=1e-4)

    assert draft_hidden.grad is not None and hidden_ref.grad is not None
    assert torch.allclose(draft_hidden.grad, hidden_ref.grad, atol=1e-5, rtol=1e-3)
    assert torch.allclose(
        draft_lm.weight.grad, draft_lm_ref.weight.grad, atol=1e-5, rtol=1e-3
    )


def test_hidden_loss_checkpoint_matches_no_checkpoint():
    (
        draft_hidden,
        draft_lm,
        teacher_hidden,
        verifier_lm,
        labels,
        token_weight,
    ) = _make_inputs(seed=3)
    shift_labels = labels[..., 1:].contiguous()
    weight = (shift_labels != -100).float() * token_weight[..., 1:]

    common = dict(
        shift_labels=shift_labels,
        shift_teacher_hidden=teacher_hidden[..., :-1, :],
        verifier_lm_weight=verifier_lm.weight,
        token_weight=weight,
        temperature=1.0,
        ce_alpha=1.0,
        kd_alpha=1.0,
        chunk_size=8,
    )

    ckpt_final, _, _ = weighted_ce_kd_loss_from_hidden_chunked(
        draft_hidden[..., :-1, :], draft_lm.weight, use_checkpoint=True, **common
    )
    no_ckpt_final, _, _ = weighted_ce_kd_loss_from_hidden_chunked(
        draft_hidden[..., :-1, :], draft_lm.weight, use_checkpoint=False, **common
    )
    assert torch.allclose(ckpt_final, no_ckpt_final, atol=1e-6)
