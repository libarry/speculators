"""DFlash GQA partial NeoX RoPE: short cos/sin tables and pass-through tail."""

from __future__ import annotations

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.models.dflash.model_definitions import (
    apply_rotary_pos_emb,
    build_gqa_rotary_embedding,
    resolve_partial_rotary_factor,
)
from speculators.models.eagle3.rotary_partial import apply_neox_rotary


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def test_apply_rotary_full_head_matches_neox():
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 8, 8)
    cos = torch.randn(1, 8, 8)
    sin = torch.randn(1, 8, 8)

    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

    cos_q = cos[:, -3:, :].unsqueeze(1)
    sin_q = sin[:, -3:, :].unsqueeze(1)
    expected_q = (q * cos_q) + (_rotate_half(q) * sin_q)
    expected_k = apply_neox_rotary(k, cos, sin)

    torch.testing.assert_close(q_out, expected_q)
    torch.testing.assert_close(k_out, expected_k)


def test_apply_rotary_partial_preserves_tail_and_uses_query_tail_positions():
    head_dim, rotary_dim = 8, 4
    q = torch.randn(1, 2, 3, head_dim)
    k = torch.randn(1, 2, 8, head_dim)
    cos = torch.randn(1, 8, rotary_dim)
    sin = torch.randn(1, 8, rotary_dim)

    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

    assert q_out[..., rotary_dim:].equal(q[..., rotary_dim:])
    assert k_out[..., rotary_dim:].equal(k[..., rotary_dim:])

    q_rot = apply_neox_rotary(q[..., :rotary_dim], cos[:, -3:, :], sin[:, -3:, :])
    torch.testing.assert_close(q_out[..., :rotary_dim], q_rot)

    k_rot = apply_neox_rotary(k[..., :rotary_dim], cos, sin)
    torch.testing.assert_close(k_out[..., :rotary_dim], k_rot)


def test_build_gqa_rotary_embedding_emits_partial_tables():
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=256,
        max_position_embeddings=32,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    )
    assert resolve_partial_rotary_factor(config) == 0.25

    emb = build_gqa_rotary_embedding(config)
    dummy = torch.randn(1, 4, 256)
    positions = torch.arange(4).unsqueeze(0)
    cos, sin = emb(dummy, positions)

    assert cos.shape[-1] == 64
    assert sin.shape[-1] == 64


def test_build_gqa_rotary_embedding_full_head_when_partial_is_one():
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    emb = build_gqa_rotary_embedding(config)
    dummy = torch.randn(1, 4, 16)
    positions = torch.arange(4).unsqueeze(0)
    cos, sin = emb(dummy, positions)

    assert cos.shape[-1] == 16
    assert sin.shape[-1] == 16
