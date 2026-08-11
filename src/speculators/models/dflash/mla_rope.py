"""Interleaved RoPE helpers for DFlash/DSpark MLA attention.

DeepSeek MLA rotates consecutive dim pairs (0,1), (2,3), ... rather than the
NeoX first-half/second-half layout used by Qwen3. Cos/sin caches are still
built in NeoX layout and converted on the fly via repeat_interleave.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

__all__ = [
    "MLARotaryEmbedding",
    "MLAYarnRotaryEmbedding",
    "apply_rope_interleaved",
    "build_mla_rotary_embedding",
    "compute_mla_softmax_scale",
    "resolve_rope_scaling",
    "resolve_rope_theta",
    "rotate_half_interleaved",
    "yarn_get_mscale",
]


def rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair rotation: pairs dims (0,1), (2,3), ..."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply interleaved RoPE to one tensor using NeoX-layout cos/sin caches."""
    cos = cos.squeeze(1).squeeze(0)[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin.squeeze(1).squeeze(0)[position_ids].unsqueeze(unsqueeze_dim)
    half = cos.shape[-1] // 2
    cos = cos[..., :half].repeat_interleave(2, dim=-1)
    sin = sin[..., :half].repeat_interleave(2, dim=-1)
    return (x * cos) + (rotate_half_interleaved(x) * sin)


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> tuple[int, int]:
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
        max_val - min_val
    )
    return torch.clamp(linear_func, 0, 1)


def resolve_rope_theta(config: Any, default: float = 10000.0) -> float:
    """Read rope base from top-level or nested rope_parameters/rope_scaling."""
    top_level = getattr(config, "rope_theta", None)
    params = resolve_rope_scaling(config)
    nested = params.get("rope_theta") if isinstance(params, dict) else None
    if top_level is not None and nested is not None and float(top_level) != float(nested):
        raise ValueError(
            f"Conflicting rope_theta: attribute is {top_level} but rope "
            f"scaling/parameters carries {nested}."
        )
    if top_level is not None:
        return float(top_level)
    if nested is not None:
        return float(nested)
    return float(default)


def resolve_rope_scaling(config: Any) -> dict | None:
    """Prefer rope_scaling, then rope_parameters (transformers 5.x)."""
    for attr in ("rope_scaling", "rope_parameters"):
        value = getattr(config, attr, None)
        if value is None:
            continue
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "to_dict"):
            return value.to_dict()
    return None


def _rope_get(rope_scaling: dict | None, key: str, default: Any = None) -> Any:
    if rope_scaling is None:
        return default
    return rope_scaling.get(key, default)


class MLARotaryEmbedding(nn.Module):
    """Cached NeoX-layout cos/sin for MLA rope-side dims."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            max_position_embeddings + 20, self.inv_freq.device, torch.float32
        )

    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(
        self, x: torch.Tensor, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        assert seq_len is not None  # noqa: S101
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class MLAYarnRotaryEmbedding(MLARotaryEmbedding):
    """YaRN-scaled rotary cache for long-context MLA drafts."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.max_seq_len_cached = seq_len
        dim = self.dim
        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached",
            (emb.cos() * mscale)[None, None, :, :].to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (emb.sin() * mscale)[None, None, :, :].to(dtype),
            persistent=False,
        )


def build_mla_rotary_embedding(
    config: Any,
    dim: int,
    max_position_embeddings: int,
) -> MLARotaryEmbedding:
    """Build default or YaRN rotary embedding for the MLA rope-side dims."""
    rope_scaling = resolve_rope_scaling(config)
    rope_theta = resolve_rope_theta(config)
    scaling_type = _rope_get(rope_scaling, "rope_type", _rope_get(rope_scaling, "type"))

    if rope_scaling is None or scaling_type in (None, "default"):
        return MLARotaryEmbedding(
            dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    if scaling_type == "yarn":
        return MLAYarnRotaryEmbedding(
            dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            original_max_position_embeddings=_rope_get(
                rope_scaling, "original_max_position_embeddings", 4096
            ),
            scaling_factor=_rope_get(rope_scaling, "factor", 1.0),
            beta_fast=_rope_get(rope_scaling, "beta_fast", 32.0),
            beta_slow=_rope_get(rope_scaling, "beta_slow", 1.0),
            mscale=_rope_get(rope_scaling, "mscale", 1.0),
            mscale_all_dim=_rope_get(rope_scaling, "mscale_all_dim", 0.0),
        )

    raise ValueError(
        f"Unsupported RoPE scaling type for MLA draft attention: {scaling_type!r}. "
        "Supported: 'default', 'yarn'."
    )


def compute_mla_softmax_scale(config: Any, qk_head_dim: int) -> float:
    """Softmax scale with optional YaRN mscale_all_dim adjustment."""
    rope_scaling = resolve_rope_scaling(config)
    if rope_scaling is not None:
        scaling_type = _rope_get(
            rope_scaling, "rope_type", _rope_get(rope_scaling, "type")
        )
        if scaling_type == "yarn":
            factor = _rope_get(rope_scaling, "factor", 1.0)
            mscale_all_dim = _rope_get(rope_scaling, "mscale_all_dim", 0)
            mscale = yarn_get_mscale(factor, mscale_all_dim)
            return (mscale * mscale) / math.sqrt(qk_head_dim)
    return 1.0 / math.sqrt(qk_head_dim)
