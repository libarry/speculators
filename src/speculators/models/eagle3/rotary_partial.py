"""Patch HF rotary helper to match vLLM partial-neox behavior.

HF ``apply_rotary_pos_emb`` rotates by splitting at ``head_dim/2``.
vLLM partial MRoPE rotates only the first ``rotary_dim`` channels and
keeps the tail unchanged. This file aligns HF training with that runtime
behavior, while keeping full-rotation paths unchanged.
"""

from __future__ import annotations

import torch

__all__ = [
    "apply_neox_rotary",
    "install_partial_neox_rotary",
    "partial_neox_apply_rotary_pos_emb",
]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF/neox "rotate_half" — splits the last dim in half and swaps."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_neox_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply NeoX RoPE to one Q/K tensor, with vLLM partial-rotary fallback.

    - If ``cos`` covers full head dim, behavior matches HF ``apply_rotary_pos_emb``.
    - If ``cos`` is shorter, rotate only the first ``rotary_dim`` channels and
      keep the remaining channels unchanged.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    if cos.shape[-1] == x.shape[-1]:
        return (x * cos) + (_rotate_half(x) * sin)

    if cos.shape[-1] > x.shape[-1]:
        raise ValueError(
            f"cos last dim ({cos.shape[-1]}) exceeds q last dim "
            f"({x.shape[-1]}); rotary tables larger than head_dim are "
            "unsupported by this partial-neox replacement."
        )

    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
    return torch.cat([x_rot, x_pass], dim=-1)


def partial_neox_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF-compatible rotary helper with partial-neox fallback.

    - If ``cos`` covers full head dim, behavior matches HF.
    - If ``cos`` is shorter, rotate only the first ``rotary_dim`` channels
      and keep the remaining channels unchanged.
    """
    return (
        apply_neox_rotary(q, cos, sin, unsqueeze_dim=unsqueeze_dim),
        apply_neox_rotary(k, cos, sin, unsqueeze_dim=unsqueeze_dim),
    )


_PATCH_STATE = {"installed": False}
_ORIGINAL_APPLY_ATTR = "_speculators_original_apply_rotary_pos_emb"


def install_partial_neox_rotary() -> None:
    """Patch ``apply_rotary_pos_emb`` in HF ``llama`` and ``qwen3`` modules.

    Idempotent. Full-rotation paths keep original behavior.
    """
    if _PATCH_STATE["installed"]:
        return

    # Local imports — keep transformers a soft dep at module import time.
    from transformers.models.llama import modeling_llama  # noqa: PLC0415
    from transformers.models.qwen3 import modeling_qwen3  # noqa: PLC0415

    for module in (modeling_llama, modeling_qwen3):
        original = module.apply_rotary_pos_emb  # type: ignore[attr-defined]
        # Cache original for tests / debugging — and to allow uninstall.
        if not hasattr(module, _ORIGINAL_APPLY_ATTR):
            setattr(module, _ORIGINAL_APPLY_ATTR, original)
        module.apply_rotary_pos_emb = partial_neox_apply_rotary_pos_emb  # type: ignore[attr-defined]

    _PATCH_STATE["installed"] = True


def uninstall_partial_neox_rotary() -> None:
    """Restore HF's original ``apply_rotary_pos_emb``. Test/debug helper."""
    if not _PATCH_STATE["installed"]:
        return
    from transformers.models.llama import modeling_llama  # noqa: PLC0415
    from transformers.models.qwen3 import modeling_qwen3  # noqa: PLC0415

    for module in (modeling_llama, modeling_qwen3):
        original = getattr(module, _ORIGINAL_APPLY_ATTR, None)
        if original is not None:
            module.apply_rotary_pos_emb = original  # type: ignore[attr-defined]
    _PATCH_STATE["installed"] = False
