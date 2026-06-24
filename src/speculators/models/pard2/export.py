"""Export PARD-2 checkpoints to PARD inference layout."""

from __future__ import annotations

import os

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from speculators.models.pard2.config import Pard2SpeculatorConfig

__all__ = ["convert_checkpoint_for_infer"]


def prepare_pard2_infer_config(
    cfg,
    scale: float | None = None,
    proj_bias: bool | None = None,
    target_dim: int | None = None,
    target_layers: list[int] | None = None,
    pard_token: int = -1,
):
    scale = scale if scale is not None else getattr(cfg, "feat_scale", 0.02)
    proj_bias = proj_bias if proj_bias is not None else getattr(cfg, "proj_bias", False)
    target_dim = target_dim if target_dim is not None else getattr(cfg, "target_feat_dim", 4096)
    target_layers = target_layers if target_layers is not None else getattr(
        cfg, "target_layer_ids", [-1, -8, -16, -24]
    )

    cfg.pard2 = True
    cfg.spd_type = "pard2"
    cfg.pard2_scale = float(scale)
    cfg.pard2_proj_bias = bool(proj_bias)
    cfg.pard2_target_dim = int(target_dim)
    cfg.pard2_target_layers = [int(layer) for layer in target_layers]
    cfg.pard_token = pard_token
    return cfg


def convert_checkpoint_for_infer(checkpoint_dir: str) -> dict:
    """Split a speculators/PARD-2 training checkpoint into pard_model + warp weights."""
    model_file = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"missing model.safetensors: {model_file}")

    model_path = os.path.join(checkpoint_dir, "pard_model")
    warp_model_path = os.path.join(checkpoint_dir, "pard_warp_model")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(warp_model_path, exist_ok=True)

    state = load_file(model_file, device="cpu")
    base_sd: dict[str, torch.Tensor] = {}
    warp_sd: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("draft_model."):
            base_sd[key[len("draft_model.") :]] = value
        elif key.startswith("base_model."):
            base_sd[key[len("base_model.") :]] = value
        elif key.startswith("target_proj."):
            warp_sd[key] = value

    metadata = {"source": model_file, "format": "pt"}
    save_file(base_sd, os.path.join(model_path, "model.safetensors"), metadata=metadata)
    save_file(warp_sd, os.path.join(warp_model_path, "model.safetensors"), metadata=metadata)
    torch.save(warp_sd, os.path.join(model_path, "warp_model.bin"))

    pard2_cfg = Pard2SpeculatorConfig.from_pretrained(checkpoint_dir)
    cfg = AutoConfig.from_pretrained(pard2_cfg.draft_name_or_path)
    prepare_pard2_infer_config(
        cfg,
        scale=pard2_cfg.feat_scale,
        proj_bias=pard2_cfg.proj_bias,
        target_dim=pard2_cfg.target_feat_dim,
        target_layers=pard2_cfg.target_layer_ids,
        pard_token=pard2_cfg.pard_token,
    )
    cfg.save_pretrained(model_path)

    return {
        "model_path": model_path,
        "base_tensors": len(base_sd),
        "warp_tensors": len(warp_sd),
    }
