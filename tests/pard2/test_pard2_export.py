"""Tests for PARD-2 checkpoint export."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("transformers")

try:
    from speculators.models.pard2.export import convert_checkpoint_for_infer
except ImportError as exc:
    pytest.skip(f"speculators import unavailable: {exc}", allow_module_level=True)


def test_convert_checkpoint_for_infer(tmp_path: Path, monkeypatch):
    state = {
        "draft_model.model.layers.0.weight": torch.randn(4, 4),
        "target_proj.weight": torch.randn(8, 16),
    }
    save_file(state, tmp_path / "model.safetensors")

    draft_cfg_dir = tmp_path / "draft_cfg"
    draft_cfg_dir.mkdir()
    draft_cfg = {
        "model_type": "gpt2",
        "vocab_size": 64,
        "n_positions": 128,
        "n_embd": 32,
        "n_layer": 1,
        "n_head": 1,
    }
    (draft_cfg_dir / "config.json").write_text(json.dumps(draft_cfg), encoding="utf-8")

    config = {
        "speculators_model_type": "pard2",
        "draft_name_or_path": str(draft_cfg_dir),
        "feat_scale": 0.02,
        "proj_bias": False,
        "target_feat_dim": 16,
        "target_layer_ids": [-1, -2],
        "pard_token": 99,
        "speculators_config": {
            "algorithm": "pard2",
            "proposal_methods": [
                {"proposal_type": "greedy", "speculative_tokens": 2}
            ],
            "default_proposal_method": "greedy",
            "verifier": {"name_or_path": str(draft_cfg_dir), "architectures": []},
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    result = convert_checkpoint_for_infer(str(tmp_path))
    assert (Path(result["model_path"]) / "warp_model.bin").exists()
    assert (Path(result["model_path"]) / "model.safetensors").exists()
    infer_cfg = json.loads((Path(result["model_path"]) / "config.json").read_text())
    assert infer_cfg.get("pard2") is True
    assert infer_cfg.get("draft_name_or_path") is None
