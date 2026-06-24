"""Smoke test for PARD-2 training loop with synthetic data."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from safetensors.torch import save_file
from torch.utils.data import DataLoader

pytest.importorskip("transformers")

try:
    from transformers import LlamaConfig

    from speculators.models.pard2.core import Pard2DraftModel
    from speculators.models.pard2.data import (
        create_pard2_collate_fn,
        preprocess_pard2_sample,
    )
    from speculators.train.data import ArrowDataset
    from speculators.train.trainer import Trainer, TrainerConfig
except ImportError as exc:
    pytest.skip(f"speculators import unavailable: {exc}", allow_module_level=True)


def _tiny_verifier_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
    )


def _make_synthetic_arrow_dataset(tmp_path: Path, num_samples: int = 4, seq_len: int = 16):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    data = {
        "input_ids": [],
        "loss_mask": [],
        "seq_len": [],
    }
    hs_dir = data_dir / "hidden_states"
    hs_dir.mkdir(parents=True)

    for idx in range(num_samples):
        input_ids = torch.arange(10, 10 + seq_len, dtype=torch.long)
        loss_mask = torch.zeros(seq_len, dtype=torch.bool)
        loss_mask[seq_len // 2 :] = True
        hs = torch.randn(seq_len, 4, 32, dtype=torch.bfloat16)
        save_file(
            {
                "hidden_states": hs,
                "token_ids": input_ids,
            },
            hs_dir / f"hs_{idx}.safetensors",
        )
        data["input_ids"].append(input_ids.tolist())
        data["loss_mask"].append(loss_mask.tolist())
        data["seq_len"].append(seq_len)

    Dataset.from_dict(data).save_to_disk(str(data_dir))
    return data_dir


def test_pard2_smoke_train(monkeypatch):
    tiny_cfg = _tiny_verifier_config()

    class _FakeBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = tiny_cfg
            self.embed = torch.nn.Embedding(64, 32)
            layer = torch.nn.Linear(32, 32)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([layer])
            self.lm_head = torch.nn.Linear(32, 64, bias=False)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None, **kwargs):
            del attention_mask, position_ids, kwargs
            logits = self.lm_head(inputs_embeds)
            return type("Out", (), {"logits": logits})()

    def _fake_from_pretrained(path, **kwargs):
        del path, kwargs
        return _FakeBase()

    monkeypatch.setattr(
        "speculators.models.pard2.core.AutoConfig.from_pretrained",
        lambda path, **kwargs: tiny_cfg,
    )
    monkeypatch.setattr(
        "speculators.models.pard2.core.AutoModelForCausalLM.from_pretrained",
        _fake_from_pretrained,
    )
    monkeypatch.setattr(
        "speculators.models.pard2.core.load_verifier_lm_head",
        lambda path: torch.nn.Linear(32, 64, bias=False),
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_path = _make_synthetic_arrow_dataset(tmp_path)
        dataset = ArrowDataset(
            datapath=data_path,
            max_len=32,
            on_missing="raise",
            concat_all_hidden_layers=True,
        )
        model = Pard2DraftModel.from_training_args(
            verifier_config=tiny_cfg,
            verifier_name_or_path="fake-verifier",
            draft_name_or_path="fake-draft",
            para_num=2,
            mask_token_ids=[63, 63],
            target_layer_ids=[-1, -2, -3, -4],
            feat_scale=0.02,
            target_feat_mask=0.0,
        )
        collate = create_pard2_collate_fn(
            max_len=32,
            hidden_size=32,
            num_target_layers=4,
            para_num=2,
            unused_tokenids=[63, 63],
            down_sample_ratio=1.0,
            down_sample_ratio_min=0.0,
            lm_head_weight=model.verifier_lm_head.weight.detach().cpu().float(),
            lm_head_bias=None,
            preprocess=preprocess_pard2_sample,
        )
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate)
        batch = next(iter(loader))
        if batch.get("attention_mask") is not None:
            batch["attention_mask"] = batch["attention_mask"].float()
        _out, loss, metrics = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            position_ids=batch["position_ids"],
            labels=batch["labels"],
            target_feat=batch["target_feat"],
            teacher_hidden=batch["teacher_hidden"],
            prev_prob=batch["prev_prob"],
        )
        assert loss.ndim == 0
        assert "loss_sum" in metrics

        save_path = tmp_path / "ckpt"
        trainer = Trainer(
            model,
            TrainerConfig(
                lr=1e-4,
                num_epochs=1,
                save_path=str(save_path),
                train_call_kwargs={},
                val_call_kwargs={},
                checkpoint_freq=1,
                log_freq=1,
            ),
            train_loader=loader,
            val_loader=None,
        )
        trainer.run_training()
