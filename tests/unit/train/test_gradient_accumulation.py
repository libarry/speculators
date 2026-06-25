"""Unit tests for gradient accumulation in the training loop."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import torch
from torch.utils.data import DataLoader

from speculators.model import SpeculatorModel
from speculators.train.trainer import Trainer, TrainerConfig


class _StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def train(self, mode: bool = True):
        return self

    def forward(self, x=None, **kwargs):
        if x is None:
            x = kwargs["x"]
        loss = (self.weight * x).pow(2).mean()
        metrics = {
            "loss_sum": loss.detach().clone(),
            "loss_total": torch.tensor(1.0),
        }
        return None, loss, metrics


class _ScalarBatchDataset:
    def __init__(self, values: list[float]):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx: int):
        return {"x": torch.tensor(self.values[idx])}


def _collate_scalar_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {"x": torch.stack([item["x"] for item in batch])}


def test_train_epoch_steps_optimizer_every_accumulation(tmp_path):
    model = _StubModel()
    loader = DataLoader(
        _ScalarBatchDataset([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        batch_size=1,
        collate_fn=_collate_scalar_batch,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    step_counter = {"count": 0}
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        step_counter["count"] += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step  # type: ignore[method-assign]

    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainerConfig(
        lr=1e-2,
        num_epochs=1,
        save_path=str(tmp_path),
        resume_from_checkpoint=False,
        is_distributed=False,
        rank=0,
        local_rank=0,
        scheduler_type="none",
        gradient_accumulation_steps=2,
        log_freq=1,
    )
    trainer.model = cast("SpeculatorModel", model)
    trainer.local_rank = torch.device("cpu")
    trainer.rank = 0
    trainer.is_distributed = False
    trainer.resume_from_checkpoint = False
    trainer.train_loader = loader
    trainer.val_loader = None
    trainer.global_step = 0
    trainer.checkpointer = MagicMock()
    trainer.optimizers = [optimizer]
    trainer.schedulers = []

    trainer.train_epoch(0)

    assert step_counter["count"] == 3
    assert trainer.global_step == 3
