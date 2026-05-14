import pytest

from speculators.train import utils as train_utils


class _FakeMesh:
    def __getitem__(self, _key):
        return self

    def get_local_rank(self):
        return 1

    def size(self):
        return 2


def test_cp_disabled_by_default():
    cp_config = train_utils._setup_context_parallel(
        cp_size=1, cp_mode="none", rank=0
    )
    assert cp_config.enabled is False
    assert cp_config.mode == "none"
    assert cp_config.size == 1


def test_cp_none_mode_requires_size_one():
    with pytest.raises(ValueError, match="`cp_size` must be 1"):
        train_utils._setup_context_parallel(cp_size=2, cp_mode="none", rank=0)


def test_cp_requires_distributed(monkeypatch):
    monkeypatch.setattr(train_utils, "is_distributed", False)
    with pytest.raises(ValueError, match="requires distributed launch"):
        train_utils._setup_context_parallel(
            cp_size=2,
            cp_mode="context_parallel",
            rank=0,
        )


def test_cp_requires_world_size_divisible(monkeypatch):
    monkeypatch.setattr(train_utils, "is_distributed", True)
    monkeypatch.setattr(train_utils, "world_size", 3)
    with pytest.raises(ValueError, match="must be divisible"):
        train_utils._setup_context_parallel(
            cp_size=2,
            cp_mode="context_parallel",
            rank=0,
        )


def test_cp_setup_success_smoke(monkeypatch):
    monkeypatch.setattr(train_utils, "is_distributed", True)
    monkeypatch.setattr(train_utils, "world_size", 2)
    monkeypatch.setattr(train_utils.torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(
        train_utils.torch.accelerator,
        "current_accelerator",
        lambda: type("FakeAcc", (), {"type": "cuda"})(),
    )
    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh",
        lambda *_args, **_kwargs: _FakeMesh(),
    )

    cp_config = train_utils._setup_context_parallel(
        cp_size=2,
        cp_mode="context_parallel",
        rank=1,
    )
    assert cp_config.enabled is True
    assert cp_config.size == 2
    assert cp_config.rank == 1


def test_cp_setup_on_npu_accelerator(monkeypatch):
    monkeypatch.setattr(train_utils, "is_distributed", True)
    monkeypatch.setattr(train_utils, "world_size", 2)
    monkeypatch.setattr(train_utils.torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(
        train_utils.torch.accelerator,
        "current_accelerator",
        lambda: type("FakeAcc", (), {"type": "npu"})(),
    )
    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh",
        lambda *_args, **_kwargs: _FakeMesh(),
    )
    cp_config = train_utils._setup_context_parallel(
        cp_size=2,
        cp_mode="context_parallel",
        rank=0,
    )
    assert cp_config.enabled is True
