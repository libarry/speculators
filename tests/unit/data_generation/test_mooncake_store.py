"""Unit tests for the Mooncake hidden-states store round-trip.

These exercise the producer/consumer payload contract without a real Mooncake
cluster by swapping in a dict-backed fake for ``MooncakeDistributedStore``.
The point is to prove the seam: a tensor dict written by the producer is read
back byte-identical by the consumer.
"""

import sys
import types

import pytest
import torch

# hs_connectors is an optional dependency (the mooncake extra); skip when absent.
pytest.importorskip("hs_connectors.mooncake_store")

from hs_connectors.mooncake_store import (
    MooncakeHiddenStatesStore,
    MooncakeStoreConfig,
)


class _FakeMooncakeStore:
    """In-memory stand-in for MooncakeDistributedStore."""

    def __init__(self):
        self._bytes: dict[str, bytes] = {}
        self._tensors: dict[str, torch.Tensor] = {}
        self.removed_keys: list[str] = []
        self.force_removed_keys: list[str] = []
        self.closed = False

    def put(self, key: str, value: bytes) -> int:
        self._bytes[key] = bytes(value)
        return 0

    def get(self, key: str) -> bytes:
        return self._bytes.get(key, b"")

    def is_exist(self, key: str) -> bool:
        return key in self._bytes or key in self._tensors

    def put_tensor(self, key: str, tensor: torch.Tensor) -> int:
        self._tensors[key] = tensor.clone()
        return 0

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._tensors[key].clone()

    def remove(self, key: str, force: bool = False) -> int:
        self.removed_keys.append(key)
        if force:
            self.force_removed_keys.append(key)
        self._bytes.pop(key, None)
        self._tensors.pop(key, None)
        return 0

    def remove_by_regex(self, pattern: str) -> int:
        import re

        rx = re.compile(pattern)
        for key in [k for k in list(self._bytes) if rx.search(k)]:
            self._bytes.pop(key, None)
        for key in [k for k in list(self._tensors) if rx.search(k)]:
            self._tensors.pop(key, None)
        return 0

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def store() -> MooncakeHiddenStatesStore:
    s = MooncakeHiddenStatesStore(MooncakeStoreConfig())
    # bypass setup(); no real cluster needed
    s._store = _FakeMooncakeStore()  # type: ignore[assignment]
    return s


def test_put_get_roundtrip_preserves_shape_and_dtype(store):
    # Mirrors the ExampleHiddenStatesConnector payload: [seq, n_layers, hidden]
    # bf16 hidden states + int64 token ids.
    hidden_states = torch.randn(7, 4, 16, dtype=torch.bfloat16)
    token_ids = torch.arange(7, dtype=torch.int64)

    store.put_sample("req-1", {"hidden_states": hidden_states, "token_ids": token_ids})
    out = store.get_sample("req-1", timeout=1.0)

    assert out.keys() == {"hidden_states", "token_ids"}
    assert out["hidden_states"].shape == hidden_states.shape
    assert out["hidden_states"].dtype == torch.bfloat16
    assert torch.equal(out["hidden_states"], hidden_states)
    assert torch.equal(out["token_ids"], token_ids)


def test_meta_written_last_gates_visibility(store):
    # get_sample keys off the meta blob, which put_sample writes last. Simulate
    # a half-written sample (tensors present, meta absent) -> consumer waits.
    store._store._tensors["req-2:hidden_states"] = torch.zeros(1)
    with pytest.raises(TimeoutError):
        store.get_sample("req-2", timeout=0.2, poll_interval=0.02)


def test_setup_raises_when_underlying_store_init_fails(monkeypatch):
    class _FailingMooncakeStore:
        def setup(self, *_args):
            return -1

    fake_store_module = types.ModuleType("mooncake.store")
    fake_store_module.MooncakeDistributedStore = _FailingMooncakeStore
    monkeypatch.setitem(
        sys.modules,
        "mooncake.store",
        fake_store_module,
    )

    s = MooncakeHiddenStatesStore(
        MooncakeStoreConfig(
            local_hostname="127.0.0.1",
            metadata_server="P2PHANDSHAKE",
            master_server_address="127.0.0.1:50051",
            protocol="tcp",
        )
    )

    with pytest.raises(RuntimeError, match="Failed to initialize MooncakeDistributedStore"):
        s.setup()


def test_pickle_drops_live_store(store):
    import pickle

    raw = pickle.dumps(store)
    restored = pickle.loads(raw)  # noqa: S301
    assert restored._store is None
    assert restored.config == store.config


def test_reset_closes_live_store(store):
    live_store = store._store
    assert live_store is not None
    store.reset()
    assert live_store.closed
    assert not store.is_setup


def test_segment_sizes_shrink_for_dataloader_workers(monkeypatch):
    s = MooncakeHiddenStatesStore(
        MooncakeStoreConfig(
            global_segment_size=4 * 1024 * 1024 * 1024,
            local_buffer_size=2 * 1024 * 1024 * 1024,
        )
    )

    class _WorkerInfo:
        num_workers = 8
        id = 0

    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: _WorkerInfo())
    segment, buffer = s._segment_sizes_for_process()
    assert segment == 512 * 1024 * 1024
    assert buffer == 256 * 1024 * 1024


def test_setup_client_does_not_stick_half_initialized(monkeypatch, tmp_path):
    from hs_connectors.transfer import MooncakeTransfer
    from speculators.train.data import ArrowDataset

    ds = __import__("datasets").Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "loss_mask": [[1, 1, 1]],
            "seq_len": [3],
        }
    )
    ds.save_to_disk(str(tmp_path / "data"))

    class _FailTransfer(MooncakeTransfer):
        def __init__(self):
            self.store = MooncakeHiddenStatesStore(MooncakeStoreConfig())

        def setup(self):
            raise RuntimeError("boom-setup")

    transfer = _FailTransfer()
    dataset = ArrowDataset(
        datapath=str(tmp_path / "data"),
        max_len=8,
        transfer=transfer,
        vllm_endpoint="http://127.0.0.1:9/v1",
        on_missing="generate",
    )

    class _FakeClient:
        @property
        def models(self):
            return types.SimpleNamespace(
                list=lambda: types.SimpleNamespace(
                    data=[types.SimpleNamespace(id="m")]
                )
            )

    monkeypatch.setattr(
        "speculators.train.data.openai.OpenAI",
        lambda **_kwargs: _FakeClient(),
    )

    assert dataset.client is None
    with pytest.raises(RuntimeError, match="boom-setup"):
        dataset._setup_client()
    assert dataset.client is None
    assert not transfer.is_setup


def test_remove_sample_deletes_meta_and_tensors(store):
    hidden_states = torch.randn(2, 1, 4, dtype=torch.bfloat16)
    token_ids = torch.arange(2, dtype=torch.int64)
    store.put_sample("req-del", {"hidden_states": hidden_states, "token_ids": token_ids})
    assert store._store.get("req-del:meta")
    store.remove_sample("req-del")
    assert store._store.get("req-del:meta") == b""
    assert "req-del:hidden_states" not in store._store._tensors
    assert store._store.removed_keys == [
        "req-del:hidden_states",
        "req-del:token_ids",
        "req-del:meta",
    ]
    assert store._store.force_removed_keys == store._store.removed_keys


def test_put_sample_does_not_publish_meta_after_tensor_put_failure(store):
    original_put_tensor = store._store.put_tensor
    calls = 0

    def fail_second_tensor(key, tensor):
        nonlocal calls
        calls += 1
        if calls == 2:
            return -200
        return original_put_tensor(key, tensor)

    store._store.put_tensor = fail_second_tensor
    with pytest.raises(RuntimeError, match="put_tensor.*-200"):
        store.put_sample(
            "req-full",
            {
                "hidden_states": torch.zeros(2, 1, 4),
                "token_ids": torch.arange(2),
            },
        )

    assert store._store.get("req-full:meta") == b""
    assert "req-full:hidden_states" not in store._store._tensors


def test_thread_prefetch_loader_preserves_order_and_len():
    from torch.utils.data import DataLoader, TensorDataset

    from speculators.train.dataloader import ParallelBatchPrefetchLoader

    class _BatchSampler:
        def __init__(self, n: int, bs: int):
            self.batches = [list(range(i, min(i + bs, n))) for i in range(0, n, bs)]

        def __iter__(self):
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)

    class _DS:
        def __getitem__(self, idx):
            return {"x": torch.tensor([idx]), "lengths": torch.tensor([1])}

    def _collate(samples):
        return {
            "x": torch.cat([s["x"] for s in samples]),
            "lengths": torch.cat([s["lengths"] for s in samples]),
        }

    sampler = _BatchSampler(6, 2)
    wrapped = ParallelBatchPrefetchLoader(
        dataset=_DS(),  # type: ignore[arg-type]
        batch_sampler=sampler,
        collate_fn=_collate,
        prefetch_factor=2,
    )
    assert len(wrapped) == 3
    out = [int(batch["x"][0]) for batch in wrapped]
    assert out == [0, 2, 4]
