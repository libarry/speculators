"""Unit tests for the Mooncake hidden-states store round-trip.

These exercise the producer/consumer payload contract without a real Mooncake
cluster by swapping in a dict-backed fake for ``MooncakeDistributedStore``.
The point is to prove the seam: a tensor dict written by the producer is read
back byte-identical by the consumer.
"""

import json

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

    def put(self, key: str, value: bytes) -> int:
        self._bytes[key] = bytes(value)
        return 0

    def get(self, key: str) -> bytes:
        return self._bytes.get(key, b"")

    def put_tensor(self, key: str, tensor: torch.Tensor) -> int:
        self._tensors[key] = tensor.clone()
        return 0

    def get_tensor(self, key: str) -> torch.Tensor | None:
        t = self._tensors.get(key)
        return t.clone() if t is not None else None

    def is_exist(self, key: str) -> bool:
        return key in self._bytes or key in self._tensors

    def batch_remove(self, keys: list[str], force: bool = False) -> list[int]:
        results = []
        for key in keys:
            results.append(self.remove(key, force=force))
        return results

    def remove(self, key: str, force: bool = False) -> int:  # noqa: ARG002
        removed = key in self._bytes or key in self._tensors
        self._bytes.pop(key, None)
        self._tensors.pop(key, None)
        return 0 if removed else -1


class _FakeMooncakeStoreRemoveOnly(_FakeMooncakeStore):
    """Mimics Ascend/CANN Mooncake which has remove() but not batch_remove()."""

    batch_remove = None  # type: ignore[assignment]


@pytest.fixture
def store() -> MooncakeHiddenStatesStore:
    s = MooncakeHiddenStatesStore(MooncakeStoreConfig())
    # bypass setup(); no real cluster needed
    s._store = _FakeMooncakeStore()  # type: ignore[assignment]
    return s


@pytest.fixture
def store_remove_only() -> MooncakeHiddenStatesStore:
    s = MooncakeHiddenStatesStore(MooncakeStoreConfig())
    s._store = _FakeMooncakeStoreRemoveOnly()  # type: ignore[assignment]
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


def test_delete_sample_removes_all_keys(store):
    hs = torch.randn(4, 2, 8, dtype=torch.bfloat16)
    tids = torch.arange(4, dtype=torch.int64)
    store.put_sample("req-del", {"hidden_states": hs, "token_ids": tids})

    store.delete_sample("req-del")

    assert store._store.get("req-del:meta") == b""
    assert store._store.get_tensor("req-del:hidden_states") is None
    assert store._store.get_tensor("req-del:token_ids") is None


def test_delete_sample_noop_when_missing(store):
    store.delete_sample("nonexistent-key")


def test_setup_raises_when_mooncake_returns_nonzero(monkeypatch):
    class _FailingStore:
        def setup(self, *args, **kwargs):  # noqa: ARG002
            return -1

    monkeypatch.setattr(
        "mooncake.store.MooncakeDistributedStore",
        _FailingStore,
        raising=False,
    )
    # Patch the import path used inside setup()
    import sys
    import types

    fake_mod = types.ModuleType("mooncake.store")
    fake_mod.MooncakeDistributedStore = _FailingStore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake.store", fake_mod)
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))

    s = MooncakeHiddenStatesStore(MooncakeStoreConfig())
    with pytest.raises(RuntimeError, match="setup failed with rc=-1"):
        s.setup()


def test_delete_sample_falls_back_to_remove(store_remove_only):
    hs = torch.randn(4, 2, 8, dtype=torch.bfloat16)
    tids = torch.arange(4, dtype=torch.int64)
    store_remove_only.put_sample(
        "req-rm", {"hidden_states": hs, "token_ids": tids}
    )

    store_remove_only.delete_sample("req-rm")

    assert store_remove_only._store.get("req-rm:meta") == b""
    assert store_remove_only._store.get_tensor("req-rm:hidden_states") is None
    assert store_remove_only._store.get_tensor("req-rm:token_ids") is None


def test_get_sample_reports_missing_tensor(store):
    hs = torch.randn(4, 2, 8, dtype=torch.bfloat16)
    tids = torch.arange(4, dtype=torch.int64)
    store.put_sample("req-evict", {"hidden_states": hs, "token_ids": tids})

    # Simulate eviction: meta key survives but tensor data is gone
    del store._store._tensors["req-evict:hidden_states"]

    with pytest.raises(RuntimeError, match="key missing/evicted"):
        store.get_sample("req-evict", timeout=1.0)


def test_chunked_put_get_roundtrip():
    # Force multiple chunks along seq dim: 16 rows * 2 * 8 * 2 bytes = 512 B
    # with max_bytes=200 -> ~6 rows/chunk.
    s = MooncakeHiddenStatesStore(MooncakeStoreConfig(hs_chunk_bytes=200))
    s._store = _FakeMooncakeStore()  # type: ignore[assignment]

    hidden_states = torch.randn(16, 2, 8, dtype=torch.bfloat16)
    token_ids = torch.arange(16, dtype=torch.int64)
    s.put_sample("req-chunk", {"hidden_states": hidden_states, "token_ids": token_ids})

    # Meta must be v2 because hidden_states was split.
    meta = json.loads(s._store.get("req-chunk:meta"))
    assert isinstance(meta, dict)
    assert meta["v"] == 2
    hs_entry = next(e for e in meta["tensors"] if e["name"] == "hidden_states")
    assert hs_entry["n_chunks"] > 1
    # token_ids is tiny → single unchunked key
    assert "req-chunk:token_ids" in s._store._tensors
    assert "req-chunk:hidden_states:0" in s._store._tensors

    out = s.get_sample("req-chunk", timeout=1.0)
    assert out["hidden_states"].shape == hidden_states.shape
    assert torch.equal(out["hidden_states"], hidden_states)
    assert torch.equal(out["token_ids"], token_ids)


def test_chunked_delete_removes_part_keys():
    s = MooncakeHiddenStatesStore(MooncakeStoreConfig(hs_chunk_bytes=200))
    s._store = _FakeMooncakeStore()  # type: ignore[assignment]
    hs = torch.randn(16, 2, 8, dtype=torch.bfloat16)
    tids = torch.arange(16, dtype=torch.int64)
    s.put_sample("req-cdel", {"hidden_states": hs, "token_ids": tids})
    part_keys = [k for k in s._store._tensors if k.startswith("req-cdel:hidden_states")]
    assert len(part_keys) > 1

    s.delete_sample("req-cdel")
    assert s._store.get("req-cdel:meta") == b""
    for k in part_keys:
        assert s._store.get_tensor(k) is None


def test_legacy_meta_still_readable(store):
    # Old producers wrote a bare name list + single-key tensors.
    hs = torch.randn(3, 2, 4, dtype=torch.bfloat16)
    tids = torch.arange(3, dtype=torch.int64)
    store._store.put_tensor("req-legacy:hidden_states", hs)
    store._store.put_tensor("req-legacy:token_ids", tids)
    store._store.put(
        "req-legacy:meta", json.dumps(["hidden_states", "token_ids"]).encode()
    )

    out = store.get_sample("req-legacy", timeout=1.0)
    assert torch.equal(out["hidden_states"], hs)
    assert torch.equal(out["token_ids"], tids)


def test_get_tensor_retries_transient_transfer_failure():
    class _FailOnceStore(_FakeMooncakeStore):
        def __init__(self):
            super().__init__()
            self.get_attempts: dict[str, int] = {}

        def get_tensor(self, key: str) -> torch.Tensor | None:
            self.get_attempts[key] = self.get_attempts.get(key, 0) + 1
            if self.get_attempts[key] == 1:
                return None
            return super().get_tensor(key)

    s = MooncakeHiddenStatesStore(
        MooncakeStoreConfig(
            hs_transfer_max_retries=2,
            hs_transfer_retry_backoff=0,
        )
    )
    s._store = _FailOnceStore()  # type: ignore[assignment]
    hs = torch.randn(4, 2, 8, dtype=torch.bfloat16)
    tids = torch.arange(4, dtype=torch.int64)
    s.put_sample("req-retry", {"hidden_states": hs, "token_ids": tids})

    out = s.get_sample("req-retry", timeout=1.0)

    assert torch.equal(out["hidden_states"], hs)
    assert s._store.get_attempts["req-retry:hidden_states"] == 2


def test_put_failure_cleans_partial_keys_and_publishes_error():
    class _FailSecondChunkStore(_FakeMooncakeStore):
        def put_tensor(self, key: str, tensor: torch.Tensor) -> int:
            if key.endswith("hidden_states:1"):
                return -800
            return super().put_tensor(key, tensor)

    s = MooncakeHiddenStatesStore(
        MooncakeStoreConfig(
            hs_chunk_bytes=200,
            hs_transfer_max_retries=1,
            hs_transfer_retry_backoff=0,
        )
    )
    s._store = _FailSecondChunkStore()  # type: ignore[assignment]
    hs = torch.randn(16, 2, 8, dtype=torch.bfloat16)
    tids = torch.arange(16, dtype=torch.int64)

    with pytest.raises(RuntimeError, match="put_tensor failed"):
        s.put_sample("req-put-fail", {"hidden_states": hs, "token_ids": tids})

    assert s._store.get("req-put-fail:meta") == b""
    assert s._store.get("req-put-fail:error")
    assert not any(k.startswith("req-put-fail:") for k in s._store._tensors)


def test_consumer_reports_producer_error_without_meta_timeout():
    s = MooncakeHiddenStatesStore(MooncakeStoreConfig())
    s._store = _FakeMooncakeStore()  # type: ignore[assignment]
    s._store.put(
        "req-producer-error:error",
        json.dumps({"error": "put chunk 1 rc=-800"}).encode(),
    )

    with pytest.raises(RuntimeError, match="producer failed.*rc=-800"):
        s.get_sample("req-producer-error", timeout=1.0)
