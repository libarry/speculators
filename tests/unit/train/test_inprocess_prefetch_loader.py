"""Tests for Ascend-friendly concurrent in-process DataLoader prefetch."""

from __future__ import annotations

import threading
import time

import pytest

from speculators.train.dataloader import (
    ConcurrentInProcessLoader,
    InProcessPrefetchLoader,
    _use_inprocess_prefetch,
)


class _FakeLoader:
    def __init__(self, items: list[int], delay_s: float = 0.0):
        self._items = items
        self._delay_s = delay_s
        self.batch_sampler = object()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        for item in self._items:
            if self._delay_s:
                time.sleep(self._delay_s)
            yield item


class _ParallelDataset:
    """Dataset that records peak concurrent __getitem__ calls."""

    def __init__(self, delay_s: float = 0.05):
        self.delay_s = delay_s
        self._lock = threading.Lock()
        self._inflight = 0
        self.peak_inflight = 0
        self.approx_lengths = [1] * 8
        self.hidden_states_dtype = None

    def __getitem__(self, index: int) -> dict:
        with self._lock:
            self._inflight += 1
            self.peak_inflight = max(self.peak_inflight, self._inflight)
        try:
            time.sleep(self.delay_s)
            return {"idx": index}
        finally:
            with self._lock:
                self._inflight -= 1


class _ListBatchSampler:
    def __init__(self, batches: list[list[int]]):
        self._batches = batches

    def __iter__(self):
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


def test_legacy_inprocess_prefetch_yields_all_batches_in_order():
    base = _FakeLoader([1, 2, 3, 4])
    loader = InProcessPrefetchLoader(base, num_prefetch_batches=2)  # type: ignore[arg-type]
    assert loader.batch_sampler is base.batch_sampler
    assert len(loader) == 4
    assert list(loader) == [1, 2, 3, 4]


def test_legacy_inprocess_prefetch_propagates_producer_errors():
    class _BoomLoader:
        batch_sampler = None

        def __len__(self):
            return 1

        def __iter__(self):
            yield 1
            raise RuntimeError("boom")

    loader = InProcessPrefetchLoader(_BoomLoader(), num_prefetch_batches=1)  # type: ignore[arg-type]
    it = iter(loader)
    assert next(it) == 1
    with pytest.raises(RuntimeError, match="boom"):
        next(it)


def test_concurrent_loader_runs_num_workers_in_parallel():
    dataset = _ParallelDataset(delay_s=0.08)
    sampler = _ListBatchSampler([[0, 1, 2, 3], [4, 5, 6, 7]])

    def collate(samples):
        return {"idxs": [s["idx"] for s in samples]}

    loader = ConcurrentInProcessLoader(
        dataset,  # type: ignore[arg-type]
        sampler,
        collate,
        num_workers=4,
        prefetch_factor=1,
        pin_memory=False,
    )
    batches = list(loader)
    assert batches == [{"idxs": [0, 1, 2, 3]}, {"idxs": [4, 5, 6, 7]}]
    # 4 workers should overlap; allow some scheduling slack.
    assert dataset.peak_inflight >= 3


def test_concurrent_loader_prefetch_factor_keeps_batches_ahead():
    dataset = _ParallelDataset(delay_s=0.02)
    sampler = _ListBatchSampler([[0, 1], [2, 3], [4, 5]])

    def collate(samples):
        return {"idxs": [s["idx"] for s in samples]}

    loader = ConcurrentInProcessLoader(
        dataset,  # type: ignore[arg-type]
        sampler,
        collate,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=False,
    )
    t0 = time.perf_counter()
    out = []
    for batch in loader:
        out.append(batch)
        time.sleep(0.03)
    elapsed = time.perf_counter() - t0
    assert out == [
        {"idxs": [0, 1]},
        {"idxs": [2, 3]},
        {"idxs": [4, 5]},
    ]
    # Overlapped fetch+train should beat fully serial ~3*((2*0.02)+0.03).
    assert elapsed < 0.35


def test_use_inprocess_prefetch_env_and_ascend_protocol(monkeypatch):
    class _Cfg:
        protocol = "ascend"

    class _Store:
        config = _Cfg()

    class _Transfer:
        store = _Store()

    monkeypatch.delenv("SPECULATORS_INPROCESS_PREFETCH", raising=False)
    assert _use_inprocess_prefetch(_Transfer()) is True

    # Non-ascend protocols never use in-process mode (even with env=1).
    _Cfg.protocol = "tcp"
    monkeypatch.setenv("SPECULATORS_INPROCESS_PREFETCH", "1")
    assert _use_inprocess_prefetch(_Transfer()) is False
    _Cfg.protocol = "rdma"
    assert _use_inprocess_prefetch(_Transfer()) is False
    assert _use_inprocess_prefetch(None) is False

    # Ascend can be force-disabled.
    _Cfg.protocol = "ascend"
    monkeypatch.setenv("SPECULATORS_INPROCESS_PREFETCH", "0")
    assert _use_inprocess_prefetch(_Transfer()) is False
