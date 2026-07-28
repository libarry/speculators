"""Mooncake-backed store for hidden states, keyed by request id.

The file backend (``ExampleHiddenStatesConnector``) needs the vLLM target and
the trainer to share a filesystem; this stores the same
``{"hidden_states", "token_ids"}`` payload in a Mooncake store instead, so they
can run on different nodes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass

import torch


@dataclass
class MooncakeStoreConfig:
    """Connection settings, passed straight to ``MooncakeDistributedStore.setup``."""

    local_hostname: str = "localhost"
    metadata_server: str = "http://localhost:8080/metadata"
    master_server_address: str = "localhost:50051"
    global_segment_size: int = 4 * 1024 * 1024 * 1024
    local_buffer_size: int = 2 * 1024 * 1024 * 1024
    protocol: str = "tcp"
    device_name: str = ""
    num_writer_threads: int = 16

    @classmethod
    def from_dict(cls, d: dict | None) -> MooncakeStoreConfig:
        d = d or {}
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class MooncakeHiddenStatesStore:
    """Stores/loads tensor dicts in a Mooncake store.

    Each sample is written via ``put_tensor`` under ``{key}:{name}`` plus a
    ``{key}:meta`` JSON marker listing tensor names. ``meta`` is written last,
    so its presence marks the sample complete and ``get_sample`` can poll for it.
    """

    def __init__(self, config: MooncakeStoreConfig):
        self.config = config
        self._store = None

    def __getstate__(self):
        # Never pickle a live Mooncake client across DataLoader workers.
        state = self.__dict__.copy()
        state["_store"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._store = None

    @property
    def is_setup(self):
        return self._store is not None

    def reset(self) -> None:
        """Close and drop the process-local client and its mounted segment."""
        store, self._store = self._store, None
        if store is None:
            return
        close = getattr(store, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - reset must remain best-effort
                logging.getLogger(__name__).warning(
                    "Failed to close Mooncake client during reset",
                    exc_info=True,
                )

    def _ensure_accelerator_context(self) -> None:
        """Ascend Mooncake requires a live ACL device before store.setup().

        On Ascend builds, even ``protocol=tcp`` still installs
        ``AscendDirectTransport``; without a current NPU context,
        ``aclrtGetDevice`` fails and setup returns non-zero.
        """
        npu = getattr(torch, "npu", None)
        if npu is None or not callable(getattr(npu, "is_available", None)):
            return
        try:
            if not npu.is_available():
                return
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device_count = int(npu.device_count())
            if device_count <= 0:
                return
            # Dataloader workers share LOCAL_RANK with the trainer process;
            # ASCEND_RT_VISIBLE_DEVICES already isolates the visible device.
            npu.set_device(local_rank % device_count)
        except Exception:  # noqa: BLE001 - best-effort; setup() will report real failure
            return

    def _segment_sizes_for_process(self) -> tuple[int, int]:
        """Shrink mounted buffers when many DataLoader workers each own a client."""
        global_segment_size = self.config.global_segment_size
        local_buffer_size = self.config.local_buffer_size
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or worker_info.num_workers <= 1:
            return global_segment_size, local_buffer_size
        # Keep aggregate reservation near the single-client budget.
        n = worker_info.num_workers
        min_segment = 256 * 1024 * 1024
        min_buffer = 128 * 1024 * 1024
        return (
            max(min_segment, global_segment_size // n),
            max(min_buffer, local_buffer_size // n),
        )

    def setup(self) -> MooncakeHiddenStatesStore:
        if self._store is not None:
            return self
        try:
            from mooncake.store import (  # type: ignore[import-not-found] # noqa: PLC0415
                MooncakeDistributedStore,
            )
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "Mooncake is required for the Mooncake hidden-states backend. "
                "Install it with `pip install mooncake-transfer-engine`."
            ) from e

        self._ensure_accelerator_context()
        global_segment_size, local_buffer_size = self._segment_sizes_for_process()
        store = MooncakeDistributedStore()
        ret = store.setup(
            self.config.local_hostname,
            self.config.metadata_server,
            global_segment_size,
            local_buffer_size,
            self.config.protocol,
            self.config.device_name,
            self.config.master_server_address,
        )
        if ret != 0:
            raise RuntimeError(
                "Failed to initialize MooncakeDistributedStore "
                f"(return code {ret}). local_hostname={self.config.local_hostname!r} "
                f"metadata_server={self.config.metadata_server!r} "
                f"master_server_address={self.config.master_server_address!r} "
                f"protocol={self.config.protocol!r} "
                f"device_name={self.config.device_name!r} "
                f"global_segment_size={global_segment_size} "
                f"local_buffer_size={local_buffer_size}. "
                "On Ascend, use --mooncake-protocol ascend, a routable "
                "MOONCAKE_LOCAL_HOSTNAME (not 127.0.0.1), ensure an NPU is visible, "
                "and export HCCL_INTRA_ROCE_ENABLE=1 to avoid ADXL status 503900 "
                "on intra-node transfers."
            )
        self._store = store
        return self

    def put_sample(self, key: str, tensors: dict[str, torch.Tensor]) -> None:
        assert self._store is not None, "call setup() first"
        written_keys: list[str] = []
        try:
            for name, tensor in tensors.items():
                tensor_key = f"{key}:{name}"
                ret = self._store.put_tensor(
                    tensor_key, tensor.detach().to("cpu").contiguous()
                )
                self._check_result("put_tensor", tensor_key, ret)
                written_keys.append(tensor_key)

            # Publish the completion marker only after every tensor is durable.
            # Ignoring a failed tensor put used to publish a dangling meta key;
            # readers then repeatedly connected to an evicted/dead ADXL segment.
            meta_key = f"{key}:meta"
            ret = self._store.put(
                meta_key, json.dumps(list(tensors)).encode("utf-8")
            )
            self._check_result("put", meta_key, ret)
        except Exception:
            # Best effort rollback. Exact-key remove is intentional: on the
            # Ascend Mooncake build remove_by_regex may return without issuing
            # any Del RPC for a prefix-only regex.
            remove = getattr(self._store, "remove", None)
            if callable(remove):
                for written_key in written_keys:
                    try:
                        remove(written_key)
                    except Exception:  # noqa: BLE001
                        pass
            raise

    @staticmethod
    def _check_result(operation: str, key: str, result: object) -> None:
        """Raise when a Mooncake mutating operation reports failure."""
        # Some test doubles/older bindings return None on success; current
        # Mooncake bindings return integer 0.
        if result is not None and result != 0:
            raise RuntimeError(
                f"Mooncake {operation} failed for key {key!r}: return code {result}"
            )

    def get_sample(
        self,
        key: str,
        timeout: float = 120.0,
        poll_interval: float = 0.1,
        lock: threading.Lock | None = None,
    ) -> dict[str, torch.Tensor]:
        assert self._store is not None, "call setup() first"
        names = json.loads(
            self._wait_for(f"{key}:meta", timeout, poll_interval, lock=lock)
        )
        out: dict[str, torch.Tensor] = {}
        for name in names:
            if lock is not None:
                with lock:
                    out[name] = self._store.get_tensor(f"{key}:{name}")
            else:
                out[name] = self._store.get_tensor(f"{key}:{name}")
        return out

    def remove_sample(self, key: str) -> None:
        """Delete a sample with exact-key Del RPCs and verify return codes."""
        assert self._store is not None, "call setup() first"
        remove = getattr(self._store, "remove", None)
        if not callable(remove):
            raise RuntimeError("Mooncake store does not provide remove()")

        meta_key = f"{key}:meta"
        meta = self._store.get(meta_key)
        try:
            names = json.loads(meta) if meta else []
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(f"Invalid Mooncake metadata for {key!r}") from exc

        # Delete data first and the completion marker last. Using exact keys
        # makes deletion visible as Del RPCs in mooncake_master metrics and
        # avoids the no-op prefix-regex behavior observed on Ascend. These keys
        # were created by the vLLM producer, so the trainer is a different
        # Mooncake client and must use force=True; force=False returns -706.
        for name in names:
            tensor_key = f"{key}:{name}"
            self._check_result("remove", tensor_key, remove(tensor_key, True))
        self._check_result("remove", meta_key, remove(meta_key, True))

    def _wait_for(
        self,
        key: str,
        timeout: float,
        poll_interval: float,
        lock: threading.Lock | None = None,
    ) -> bytes:
        """Poll until ``key`` is readable.

        Prefer ``is_exist`` (master metadata, no ADXL) before ``get`` so we do
        not hammer AscendDirectTransport while the producer is still writing.
        Only hold ``lock`` around each store call, never across sleep.
        """
        assert self._store is not None, "call setup() first"
        deadline = time.monotonic() + timeout
        interval = max(poll_interval, 0.05)
        fail_streak = 0
        is_exist = getattr(self._store, "is_exist", None)

        while True:
            try:
                if lock is not None:
                    with lock:
                        exists = True
                        if callable(is_exist):
                            exists = bool(is_exist(key))
                        raw = self._store.get(key) if exists else b""
                else:
                    exists = True
                    if callable(is_exist):
                        exists = bool(is_exist(key))
                    raw = self._store.get(key) if exists else b""
                if raw:
                    return raw
                fail_streak = 0
            except Exception as exc:  # noqa: BLE001 - transient ADXL/master errors
                fail_streak += 1
                # Mooncake C++ already logs each failure; only surface occasionally.
                if fail_streak == 1 or fail_streak % 25 == 0:
                    logging.getLogger(__name__).warning(
                        "Mooncake get(%s) failed (retry %s): %s",
                        key,
                        fail_streak,
                        exc,
                    )

            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Mooncake key: {key}")
            time.sleep(interval)
            # Back off under repeated ADXL connect failures to avoid log storms.
            if fail_streak:
                interval = min(interval * 1.5, 1.0)
            else:
                interval = min(interval * 1.2, 0.5)
