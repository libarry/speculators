"""Mooncake-backed store for hidden states, keyed by request id.

The file backend (``ExampleHiddenStatesConnector``) needs the vLLM target and
the trainer to share a filesystem; this stores the same
``{"hidden_states", "token_ids"}`` payload in a Mooncake store instead, so they
can run on different nodes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Train-side consumers only pull sample tensors; huge segments multiply ADXL
# pressure when many clients are present. Producer (vLLM) keeps the large default.
# Used only when protocol == "ascend".
ASCEND_CONSUMER_GLOBAL_SEGMENT_SIZE = 1 * 1024 * 1024 * 1024
ASCEND_CONSUMER_LOCAL_BUFFER_SIZE = 512 * 1024 * 1024


@dataclass
class MooncakeStoreConfig:
    """Connection settings, passed straight to ``MooncakeDistributedStore.setup``."""

    local_hostname: str = "localhost"
    metadata_server: str = "P2PHANDSHAKE"
    master_server_address: str = "127.0.0.1:50051"
    global_segment_size: int = 4 * 1024 * 1024 * 1024
    local_buffer_size: int = 2 * 1024 * 1024 * 1024
    protocol: str = "tcp"
    device_name: str = ""
    num_writer_threads: int = 16

    @classmethod
    def from_dict(cls, d: dict | None) -> MooncakeStoreConfig:
        d = d or {}
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            logger.warning("Unknown MooncakeStoreConfig keys ignored: %s", unknown)
        return cls(**{k: v for k, v in d.items() if k in known})


class MooncakeHiddenStatesStore:
    """Stores/loads tensor dicts in a Mooncake store.

    Each sample is written via ``put_tensor`` under ``{key}:{name}`` plus a
    ``{key}:meta`` JSON marker listing tensor names. ``meta`` is written last,
    so its presence marks the sample complete and ``get_sample`` can poll for it.

    Thread locking is enabled only for ``protocol=ascend`` (in-process threaded
    prefetch). ``tcp``/``rdma`` keep a picklable no-op lock so DataLoader spawn
    workers behave as before.
    """

    def __init__(self, config: MooncakeStoreConfig):
        self.config = config
        self._store = None
        self._lock: Any
        if config.protocol == "ascend":
            # Ascend ADXL clients are not documented as multi-thread safe.
            self._lock = threading.RLock()
        else:
            self._lock = nullcontext()

    @property
    def is_setup(self):
        return self._store is not None

    def setup(self) -> MooncakeHiddenStatesStore:
        with self._lock:
            if self._store is not None:
                return self
            try:
                from mooncake.store import (  # type: ignore[import-not-found] # noqa: PLC0415
                    MooncakeDistributedStore,
                )
            except ImportError as e:  # pragma: no cover - optional dependency
                raise ImportError(
                    "Mooncake is required for the Mooncake hidden-states backend. "
                    "Install it with `pip install mooncake-transfer-engine` or "
                    "`pip install mooncake-transfer-engine-cuda-13`."
                ) from e

            store = MooncakeDistributedStore()
            rc = store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                self.config.global_segment_size,
                self.config.local_buffer_size,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
            )
            if rc != 0:
                raise RuntimeError(
                    f"MooncakeDistributedStore.setup failed with rc={rc} "
                    f"(protocol={self.config.protocol!r}, "
                    f"master={self.config.master_server_address!r}, "
                    f"hostname={self.config.local_hostname!r}). "
                    "On Ascend, ensure the calling process has an NPU context "
                    "(e.g. DataLoader worker_init_fn calls set_device_index)."
                )
            self._store = store
            return self

    def put_sample(self, key: str, tensors: dict[str, torch.Tensor]) -> None:
        if self._store is None:
            raise RuntimeError("call setup() first")
        names = []
        with self._lock:
            for name, tensor in tensors.items():
                self._store.put_tensor(
                    f"{key}:{name}", tensor.detach().to("cpu").contiguous()
                )
                names.append(name)
            self._store.put(f"{key}:meta", json.dumps(names).encode("utf-8"))

    def delete_sample(self, key: str) -> None:
        """Remove all keys for a sample from the store."""
        if self._store is None:
            raise RuntimeError("call setup() first")
        with self._lock:
            raw = self._store.get(f"{key}:meta")
            if not raw:
                return
            names = json.loads(raw)
            keys_to_remove = [f"{key}:{name}" for name in names] + [f"{key}:meta"]
            # Ascend/CANN Mooncake builds expose remove() but not batch_remove().
            batch_remove = getattr(self._store, "batch_remove", None)
            if callable(batch_remove):
                batch_remove(keys_to_remove, force=True)
                return
            for store_key in keys_to_remove:
                self._store.remove(store_key, force=True)

    def get_sample(
        self, key: str, timeout: float = 120.0, poll_interval: float = 0.05
    ) -> dict[str, torch.Tensor]:
        if self._store is None:
            raise RuntimeError("call setup() first")
        names = json.loads(self._wait_for(f"{key}:meta", timeout, poll_interval))
        result = {}
        with self._lock:
            for name in names:
                tensor = self._store.get_tensor(f"{key}:{name}")
                if tensor is None:
                    raise RuntimeError(f"Mooncake tensor evicted for key={key}:{name}")
                result[name] = tensor
        return result

    def _wait_for(self, key: str, timeout: float, poll_interval: float) -> bytes:
        if self._store is None:
            raise RuntimeError("call setup() first")
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                raw = self._store.get(key)
            if raw:
                return raw
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Mooncake key: {key}")
            time.sleep(poll_interval)
