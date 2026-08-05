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

# Meta schema version for chunked payloads. Legacy meta is a bare JSON list of
# tensor names (single key ``{sample}:{name}`` per tensor).
_META_VERSION = 2


@dataclass
class MooncakeStoreConfig:
    """Connection settings, passed straight to ``MooncakeDistributedStore.setup``."""

    local_hostname: str = "localhost"
    metadata_server: str = "P2PHANDSHAKE"
    master_server_address: str = "127.0.0.1:50051"
    global_segment_size: int = 16 * 1024 * 1024 * 1024
    local_buffer_size: int = 8 * 1024 * 1024 * 1024
    protocol: str = "tcp"
    device_name: str = ""
    num_writer_threads: int = 16
    # Soft upper bound (bytes) per Mooncake put/get for large tensors such as
    # hidden_states. Split roughly along dim 0; 0 disables chunking.
    hs_chunk_bytes: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> MooncakeStoreConfig:
        d = d or {}
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            logger.warning("Unknown MooncakeStoreConfig keys ignored: %s", unknown)
        return cls(**{k: v for k, v in d.items() if k in known})


def _split_tensor_rough(tensor: torch.Tensor, max_bytes: int) -> list[torch.Tensor]:
    """Split ``tensor`` along dim 0 so each piece is roughly <= ``max_bytes``.

    Uses ``nbytes / shape[0]`` as the per-row size; does not aim for an exact
    byte budget (padding / non-contiguous layouts can differ slightly).
    """
    if max_bytes <= 0 or tensor.numel() == 0 or tensor.nbytes <= max_bytes:
        return [tensor]
    if tensor.dim() == 0 or tensor.shape[0] <= 1:
        return [tensor]

    row_nbytes = max(1, tensor.nbytes // int(tensor.shape[0]))
    rows_per_chunk = max(1, max_bytes // row_nbytes)
    if rows_per_chunk >= int(tensor.shape[0]):
        return [tensor]
    return list(tensor.split(rows_per_chunk, dim=0))


def _tensor_keys(sample_key: str, name: str, n_chunks: int) -> list[str]:
    if n_chunks <= 1:
        return [f"{sample_key}:{name}"]
    return [f"{sample_key}:{name}:{i}" for i in range(n_chunks)]


def _encode_meta(entries: list[dict[str, Any]], *, chunked: bool) -> bytes:
    if not chunked:
        # Legacy: list of names. Consumers that only understand this format
        # still work when nothing was split.
        return json.dumps([e["name"] for e in entries]).encode("utf-8")
    return json.dumps({"v": _META_VERSION, "tensors": entries}).encode("utf-8")


def _parse_meta(raw: bytes) -> list[tuple[str, int]]:
    """Return ``[(name, n_chunks), ...]`` from either legacy or v2 meta."""
    meta = json.loads(raw)
    if isinstance(meta, list):
        return [(name, 1) for name in meta]
    if isinstance(meta, dict) and "tensors" in meta:
        out: list[tuple[str, int]] = []
        for entry in meta["tensors"]:
            out.append((entry["name"], int(entry.get("n_chunks", 1))))
        return out
    raise ValueError(f"Unrecognized Mooncake sample meta: {meta!r}")


class MooncakeHiddenStatesStore:
    """Stores/loads tensor dicts in a Mooncake store.

    Each sample is written via ``put_tensor`` under ``{key}:{name}`` (or
    ``{key}:{name}:{i}`` when chunked) plus a ``{key}:meta`` JSON marker.
    ``meta`` is written last, so its presence marks the sample complete and
    ``get_sample`` can poll for it.

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
        max_bytes = int(self.config.hs_chunk_bytes)
        entries: list[dict[str, Any]] = []
        any_chunked = False
        with self._lock:
            for name, tensor in tensors.items():
                host = tensor.detach().to("cpu").contiguous()
                parts = _split_tensor_rough(host, max_bytes)
                n_chunks = len(parts)
                if n_chunks > 1:
                    any_chunked = True
                for store_key, part in zip(
                    _tensor_keys(key, name, n_chunks), parts, strict=True
                ):
                    self._store.put_tensor(store_key, part)
                entries.append({"name": name, "n_chunks": n_chunks})
            self._store.put(
                f"{key}:meta", _encode_meta(entries, chunked=any_chunked)
            )

    def delete_sample(self, key: str) -> None:
        """Remove all keys for a sample from the store."""
        if self._store is None:
            raise RuntimeError("call setup() first")
        with self._lock:
            raw = self._store.get(f"{key}:meta")
            if not raw:
                return
            keys_to_remove: list[str] = [f"{key}:meta"]
            for name, n_chunks in _parse_meta(raw):
                keys_to_remove.extend(_tensor_keys(key, name, n_chunks))
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
        entries = _parse_meta(self._wait_for(f"{key}:meta", timeout, poll_interval))
        result: dict[str, torch.Tensor] = {}
        with self._lock:
            for name, n_chunks in entries:
                parts: list[torch.Tensor] = []
                for store_key in _tensor_keys(key, name, n_chunks):
                    tensor = self._store.get_tensor(store_key)
                    if tensor is None:
                        raise RuntimeError(
                            f"Mooncake tensor evicted for key={store_key}"
                        )
                    parts.append(tensor)
                result[name] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
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
