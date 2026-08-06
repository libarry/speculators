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
    # Number of retries after the first failed put/get attempt.
    hs_transfer_max_retries: int = 3
    hs_transfer_retry_backoff: float = 1.0

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


def _parse_meta(raw: bytes) -> list[dict[str, Any]]:
    """Return normalized tensor entries from either legacy or v2 meta."""
    meta = json.loads(raw)
    if isinstance(meta, list):
        return [{"name": name, "n_chunks": 1} for name in meta]
    if isinstance(meta, dict) and "tensors" in meta:
        out: list[dict[str, Any]] = []
        for entry in meta["tensors"]:
            normalized = dict(entry)
            normalized["n_chunks"] = int(entry.get("n_chunks", 1))
            out.append(normalized)
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
            logger.info(
                "Mooncake store ready: protocol=%s master=%s hostname=%s "
                "global_segment_size=%d local_buffer_size=%d hs_chunk_bytes=%d "
                "hs_transfer_max_retries=%d",
                self.config.protocol,
                self.config.master_server_address,
                self.config.local_hostname,
                self.config.global_segment_size,
                self.config.local_buffer_size,
                self.config.hs_chunk_bytes,
                self.config.hs_transfer_max_retries,
            )
            return self

    def _retry_delay(self, failed_attempt: int) -> float:
        return max(0.0, self.config.hs_transfer_retry_backoff) * (2**failed_attempt)

    def _put_tensor(self, store_key: str, tensor: torch.Tensor) -> None:
        if self._store is None:
            raise RuntimeError("call setup() first")
        max_attempts = max(1, int(self.config.hs_transfer_max_retries) + 1)
        last_rc: Any = None
        for attempt in range(max_attempts):
            started = time.monotonic()
            last_rc = self._store.put_tensor(store_key, tensor)
            elapsed = time.monotonic() - started
            if last_rc == 0:
                logger.info(
                    "Mooncake put_tensor succeeded: key=%s bytes=%d "
                    "attempt=%d/%d elapsed=%.3fs",
                    store_key,
                    tensor.nbytes,
                    attempt + 1,
                    max_attempts,
                    elapsed,
                )
                return
            logger.warning(
                "Mooncake put_tensor failed: key=%s bytes=%d rc=%r "
                "attempt=%d/%d elapsed=%.3fs protocol=%s master=%s",
                store_key,
                tensor.nbytes,
                last_rc,
                attempt + 1,
                max_attempts,
                elapsed,
                self.config.protocol,
                self.config.master_server_address,
            )
            if attempt + 1 < max_attempts:
                time.sleep(self._retry_delay(attempt))
        raise RuntimeError(
            "Mooncake put_tensor failed after "
            f"{max_attempts} attempts: key={store_key}, bytes={tensor.nbytes}, "
            f"last_rc={last_rc!r}, protocol={self.config.protocol}, "
            f"master={self.config.master_server_address}"
        )

    def _key_exists(self, store_key: str) -> bool | None:
        if self._store is None:
            return None
        is_exist = getattr(self._store, "is_exist", None)
        if not callable(is_exist):
            return None
        try:
            return bool(is_exist(store_key))
        except Exception:
            logger.exception("Mooncake is_exist failed for key=%s", store_key)
            return None

    def _get_tensor(
        self, store_key: str, *, expected_bytes: int | None = None
    ) -> torch.Tensor:
        if self._store is None:
            raise RuntimeError("call setup() first")
        max_attempts = max(1, int(self.config.hs_transfer_max_retries) + 1)
        last_exists: bool | None = None
        for attempt in range(max_attempts):
            started = time.monotonic()
            tensor = self._store.get_tensor(store_key)
            elapsed = time.monotonic() - started
            if tensor is not None:
                logger.info(
                    "Mooncake get_tensor succeeded: key=%s bytes=%d "
                    "expected_bytes=%s attempt=%d/%d elapsed=%.3fs",
                    store_key,
                    tensor.nbytes,
                    expected_bytes,
                    attempt + 1,
                    max_attempts,
                    elapsed,
                )
                return tensor
            last_exists = self._key_exists(store_key)
            logger.warning(
                "Mooncake get_tensor returned None: key=%s expected_bytes=%s "
                "exists=%s attempt=%d/%d elapsed=%.3fs protocol=%s master=%s "
                "local_buffer_size=%d. Inspect Mooncake C++ logs for "
                "TRANSFER_FAIL, LEASE_EXPIRED, OBJECT_NOT_FOUND, or allocation errors.",
                store_key,
                expected_bytes,
                last_exists,
                attempt + 1,
                max_attempts,
                elapsed,
                self.config.protocol,
                self.config.master_server_address,
                self.config.local_buffer_size,
            )
            if attempt + 1 < max_attempts:
                time.sleep(self._retry_delay(attempt))
        reason = (
            "key missing/evicted or producer never committed it"
            if last_exists is False
            else "key exists but transfer/allocation/tensor decoding failed"
            if last_exists is True
            else "Mooncake returned no tensor (existence API unavailable)"
        )
        raise RuntimeError(
            "Mooncake get_tensor failed after "
            f"{max_attempts} attempts: key={store_key}, "
            f"expected_bytes={expected_bytes}, exists={last_exists}, "
            f"reason={reason}, protocol={self.config.protocol}, "
            f"master={self.config.master_server_address}, "
            f"local_buffer_size={self.config.local_buffer_size}"
        )

    def put_sample(self, key: str, tensors: dict[str, torch.Tensor]) -> None:
        if self._store is None:
            raise RuntimeError("call setup() first")
        max_bytes = int(self.config.hs_chunk_bytes)
        entries: list[dict[str, Any]] = []
        any_chunked = False
        written_keys: list[str] = []
        with self._lock:
            try:
                for name, tensor in tensors.items():
                    host = tensor.detach().to("cpu").contiguous()
                    parts = _split_tensor_rough(host, max_bytes)
                    n_chunks = len(parts)
                    if n_chunks > 1:
                        any_chunked = True
                    chunk_nbytes = [part.nbytes for part in parts]
                    logger.info(
                        "Mooncake put tensor plan: sample=%s name=%s "
                        "shape=%s dtype=%s total_bytes=%d n_chunks=%d "
                        "chunk_bytes=%s",
                        key,
                        name,
                        tuple(host.shape),
                        host.dtype,
                        host.nbytes,
                        n_chunks,
                        chunk_nbytes,
                    )
                    for store_key, part in zip(
                        _tensor_keys(key, name, n_chunks), parts, strict=True
                    ):
                        self._put_tensor(store_key, part)
                        written_keys.append(store_key)
                    entries.append(
                        {
                            "name": name,
                            "n_chunks": n_chunks,
                            "chunk_nbytes": chunk_nbytes,
                            "shape": list(host.shape),
                            "dtype": str(host.dtype),
                        }
                    )
                meta_key = f"{key}:meta"
                rc = self._store.put(
                    meta_key, _encode_meta(entries, chunked=any_chunked)
                )
                if rc != 0:
                    raise RuntimeError(
                        f"Mooncake meta put failed: key={meta_key}, rc={rc!r}"
                    )
                logger.info(
                    "Mooncake sample committed: sample=%s tensors=%d keys=%d",
                    key,
                    len(entries),
                    len(written_keys),
                )
            except Exception as exc:
                logger.exception(
                    "Mooncake sample write failed: sample=%s written_keys=%s",
                    key,
                    written_keys,
                )
                for written_key in written_keys:
                    try:
                        self._store.remove(written_key, force=True)
                    except Exception:
                        logger.exception(
                            "Failed to clean partial Mooncake key=%s", written_key
                        )
                error_payload = json.dumps(
                    {
                        "error": str(exc),
                        "written_keys": written_keys,
                        "protocol": self.config.protocol,
                    }
                ).encode("utf-8")
                try:
                    error_rc = self._store.put(f"{key}:error", error_payload)
                    if error_rc != 0:
                        logger.error(
                            "Failed to publish Mooncake error marker: "
                            "sample=%s rc=%r",
                            key,
                            error_rc,
                        )
                except Exception:
                    logger.exception(
                        "Failed to publish Mooncake error marker for sample=%s", key
                    )
                raise

    def delete_sample(self, key: str) -> None:
        """Remove all keys for a sample from the store."""
        if self._store is None:
            raise RuntimeError("call setup() first")
        with self._lock:
            raw = self._store.get(f"{key}:meta")
            if not raw:
                self._store.remove(f"{key}:error", force=True)
                return
            keys_to_remove: list[str] = [f"{key}:meta"]
            for entry in _parse_meta(raw):
                keys_to_remove.extend(
                    _tensor_keys(key, entry["name"], entry["n_chunks"])
                )
            keys_to_remove.append(f"{key}:error")
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
        entries = _parse_meta(self._wait_for_meta(key, timeout, poll_interval))
        result: dict[str, torch.Tensor] = {}
        with self._lock:
            for entry in entries:
                name = entry["name"]
                n_chunks = entry["n_chunks"]
                chunk_nbytes = entry.get("chunk_nbytes", [])
                parts: list[torch.Tensor] = []
                for index, store_key in enumerate(_tensor_keys(key, name, n_chunks)):
                    expected_bytes = (
                        int(chunk_nbytes[index]) if index < len(chunk_nbytes) else None
                    )
                    parts.append(
                        self._get_tensor(store_key, expected_bytes=expected_bytes)
                    )
                result[name] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return result

    def _wait_for_meta(
        self, sample_key: str, timeout: float, poll_interval: float
    ) -> bytes:
        if self._store is None:
            raise RuntimeError("call setup() first")
        deadline = time.monotonic() + timeout
        meta_key = f"{sample_key}:meta"
        error_key = f"{sample_key}:error"
        while True:
            with self._lock:
                raw = self._store.get(meta_key)
                error_raw = b"" if raw else self._store.get(error_key)
            if raw:
                return raw
            if error_raw:
                try:
                    detail = json.loads(error_raw)
                except Exception:
                    detail = error_raw.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Mooncake producer failed for sample={sample_key}: {detail}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for Mooncake sample completion: "
                    f"meta_key={meta_key}, error_key={error_key}, timeout={timeout}s, "
                    f"protocol={self.config.protocol}, "
                    f"master={self.config.master_server_address}"
                )
            time.sleep(poll_interval)
