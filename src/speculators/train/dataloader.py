from __future__ import annotations

import logging
import os
import queue
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

import torch
from torch.utils.data import DataLoader

from hs_connectors import HiddenStatesTransfer
from speculators.train.data import (
    ArrowDataset,
    BaseDataset,
    CollateFn,
    SampleFileDataset,
    split_files,
)
from speculators.train.distributed import get_dp_rank, get_dp_size
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.noise_transforms import AddUniformNoise

logger = logging.getLogger(__name__)

BatchType = dict[str, Any]

_SENTINEL = object()


def _pin_memory_batch(batch: BatchType) -> BatchType:
    pinned: BatchType = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.device.type == "cpu":
            pinned[key] = value.pin_memory()
        else:
            pinned[key] = value
    return pinned


class ConcurrentInProcessLoader:
    """In-process loader: ``num_workers`` concurrent HTTP/sample fetches.

    Ascend Mooncake cannot safely use PyTorch spawn workers (each would open an
    ADXL client and hit ``503900``). This loader keeps **one** Mooncake/ADXL
    client in the training process and instead runs a thread pool where:

    * ``num_workers`` = max concurrent ``dataset[i]`` calls
      (each call = one vLLM HTTP request + one Mooncake get / file read)
    * ``prefetch_factor`` = how many batches to build ahead of the trainer

    Mooncake store RPCs remain serialized via the store lock; HTTP waits do not
    hold that lock, so vLLM concurrency tracks ``num_workers``.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        batch_sampler: Any,
        collate_fn: Callable[[Sequence[BatchType | None]], BatchType],
        *,
        num_workers: int,
        prefetch_factor: int,
        pin_memory: bool = True,
    ):
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        if prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self) -> Iterator[BatchType]:
        batches: queue.Queue[Any] = queue.Queue(maxsize=self.prefetch_factor)
        errors: list[BaseException] = []
        stop = threading.Event()

        def _producer() -> None:
            # Shared pool so in-flight batches compete for the same worker budget:
            # at most ``num_workers`` HTTP/get calls run at once across all
            # prefetched batches.
            with ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="hs-fetch",
            ) as pool:
                try:
                    sampler_iter = iter(self.batch_sampler)
                    in_flight: list[tuple[list[int], list[Future[Any]]]] = []

                    def _submit_batch(indices: list[int]) -> None:
                        futures = [pool.submit(self.dataset.__getitem__, i) for i in indices]
                        in_flight.append((indices, futures))

                    for _ in range(self.prefetch_factor):
                        if stop.is_set():
                            break
                        try:
                            indices = list(next(sampler_iter))
                        except StopIteration:
                            break
                        _submit_batch(indices)

                    while in_flight and not stop.is_set():
                        _indices, futures = in_flight.pop(0)
                        try:
                            samples = [fut.result() for fut in futures]
                            batch = self.collate_fn(samples)
                            if self.pin_memory:
                                batch = _pin_memory_batch(batch)
                            batches.put(batch)
                        except BaseException as exc:  # noqa: BLE001
                            errors.append(exc)
                            break

                        try:
                            indices = list(next(sampler_iter))
                        except StopIteration:
                            continue
                        if not stop.is_set():
                            _submit_batch(indices)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
                finally:
                    # Cancel leftovers so interpreter shutdown stays clean.
                    for _indices, futures in in_flight:
                        for fut in futures:
                            fut.cancel()
                    batches.put(_SENTINEL)

        thread = threading.Thread(
            target=_producer,
            name="concurrent-inprocess-loader",
            daemon=True,
        )
        thread.start()
        try:
            while True:
                item = batches.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            stop.set()
            thread.join(timeout=30.0)
        if errors:
            raise errors[0]


# Back-compat alias used by older tests / imports.
class InProcessPrefetchLoader(ConcurrentInProcessLoader):
    """Deprecated name for :class:`ConcurrentInProcessLoader`."""

    def __init__(self, loader: DataLoader, num_prefetch_batches: int):
        # Legacy wrapper around a fully-built DataLoader (serial producer).
        # Kept so existing unit tests that pass a fake loader still work via
        # a thin adapter path in tests; training uses ConcurrentInProcessLoader
        # directly.
        if num_prefetch_batches < 1:
            raise ValueError(
                f"num_prefetch_batches must be >= 1, got {num_prefetch_batches}"
            )
        self._legacy_loader = loader
        self.num_prefetch_batches = num_prefetch_batches
        self.batch_sampler = loader.batch_sampler
        self.num_workers = 1
        self.prefetch_factor = num_prefetch_batches
        self.pin_memory = False
        self.dataset = None  # type: ignore[assignment]
        self.collate_fn = None  # type: ignore[assignment]

    def __len__(self) -> int:
        return len(self._legacy_loader)

    def __iter__(self) -> Iterator[BatchType]:
        batches: queue.Queue[Any] = queue.Queue(maxsize=self.num_prefetch_batches)
        errors: list[BaseException] = []

        def _producer() -> None:
            try:
                for batch in self._legacy_loader:
                    batches.put(batch)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                batches.put(_SENTINEL)

        thread = threading.Thread(
            target=_producer, name="legacy-inprocess-prefetch", daemon=True
        )
        thread.start()
        try:
            while True:
                item = batches.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            thread.join(timeout=5.0)
        if errors:
            raise errors[0]


def _use_inprocess_prefetch(transfer: HiddenStatesTransfer | None) -> bool:
    """Prefer in-process concurrent fetch for Ascend Mooncake unless overridden."""
    flag = os.environ.get("SPECULATORS_INPROCESS_PREFETCH", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    store = getattr(transfer, "store", None)
    protocol = getattr(getattr(store, "config", None), "protocol", None)
    return protocol == "ascend"


def _limit_worker_threads() -> None:
    """Limit per-worker thread pools to avoid thread exhaustion.

    With ``multiprocessing_context='spawn'``, each worker is a full process
    that re-imports numpy (OpenBLAS) and torch, each creating thread pools
    sized to the core count.  DataLoader workers only do I/O and tensor
    slicing — they don't benefit from intra-op parallelism.

    The env vars must be set before numpy/torch are imported to take effect
    on OpenBLAS/OMP.  Call this at the top of the training entry point,
    before DataLoader construction.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    """Initialize each DataLoader spawn worker.

    AscendDirect (and similar device-bound Mooncake transports) call
    ``aclrtGetDevice`` during ``store.setup()``. Spawn workers do not inherit
    the parent process ACL/CUDA context, so bind the local device here before
    any Mooncake client is created in ``ArrowDataset._setup_client``.
    """
    torch.set_num_threads(1)
    if torch.accelerator.current_accelerator() is None:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.accelerator.set_device_index(local_rank)


def _setup_dataloader(
    dataset: BaseDataset,
    total_seq_len: int,
    hidden_size: int,
    num_workers: int = 12,
    num_target_layers: int = 3,
    prefetch_factor: int | None = 4,
    preprocess: Callable[[BatchType], BatchType] | None = None,
    *,
    inprocess_prefetch: bool = False,
) -> DataLoader | ConcurrentInProcessLoader:
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=get_dp_size(),
        rank=get_dp_rank(),
    )
    collate_fn = CollateFn(
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        dtype=dataset.hidden_states_dtype,
        preprocess=preprocess,
    )

    if inprocess_prefetch:
        workers = max(1, num_workers)
        factor = max(1, prefetch_factor or 1)
        logger.info(
            "Using concurrent in-process loader: fetch_workers=%d "
            "(HTTP + HS get concurrency), prefetch_batches=%d, "
            "single Mooncake/ADXL client per rank.",
            workers,
            factor,
        )
        return ConcurrentInProcessLoader(
            dataset,
            batch_sampler,
            collate_fn,
            num_workers=workers,
            prefetch_factor=factor,
            pin_memory=True,
        )

    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=use_workers,
        multiprocessing_context="spawn" if use_workers else None,
        worker_init_fn=_worker_init_fn if use_workers else None,
    )


def create_train_val_loaders(
    *,
    data_path: str,
    total_seq_len: int,
    hidden_states_dtype: torch.dtype,
    noise_std: float,
    legacy_data: bool,
    transfer: HiddenStatesTransfer | None = None,
    vllm_endpoint: str,
    on_missing: Literal["generate", "skip", "warn", "raise"],
    on_generate: Literal["cache", "delete"],
    verifier_name_or_path: str,
    request_timeout: float | None,
    max_retries: int,
    hidden_size: int,
    num_target_layers: int,
    num_workers: int,
    prefetch_factor: int,
    preprocess: Callable[[BatchType], BatchType] | None,
    train_data_ratio: float = 0.9,
) -> tuple[
    DataLoader | ConcurrentInProcessLoader,
    DataLoader | ConcurrentInProcessLoader,
]:
    """Create training and validation DataLoaders.

    Handles dataset construction (legacy vs Arrow) and dataloader wiring.
    Non-data SP ranks get lightweight loaders with no workers (they receive
    batches via scatter).  Reads DP/SP topology from
    :mod:`speculators.train.distributed`.

    When ``transfer`` is Ascend Mooncake (``protocol=ascend``), process workers
    are replaced by :class:`ConcurrentInProcessLoader`: ``num_workers`` threads
    each run one concurrent HTTP + HS-get, sharing a single ADXL client.
    Override with ``SPECULATORS_INPROCESS_PREFETCH=0/1``.
    """
    _limit_worker_threads()
    inprocess_prefetch = _use_inprocess_prefetch(transfer)
    if inprocess_prefetch and transfer is not None:
        # Create the sole Mooncake/ADXL client in the training process up front.
        transfer.setup()
    noise_transform = AddUniformNoise(std=noise_std)

    if not (0.0 < train_data_ratio < 1.0):
        raise ValueError(f"train_data_ratio must be in (0, 1), got {train_data_ratio}")

    if legacy_data:
        warnings.warn(
            "Using '--legacy-data' is deprecated and will be removed soon.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        train_files, val_files = split_files(data_path, ratio=train_data_ratio)
        train_dataset: BaseDataset = SampleFileDataset(
            file_list=train_files,
            max_len=total_seq_len,
            transform=noise_transform,
            hidden_states_dtype=hidden_states_dtype,
        )
        val_dataset: BaseDataset = SampleFileDataset(
            file_list=val_files,
            max_len=total_seq_len,
            hidden_states_dtype=hidden_states_dtype,
        )
    else:
        train_dataset = ArrowDataset(
            datapath=data_path,
            max_len=total_seq_len,
            transfer=transfer,
            vllm_endpoint=vllm_endpoint,
            on_missing=on_missing,
            on_generate=on_generate,
            transform=noise_transform,
            train_ratio=train_data_ratio,
            split="train",
            model=verifier_name_or_path,
            hidden_states_dtype=hidden_states_dtype,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
        val_dataset = ArrowDataset(
            datapath=data_path,
            max_len=total_seq_len,
            transfer=transfer,
            vllm_endpoint=vllm_endpoint,
            on_missing=on_missing,
            on_generate=on_generate,
            train_ratio=train_data_ratio,
            split="val",
            model=verifier_name_or_path,
            hidden_states_dtype=hidden_states_dtype,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )

    train_loader = _setup_dataloader(
        train_dataset,
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        preprocess=preprocess,
        inprocess_prefetch=inprocess_prefetch,
    )
    val_loader = _setup_dataloader(
        val_dataset,
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        preprocess=preprocess,
        inprocess_prefetch=inprocess_prefetch,
    )

    return train_loader, val_loader
