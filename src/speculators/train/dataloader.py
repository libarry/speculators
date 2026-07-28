from __future__ import annotations

import logging
import warnings
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

import torch
from torch.utils.data import DataLoader

from hs_connectors import HiddenStatesTransfer, MooncakeTransfer
from speculators.train.data import (
    ArrowDataset,
    BaseDataset,
    SampleFileDataset,
    create_collate_fn,
    split_files,
)
from speculators.train.distributed import get_dp_rank, get_dp_size
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.noise_transforms import AddUniformNoise

logger = logging.getLogger(__name__)

BatchType = dict[str, Any]


class ParallelBatchPrefetchLoader:
    """Parallel batch prefetch with a **single** Ascend ADXL/Mooncake client.

    Why not DataLoader ``num_workers>0``?
      Each process would mount its own ADXL segment; vLLM put then hits
      ``status=503900`` / ``TRANSFER_FAIL``.

    What is parallel here?
      Up to ``prefetch_factor`` batches are built concurrently. Within a batch,
      samples are also fetched concurrently. vLLM HTTP generates overlap in
      parallel; Mooncake get/delete are serialized by ``MooncakeTransfer``'s
      ADXL lock (one train-side client, one in-flight RDMA/ADXL op).

    Training still overlaps with prefetch: while the GPU trains on batch N,
    workers prepare N+1…N+prefetch_factor.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        batch_sampler,
        collate_fn: Callable[[list], BatchType],
        prefetch_factor: int = 4,
        sample_workers: int | None = None,
    ):
        if prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn
        self.prefetch_factor = prefetch_factor
        # Parallel sample fetches inside one batch (HTTP overlap).
        self.sample_workers = sample_workers or max(4, prefetch_factor)

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self) -> Iterator[BatchType]:
        return _ParallelBatchPrefetchIterator(
            dataset=self.dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            prefetch_factor=self.prefetch_factor,
            sample_workers=self.sample_workers,
        )


class _ParallelBatchPrefetchIterator:
    def __init__(
        self,
        dataset: BaseDataset,
        batch_sampler,
        collate_fn: Callable[[list], BatchType],
        prefetch_factor: int,
        sample_workers: int,
    ):
        self._dataset = dataset
        self._collate_fn = collate_fn
        self._sample_workers = sample_workers
        self._index_batches = iter(batch_sampler)
        self._prefetch_factor = prefetch_factor
        self._batch_pool = ThreadPoolExecutor(
            max_workers=prefetch_factor,
            thread_name_prefix="mc-batch",
        )
        self._sample_pool = ThreadPoolExecutor(
            max_workers=sample_workers,
            thread_name_prefix="mc-sample",
        )
        self._pending: deque[Future] = deque()
        self._closed = False
        self._fill()

    def _build_batch(self, indices: list[int]) -> BatchType:
        # Parallel sample load: vLLM HTTP overlaps; ADXL get is lock-serialized.
        if len(indices) == 1:
            samples = [self._dataset[indices[0]]]
        else:
            samples = list(
                self._sample_pool.map(self._dataset.__getitem__, indices)
            )
        return self._collate_fn(samples)

    def _fill(self) -> None:
        while len(self._pending) < self._prefetch_factor:
            try:
                indices = next(self._index_batches)
            except StopIteration:
                break
            # Multipack sampler may yield numpy ints; dataset needs Python int.
            indices = [int(i) for i in indices]
            self._pending.append(self._batch_pool.submit(self._build_batch, indices))

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._batch_pool.shutdown(wait=False, cancel_futures=True)
        self._sample_pool.shutdown(wait=False, cancel_futures=True)

    def __iter__(self) -> _ParallelBatchPrefetchIterator:
        return self

    def __next__(self) -> BatchType:
        if not self._pending:
            self._close()
            raise StopIteration
        fut = self._pending.popleft()
        try:
            batch = fut.result()
        except Exception:
            self._close()
            raise
        self._fill()
        return batch

    def __del__(self) -> None:
        try:
            self._close()
        except Exception:  # noqa: BLE001
            pass


def _setup_dataloader(
    dataset: BaseDataset,
    total_seq_len: int,
    hidden_size: int,
    num_workers: int = 12,
    num_target_layers: int = 3,
    prefetch_factor: int | None = 4,
    preprocess: Callable[[BatchType], BatchType] | None = None,
) -> DataLoader | ParallelBatchPrefetchLoader:
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=get_dp_size(),
        rank=get_dp_rank(),
    )
    transfer = getattr(dataset, "transfer", None)
    use_mooncake = isinstance(transfer, MooncakeTransfer)
    collate_fn = create_collate_fn(
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        dtype=dataset.hidden_states_dtype,
        preprocess=preprocess,
    )

    if use_mooncake:
        pf = prefetch_factor or 4
        # Cap batch concurrency: too many in-flight vLLM puts + ADXL gets
        # overwhelm a single Ascend endpoint (103900 / TRANSFER_FAIL storms).
        capped_pf = min(pf, 2)
        sample_workers = min(4, max(2, pf))
        if pf > capped_pf:
            logger.warning(
                "Mooncake/Ascend: prefetch_factor=%s capped to %s to reduce ADXL "
                "connect pressure (set PREFETCH_FACTOR<=2 to silence).",
                pf,
                capped_pf,
            )
        if num_workers > 0:
            logger.warning(
                "Mooncake/Ascend ADXL cannot use multiprocess DataLoader workers "
                "(num_workers=%s ignored). Using ParallelBatchPrefetchLoader with "
                "prefetch_factor=%s: parallel vLLM HTTP + single locked ADXL client.",
                num_workers,
                capped_pf,
            )
        else:
            logger.info(
                "Mooncake backend: ParallelBatchPrefetchLoader "
                "(prefetch_factor=%s, sample_workers=%s, single ADXL client)",
                capped_pf,
                sample_workers,
            )
        return ParallelBatchPrefetchLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            prefetch_factor=capped_pf,
            sample_workers=sample_workers,
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
) -> tuple[DataLoader | ParallelBatchPrefetchLoader, DataLoader | ParallelBatchPrefetchLoader]:
    """Create training and validation DataLoaders.

    For Mooncake/Ascend, multiprocess workers are replaced by
    :class:`ParallelBatchPrefetchLoader` (parallel HTTP, one ADXL client).
    """
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
        train_transfer = transfer
        # A cloned Mooncake transfer mounts another 4 GiB ADXL segment. With
        # validation initialized lazily after epoch 0, that doubled the
        # train-side clients exactly at the epoch boundary and left producer
        # replicas on short-lived 20xxx endpoints. Train and validation are
        # sequential, so they can safely share one locked Mooncake client.
        if isinstance(transfer, MooncakeTransfer):
            val_transfer = transfer
        else:
            val_transfer = transfer.clone() if transfer is not None else None
        train_dataset = ArrowDataset(
            datapath=data_path,
            max_len=total_seq_len,
            transfer=train_transfer,
            vllm_endpoint=vllm_endpoint,
            on_missing=on_missing,
            on_generate=on_generate,
            transform=noise_transform,
            split_ratio=train_data_ratio,
            model=verifier_name_or_path,
            hidden_states_dtype=hidden_states_dtype,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
        val_dataset = ArrowDataset(
            datapath=data_path,
            max_len=total_seq_len,
            transfer=val_transfer,
            vllm_endpoint=vllm_endpoint,
            on_missing=on_missing,
            on_generate=on_generate,
            split_ratio=train_data_ratio - 1.0,
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
    )
    val_loader = _setup_dataloader(
        val_dataset,
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        preprocess=preprocess,
    )

    return train_loader, val_loader
