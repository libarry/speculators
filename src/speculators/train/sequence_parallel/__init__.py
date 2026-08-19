"""Sequence-parallel helpers (sharding + Ulysses all-to-all)."""

from speculators.train.sequence_parallel.shard import (
    pad_seq_to_sp_multiple,
    shard_batch_for_sp,
    sync_and_shard_sp_batch,
    validate_sp_head_divisibility,
)
from speculators.train.sequence_parallel.ulysses import (
    SeqAllToAll4D,
    gather_document_ids,
    ulysses_enabled,
)

__all__ = [
    "SeqAllToAll4D",
    "gather_document_ids",
    "pad_seq_to_sp_multiple",
    "shard_batch_for_sp",
    "sync_and_shard_sp_batch",
    "ulysses_enabled",
    "validate_sp_head_divisibility",
]
