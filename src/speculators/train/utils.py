import logging
import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
is_distributed = "LOCAL_RANK" in os.environ

logger = logging.getLogger("speculators")


@dataclass(frozen=True)
class ContextParallelConfig:
    enabled: bool = False
    mode: Literal["none", "context_parallel"] = "none"
    mesh: object | None = None
    size: int = 1
    rank: int = 0
    group_world_size: int = 1


def _setup_context_parallel(
    *,
    cp_size: int,
    cp_mode: Literal["none", "context_parallel"],
    rank: int,
) -> ContextParallelConfig:
    if cp_size < 1:
        raise ValueError(f"`cp_size` must be >= 1, got {cp_size}.")

    if cp_mode == "none":
        if cp_size != 1:
            raise ValueError(
                "`cp_size` must be 1 when `cp_mode='none'`. "
                f"Received cp_size={cp_size}."
            )
        return ContextParallelConfig()

    if cp_mode != "context_parallel":
        raise ValueError(
            f"Unknown cp mode: {cp_mode}. Expected one of ['none', 'context_parallel']."
        )

    if not is_distributed:
        raise ValueError(
            "Context parallel requires distributed launch with torchrun. "
            "Please launch with `torchrun` and set `--cp-size > 1`."
        )

    if cp_size == 1:
        return ContextParallelConfig()

    if world_size % cp_size != 0:
        raise ValueError(
            f"Invalid CP topology: WORLD_SIZE={world_size} must be divisible by "
            f"cp_size={cp_size}."
        )

    try:
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError as exc:
        raise RuntimeError(
            "Context parallel requires torch distributed DeviceMesh support. "
            "Please upgrade to a compatible torch version or run with `--cp-size 1`."
        ) from exc

    try:
        from torch.distributed.tensor.experimental import context_parallel as _cp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Context parallel runtime not found in this torch build. "
            "Please upgrade torch or run with `--cp-size 1`."
        ) from exc

    if not torch.accelerator.is_available():
        raise RuntimeError(
            "Context parallel requires an available accelerator device. "
            "Run with accelerator(s) or disable CP via `--cp-size 1`."
        )

    acc = torch.accelerator.current_accelerator()
    if acc is None:
        raise RuntimeError(
            "Context parallel could not determine the current accelerator. "
            "Please ensure torchrun sets the local device correctly, or disable CP "
            "via `--cp-size 1`."
        )
    device_type = acc.type

    dp_size = world_size // cp_size
    if dp_size == 1:
        cp_mesh = init_device_mesh(
            device_type,
            (cp_size,),
            mesh_dim_names=("cp",),
        )
    else:
        full_mesh = init_device_mesh(
            device_type,
            (dp_size, cp_size),
            mesh_dim_names=("dp", "cp"),
        )
        cp_mesh = full_mesh["cp"]

    cp_rank = cp_mesh.get_local_rank()
    cp_group_world_size = cp_mesh.size()
    logger.info(
        "Enabled context parallel: "
        f"cp_rank={cp_rank}, cp_size={cp_group_world_size}, "
        f"global_rank={rank}, world_size={world_size}, device_type={device_type}",
        extra={"override_rank0_filter": True},
    )

    return ContextParallelConfig(
        enabled=True,
        mode="context_parallel",
        mesh=cp_mesh,
        size=cp_group_world_size,
        rank=cp_rank,
        group_world_size=cp_group_world_size,
    )


def maybe_setup_distributed(
    cp_size: int = 1, cp_mode: Literal["none", "context_parallel"] = "none"
) -> tuple[int, int, int, bool, ContextParallelConfig]:
    """Sets up distributed training if the process was launched with `torchrun`.
    If not, returns single process training.

    Based on of https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun

    Returns:
        tuple[int, int, int, bool]: Local rank, world size, rank, and is_distributed.
    """
    if not is_distributed:
        # No distributed training
        cp_config = _setup_context_parallel(cp_size=cp_size, cp_mode=cp_mode, rank=0)
        return 0, 1, 0, False, cp_config

    torch.accelerator.set_device_index(local_rank)
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        raise ValueError("No accelerator found")
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend, device_id=local_rank)

    rank = dist.get_rank()

    logger.info(
        f"Started distributed with local_rank={local_rank}, world_size={world_size}",
        extra={"override_rank0_filter": True},
    )
    cp_config = _setup_context_parallel(cp_size=cp_size, cp_mode=cp_mode, rank=rank)
    return local_rank, world_size, rank, True, cp_config


def maybe_destroy_distributed():
    """Destroys the distributed process group if using distributed training."""
    if not is_distributed:
        # No distributed training
        return

    dist.destroy_process_group()
    logger.info(
        f"Destroyed distributed with local_rank={local_rank}, world_size={world_size}",
        extra={"override_rank0_filter": True},
    )


def apply_fully_sharded(model: torch.nn.Module):
    """Applies torch FSDP fully_shard to the model, wrapping layers in FSDPModule.

    Assumes the model has a `layers` attribute containing the decoder layers.
    Model should be validated with SpeculatorModel.verify_training_compatible()
    before calling this function.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    for layer in model.layers:  # type: ignore[union-attr]
        fully_shard(layer, mp_policy=mp_policy)

    fully_shard(model)

    return model
