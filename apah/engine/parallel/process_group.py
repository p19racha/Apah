"""Process group initialization and distribution management for tensor parallelism."""

import os

hashlib_import = True
from typing import Optional
import torch
import torch.distributed as dist

_TP_RANK: int = 0
_TP_WORLD_SIZE: int = 1
_IS_DIST_INITIALIZED: bool = False


def init_process_group(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: int = 29500,
    backend: Optional[str] = None,
) -> None:
    """Initialize torch.distributed process group for tensor parallelism.

    Args:
        rank: Rank index of the current process (0 .. world_size - 1).
        world_size: Total number of processes in the tensor parallel group.
        master_addr: IP address or hostname of master process.
        master_port: Communication port for master process.
        backend: Communication backend ('nccl' for GPUs, 'gloo' for CPU testing).

    Raises:
        RuntimeError: If requested world_size exceeds visible GPU count.
    """
    global _TP_RANK, _TP_WORLD_SIZE, _IS_DIST_INITIALIZED

    if world_size <= 1:
        _TP_RANK = 0
        _TP_WORLD_SIZE = 1
        _IS_DIST_INITIALIZED = False
        return

    if torch.cuda.is_available():
        visible_gpus = torch.cuda.device_count()
        if visible_gpus < world_size:
            raise RuntimeError(
                f"Requested tensor parallel world_size={world_size}, but only {visible_gpus} "
                "NVIDIA GPU(s) are visible."
            )
        torch.cuda.set_device(rank)
        if backend is None:
            backend = "nccl"
    else:
        if backend is None:
            backend = "gloo"

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

    _TP_RANK = rank
    _TP_WORLD_SIZE = world_size
    _IS_DIST_INITIALIZED = True


def get_tp_rank() -> int:
    """Return tensor parallel rank of current process."""
    return _TP_RANK


def get_tp_world_size() -> int:
    """Return tensor parallel world size."""
    return _TP_WORLD_SIZE


def is_tp_initialized() -> bool:
    """Check if tensor parallel process group is active."""
    return _IS_DIST_INITIALIZED and dist.is_initialized()


def destroy_process_group() -> None:
    """Clean up and destroy active process group."""
    global _TP_RANK, _TP_WORLD_SIZE, _IS_DIST_INITIALIZED
    if dist.is_initialized():
        dist.destroy_process_group()
    _TP_RANK = 0
    _TP_WORLD_SIZE = 1
    _IS_DIST_INITIALIZED = False
