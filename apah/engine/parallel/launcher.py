"""Multi-process launcher for tensor-parallel model execution across GPUs.

Architecture Choice:
    - Rank 0 acts as the Orchestrator Process: interacts directly with the Apah scheduler,
      FastAPI routes, and user client.
    - Ranks 1..N-1 act as Compute Worker Processes: run a minimal background loop waiting for
      broadcast commands (forward step execution, sequence tensor IDs, or shutdown signals)
      from Rank 0 via torch.distributed NCCL collective operations.
"""

import logging
from typing import Optional, Tuple
import torch
import torch.multiprocessing as mp

from apah.engine.parallel.process_group import (
    destroy_process_group,
    get_tp_rank,
    get_tp_world_size,
    init_process_group,
)
from apah.engine.parallel.tensor_parallel import convert_model_to_tensor_parallel

logger = logging.getLogger("apah.engine.parallel.launcher")

# Commands for Rank 0 -> Worker Ranks broadcast orchestration
CMD_EXIT = 0
CMD_FORWARD_STEP = 1


def _worker_loop(model: torch.nn.Module, device: str) -> None:
    """Compute worker loop executed on ranks 1..N-1 waiting for Rank 0 signals."""
    rank = get_tp_rank()
    dev_obj = torch.device(device)

    logger.info(f"Worker process rank={rank} entered tensor parallel compute loop.")

    cmd_tensor = torch.zeros(1, dtype=torch.long, device=dev_obj if "cuda" in device else "cpu")

    while True:
        try:
            # Broadcast command from Rank 0
            torch.distributed.broadcast(cmd_tensor, src=0)
            cmd = int(cmd_tensor.item())

            if cmd == CMD_EXIT:
                logger.info(f"Worker process rank={rank} received exit command. Shutting down worker loop.")
                break

            elif cmd == CMD_FORWARD_STEP:
                # Receive input token tensor length and contents
                len_tensor = torch.zeros(1, dtype=torch.long, device=dev_obj if "cuda" in device else "cpu")
                torch.distributed.broadcast(len_tensor, src=0)
                seq_len = int(len_tensor.item())

                input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=dev_obj if "cuda" in device else "cpu")
                torch.distributed.broadcast(input_ids, src=0)

                with torch.no_grad():
                    # Compute forward pass in lockstep across GPUs
                    model(input_ids)

        except Exception as e:
            logger.error(f"Error in worker process rank={rank} loop: {e}")
            break

    destroy_process_group()


def _mp_entrypoint(
    rank: int,
    world_size: int,
    model_path: str,
    dtype: str,
    device: str,
    master_port: int,
    return_dict: dict,
) -> None:
    """Multiprocessing entrypoint for rank process initialization."""
    device_name = f"cuda:{rank}" if "cuda" in device and torch.cuda.is_available() else "cpu"
    try:
        init_process_group(rank=rank, world_size=world_size, master_port=master_port)

        from apah.engine.model_loader import load_model

        # Load model weights on current rank
        model, tokenizer = load_model(model_path=model_path, dtype=dtype, device=device_name, tp_world_size=1)

        # Convert linear layers to tensor parallel shards
        model = convert_model_to_tensor_parallel(model, world_size, rank)

        if rank == 0:
            return_dict["model"] = model
            return_dict["tokenizer"] = tokenizer
        else:
            _worker_loop(model, device_name)

    except Exception as e:
        logger.error(f"Failed to initialize tensor parallel rank={rank}: {e}")
        return_dict["error"] = str(e)


def launch_tensor_parallel(
    model_path: str,
    tp_world_size: int,
    dtype: str = "bfloat16",
    device: str = "cuda",
    master_port: int = 29500,
) -> Tuple[torch.nn.Module, any]:
    """Launch multi-process tensor parallel group and return Rank 0 model & tokenizer.

    Args:
        model_path: Path or identifier of Hugging Face model.
        tp_world_size: Number of GPUs / ranks in tensor parallel group.
        dtype: Weight data type.
        device: Base device string.
        master_port: Communication port for process group.

    Returns:
        Tuple of (Rank 0 sharded PreTrainedModel, PreTrainedTokenizer).
    """
    if tp_world_size <= 1:
        from apah.engine.model_loader import load_model
        return load_model(model_path=model_path, dtype=dtype, device=device, tp_world_size=1)

    manager = mp.Manager()
    return_dict = manager.dict()

    logger.info(f"Spawning {tp_world_size} tensor parallel worker processes for model '{model_path}'...")
    ctx = mp.spawn(
        _mp_entrypoint,
        args=(tp_world_size, model_path, dtype, device, master_port, return_dict),
        nprocs=tp_world_size,
        join=False,
    )

    if "error" in return_dict:
        ctx.join()
        raise RuntimeError(f"Tensor parallel spawn error: {return_dict['error']}")

    model = return_dict.get("model")
    tokenizer = return_dict.get("tokenizer")

    if model is None or tokenizer is None:
        ctx.join()
        raise RuntimeError("Rank 0 failed to return initialized model and tokenizer.")

    return model, tokenizer
