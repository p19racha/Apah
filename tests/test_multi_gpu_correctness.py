"""Integration tests for multi-GPU tensor parallel output correctness and memory sharding.

Skipped automatically on single-GPU or CPU execution environments.
"""

import pytest
import torch

from apah.engine.model_loader import load_model
from apah.engine.parallel.process_group import destroy_process_group


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Multi-GPU correctness tests require at least 2 visible CUDA GPUs.",
)
def test_multi_gpu_correctness_and_memory_sharding(tmp_path):
    """Test tp_world_size=1 vs tp_world_size=2 numerical output equivalence and memory reduction."""
    # 1. Single GPU load (tp=1)
    model_tp1, tokenizer = load_model(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        device="cuda:0",
        tp_world_size=1,
    )
    mem_tp1 = torch.cuda.memory_allocated(0)

    prompt = "Tensor parallelism testing"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        out_tp1 = model_tp1(input_ids).logits

    del model_tp1
    torch.cuda.empty_cache()

    # 2. Dual GPU load (tp=2)
    try:
        model_tp2, _ = load_model(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            dtype="bfloat16",
            device="cuda",
            tp_world_size=2,
        )
        mem_tp2 = torch.cuda.memory_allocated(0)

        with torch.no_grad():
            out_tp2 = model_tp2(input_ids).logits

        # Verify logits match within floating point tolerance
        assert torch.allclose(out_tp1, out_tp2, atol=1e-2, rtol=1e-2)

        # Verify per-GPU memory usage with tp=2 is significantly less than tp=1 (roughly half)
        assert mem_tp2 < mem_tp1 * 0.75

    finally:
        destroy_process_group()
