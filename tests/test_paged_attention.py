"""Tests for PagedAttention numerical correctness and memory efficiency."""

import asyncio
import pytest
import torch
from httpx import ASGITransport, AsyncClient

from apah.engine.attention import paged_attention_forward, read_past_kv_from_paged_cache, write_past_kv_to_paged_cache
from apah.engine.paged_cache import BlockPool, BlockTable
from apah.engine.runtime import ApahRuntime
from apah.engine.server import create_app, server_state

TEST_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def runtime(device):
    rt = ApahRuntime.from_model_path(TEST_MODEL_PATH, dtype="float16", device=device)
    yield rt
    rt.unload()


def test_paged_attention_numerical_correctness(runtime, device):
    """Verify paged attention output matches standard scaled-dot-product attention within float tolerance."""
    tokenizer = runtime.tokenizer
    model = runtime.model
    config = model.config

    prompt = "Artificial Intelligence is transforming global technology"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]

    # 1. Standard forward pass
    with torch.no_grad():
        outputs_std = model(input_ids=inputs.input_ids, use_cache=True)
    past_std = outputs_std.past_key_values

    # 2. Paged KV Cache store & gather
    pool = BlockPool.create_from_model_config(config, block_size=16, dtype=torch.float16, device=device)
    table = BlockTable(block_pool=pool, block_size=16)
    table.allocate_prompt_blocks(seq_len)

    write_past_kv_to_paged_cache(pool, table, past_std, is_prefill=True)
    past_paged = read_past_kv_from_paged_cache(pool, table)

    # Compare K/V tensors for layer 0
    from apah.engine.attention import extract_kv_pair_for_layer
    k_std, v_std = extract_kv_pair_for_layer(past_std, 0)
    k_paged, v_paged = extract_kv_pair_for_layer(past_paged, 0)

    assert torch.allclose(k_std, k_paged, atol=1e-3, rtol=1e-3)
    assert torch.allclose(v_std, v_paged, atol=1e-3, rtol=1e-3)

    # 3. Test paged_attention_forward output
    num_attn_heads = getattr(config, "num_attention_heads", 14)
    head_dim = getattr(config, "head_dim", config.hidden_size // num_attn_heads)
    query = torch.randn(1, num_attn_heads, 1, head_dim, dtype=torch.float16, device=device)

    num_queries_per_kv = query.shape[1] // k_std.shape[1]
    if num_queries_per_kv > 1:
        k_std_gqa = k_std.repeat_interleave(num_queries_per_kv, dim=1)
        v_std_gqa = v_std.repeat_interleave(num_queries_per_kv, dim=1)
    else:
        k_std_gqa, v_std_gqa = k_std, v_std

    attn_out_paged = paged_attention_forward(query, table, pool, layer_idx=0)
    attn_out_std = torch.nn.functional.scaled_dot_product_attention(query, k_std_gqa, v_std_gqa)

    assert torch.allclose(attn_out_paged, attn_out_std, atol=1e-3, rtol=1e-3)
    table.free()


@pytest.mark.asyncio
async def test_paged_memory_footprint_comparison():
    """Compare peak GPU memory usage of continuous batching load test under Paged Cache."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for GPU memory comparison test")

    app = create_app()
    device = "cuda"

    runtime = ApahRuntime.from_model_path(TEST_MODEL_PATH, dtype="float16", device=device)
    server_state.set_runtime(runtime, TEST_MODEL_PATH)
    server_state.scheduler.start()

    torch.cuda.reset_peak_memory_stats(device)
    initial_mem_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as async_client:
        tasks = []
        for i in range(10):
            # Varied prompt lengths to demonstrate paged block efficiency
            prompt_text = "Word " * ((i + 1) * 5) + f"Explain topic {i}:"
            payload = {
                "model": TEST_MODEL_PATH,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": 15,
                "temperature": 0.0,
                "stream": False,
            }
            tasks.append(async_client.post("/v1/chat/completions", json=payload))

        responses = await asyncio.gather(*tasks)

        assert len(responses) == 10
        for resp in responses:
            assert resp.status_code == 200

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    paged_kv_mem_mb = peak_mem_mb - initial_mem_mb

    print("\n--- PAGED CACHE MEMORY REPORT ---")
    print(f"Initial GPU memory (Model weights): {initial_mem_mb:.2f} MB")
    print(f"Peak GPU memory (10 concurrent requests): {peak_mem_mb:.2f} MB")
    print(f"KV Cache Overhead: {paged_kv_mem_mb:.2f} MB")
    if server_state.scheduler.block_pool:
        print(f"Total Block Pool Blocks: {server_state.scheduler.block_pool.num_total_blocks()}")
    print("----------------------------------")

    server_state.unload()
