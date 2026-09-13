"""Tests for Apah Continuous Batching Scheduler."""

import asyncio
import time
import pytest
import torch
from httpx import ASGITransport, AsyncClient

from apah.engine.runtime import ApahRuntime
from apah.engine.scheduler import ContinuousBatchingScheduler
from apah.engine.sequence import Sequence, SequenceStatus
from apah.engine.server import create_app, server_state

TEST_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def loaded_runtime():
    """Load model once for scheduler module tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime = ApahRuntime.from_model_path(TEST_MODEL_PATH, dtype="float16", device=device)
    yield runtime
    runtime.unload()


class SchedulerContext:
    """Async context manager helper for managing scheduler lifecycle in tests."""

    def __init__(self, runtime: ApahRuntime, max_batch_size: int = 4) -> None:
        self.sched = ContinuousBatchingScheduler(runtime=runtime, max_batch_size=max_batch_size)

    async def __aenter__(self) -> ContinuousBatchingScheduler:
        self.sched.start()
        return self.sched

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.sched.stop()


@pytest.mark.asyncio
async def test_concurrent_3_requests_correct_outputs(loaded_runtime):
    """Test 3 concurrent requests submitted together complete correctly with independent outputs."""
    async with SchedulerContext(loaded_runtime, max_batch_size=4) as scheduler:
        tokenizer = loaded_runtime.tokenizer
        prompts = [
            "One plus one is",
            "The primary color of the sky is",
            "Capital city of France is",
        ]

        sequences = []
        for p in prompts:
            seq = Sequence(
                prompt=p,
                prompt_token_ids=tokenizer.encode(p),
                max_new_tokens=10,
                temperature=0.0,
            )
            sequences.append(seq)
            scheduler.add_request(seq)

        results = await asyncio.gather(*[seq.completion_future for seq in sequences])

        assert len(results) == 3
        for text in results:
            assert isinstance(text, str)
            assert len(text.strip()) > 0


@pytest.mark.asyncio
async def test_continuous_admission_mid_generation(loaded_runtime):
    """Test a request submitted while others are mid-generation gets admitted continuously."""
    async with SchedulerContext(loaded_runtime, max_batch_size=4) as scheduler:
        tokenizer = loaded_runtime.tokenizer

        p1 = "Write a long story about space exploration"
        seq1 = Sequence(
            prompt=p1,
            prompt_token_ids=tokenizer.encode(p1),
            max_new_tokens=25,
            temperature=0.0,
        )
        scheduler.add_request(seq1)

        # Wait until seq1 is running mid-generation
        await asyncio.sleep(0.05)

        p2 = "2 + 2 ="
        seq2 = Sequence(
            prompt=p2,
            prompt_token_ids=tokenizer.encode(p2),
            max_new_tokens=10,
            temperature=0.0,
        )
        scheduler.add_request(seq2)

        results = await asyncio.gather(seq1.completion_future, seq2.completion_future)

        assert len(results) == 2
        assert len(results[1].strip()) > 0


@pytest.mark.asyncio
async def test_max_batch_size_enforced(loaded_runtime):
    """Test max_batch_size is respected (excess requests wait in queue)."""
    async with SchedulerContext(loaded_runtime, max_batch_size=1) as sched:
        tokenizer = loaded_runtime.tokenizer
        prompts = ["Count 1:", "Count 2:", "Count 3:"]
        sequences = [
            Sequence(
                prompt=p,
                prompt_token_ids=tokenizer.encode(p),
                max_new_tokens=10,
                temperature=0.0,
            )
            for p in prompts
        ]

        for seq in sequences:
            sched.add_request(seq)

        await asyncio.sleep(0.02)
        assert sched.active_count <= 1
        assert sched.waiting_count >= 1

        await asyncio.gather(*[s.completion_future for s in sequences])


@pytest.mark.asyncio
async def test_throughput_benchmark_report(loaded_runtime):
    """Measure and print continuous batching throughput vs sequential execution."""
    tokenizer = loaded_runtime.tokenizer
    prompts = ["Explain gravity:", "Explain magnetism:", "Explain electricity:"]

    # 1. Measure sequential execution time
    t0 = time.perf_counter()
    for p in prompts:
        loaded_runtime.generate(p, max_new_tokens=20, temperature=0.0)
    seq_duration = time.perf_counter() - t0

    # 2. Measure batched execution time
    async with SchedulerContext(loaded_runtime, max_batch_size=4) as sched:
        sequences = [
            Sequence(
                prompt=p,
                prompt_token_ids=tokenizer.encode(p),
                max_new_tokens=20,
                temperature=0.0,
            )
            for p in prompts
        ]
        t1 = time.perf_counter()
        for seq in sequences:
            sched.add_request(seq)

        await asyncio.gather(*[s.completion_future for s in sequences])
        batch_duration = time.perf_counter() - t1

    speedup = seq_duration / batch_duration if batch_duration > 0 else 1.0

    print("\n--- BENCHMARK REPORT ---")
    print(f"Sequential 3-request duration: {seq_duration:.3f}s")
    print(f"Batched 3-request duration:    {batch_duration:.3f}s")
    print(f"Speedup ratio:                 {speedup:.2f}x")
    print("------------------------")

    assert batch_duration < seq_duration * 1.5


@pytest.mark.asyncio
async def test_load_test_10_concurrent_requests():
    """Load test: fire 10 concurrent requests via FastAPI AsyncClient."""
    app = create_app()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runtime = ApahRuntime.from_model_path(TEST_MODEL_PATH, dtype="float16", device=device)
    server_state.set_runtime(runtime, TEST_MODEL_PATH)
    server_state.scheduler.start()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as async_client:
        tasks = []
        for i in range(10):
            payload = {
                "model": TEST_MODEL_PATH,
                "messages": [{"role": "user", "content": f"Number {i}:"}],
                "max_tokens": 10,
                "temperature": 0.0,
                "stream": False,
            }
            tasks.append(async_client.post("/v1/chat/completions", json=payload))

        responses = await asyncio.gather(*tasks)

        assert len(responses) == 10
        for resp in responses:
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "chat.completion"
            content = data["choices"][0]["message"]["content"]
            assert len(content.strip()) > 0

    server_state.unload()
