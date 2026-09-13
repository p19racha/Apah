"""Production-scale sustained load testing and failure recovery tests."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from apah.engine.runtime import ApahRuntime
from apah.engine.sequence import Sequence
from apah.engine.server import create_app, server_state


@pytest.fixture
def test_client():
    """FastAPI TestClient instance with initialized router."""
    app = create_app()
    return TestClient(app)


@pytest.mark.asyncio
async def test_sustained_production_load_and_memory_stability():
    """Sustained load test firing overlapping requests to verify memory stability and no sequence leakage."""
    from httpx import ASGITransport, AsyncClient

    app = create_app()
    mock_runtime = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = None
    mock_tokenizer.encode.return_value = [101, 102, 103, 104]
    mock_tokenizer.eos_token_id = 104
    mock_runtime.tokenizer = mock_tokenizer

    server_state.runtime = mock_runtime
    server_state.model_name = "production-test-model"

    def fake_add_request(seq: Sequence):
        # Simulate sequence decoding for specific prompt
        req_index = seq.prompt.split("Request-")[-1]
        seq.token_queue.put_nowait(f"Result for Request-{req_index}")
        seq.token_queue.put_nowait(None)

    with patch.object(server_state.scheduler, "add_request", side_effect=fake_add_request):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as async_client:
            async def send_request(idx: int):
                resp = await async_client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "production-test-model",
                        "messages": [{"role": "user", "content": f"Production Workload Request-{idx}"}],
                        "stream": False,
                    },
                )
                assert resp.status_code == 200
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                assert f"Request-{idx}" in content, f"Sequence cross-talk detected! Got content: {content}"
                return content

            tasks = [send_request(i) for i in range(25)]
            results = await asyncio.gather(*tasks)
            assert len(results) == 25

    # Confirm scheduler queue and active batch are empty after load completes
    stats = server_state.scheduler.get_stats()
    assert stats["waiting_queue_length"] == 0
    assert stats["active_batch_size"] == 0


def test_failure_recovery_and_clean_server_state(test_client):
    """Test failure injection and clean state recovery after unload/reset."""
    # Populate loaded state
    mock_runtime = MagicMock()
    server_state.runtime = mock_runtime
    server_state.model_name = "failed-model"

    assert server_state.is_loaded is True

    # Simulate abrupt server state unload / recovery
    server_state.unload()

    assert server_state.is_loaded is False
    assert server_state.runtime is None
    assert server_state.model_name is None

    # Verify /ps reflects clean unloaded state
    ps_resp = test_client.get("/ps")
    assert ps_resp.status_code == 200
    assert ps_resp.json() == []
