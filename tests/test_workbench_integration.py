"""End-to-end integration tests for Sovereign On-Premise Agentic AI Workbench migration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from apah.engine.server import create_app, server_state


@pytest.fixture
def test_client():
    """FastAPI TestClient instance with initialized router."""
    app = create_app()
    return TestClient(app)


def test_openai_embeddings_endpoint(test_client):
    """Test POST /v1/embeddings returning OpenAI-compatible vector array."""
    with patch("apah.compat.embeddings.EmbeddingEngine.embed", return_value=[[0.1, 0.2, 0.3]]):
        resp = test_client.post(
            "/v1/embeddings",
            json={"model": "bge-small-en", "input": "Refinery pressure sensor alert"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["object"] == "embedding"
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_ollama_embeddings_endpoint(test_client):
    """Test Ollama-compatible POST /api/embeddings endpoint."""
    with patch("apah.compat.embeddings.EmbeddingEngine.embed", return_value=[[0.5, 0.6, 0.7]]):
        resp = test_client.post(
            "/api/embeddings",
            json={"model": "bge-small-en", "prompt": "Industrial turbine status"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "embedding" in data
        assert data["embedding"] == [0.5, 0.6, 0.7]


def test_ollama_tags_and_show_endpoints(test_client):
    """Test Ollama-compatible GET /api/tags and POST /api/show model info endpoints."""
    tags_resp = test_client.get("/api/tags")
    assert tags_resp.status_code == 200
    assert "models" in tags_resp.json()

    show_resp = test_client.post("/api/show", json={"name": "default"})
    assert show_resp.status_code == 200
    show_data = show_resp.json()
    assert "modelfile" in show_data
    assert "parameters" in show_data


def test_ollama_chat_endpoint_non_streaming(test_client):
    """Test Ollama-compatible POST /api/chat non-streaming request forwarding."""
    mock_runtime = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [101, 102]
    mock_tokenizer.eos_token_id = 102
    mock_runtime.tokenizer = mock_tokenizer

    server_state.runtime = mock_runtime
    server_state.model_name = "test-model"

    def fake_add_request(seq):
        seq.token_queue.put_nowait("Turbine pressure is nominal.")
        seq.token_queue.put_nowait(None)

    with patch.object(server_state.scheduler, "add_request", side_effect=fake_add_request):
        resp = test_client.post(
            "/api/chat",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "What is turbine status?"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("model") == "test-model"
        assert data.get("message", {}).get("content") == "Turbine pressure is nominal."
        assert data.get("done") is True


def test_tool_calling_openai_format(test_client):
    """Test OpenAI-compatible POST /v1/chat/completions with tools returning tool_calls payload."""
    mock_runtime = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [101, 102]
    mock_tokenizer.eos_token_id = 102
    mock_runtime.tokenizer = mock_tokenizer

    server_state.runtime = mock_runtime
    server_state.model_name = "test-agent-model"

    tool_call_text = '```json\n{"name": "query_refinery_db", "arguments": {"sensor_id": "T-104"}}\n```'

    def fake_add_request(seq):
        seq.token_queue.put_nowait(tool_call_text)
        seq.token_queue.put_nowait(None)

    tools_payload = [
        {
            "type": "function",
            "function": {
                "name": "query_refinery_db",
                "description": "Query refinery SCADA telemetry database",
                "parameters": {
                    "type": "object",
                    "properties": {"sensor_id": {"type": "string"}},
                    "required": ["sensor_id"],
                },
            },
        }
    ]

    with patch.object(server_state.scheduler, "add_request", side_effect=fake_add_request):
        resp = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-agent-model",
                "messages": [{"role": "user", "content": "Get telemetry for sensor T-104"}],
                "tools": tools_payload,
                "stream": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"] is not None
        tool_call = choice["message"]["tool_calls"][0]
        assert tool_call["function"]["name"] == "query_refinery_db"
        assert '"sensor_id": "T-104"' in tool_call["function"]["arguments"]


def test_workbench_rollback_path(test_client):
    """Test side-by-side compatibility and seamless rollback configuration."""
    health_resp = test_client.get("/health")
    assert health_resp.status_code == 200

    tags_resp = test_client.get("/api/tags")
    assert tags_resp.status_code == 200

    embed_resp = test_client.post("/v1/embeddings", json={"input": "test"})
    assert embed_resp.status_code == 200

