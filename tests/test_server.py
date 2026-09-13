"""Tests for Apah FastAPI Server endpoints."""

import json
import pytest
import torch
from fastapi.testclient import TestClient

from apah.engine.server import create_app, server_state

TEST_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    server_state.unload()


def test_health_endpoint(client):
    """Verify /health returns 200 OK without requiring a loaded model."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ps_empty_when_no_model_loaded(client):
    """Verify /ps returns empty list when no model is loaded."""
    server_state.unload()
    response = client.get("/ps")
    assert response.status_code == 200
    assert response.json() == []


def test_chat_completions_error_no_model_loaded(client):
    """Verify /v1/chat/completions returns 400 error when no model is loaded."""
    server_state.unload()
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "model_not_loaded"


def test_load_model_success(client):
    """Verify POST /load successfully loads a model."""
    payload = {
        "model_path": TEST_MODEL_PATH,
        "dtype": "float16",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    response = client.post("/load", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert server_state.is_loaded
    assert server_state.model_name == TEST_MODEL_PATH


def test_ps_loaded_state(client):
    """Verify /ps reflects loaded model status."""
    response = client.get("/ps")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == TEST_MODEL_PATH
    assert data[0]["loaded"] is True


def test_load_model_conflict(client):
    """Verify POST /load returns 409 Conflict if a different model load is requested."""
    payload = {
        "model_path": "different/model-id",
        "dtype": "float16",
    }
    response = client.post("/load", json=payload)
    assert response.status_code == 409
    assert "already loaded" in response.json()["error"]


def test_chat_completions_non_streaming(client):
    """Verify POST /v1/chat/completions returns valid non-streaming OpenAI response."""
    payload = {
        "model": TEST_MODEL_PATH,
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"].strip()) > 0
    assert data["usage"]["total_tokens"] > 0


def test_chat_completions_streaming(client):
    """Verify POST /v1/chat/completions with stream=True yields SSE chunks ending in [DONE]."""
    payload = {
        "model": TEST_MODEL_PATH,
        "messages": [{"role": "user", "content": "Count from 1 to 3."}],
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    raw_text = response.text
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

    data_lines = [line for line in lines if line.startswith("data:")]
    assert len(data_lines) >= 2
    assert data_lines[-1] == "data: [DONE]"

    json_chunk = json.loads(data_lines[0].replace("data: ", ""))
    assert json_chunk["object"] == "chat.completion.chunk"
    assert "choices" in json_chunk


def test_unload_model_frees_gpu_memory(client):
    """Verify POST /unload frees GPU memory and clears process status."""
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated()
        assert mem_before > 0

    response = client.post("/unload")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert not server_state.is_loaded

    ps_resp = client.get("/ps")
    assert ps_resp.json() == []

    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated()
        assert mem_after < mem_before
