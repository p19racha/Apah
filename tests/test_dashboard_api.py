"""Unit tests for Phase 12 dashboard backend endpoints and WebSockets."""

import json
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from apah.engine.server import create_app, server_state
from apah.security.audit_log import AuditLogger


@pytest.fixture
def app_client(tmp_path):
    """Fixture initializing FastAPI app client with temporary audit log dir."""
    test_log_dir = tmp_path / "audit_logs"
    test_log_dir.mkdir(parents=True, exist_ok=True)
    server_state.audit_logger = AuditLogger(log_dir=test_log_dir)

    app = create_app()
    client = TestClient(app)
    return client, test_log_dir


def test_dashboard_index_served(app_client):
    """Test /dashboard returns 200 HTTP status and serves HTML index page."""
    client, _ = app_client
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Apah Engine" in response.text or "Dashboard" in response.text


def test_models_endpoint(app_client):
    """Test /models returns JSON array of local model manifests."""
    client, _ = app_client
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_websocket_stats(app_client):
    """Test WebSocket /ws/stats delivers well-formed JSON with expected shape."""
    client, _ = app_client
    with client.websocket_connect("/ws/stats") as websocket:
        data = websocket.receive_json()
        assert "gpu" in data
        assert "scheduler" in data
        assert "is_loaded" in data
        assert "gpu_memory_mb" in data
        assert isinstance(data["gpu"], list)
        assert isinstance(data["scheduler"], dict)


def test_websocket_logs(app_client):
    """Test WebSocket /ws/logs streams new lines written to audit log."""
    client, test_log_dir = app_client

    # Write a test event via audit logger
    server_state.audit_logger.log_event(
        event_type="test_event",
        details={"message": "Phase 12 dashboard audit log test"},
    )

    with client.websocket_connect("/ws/logs") as websocket:
        # Receive log line
        message = websocket.receive_text()
        assert message is not None
        payload = json.loads(message)
        assert "event_type" in payload or "details" in payload
