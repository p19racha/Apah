"""Tests for Apah Dashboard serving, WebSockets (/ws/logs, /ws/stats), /models, and /config APIs."""

import json
import pytest
from fastapi.testclient import TestClient

from apah.engine.server import create_app, server_state


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    server_state.unload()


def test_dashboard_index_served(client):
    """Verify GET /dashboard serves static index.html page (200 OK, text/html)."""
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Apah" in resp.text or "Dashboard" in resp.text

    resp_file = client.get("/dashboard/index.html")
    assert resp_file.status_code == 200
    assert "text/html" in resp_file.headers["content-type"]


def test_dashboard_models_list(client):
    """Verify GET /models returns list of model manifests without drift from /v1/models."""
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)

    resp_v1 = client.get("/v1/models")
    assert resp_v1.status_code == 200
    assert resp_v1.json() == data


def test_config_endpoint_get_and_post(client):
    """Verify GET /config returns configuration and POST /config updates settings."""
    resp = client.get("/config")
    assert resp.status_code == 200
    cfg = resp.json()

    assert "idle_timeout_seconds" in cfg
    assert "auto_unload_enabled" in cfg
    assert "airgap_mode_enabled" in cfg

    # Update idle timeout
    update_resp = client.post("/config", json={"idle_timeout_seconds": 600, "auto_unload_enabled": True})
    assert update_resp.status_code == 200
    assert update_resp.json()["idle_timeout_seconds"] == 600
    assert server_state.idle_manager.idle_timeout_seconds == 600


def test_ws_stats_stream(client):
    """Verify WebSocket /ws/stats delivers well-formed telemetry JSON payload."""
    with client.websocket_connect("/ws/stats") as websocket:
        data_str = websocket.receive_text()
        data = json.loads(data_str)

        assert "gpu" in data
        assert "scheduler" in data
        assert "server" in data

        assert isinstance(data["gpu"], list)
        assert "active_batch_size" in data["scheduler"]
        assert "aggregate_throughput_tok_s" in data["scheduler"]
        assert "is_loaded" in data["server"]


def test_ws_logs_stream(client):
    """Verify WebSocket /ws/logs streams new audit log entries as written."""
    server_state.audit_logger.log_event("dashboard_test_event", details={"phase": 12})
    with client.websocket_connect("/ws/logs") as websocket:
        found = False
        while True:
            try:
                line = websocket.receive_text()
                if "dashboard_test_event" in line:
                    found = True
                    break
            except Exception:
                break

        assert found, "WebSocket /ws/logs failed to receive 'dashboard_test_event'"
