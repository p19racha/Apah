"""Unit tests for IdleManager auto-unloading and race condition protection."""

import asyncio
import time
from unittest.mock import MagicMock
import pytest

from apah.engine.idle_manager import IdleManager
from apah.engine.server import server_state


@pytest.mark.asyncio
async def test_idle_manager_auto_unload(monkeypatch):
    """Test IdleManager triggers auto-unload after idle timeout when no requests are active."""
    mock_scheduler = MagicMock()
    mock_scheduler.get_stats.return_value = {
        "active_batch_size": 0,
        "waiting_queue_length": 0,
    }

    idle_mgr = IdleManager(
        scheduler=mock_scheduler,
        idle_timeout_seconds=2,
        enabled=True,
        check_interval_seconds=0.1,
    )

    # Mock runtime and server_state
    mock_runtime = MagicMock()
    monkeypatch.setattr(server_state, "runtime", mock_runtime)
    monkeypatch.setattr(server_state, "model_name", "test-idle-model")
    unload_called = False

    def mock_unload():
        nonlocal unload_called
        unload_called = True
        monkeypatch.setattr(server_state, "runtime", None)
        monkeypatch.setattr(server_state, "model_name", None)

    monkeypatch.setattr(server_state, "unload", mock_unload)

    # Set last activity to 3 seconds ago
    idle_mgr._last_activity_time = time.time() - 3.0

    task = asyncio.create_task(idle_mgr.run_loop())
    await asyncio.sleep(0.3)
    idle_mgr.stop()
    await asyncio.sleep(0.1)

    assert unload_called is True


@pytest.mark.asyncio
async def test_idle_manager_race_condition_guard(monkeypatch):
    """Test IdleManager does NOT auto-unload if requests are active or waiting."""
    mock_scheduler = MagicMock()
    mock_scheduler.get_stats.return_value = {
        "active_batch_size": 1,
        "waiting_queue_length": 0,
    }

    idle_mgr = IdleManager(
        scheduler=mock_scheduler,
        idle_timeout_seconds=2,
        enabled=True,
        check_interval_seconds=0.1,
    )

    mock_runtime = MagicMock()
    monkeypatch.setattr(server_state, "runtime", mock_runtime)
    monkeypatch.setattr(server_state, "model_name", "busy-model")
    unload_called = False

    def mock_unload():
        nonlocal unload_called
        unload_called = True

    monkeypatch.setattr(server_state, "unload", mock_unload)
    idle_mgr._last_activity_time = time.time() - 5.0

    task = asyncio.create_task(idle_mgr.run_loop())
    await asyncio.sleep(0.3)
    idle_mgr.stop()

    assert unload_called is False


@pytest.mark.asyncio
async def test_idle_manager_disabled(monkeypatch):
    """Test --no-idle-unload (enabled=False) prevents auto-unloading."""
    mock_scheduler = MagicMock()
    mock_scheduler.get_stats.return_value = {"active_batch_size": 0, "waiting_queue_length": 0}

    idle_mgr = IdleManager(
        scheduler=mock_scheduler,
        idle_timeout_seconds=1,
        enabled=False,
        check_interval_seconds=0.1,
    )

    mock_runtime = MagicMock()
    monkeypatch.setattr(server_state, "runtime", mock_runtime)
    monkeypatch.setattr(server_state, "model_name", "test-model")
    unload_called = False

    def mock_unload():
        nonlocal unload_called
        unload_called = True

    monkeypatch.setattr(server_state, "unload", mock_unload)
    idle_mgr._last_activity_time = time.time() - 10.0

    task = asyncio.create_task(idle_mgr.run_loop())
    await asyncio.sleep(0.3)
    idle_mgr.stop()

    assert unload_called is False


def test_idle_manager_status(monkeypatch):
    """Test get_idle_status reports active model idle seconds."""
    mock_scheduler = MagicMock()
    idle_mgr = IdleManager(scheduler=mock_scheduler)

    mock_runtime = MagicMock()
    monkeypatch.setattr(server_state, "runtime", mock_runtime)
    monkeypatch.setattr(server_state, "model_name", "active-model")
    idle_mgr._last_activity_time = time.time() - 15.0

    status = idle_mgr.get_idle_status()
    assert "active-model" in status
    assert status["active-model"] >= 14.5
