"""Unit tests for process-level network isolation guard."""

import os
import socket
import pytest
from unittest.mock import MagicMock, patch

from apah.security.network_guard import (
    AirgapViolationError,
    check_airgap_pull_source,
    disable_airgap_mode,
    enable_airgap_mode,
    is_airgap_enabled,
)


@pytest.fixture(autouse=True)
def cleanup_airgap():
    """Ensure airgap mode is restored to clean state after each test."""
    yield
    disable_airgap_mode()
    if "APAH_AIRGAP_MODE" in os.environ:
        del os.environ["APAH_AIRGAP_MODE"]


def test_airgap_outbound_connection_blocked():
    """Test that outbound socket connection to non-loopback address raises AirgapViolationError."""
    enable_airgap_mode()
    assert is_airgap_enabled()

    with pytest.raises(AirgapViolationError) as exc_info:
        socket.create_connection(("example.com", 80))

    assert "Outbound connection attempt" in str(exc_info.value)
    assert "blocked by Apah airgap network guard" in str(exc_info.value)


def test_airgap_loopback_connection_permitted():
    """Test that loopback (127.0.0.1) socket connections are allowed through the guard."""
    enable_airgap_mode()

    # Mock low level original socket connect to avoid needing an actual open local server
    with patch("apah.security.network_guard._ORIGINAL_CONNECT", return_value=True):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Connecting to 127.0.0.1 should pass guard check without raising AirgapViolationError
            sock.connect(("127.0.0.1", 11500))
        except AirgapViolationError:
            pytest.fail("AirgapViolationError raised unexpectedly for loopback connection 127.0.0.1")
        except Exception:
            # Other OS connection error (e.g. connection refused) is acceptable, as long as AirgapViolationError is not raised
            pass
        finally:
            sock.close()


def test_airgap_pull_hf_rejected():
    """Test that pulling from Hugging Face is rejected when APAH_AIRGAP_MODE=1."""
    os.environ["APAH_AIRGAP_MODE"] = "1"
    enable_airgap_mode()

    with pytest.raises(AirgapViolationError) as exc_info:
        check_airgap_pull_source("hf")

    assert "Downloading from Hugging Face" in str(exc_info.value)

    # Local and registry pulls should not raise AirgapViolationError
    try:
        check_airgap_pull_source("local")
        check_airgap_pull_source("registry")
    except AirgapViolationError:
        pytest.fail("check_airgap_pull_source raised AirgapViolationError for local/registry source")


def test_disable_airgap_mode():
    """Test disabling airgap mode restores standard socket connectivity."""
    enable_airgap_mode()
    assert is_airgap_enabled()

    disable_airgap_mode()
    assert not is_airgap_enabled()
