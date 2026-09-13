"""Process-level socket patching and network guard enforcing air-gap isolation."""

import logging
import os
import socket
from typing import Any, Callable, Optional, Tuple, Union

logger = logging.getLogger("apah.security.network_guard")


class AirgapViolationError(Exception):
    """Raised when an outbound non-loopback connection is attempted in air-gap mode."""
    pass


_AIRGAP_ENABLED: bool = False
_ORIGINAL_CONNECT: Optional[Callable] = None
_ORIGINAL_CREATE_CONNECTION: Optional[Callable] = None


def _is_loopback(host: str) -> bool:
    """Check if target host or IP is a local loopback interface."""
    if host in ("127.0.0.1", "localhost", "::1", "0.0.0.0", ""):
        return True
    try:
        # Resolve host to IP address
        ip = socket.gethostbyname(host)
        if ip in ("127.0.0.1", "0.0.0.0"):
            return True
    except Exception:
        pass
    return False


def _patched_connect(self: socket.socket, address: Union[Tuple[Any, ...], str, bytes]) -> Any:
    """Patched socket.socket.connect blocking non-loopback outbound connections in air-gap mode."""
    if _AIRGAP_ENABLED:
        host = ""
        port = None
        if isinstance(address, tuple) and len(address) >= 1:
            host = str(address[0])
            if len(address) >= 2:
                port = address[1]
        elif isinstance(address, (str, bytes)):
            host = str(address)

        if not _is_loopback(host):
            err_msg = f"Outbound connection attempt to '{host}:{port}' blocked by Apah airgap network guard."
            logger.error(f"AIRGAP VIOLATION: {err_msg}")
            raise AirgapViolationError(err_msg)

    return _ORIGINAL_CONNECT(self, address)


def _patched_create_connection(
    address: Tuple[str, int],
    timeout: float = socket._GLOBAL_DEFAULT_TIMEOUT,
    source_address: Optional[Tuple[str, int]] = None,
    *,
    all_errors: bool = False,
) -> socket.socket:
    """Patched socket.create_connection blocking non-loopback connections."""
    if _AIRGAP_ENABLED and isinstance(address, tuple):
        host = str(address[0])
        if not _is_loopback(host):
            err_msg = f"Outbound connection attempt to '{host}' blocked by Apah airgap network guard."
            logger.error(f"AIRGAP VIOLATION: {err_msg}")
            raise AirgapViolationError(err_msg)

    return _ORIGINAL_CREATE_CONNECTION(
        address, timeout=timeout, source_address=source_address, all_errors=all_errors
    )


def enable_airgap_mode() -> None:
    """Enable process-level air-gap network isolation by patching socket connection methods."""
    global _AIRGAP_ENABLED, _ORIGINAL_CONNECT, _ORIGINAL_CREATE_CONNECTION

    if _ORIGINAL_CONNECT is None:
        _ORIGINAL_CONNECT = socket.socket.connect
    if _ORIGINAL_CREATE_CONNECTION is None:
        _ORIGINAL_CREATE_CONNECTION = socket.create_connection

    socket.socket.connect = _patched_connect
    socket.create_connection = _patched_create_connection
    _AIRGAP_ENABLED = True
    logger.info("Airgap network guard ENABLED. Non-loopback outbound socket connections are forbidden.")


def disable_airgap_mode() -> None:
    """Disable air-gap network isolation and restore original socket methods."""
    global _AIRGAP_ENABLED, _ORIGINAL_CONNECT, _ORIGINAL_CREATE_CONNECTION

    if _ORIGINAL_CONNECT is not None:
        socket.socket.connect = _ORIGINAL_CONNECT
    if _ORIGINAL_CREATE_CONNECTION is not None:
        socket.create_connection = _ORIGINAL_CREATE_CONNECTION

    _AIRGAP_ENABLED = False
    logger.info("Airgap network guard DISABLED.")


def is_airgap_enabled() -> bool:
    """Check if air-gap mode is active via environment variable or explicit invocation."""
    env_val = os.environ.get("APAH_AIRGAP_MODE", "0").lower()
    return _AIRGAP_ENABLED or env_val in ("1", "true", "yes")


def check_airgap_pull_source(source: str) -> None:
    """Enforce airgap rules on model pull source."""
    if is_airgap_enabled() and source == "hf":
        raise AirgapViolationError(
            "Downloading from Hugging Face ('--source hf') is forbidden when APAH_AIRGAP_MODE is enabled. "
            "Use '--source registry' or '--source local'."
        )


def format_airgap_banner() -> str:
    """Format server startup airgap status banner."""
    active = is_airgap_enabled()
    if active:
        return "[bold red]AIR-GAP MODE: ENABLED (Process-level network isolation active)[/bold red]"
    return "[yellow]AIR-GAP MODE: DISABLED (Standard network connectivity permitted)[/yellow]"
