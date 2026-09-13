"""Unit tests for Python-level sandbox path guard and systemd service configuration."""

from pathlib import Path
import pytest

from apah.security.sandbox import FilesystemAccessError, PathGuard, restrict_filesystem_access


def test_path_guard_allowed_and_denied(tmp_path: Path):
    """Test that PathGuard allows access within permitted directories and denies access outside."""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir(parents=True, exist_ok=True)
    denied_dir = tmp_path / "denied"
    denied_dir.mkdir(parents=True, exist_ok=True)

    guard = PathGuard(allowed_paths=[allowed_dir])

    # Allowed path check
    sub_file = allowed_dir / "models" / "model.bin"
    assert guard.is_path_allowed(sub_file) is True
    validated = guard.validate_path(sub_file)
    assert validated == sub_file.resolve()

    # Denied path check
    forbidden_file = denied_dir / "secret.txt"
    assert guard.is_path_allowed(forbidden_file) is False
    with pytest.raises(FilesystemAccessError) as exc_info:
        guard.validate_path(forbidden_file)

    assert "Access denied" in str(exc_info.value)


def test_systemd_service_file_exists_and_valid():
    """Verify that deploy/apah.service is present and contains required security directives."""
    service_path = Path(__file__).parent.parent / "deploy" / "apah.service"
    assert service_path.exists(), f"Service file missing at {service_path}"

    content = service_path.read_text()

    # Verify critical OS-level hardening directives are present
    assert "NoNewPrivileges=true" in content
    assert "ProtectSystem=strict" in content
    assert "ProtectHome=true" in content
    assert "PrivateNetwork=true" in content
    assert "ReadWritePaths=" in content
    assert "APAH_AIRGAP_MODE=1" in content
