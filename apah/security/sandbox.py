"""Python-level filesystem path guard and defense-in-depth sandbox helpers.

SECURITY NOTE:
Userspace Python-level path validation is NOT a security boundary against arbitrary code execution
or malicious dependencies that can invoke C/syscalls directly. OS-level enforcement (systemd unit
hardening, cgroups, AppArmor, SELinux, mount namespaces, seccomp filters) MUST be deployed as the
primary security boundary in industrial production environments.

Apah ships a reference hardened systemd service definition in `deploy/apah.service`.
"""

import logging
import os
from pathlib import Path
from typing import List, Union

logger = logging.getLogger("apah.security.sandbox")


class FilesystemAccessError(PermissionError):
    """Raised when access to a path outside allowed directories is attempted."""

    pass


class PathGuard:
    """Python-level defense-in-depth path validator restricting operations to allowed directories."""

    def __init__(self, allowed_paths: List[Path]):
        self.allowed_paths = [Path(p).expanduser().resolve() for p in allowed_paths]

    def is_path_allowed(self, path: Union[str, Path]) -> bool:
        """Check whether target path falls strictly within one of the allowed directories."""
        try:
            target = Path(path).expanduser().resolve()
        except Exception as e:
            logger.warning(f"Failed to resolve path '{path}': {e}")
            return False

        for allowed in self.allowed_paths:
            try:
                # Check if target is relative to allowed path
                target.relative_to(allowed)
                return True
            except ValueError:
                continue

        return False

    def validate_path(self, path: Union[str, Path]) -> Path:
        """Validate target path and return resolved Path object, or raise FilesystemAccessError."""
        target = Path(path).expanduser().resolve()
        if not self.is_path_allowed(target):
            err_msg = (
                f"Access denied: path '{target}' is outside configured allowed directories: "
                f"{[str(p) for p in self.allowed_paths]}"
            )
            logger.error(f"SANDBOX VIOLATION: {err_msg}")
            raise FilesystemAccessError(err_msg)
        return target


_GLOBAL_PATH_GUARD: PathGuard = None


def restrict_filesystem_access(allowed_paths: List[Path]) -> PathGuard:
    """Set up process-level allowed directory rules for Python defense-in-depth path validation."""
    global _GLOBAL_PATH_GUARD
    # Ensure default model store and audit log dirs are included if not present
    default_model_dir = (Path.home() / ".apah" / "models").expanduser().resolve()
    default_audit_dir = (Path.home() / ".apah" / "audit_logs").expanduser().resolve()

    all_paths = list(allowed_paths)
    if default_model_dir not in all_paths:
        all_paths.append(default_model_dir)
    if default_audit_dir not in all_paths:
        all_paths.append(default_audit_dir)

    _GLOBAL_PATH_GUARD = PathGuard(all_paths)
    logger.info(
        f"Python path sandbox configured with allowed directories: {[str(p) for p in _GLOBAL_PATH_GUARD.allowed_paths]}"
    )
    return _GLOBAL_PATH_GUARD


def get_path_guard() -> PathGuard:
    """Get current active PathGuard instance, or initialize default if none exists."""
    global _GLOBAL_PATH_GUARD
    if _GLOBAL_PATH_GUARD is None:
        default_model_dir = Path.home() / ".apah" / "models"
        default_audit_dir = Path.home() / ".apah" / "audit_logs"
        restrict_filesystem_access([default_model_dir, default_audit_dir])
    return _GLOBAL_PATH_GUARD


def format_sandbox_banner() -> str:
    """Format server startup sandbox status banner."""
    guard = get_path_guard()
    allowed_str = ", ".join([str(p) for p in guard.allowed_paths])
    return (
        f"[green]SANDBOX STATUS: Python Path Guard ACTIVE (Allowed paths: {allowed_str})[/green]\n"
        f"  [italic dim](Note: OS-level systemd/container isolation recommended for true process isolation)[/italic dim]"
    )
