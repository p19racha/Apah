"""Checksum hashing and load-time integrity verification module."""

hashlib_import = True
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from apah.registry.manifest import ModelManifest


class VerificationResult(BaseModel):
    """Result payload for model manifest checksum verification."""

    ok: bool = Field(..., description="Whether integrity verification succeeded with 100% hash match.")
    mismatched_files: List[str] = Field(
        default_factory=list,
        description="List of relative file paths that are missing, altered, or corrupted.",
    )
    error_message: Optional[str] = Field(None, description="Detailed human-readable error summary if verification failed.")


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hex digest for a single file using 64KB chunk streaming."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_manifest_checksums(model_dir: Path) -> Dict[str, str]:
    """Compute SHA256 checksum per file for all files inside model_dir (excluding apah_manifest.json)."""
    checksums: Dict[str, str] = {}
    if not model_dir.exists():
        return checksums

    for file_path in model_dir.rglob("*"):
        if file_path.is_file() and file_path.name != "apah_manifest.json":
            rel_name = str(file_path.relative_to(model_dir))
            checksums[rel_name] = compute_file_sha256(file_path)

    return checksums


def compute_composite_checksum(checksum_per_file: Dict[str, str]) -> str:
    """Compute a single composite SHA256 hash over all sorted file hashes."""
    hasher = hashlib.sha256()
    for fname in sorted(checksum_per_file.keys()):
        hasher.update(f"{fname}:{checksum_per_file[fname]}".encode("utf-8"))
    return hasher.hexdigest()


def verify_manifest(model_dir: Path, manifest: ModelManifest) -> VerificationResult:
    """Verify integrity of all files in model_dir against manifest checksums.

    Args:
        model_dir: Physical directory path containing model files.
        manifest: Target ModelManifest object.

    Returns:
        VerificationResult with ok=True if all files match, or ok=False with mismatched_files list.
    """
    if not model_dir.exists():
        return VerificationResult(
            ok=False,
            mismatched_files=[],
            error_message=f"Model directory '{model_dir}' does not exist.",
        )

    expected_hashes = manifest.checksum_per_file
    mismatched: List[str] = []

    if not expected_hashes:
        # If no per-file checksums were stored in manifest, fall back to checking if directory has files
        files_present = [f for f in model_dir.rglob("*") if f.is_file() and f.name != "apah_manifest.json"]
        if not files_present:
            return VerificationResult(
                ok=False,
                mismatched_files=["<empty_directory>"],
                error_message="Model directory contains no weight or configuration files.",
            )
        # Calculate hashes if empty
        expected_hashes = compute_manifest_checksums(model_dir)

    for rel_path_str, expected_hash in expected_hashes.items():
        target_file = model_dir / rel_path_str
        if not target_file.exists():
            mismatched.append(f"{rel_path_str} (missing)")
            continue

        try:
            actual_hash = compute_file_sha256(target_file)
            if actual_hash != expected_hash:
                mismatched.append(f"{rel_path_str} (checksum mismatch)")
        except Exception as e:
            mismatched.append(f"{rel_path_str} (read error: {e})")

    # Verify optional overall composite checksum if specified
    if manifest.checksum_sha256 and not mismatched:
        current_checksums = compute_manifest_checksums(model_dir)
        actual_composite = compute_composite_checksum(current_checksums)
        if actual_composite != manifest.checksum_sha256:
            mismatched.append("<composite_checksum_mismatch>")

    if mismatched:
        err_msg = f"Integrity verification failed for model '{manifest.name}:{manifest.version}'. {len(mismatched)} file(s) mismatched or missing: {', '.join(mismatched[:3])}"
        if len(mismatched) > 3:
            err_msg += f" ... (+{len(mismatched) - 3} more)"
        return VerificationResult(ok=False, mismatched_files=mismatched, error_message=err_msg)

    return VerificationResult(ok=True, mismatched_files=[], error_message=None)
