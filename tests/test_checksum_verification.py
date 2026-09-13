"""Unit tests for checksum hashing, manifest verification, and pre-load integrity enforcement."""

import json
from pathlib import Path
from typer.testing import CliRunner

from apah.cli.client import ApahClient
from apah.cli.main import app
from apah.registry.checksum import compute_manifest_checksums, verify_manifest
from apah.registry.manifest import ModelManifest

runner = CliRunner()


def test_checksum_verification_success(tmp_path):
    """Test verify_manifest returns ok=True when all files match checksums."""
    model_dir = tmp_path / "model_v1"
    model_dir.mkdir()

    file1 = model_dir / "config.json"
    file2 = model_dir / "model.safetensors"

    file1.write_text('{"model_type": "qwen2"}')
    file2.write_bytes(b"sample_weight_bytes_123456789")

    checksums = compute_manifest_checksums(model_dir)
    manifest = ModelManifest(
        name="test-model",
        version="v1.0.0",
        size_bytes=sum(f.stat().st_size for f in [file1, file2]),
        checksum_per_file=checksums,
    )

    result = verify_manifest(model_dir, manifest)
    assert result.ok is True
    assert len(result.mismatched_files) == 0


def test_checksum_verification_corrupted_file(tmp_path):
    """Test verify_manifest catches byte corruption in weight files."""
    model_dir = tmp_path / "model_v1"
    model_dir.mkdir()

    file1 = model_dir / "config.json"
    file2 = model_dir / "model.safetensors"

    file1.write_text('{"model_type": "qwen2"}')
    file2.write_bytes(b"original_uncorrupted_bytes")

    checksums = compute_manifest_checksums(model_dir)
    manifest = ModelManifest(
        name="test-model",
        version="v1.0.0",
        checksum_per_file=checksums,
    )

    # Corrupt 1 byte in weight file
    file2.write_bytes(b"original_Xorrupted_bytes")

    result = verify_manifest(model_dir, manifest)
    assert result.ok is False
    assert any("model.safetensors" in item for item in result.mismatched_files)
    assert result.error_message is not None
    assert "Integrity verification failed" in result.error_message


def test_apah_run_refuses_corrupted_model(tmp_path, monkeypatch):
    """Test apah run detects corrupted model on disk and refuses to load into GPU memory."""
    models_root = tmp_path / ".apah" / "models"
    model_dir = models_root / "corrupted_model" / "v1.0.0"
    model_dir.mkdir(parents=True)

    file1 = model_dir / "config.json"
    file2 = model_dir / "model.safetensors"
    file1.write_text('{"model_type": "qwen2"}')
    file2.write_bytes(b"original_valid_data")

    checksums = compute_manifest_checksums(model_dir)
    manifest = ModelManifest(
        name="corrupted_model",
        version="v1.0.0",
        checksum_per_file=checksums,
    )
    with open(model_dir / "apah_manifest.json", "w") as f:
        json.dump(manifest.model_dump(), f)

    # Corrupt weight file
    file2.write_bytes(b"TAMPERED_DATA_BYTES")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(ApahClient, "ps", lambda self: [])

    result = runner.invoke(app, ["run", "corrupted_model:v1.0.0", "Hello"])
    assert result.exit_code == 1
    assert "Pre-load integrity verification failed" in result.stdout
    assert "Refusing to load corrupted model" in result.stdout
