"""Tests for Apah CLI entrypoint and client integration."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

from apah.cli.client import (
    ApahAPIError,
    ApahClient,
    ModelNotLoadedError,
    ServerNotRunningError,
)
from apah.cli.formatting import human_duration, human_size
from apah.cli.main import app

runner = CliRunner()


def test_formatting_helpers():
    """Test human_size and human_duration utility functions."""
    assert human_size(0) == "0 B"
    assert human_size(500) == "500 B"
    assert human_size(1024) == "1 KB"
    assert human_size(1572864) == "1.5 MB"
    assert human_size(4508123136) == "4.2 GB"

    assert human_duration(0) == "0s"
    assert human_duration(45) == "45s"
    assert human_duration(125) == "2m 5s"
    assert human_duration(3725) == "1h 2m 5s"


def test_version():
    """Test apah --version and apah -v commands."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Apah version" in result.stdout

    result_v = runner.invoke(app, ["-v"])
    assert result_v.exit_code == 0
    assert "Apah version" in result_v.stdout


def test_help():
    """Test apah --help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Apah: Custom LLM inference runtime" in result.stdout
    assert "serve" in result.stdout
    assert "run" in result.stdout
    assert "pull" in result.stdout
    assert "list" in result.stdout
    assert "ps" in result.stdout
    assert "stop" in result.stdout
    assert "rm" in result.stdout
    assert "show" in result.stdout


def test_ps_server_not_running(monkeypatch):
    """Test apah ps prints clear error when server is unreachable without a stack trace."""
    def mock_ps(self):
        raise ServerNotRunningError("Could not connect to Apah server at http://localhost:11500. Run 'apah serve' first.")

    monkeypatch.setattr(ApahClient, "ps", mock_ps)
    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 1
    assert "Could not connect to Apah server" in result.stdout
    assert "Traceback" not in result.stdout


def test_ps_server_running(monkeypatch):
    """Test apah ps formatted table when server is active."""
    def mock_ps(self):
        return [
            {
                "name": "Qwen/Qwen2.5-0.5B-Instruct",
                "gpu_memory_mb": 1024.5,
                "active_batch_size": 2,
                "waiting_queue_length": 1,
                "uptime_seconds": 3725.0,
            }
        ]

    monkeypatch.setattr(ApahClient, "ps", mock_ps)
    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 0
    assert "Qwen/Qwen2.5-0.5B-Instruct" in result.stdout
    assert "1024.5 MB" in result.stdout
    assert "1h 2m 5s" in result.stdout


def test_list_models(tmp_path, monkeypatch):
    """Test apah list scans ~/.apah/models/ and prints rich table."""
    fake_models_dir = tmp_path / ".apah" / "models"
    fake_model_dir = fake_models_dir / "test_model" / "v1.0.0"
    fake_model_dir.mkdir(parents=True)

    manifest_data = {
        "name": "test-model",
        "version": "v1.0.0",
        "source": "hf",
        "revision": "main",
        "pulled_at": "2026-09-13T00:00:00Z",
        "size_bytes": 4508123136,
        "quant": "bfloat16",
        "checksum_per_file": {"config.json": "abcd"},
    }
    with open(fake_model_dir / "apah_manifest.json", "w") as f:
        json.dump(manifest_data, f)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "test-model" in result.stdout
    assert "v1.0.0" in result.stdout
    assert "4.2 GB" in result.stdout
    assert "bfloat16" in result.stdout


def test_rm_model_confirmation_and_yes(tmp_path, monkeypatch):
    """Test apah rm prompts for confirmation and respects --yes."""
    fake_models_dir = tmp_path / ".apah" / "models"
    fake_model_dir = fake_models_dir / "model_to_remove" / "v1.0.0"
    fake_model_dir.mkdir(parents=True)

    manifest_data = {"name": "model_to_remove", "version": "v1.0.0"}
    with open(fake_model_dir / "apah_manifest.json", "w") as f:
        json.dump(manifest_data, f)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # 1. Abort on confirmation 'n'
    result_n = runner.invoke(app, ["rm", "model_to_remove"], input="n\n")
    assert result_n.exit_code == 0
    assert "Aborted" in result_n.stdout
    assert fake_model_dir.exists()

    # 2. Confirm with --yes
    result_yes = runner.invoke(app, ["rm", "model_to_remove", "--yes"])
    assert result_yes.exit_code == 0
    assert "Successfully removed" in result_yes.stdout
    assert not fake_model_dir.exists()


def test_show_model(tmp_path, monkeypatch):
    """Test apah show displays manifest info and live status."""
    fake_models_dir = tmp_path / ".apah" / "models"
    fake_model_dir = fake_models_dir / "show_model" / "v1.0.0"
    fake_model_dir.mkdir(parents=True)

    manifest_data = {
        "name": "show_model",
        "version": "v1.0.0",
        "source": "hf",
        "revision": "main",
        "pulled_at": "2026-09-13T00:00:00Z",
        "size_bytes": 1048576,
        "quant": "none",
        "checksum_per_file": {"file.bin": "1234"},
    }
    with open(fake_model_dir / "apah_manifest.json", "w") as f:
        json.dump(manifest_data, f)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(ApahClient, "ps", lambda self: [{"name": "show_model"}])

    result = runner.invoke(app, ["show", "show_model"])
    assert result.exit_code == 0
    assert "show_model" in result.stdout
    assert "Loaded on server" in result.stdout


def test_stop_model(monkeypatch):
    """Test apah stop calls server unload endpoint."""
    def mock_unload(self, model_name=None):
        return {"status": "success", "message": f"Model '{model_name}' unloaded successfully."}

    monkeypatch.setattr(ApahClient, "unload", mock_unload)

    result = runner.invoke(app, ["stop", "test-model"])
    assert result.exit_code == 0
    assert "Model 'test-model' unloaded successfully." in result.stdout


def test_run_oneshot(monkeypatch):
    """Test apah run with one-shot prompt."""
    monkeypatch.setattr(ApahClient, "ps", lambda self: [{"name": "loaded-model"}])

    def mock_stream(self, model, messages, temperature=0.7, max_tokens=512, top_p=0.9):
        yield "Hello "
        yield "world!"

    monkeypatch.setattr(ApahClient, "chat_completion_stream", mock_stream)

    result = runner.invoke(app, ["run", "loaded-model", "Say hi", "--stream"])
    assert result.exit_code == 0
    assert "Hello world!" in result.stdout


def test_pull_local(tmp_path, monkeypatch):
    """Test apah pull from local directory source."""
    src_dir = tmp_path / "my_source_model"
    src_dir.mkdir()
    (src_dir / "config.json").write_text('{"torch_dtype": "float16"}')
    (src_dir / "model.safetensors").write_bytes(b"1234567890")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = runner.invoke(app, ["pull", "my-local-model", "--source", "local", "--local-path", str(src_dir)])
    assert result.exit_code == 0
    assert "Successfully pulled" in result.stdout

    manifest_file = tmp_path / ".apah" / "models" / "my-local-model" / "v1.0.0" / "apah_manifest.json"
    assert manifest_file.exists()
    with open(manifest_file) as f:
        data = json.load(f)
        assert data["name"] == "my-local-model"
        assert data["version"] == "v1.0.0"
        assert "config.json" in data["checksum_per_file"]
        assert "model.safetensors" in data["checksum_per_file"]
