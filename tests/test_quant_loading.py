"""Unit tests for quantization format detection and branched model loading."""

import json
from pathlib import Path
import pytest

from apah.engine.model_loader import load_model
from apah.registry.quant_detect import QuantFormat, detect_quant_format, estimate_quant_memory_mb


def test_detect_quant_format_awq(tmp_path):
    """Test detect_quant_format identifies AWQ quantization format from config.json."""
    model_dir = tmp_path / "qwen2-awq"
    model_dir.mkdir()
    config = {
        "model_type": "qwen2",
        "quantization_config": {"quant_method": "awq", "bits": 4},
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model-awq.safetensors").write_bytes(b"awq_dummy_weights")

    fmt = detect_quant_format(model_dir)
    assert fmt == QuantFormat.AWQ

    est_mb = estimate_quant_memory_mb(model_dir, fmt)
    assert est_mb > 0.0


def test_detect_quant_format_gptq(tmp_path):
    """Test detect_quant_format identifies GPTQ quantization format from config.json."""
    model_dir = tmp_path / "llama-gptq"
    model_dir.mkdir()
    config = {
        "model_type": "llama",
        "quantization_config": {"quant_method": "gptq", "bits": 4},
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    fmt = detect_quant_format(model_dir)
    assert fmt == QuantFormat.GPTQ


def test_detect_quant_format_fp8(tmp_path):
    """Test detect_quant_format identifies FP8 quantization format."""
    model_dir = tmp_path / "mistral-fp8"
    model_dir.mkdir()
    config = {
        "model_type": "mistral",
        "torch_dtype": "float8_e4m3fn",
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    fmt = detect_quant_format(model_dir)
    assert fmt == QuantFormat.FP8


def test_quant_missing_library_error(tmp_path, monkeypatch):
    """Test load_model raises clear ImportError when autoawq is required but missing."""
    model_dir = tmp_path / "model_awq"
    model_dir.mkdir()
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "quantization_config": {"quant_method": "awq"},
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"dummy_weights")

    # Mock CUDA check to pass
    monkeypatch.setattr("apah.engine.model_loader._check_cuda_available", lambda device: None)

    # Force autoawq import to fail and AutoModelForCausalLM fallback to fail
    def mock_from_pretrained(*args, **kwargs):
        raise RuntimeError("Transformers cannot load AWQ directly")

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", mock_from_pretrained)

    with pytest.raises(ImportError) as exc_info:
        load_model(str(model_dir), dtype="bfloat16", device="cuda")

    assert "autoawq" in str(exc_info.value)
    assert "pip install autoawq" in str(exc_info.value)
