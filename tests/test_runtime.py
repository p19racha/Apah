"""Tests for Apah LLM Inference Runtime."""

import gc
import pytest
import torch

from apah.engine.model_loader import (
    load_model,
    _validate_architecture,
    _validate_safetensors_format,
)
from apah.engine.runtime import ApahRuntime
from transformers import AutoConfig

TEST_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def loaded_model_and_tokenizer(device):
    """Fixture to load model once for module tests."""
    model, tokenizer = load_model(TEST_MODEL_PATH, dtype="float16", device=device)
    yield model, tokenizer
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_model_loader_success(loaded_model_and_tokenizer):
    """Verify model loads without error and returns model + tokenizer tuple."""
    model, tokenizer = loaded_model_and_tokenizer
    assert model is not None
    assert tokenizer is not None


def test_runtime_generate_non_empty(loaded_model_and_tokenizer, device):
    """Verify generate() returns non-empty text."""
    model, tokenizer = loaded_model_and_tokenizer
    runtime = ApahRuntime(model, tokenizer, device=device)

    prompt = "Artificial Intelligence is"
    output = runtime.generate(prompt=prompt, max_new_tokens=20, temperature=0.7)

    assert isinstance(output, str)
    assert len(output.strip()) > 0
    assert runtime.last_metrics is not None
    assert runtime.last_metrics.generated_tokens > 0
    assert runtime.last_metrics.ttft_sec > 0


def test_runtime_generate_stream_incremental(loaded_model_and_tokenizer, device):
    """Verify generate_stream() yields tokens incrementally."""
    model, tokenizer = loaded_model_and_tokenizer
    runtime = ApahRuntime(model, tokenizer, device=device)

    prompt = "The sky is"
    tokens = list(runtime.generate_stream(prompt=prompt, max_new_tokens=15, temperature=1.0))

    assert len(tokens) > 0
    full_text = "".join(tokens)
    assert len(full_text.strip()) > 0
    assert isinstance(tokens[0], str)


def test_gpu_memory_freed_on_delete(device):
    """Verify GPU memory is freed after the runtime object is deleted."""
    if device != "cuda" or not torch.cuda.is_available():
        pytest.skip("CUDA device not available for GPU memory test")

    gc.collect()
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(device)

    # Load model and create runtime
    runtime = ApahRuntime.from_model_path(TEST_MODEL_PATH, dtype="float16", device=device)
    loaded_memory = torch.cuda.memory_allocated(device)
    assert loaded_memory > initial_memory

    # Explicitly unload/delete runtime
    runtime.unload()
    del runtime
    gc.collect()
    torch.cuda.empty_cache()

    freed_memory = torch.cuda.memory_allocated(device)
    assert freed_memory < loaded_memory
    assert abs(freed_memory - initial_memory) < 50 * 1024 * 1024  # Within 50MB margin


def test_error_unsupported_architecture():
    """Verify clear error is raised for unsupported model architectures."""
    fake_config = AutoConfig.for_model("bert")
    with pytest.raises(ValueError, match="Unsupported model architecture"):
        _validate_architecture(fake_config)


def test_error_non_safetensors_format(tmp_path):
    """Verify clear error is raised if directory contains no safetensors files."""
    dummy_dir = tmp_path / "dummy_bin_model"
    dummy_dir.mkdir()
    (dummy_dir / "pytorch_model.bin").write_bytes(b"dummy weights")

    with pytest.raises(ValueError, match="does not contain '.safetensors'"):
        _validate_safetensors_format(str(dummy_dir))
