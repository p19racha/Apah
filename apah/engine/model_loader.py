"""Model loader for Apah LLM inference runtime.

Loads Hugging Face causal LMs in safetensors format onto NVIDIA GPUs with safety checks:
- CUDA availability check
- Model architecture compatibility check (Llama, Mistral, Qwen)
- Safetensors format enforcement
- Pre-load GPU memory check
- Quantization format detection and branched loading (AWQ, GPTQ, FP8, FP16/BF16)
"""

import logging
import os
from pathlib import Path
from typing import Set, Tuple
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from apah.registry.quant_detect import QuantFormat, detect_quant_format, estimate_quant_memory_mb

logger = logging.getLogger("apah.engine.model_loader")

# List of tested and supported architectures / model types
TESTED_ARCHITECTURES: Set[str] = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
}

TESTED_MODEL_TYPES: Set[str] = {
    "llama",
    "mistral",
    "qwen2",
    "qwen2_moe",
    "tiny_llama",
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

BYTES_PER_PARAM = {
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.float32: 4,
}


def _check_cuda_available(device: str) -> None:
    """Validate CUDA availability if CUDA device is specified."""
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA execution requested on device '{device}', but CUDA is not available on this system. "
            "Apah requires an NVIDIA GPU with working CUDA drivers."
        )


def _validate_architecture(config: AutoConfig) -> None:
    """Validate if the model architecture is supported."""
    model_type = getattr(config, "model_type", "") or ""
    model_type = model_type.lower()
    raw_archs = getattr(config, "architectures", None)
    architectures = set(raw_archs) if raw_archs is not None else set()

    is_supported_type = model_type in TESTED_MODEL_TYPES
    is_supported_arch = bool(architectures.intersection(TESTED_ARCHITECTURES))

    if not (is_supported_type or is_supported_arch):
        tested_str = ", ".join(sorted(TESTED_ARCHITECTURES))
        raise ValueError(
            f"Unsupported model architecture '{raw_archs}' (model_type: '{model_type}'). "
            f"Apah currently supports tested architectures: [{tested_str}]."
        )


def _validate_safetensors_format(model_path: str) -> None:
    """Verify that the model file is in safetensors format."""
    if os.path.isdir(model_path):
        files = os.listdir(model_path)
        has_safetensors = any(f.endswith(".safetensors") for f in files)
        has_bin = any(f.endswith(".bin") for f in files)

        if not has_safetensors:
            raise ValueError(
                f"Model directory '{model_path}' does not contain '.safetensors' weight files. "
                "Apah strictly requires models in safetensors format."
            )
        if has_bin and not has_safetensors:
            raise ValueError(
                f"Model at '{model_path}' contains legacy PyTorch '.bin' files instead of safetensors. "
                "Only safetensors format is supported."
            )
    else:
        try:
            from huggingface_hub import list_repo_files

            files = list_repo_files(model_path)
            has_safetensors = any(f.endswith(".safetensors") for f in files)
            if not has_safetensors:
                raise ValueError(
                    f"Hugging Face repository '{model_path}' does not contain safetensors weight files. "
                    "Apah strictly requires safetensors format."
                )
        except ValueError:
            raise
        except Exception:
            pass


def _estimate_model_memory_bytes(model_path: str, config: AutoConfig, torch_dtype: torch.dtype) -> int:
    """Estimate required memory in bytes for storing model weights."""
    if os.path.isdir(model_path):
        safetensors_bytes = sum(
            os.path.getsize(os.path.join(model_path, f))
            for f in os.listdir(model_path)
            if f.endswith(".safetensors")
        )
        if safetensors_bytes > 0:
            return safetensors_bytes

    hidden_size = getattr(config, "hidden_size", 4096)
    num_layers = getattr(config, "num_hidden_layers", 32)
    vocab_size = getattr(config, "vocab_size", 32000)
    intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)

    approx_params = vocab_size * hidden_size + num_layers * (
        4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size
    )
    bytes_per_param = BYTES_PER_PARAM.get(torch_dtype, 2)
    return int(approx_params * bytes_per_param)


def _check_gpu_memory(model_path: str, config: AutoConfig, torch_dtype: torch.dtype, device: str) -> None:
    """Check if available GPU memory is sufficient for the model before loading."""
    if not torch.cuda.is_available() or "cuda" not in device:
        return

    try:
        device_obj = torch.device(device)
        device_index = device_obj.index if device_obj.index is not None else 0
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)

        est_weight_bytes = _estimate_model_memory_bytes(model_path, config, torch_dtype)
        safety_margin_bytes = 500 * 1024 * 1024
        required_bytes = est_weight_bytes + safety_margin_bytes

        if required_bytes > free_bytes:
            req_gb = required_bytes / (1024**3)
            free_gb = free_bytes / (1024**3)
            raise MemoryError(
                f"Insufficient GPU memory on device '{device}'. "
                f"Estimated required memory: {req_gb:.2f} GB, "
                f"Available free GPU memory: {free_gb:.2f} GB."
            )
    except MemoryError:
        raise
    except Exception as e:
        logger.warning(f"Could not pre-check GPU memory: {e}")


def load_model(
    model_path: str,
    dtype: str = "bfloat16",
    device: str = "cuda",
    tp_world_size: int = 1,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a Hugging Face Causal LM and Tokenizer onto specified device(s).

    Args:
        model_path: Hugging Face model ID or local directory path.
        dtype: Data type for model weights ("bfloat16", "float16", "float32").
        device: Target execution device (e.g. "cuda", "cuda:0").
        tp_world_size: Tensor parallel world size (number of GPUs).

    Returns:
        Tuple of (loaded PreTrainedModel, loaded PreTrainedTokenizerBase).

    Raises:
        RuntimeError: If CUDA is requested but not available or requested GPUs are insufficient.
        ValueError: If model architecture is unsupported, dtype is invalid, or weights are non-safetensors.
        ImportError: If required quantization libraries (autoawq, auto_gptq) are missing.
        MemoryError: If available GPU memory is insufficient for the model.
    """
    if tp_world_size > 1:
        from apah.engine.parallel.launcher import launch_tensor_parallel
        return launch_tensor_parallel(model_path=model_path, tp_world_size=tp_world_size, dtype=dtype, device=device)

    _check_cuda_available(device)

    if dtype not in DTYPE_MAP:
        supported = ", ".join(DTYPE_MAP.keys())
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported dtypes: [{supported}]")
    torch_dtype = DTYPE_MAP[dtype]

    # Validate safetensors format
    _validate_safetensors_format(model_path)

    # Detect quantization format and estimate GPU memory footprint
    model_path_obj = Path(model_path)
    quant_format = detect_quant_format(model_path_obj)
    if model_path_obj.exists():
        est_mem_mb = estimate_quant_memory_mb(model_path_obj, quant_format)
        logger.info(f"Detected quantization format: '{quant_format.value}'. Expected GPU memory footprint: {est_mem_mb:.2f} MB")

    # Load configuration to validate architecture and check memory
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    _validate_architecture(config)

    # Pre-check GPU memory before loading weights
    _check_gpu_memory(model_path, config, torch_dtype, device)

    logger.info(f"Loading tokenizer from '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading model from '{model_path}' with quant='{quant_format.value}' onto device={device}...")

    # Branched loading based on quantization format
    if quant_format == QuantFormat.AWQ:
        try:
            from awq import AutoAWQForCausalLM
            logger.info("Loading AWQ model via AutoAWQForCausalLM...")
            model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=True, trust_remote_code=False)
        except ImportError:
            try:
                # Fall back to AutoModelForCausalLM if transformers has AWQ support compiled
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=torch_dtype,
                    use_safetensors=True,
                    trust_remote_code=False,
                )
            except Exception as e:
                raise ImportError(
                    f"Loading AWQ quantized model '{model_path}' requires the 'autoawq' package. "
                    "Please run 'pip install autoawq' to enable AWQ quantized model loading."
                ) from e

    elif quant_format == QuantFormat.GPTQ:
        try:
            from auto_gptq import AutoGPTQForCausalLM
            logger.info("Loading GPTQ model via AutoGPTQForCausalLM...")
            model = AutoGPTQForCausalLM.from_quantized(model_path, use_safetensors=True, trust_remote_code=False)
        except ImportError:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=torch_dtype,
                    use_safetensors=True,
                    trust_remote_code=False,
                )
            except Exception as e:
                raise ImportError(
                    f"Loading GPTQ quantized model '{model_path}' requires the 'auto_gptq' package. "
                    "Please run 'pip install auto_gptq' to enable GPTQ quantized model loading."
                ) from e

    else:
        # Standard FP16/BF16/FP32 load
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            use_safetensors=True,
            trust_remote_code=False,
        )

    model.to(device)
    model.eval()

    if "cuda" in device and torch.cuda.is_available():
        allocated_bytes = torch.cuda.memory_allocated(device)
        allocated_mb = allocated_bytes / (1024 * 1024)
        logger.info(f"Model loaded successfully on {device}. GPU memory allocated: {allocated_mb:.2f} MB")

    return model, tokenizer
