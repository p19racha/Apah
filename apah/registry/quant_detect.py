"""Quantization format detection and memory footprint estimation module."""

from enum import Enum
import json
from pathlib import Path
from typing import Union


class QuantFormat(str, Enum):
    """Supported quantization formats for Apah inference runtime."""

    NONE = "none"
    AWQ = "awq"
    GPTQ = "gptq"
    FP8 = "fp8"


def detect_quant_format(model_dir_or_path: Union[str, Path]) -> QuantFormat:
    """Inspect model directory config.json and file names to detect quantization format.

    Args:
        model_dir_or_path: Path to model directory.

    Returns:
        QuantFormat enum value (NONE, AWQ, GPTQ, or FP8).
    """
    model_dir = Path(model_dir_or_path)
    if not model_dir.exists() or not model_dir.is_dir():
        return QuantFormat.NONE

    config_file = model_dir / "config.json"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                cfg = json.load(f)

            # Check explicit quantization_config dictionary
            quant_cfg = cfg.get("quantization_config")
            if isinstance(quant_cfg, dict):
                method = str(quant_cfg.get("quant_method", "")).lower()
                if "awq" in method:
                    return QuantFormat.AWQ
                if "gptq" in method:
                    return QuantFormat.GPTQ
                if "fp8" in method or "float8" in method:
                    return QuantFormat.FP8

            # Check torch_dtype / dtype
            dtype_str = str(cfg.get("torch_dtype") or cfg.get("dtype") or "").lower()
            if "fp8" in dtype_str or "float8" in dtype_str:
                return QuantFormat.FP8

        except Exception:
            pass

    # Inspect filenames for quantization markers
    for item in model_dir.iterdir():
        name_lower = item.name.lower()
        if "awq" in name_lower:
            return QuantFormat.AWQ
        if "gptq" in name_lower:
            return QuantFormat.GPTQ
        if "fp8" in name_lower:
            return QuantFormat.FP8

    return QuantFormat.NONE


def estimate_quant_memory_mb(model_dir: Path, quant_format: QuantFormat) -> float:
    """Estimate GPU memory footprint in MB for a given model directory and quantization format."""
    total_bytes = sum(
        f.stat().st_size
        for f in model_dir.rglob("*")
        if f.is_file() and f.suffix in (".safetensors", ".bin", ".pt")
    )
    if total_bytes > 0:
        # Add 20% overhead for activations, KV cache initialization, and CUDA runtime context
        return (total_bytes * 1.20) / (1024 * 1024)

    # Default fallback if weight files not found
    return 1024.0
