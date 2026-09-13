"""Model manifest schema and local versioning layout logic."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class ModelManifest(BaseModel):
    """Pydantic model representing a detailed model manifest with versioning and checksum telemetry."""

    name: str = Field(..., description="Model identifier or repository name.")
    version: str = Field("v1.0.0", description="Semver or content-hash version identifier.")
    source: str = Field("hf", description="Origin source (hf, local, registry).")
    revision: str = Field("main", description="Git revision or commit hash.")
    pulled_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp when model was pulled.",
    )
    size_bytes: int = Field(0, description="Total size in bytes of model files.")
    quant: str = Field("none", description="Quantization format (none, awq, gptq, fp8).")
    architecture: str = Field("unknown", description="Model architecture type (llama, mistral, qwen, etc.).")
    checksum_sha256: str = Field("", description="Overall manifest/directory SHA256 checksum.")
    checksum_per_file: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of relative file paths to SHA256 checksums for per-file integrity verification.",
    )


def get_default_models_dir() -> Path:
    """Return default root directory for Apah models (~/.apah/models)."""
    return Path.home() / ".apah" / "models"


def parse_model_identifier(model_identifier: str) -> Tuple[str, Optional[str]]:
    """Parse 'model_name:version' identifier into (model_name, version).

    If no version tag is attached, version returns None.
    Handles Hugging Face repo IDs containing slashes (e.g. 'Qwen/Qwen2.5-0.5B-Instruct:v1.0.0').
    """
    if ":" in model_identifier:
        # Split on the last colon to separate version tag from model name
        parts = model_identifier.rsplit(":", 1)
        name, version = parts[0], parts[1]
        # Avoid misidentifying windows path drive letters or ports
        if version and not version.startswith("/") and not version.startswith("\\"):
            return name, version
    return model_identifier, None


def sanitize_model_name(name: str) -> str:
    """Sanitize model name for filesystem directory compatibility."""
    return name.replace("/", "_").replace(":", "_")


def list_versions(model_name: str, models_root: Optional[Path] = None) -> List[ModelManifest]:
    """List all available local manifest versions for a given model name."""
    clean_name, _ = parse_model_identifier(model_name)
    root = models_root or get_default_models_dir()
    sanitized = sanitize_model_name(clean_name)
    model_base_dir = root / sanitized

    manifests: List[ModelManifest] = []
    if not model_base_dir.exists():
        return manifests

    # 1. Check version subdirectories (~/.apah/models/<name>/<version>/apah_manifest.json)
    for item in model_base_dir.iterdir():
        if item.is_dir():
            manifest_file = item / "apah_manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r") as f:
                        data = json.load(f)
                        # Normalize legacy manifest format if needed
                        if "checksum_per_file" not in data and "checksums" in data:
                            data["checksum_per_file"] = data.pop("checksums")
                        manifests.append(ModelManifest(**data))
                except Exception:
                    pass

    # 2. Legacy flat directory check (~/.apah/models/<name>/apah_manifest.json)
    legacy_manifest = model_base_dir / "apah_manifest.json"
    if legacy_manifest.exists():
        try:
            with open(legacy_manifest, "r") as f:
                data = json.load(f)
                if "checksum_per_file" not in data and "checksums" in data:
                    data["checksum_per_file"] = data.pop("checksums")
                if "version" not in data:
                    data["version"] = "v1.0.0"
                manifests.append(ModelManifest(**data))
        except Exception:
            pass

    # Sort manifests by pulled_at descending
    manifests.sort(key=lambda m: m.pulled_at, reverse=True)
    return manifests


def get_latest(model_name: str, models_root: Optional[Path] = None) -> Optional[ModelManifest]:
    """Retrieve manifest for a model name.

    If a specific version is specified (e.g. 'model:v1.0.0'), returns that version.
    Otherwise returns the version with the most recent 'pulled_at' timestamp.
    """
    clean_name, version = parse_model_identifier(model_name)
    manifests = list_versions(clean_name, models_root=models_root)

    if not manifests:
        return None

    if version is not None and version != "latest":
        for m in manifests:
            if m.version == version:
                return m
        return None

    return manifests[0]


def get_model_dir(
    model_name: str,
    version: Optional[str] = None,
    models_root: Optional[Path] = None,
) -> Path:
    """Resolve physical directory path for a model and version (~/.apah/models/<name>/<version>/)."""
    clean_name, parsed_version = parse_model_identifier(model_name)
    target_version = version or parsed_version
    root = models_root or get_default_models_dir()
    sanitized = sanitize_model_name(clean_name)

    base_dir = root / sanitized

    if target_version:
        return base_dir / target_version

    # If no version specified, check latest existing manifest directory
    latest_manifest = get_latest(clean_name, models_root=models_root)
    if latest_manifest:
        version_dir = base_dir / latest_manifest.version
        if version_dir.exists():
            return version_dir
        # Legacy flat layout fallback
        if (base_dir / "apah_manifest.json").exists():
            return base_dir

    return base_dir / "v1.0.0"
