"""Internal registry client for HTTP/S3 endpoints and air-gapped shared directory mounts."""

import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union
import httpx

from apah.registry.checksum import compute_composite_checksum, compute_manifest_checksums, verify_manifest
from apah.registry.manifest import ModelManifest, sanitize_model_name


class InternalRegistryClient:
    """Registry client for air-gapped model transfers and internal HTTP/filesystem model stores."""

    def __init__(self, base_url_or_path: Union[str, Path]):
        self.endpoint_str = str(base_url_or_path)
        self.is_url = self.endpoint_str.startswith("http://") or self.endpoint_str.startswith("https://")
        if not self.is_url:
            self.local_registry_path = Path(base_url_or_path).resolve()
        else:
            self.local_registry_path = None

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models in the registry."""
        if not self.is_url:
            models: List[Dict[str, Any]] = []
            if not self.local_registry_path or not self.local_registry_path.exists():
                return models

            # Scan registry filesystem directory (~/<name>/<version>/apah_manifest.json or <name>/apah_manifest.json)
            for model_dir in self.local_registry_path.iterdir():
                if model_dir.is_dir():
                    # Check version subdirectories first
                    found_version = False
                    for ver_dir in model_dir.iterdir():
                        if ver_dir.is_dir() and (ver_dir / "apah_manifest.json").exists():
                            try:
                                with open(ver_dir / "apah_manifest.json") as f:
                                    models.append(json.load(f))
                                    found_version = True
                            except Exception:
                                pass
                    if not found_version and (model_dir / "apah_manifest.json").exists():
                        try:
                            with open(model_dir / "apah_manifest.json") as f:
                                models.append(json.load(f))
                        except Exception:
                            pass
            return models
        else:
            url = f"{self.endpoint_str.rstrip('/')}/v1/registry/models"
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                    return data if isinstance(data, list) else data.get("models", [])
            except Exception as e:
                raise RuntimeError(f"Failed to fetch models from HTTP registry '{self.endpoint_str}': {e}") from e

    def fetch_model(
        self,
        name: str,
        version: Optional[str] = None,
        dest_dir: Optional[Path] = None,
    ) -> ModelManifest:
        """Fetch/copy a model from the registry into dest_dir, verify checksums, and write manifest.

        Args:
            name: Model identifier or name.
            version: Optional target version string (e.g. 'v1.0.0').
            dest_dir: Target destination directory where model files will be written.

        Returns:
            Verified ModelManifest instance.
        """
        sanitized = sanitize_model_name(name)
        target_version = version or "v1.0.0"

        if dest_dir is None:
            dest_dir = Path.home() / ".apah" / "models" / sanitized / target_version

        dest_dir.mkdir(parents=True, exist_ok=True)

        if not self.is_url:
            # 1. Local filesystem path mode
            if not self.local_registry_path or not self.local_registry_path.exists():
                raise FileNotFoundError(f"Local registry directory '{self.endpoint_str}' does not exist.")

            # Search source directory in local registry
            src_dir = self.local_registry_path / sanitized / target_version
            if not src_dir.exists():
                src_dir = self.local_registry_path / sanitized
            if not src_dir.exists():
                raise FileNotFoundError(
                    f"Model '{name}' (version '{target_version}') not found in registry at '{self.endpoint_str}'."
                )

            # Copy files to dest_dir
            for item in src_dir.iterdir():
                dest_item = dest_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest_item)

        else:
            # 2. HTTP registry mode
            url = f"{self.endpoint_str.rstrip('/')}/v1/registry/models/{sanitized}/{target_version}/files"
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    files_list = resp.json().get("files", [])
                    for file_info in files_list:
                        file_name = file_info["name"]
                        file_url = file_info["url"]
                        file_resp = client.get(file_url)
                        file_resp.raise_for_status()
                        target_file = dest_dir / file_name
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        target_file.write_bytes(file_resp.content)
            except Exception as e:
                raise RuntimeError(f"HTTP registry download failed for '{name}:{target_version}': {e}") from e

        # Compute checksums and write manifest
        checksum_per_file = compute_manifest_checksums(dest_dir)
        composite_sha256 = compute_composite_checksum(checksum_per_file)

        # Detect quantization and size
        total_size = sum((dest_dir / fname).stat().st_size for fname in checksum_per_file)

        # Check existing config or manifest for metadata
        manifest_file = dest_dir / "apah_manifest.json"
        existing_meta = {}
        if manifest_file.exists():
            try:
                with open(manifest_file) as f:
                    existing_meta = json.load(f)
            except Exception:
                pass

        manifest = ModelManifest(
            name=name,
            version=target_version,
            source="registry",
            revision=existing_meta.get("revision", "main"),
            size_bytes=total_size,
            quant=existing_meta.get("quant", "none"),
            architecture=existing_meta.get("architecture", "unknown"),
            checksum_sha256=composite_sha256,
            checksum_per_file=checksum_per_file,
        )

        with open(manifest_file, "w") as f:
            json.dump(manifest.model_dump(), f, indent=2)

        # Enforce load-time integrity verification right after fetch
        verification = verify_manifest(dest_dir, manifest)
        if not verification.ok:
            raise RuntimeError(
                f"Registry fetch failed integrity verification for '{name}:{target_version}': {verification.error_message}"
            )

        return manifest
