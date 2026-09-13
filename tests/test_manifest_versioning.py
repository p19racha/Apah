"""Unit tests for model manifest versioning and directory resolution."""

import json
from pathlib import Path

from apah.registry.manifest import (
    ModelManifest,
    get_latest,
    get_model_dir,
    list_versions,
    parse_model_identifier,
)


def test_parse_model_identifier():
    """Test model identifier parsing with and without version tags."""
    name, ver = parse_model_identifier("Qwen/Qwen2.5-0.5B-Instruct:v1.0.0")
    assert name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert ver == "v1.0.0"

    name2, ver2 = parse_model_identifier("Qwen/Qwen2.5-0.5B-Instruct")
    assert name2 == "Qwen/Qwen2.5-0.5B-Instruct"
    assert ver2 is None

    name3, ver3 = parse_model_identifier("my-model:v2.5")
    assert name3 == "my-model"
    assert ver3 == "v2.5"


def test_multiple_versions_coexist(tmp_path):
    """Test that multiple versions of the same model can coexist locally."""
    models_root = tmp_path / ".apah" / "models"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    v1_dir = models_root / "Qwen_Qwen2.5-0.5B-Instruct" / "v1.0.0"
    v2_dir = models_root / "Qwen_Qwen2.5-0.5B-Instruct" / "v1.1.0"
    v1_dir.mkdir(parents=True)
    v2_dir.mkdir(parents=True)

    m1 = ModelManifest(
        name=model_name,
        version="v1.0.0",
        source="hf",
        pulled_at="2026-09-10T10:00:00Z",
        size_bytes=1000,
        quant="none",
    )
    m2 = ModelManifest(
        name=model_name,
        version="v1.1.0",
        source="hf",
        pulled_at="2026-09-13T10:00:00Z",
        size_bytes=1000,
        quant="none",
    )

    with open(v1_dir / "apah_manifest.json", "w") as f:
        json.dump(m1.model_dump(), f)
    with open(v2_dir / "apah_manifest.json", "w") as f:
        json.dump(m2.model_dump(), f)

    versions = list_versions(model_name, models_root=models_root)
    assert len(versions) == 2
    assert [v.version for v in versions] == ["v1.1.0", "v1.0.0"]

    # get_latest without version returns most recent pulled_at (v1.1.0)
    latest = get_latest(model_name, models_root=models_root)
    assert latest is not None
    assert latest.version == "v1.1.0"

    # Pinning specific version returns v1.0.0
    pinned = get_latest(f"{model_name}:v1.0.0", models_root=models_root)
    assert pinned is not None
    assert pinned.version == "v1.0.0"

    # get_model_dir returns correct physical paths
    dir_v1 = get_model_dir(model_name, version="v1.0.0", models_root=models_root)
    assert dir_v1 == v1_dir

    dir_latest = get_model_dir(model_name, models_root=models_root)
    assert dir_latest == v2_dir
