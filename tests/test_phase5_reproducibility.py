from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_reproducibility_manifest import build_reproducibility_manifest  # noqa: E402
from phase5_common import (  # noqa: E402
    PHASE5_FIGURES,
    PHASE5_MANIFESTS,
    PHASE5_TABLES,
    all_phase4_training_param_paths,
    load_training_metadata,
    official_metadata_values,
)


def _manifest() -> dict:
    path = PHASE5_MANIFESTS / "reproducibility_manifest.json"
    if not path.exists():
        build_reproducibility_manifest()
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_required_fields_and_metadata_match_official_results() -> None:
    manifest = _manifest()
    metadata = official_metadata_values()
    required = {
        "git_commit",
        "source_code_hash",
        "python_version",
        "package_versions",
        "operating_system",
        "config_files",
        "evaluation_scenarios",
        "training_seeds",
        "algorithm_result_files",
        "ablation_result_files",
        "figure_files",
        "table_files",
        "created_at",
    }

    assert required.issubset(manifest)
    assert manifest["git_commit"] == metadata["git_commit"]
    assert manifest["source_code_hash"] == metadata["source_code_hash"]


def test_manifest_paths_are_relative_and_include_figures_and_tables() -> None:
    manifest = _manifest()
    for collection in ("config_files", "algorithm_result_files", "ablation_result_files", "figure_files", "table_files"):
        for path in manifest[collection]:
            assert not Path(path).is_absolute()
            assert ":" not in path

    expected_figures = {str(path.relative_to(ROOT)).replace("\\", "/") for path in PHASE5_FIGURES.glob("*") if path.is_file()}
    expected_tables = {str(path.relative_to(ROOT)).replace("\\", "/") for path in PHASE5_TABLES.glob("*") if path.is_file()}
    assert expected_figures.issubset(set(manifest["figure_files"]))
    assert expected_tables.issubset(set(manifest["table_files"]))


def test_checksum_file_can_be_verified() -> None:
    _manifest()
    checksum_path = PHASE5_MANIFESTS / "file_checksums.sha256"
    if not checksum_path.exists():
        build_reproducibility_manifest()
    lines = [line for line in checksum_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert lines
    for line in lines:
        expected_hash, relative = line.split(maxsplit=1)
        path = ROOT / relative
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash


def test_all_formal_phase4_results_are_official() -> None:
    for path in all_phase4_training_param_paths():
        metadata = load_training_metadata(path.parent)
        assert metadata["official_result"] is True
        assert metadata["git_dirty"] is False
