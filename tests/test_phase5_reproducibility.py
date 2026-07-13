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
    PHASE5_REQUIRED_SCRIPT_PATHS,
    PHASE5_TABLES,
    all_phase4_training_param_paths,
    compute_phase5_source_code_hash,
    git_commit_exists,
    git_tree_files,
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
        "phase4_result_commit",
        "phase4_source_code_hash",
        "phase5_code_commit",
        "phase5_git_dirty",
        "phase5_source_code_hash",
        "phase5_generated_at",
        "python_version",
        "package_versions",
        "operating_system",
        "config_files",
        "evaluation_scenarios",
        "training_seeds",
        "algorithm_result_files",
        "ablation_result_files",
        "analysis_files",
        "figure_files",
        "source_data_files",
        "table_files",
        "created_at",
    }

    assert required.issubset(manifest)
    assert manifest["git_commit"] == metadata["git_commit"]
    assert manifest["source_code_hash"] == metadata["source_code_hash"]
    assert manifest["phase4_result_commit"] == metadata["git_commit"]
    assert manifest["phase4_source_code_hash"] == metadata["source_code_hash"]
    assert manifest["phase5_code_commit"]
    assert manifest["phase5_source_code_hash"]
    assert manifest["phase5_git_dirty"] is False
    assert manifest["phase4_result_commit"] != manifest["phase5_code_commit"]


def test_phase5_code_commit_exists_and_contains_generation_scripts() -> None:
    manifest = _manifest()
    commit = manifest["phase5_code_commit"]
    files = git_tree_files(commit)

    assert git_commit_exists(commit)
    for script_path in PHASE5_REQUIRED_SCRIPT_PATHS:
        assert script_path in files


def test_phase5_source_hash_is_stable_and_changes_with_script_content(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    script = tmp_path / "scripts" / "phase5_common.py"
    config = tmp_path / "configs" / "comm_env_default.yaml"
    script.write_text("print('phase5')\n", encoding="utf-8")
    config.write_text("simulation:\n  max_steps: 1\n", encoding="utf-8")
    paths = ["scripts/phase5_common.py", "configs/comm_env_default.yaml"]

    first = compute_phase5_source_code_hash(paths=paths, root=tmp_path)
    second = compute_phase5_source_code_hash(paths=paths, root=tmp_path)
    script.write_text("print('phase5 changed')\n", encoding="utf-8")
    changed = compute_phase5_source_code_hash(paths=paths, root=tmp_path)

    assert first == second
    assert changed != first


def test_manifest_paths_are_relative_and_include_figures_and_tables() -> None:
    manifest = _manifest()
    for path in _manifest_paths(manifest):
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

    figure_1 = "results/phase5/figures/figure_1_system_model.png"
    assert any(line.endswith(f"  {figure_1}") for line in lines)


def test_all_formal_phase4_results_are_official() -> None:
    for path in all_phase4_training_param_paths():
        metadata = load_training_metadata(path.parent)
        assert metadata["official_result"] is True
        assert metadata["git_dirty"] is False


def _manifest_paths(value: object) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            paths.extend(_manifest_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_manifest_paths(item))
    elif isinstance(value, str) and (value.startswith("configs/") or value.startswith("results/") or value.startswith("scripts/")):
        paths.append(value)
    return paths
