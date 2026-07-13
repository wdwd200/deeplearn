from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase5_common import (  # noqa: E402
    EVAL_SCENARIOS_PATH,
    PHASE4_CONFIG_PATH,
    PHASE4_ROOT,
    PHASE5_ANALYSIS,
    PHASE5_FIGURES,
    PHASE5_MANIFESTS,
    PHASE5_SOURCE_DATA,
    PHASE5_TABLES,
    assert_clean_phase5_generation,
    checksum_paths,
    compute_phase5_source_code_hash,
    current_git_commit,
    current_phase5_git_dirty,
    ensure_phase5_dirs,
    load_phase4_config,
    model_file_paths,
    now_utc,
    official_metadata_values,
    python_environment,
    rel,
    sha256_file,
    validate_phase5_commit_contains_scripts,
    write_json,
)


def _files_under(root: Path, pattern: str = "*") -> list[str]:
    if not root.exists():
        return []
    return [rel(path) for path in sorted(root.rglob(pattern), key=rel) if path.is_file()]


def build_reproducibility_manifest(require_clean: bool = True) -> tuple[Path, Path]:
    ensure_phase5_dirs()
    if require_clean:
        assert_clean_phase5_generation()
    metadata = official_metadata_values()
    env = python_environment()
    config = load_phase4_config(PHASE4_CONFIG_PATH)
    phase5_code_commit = current_git_commit()
    phase5_source_code_hash = compute_phase5_source_code_hash()
    phase5_git_dirty = current_phase5_git_dirty()
    generated_at = now_utc()
    commit_errors = validate_phase5_commit_contains_scripts(phase5_code_commit)
    if commit_errors:
        raise RuntimeError("; ".join(commit_errors))
    if require_clean and phase5_git_dirty:
        raise RuntimeError(
            "Phase 5 formal results must be generated from a committed and clean code version. "
            "Please commit code first, then rerun."
        )

    model_files = [
        {"path": path, "sha256": sha256_file(ROOT / path)}
        for path in model_file_paths()
    ]
    manifest: dict[str, Any] = {
        "git_commit": metadata.get("git_commit", ""),
        "source_code_hash": metadata.get("source_code_hash", ""),
        "phase4_result_commit": metadata.get("git_commit", ""),
        "phase4_source_code_hash": metadata.get("source_code_hash", ""),
        "phase5_code_commit": phase5_code_commit,
        "phase5_generation_commit": phase5_code_commit,
        "phase5_git_dirty": phase5_git_dirty,
        "phase5_source_code_hash": phase5_source_code_hash,
        "phase5_generated_at": generated_at,
        "python_version": env["python_version"],
        "package_versions": env["package_versions"],
        "operating_system": env["operating_system"],
        "config_files": [
            "configs/comm_env_default.yaml",
            "configs/phase4_experiments.yaml",
            "configs/phase4_eval_scenarios.yaml",
        ],
        "evaluation_scenarios": rel(EVAL_SCENARIOS_PATH),
        "training_seeds": [int(seed) for seed in config["experiment"]["training_seeds"]],
        "algorithm_result_files": _files_under(PHASE4_ROOT / "algorithms", "*.json")
        + _files_under(PHASE4_ROOT / "algorithms", "*.csv"),
        "ablation_result_files": _files_under(PHASE4_ROOT / "ablations", "*.json")
        + _files_under(PHASE4_ROOT / "ablations", "*.csv"),
        "model_files": model_files,
        "analysis_files": _files_under(PHASE5_ANALYSIS),
        "figure_files": _files_under(PHASE5_FIGURES),
        "source_data_files": _files_under(PHASE5_SOURCE_DATA),
        "table_files": _files_under(PHASE5_TABLES),
        "created_at": generated_at,
    }
    manifest_path = PHASE5_MANIFESTS / "reproducibility_manifest.json"
    write_json(manifest_path, manifest)

    checksum_path = PHASE5_MANIFESTS / "file_checksums.sha256"
    lines = [f"{sha256_file(path)}  {rel(path)}" for path in checksum_paths()]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved manifest: {manifest_path}")
    print(f"saved checksums: {checksum_path}")
    return manifest_path, checksum_path


if __name__ == "__main__":
    build_reproducibility_manifest()
