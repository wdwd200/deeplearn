from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase5_common import (  # noqa: E402
    ABLATION_GROUPS,
    ABLATION_RESULTS_PATH,
    ABLATION_SUMMARY_PATH,
    ABLATION_SUMMARY_REQUIRED_FIELDS,
    ALGORITHM_EPISODE_PATH,
    ALGORITHM_SUMMARY_PATH,
    ALGORITHM_SUMMARY_REQUIRED_FIELDS,
    BASELINE_METHODS,
    DRL_METHODS,
    EPISODE_REQUIRED_FIELDS,
    EVAL_SCENARIO_IDS,
    PHASE4_ROOT,
    PHASE5_ANALYSIS,
    all_phase4_training_param_paths,
    assert_clean_phase5_generation,
    compare_summary_rows,
    ensure_phase5_dirs,
    finite_csv_report,
    finite_json_report,
    is_relative_recorded_path,
    load_training_metadata,
    official_metadata_values,
    read_csv_rows,
    recompute_ablation_summary,
    recompute_algorithm_summary,
    rel,
    required_fields_present,
    write_json,
)


def _check(condition: bool, name: str, details: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "details": details}


def _missing_set(actual: set[str], expected: set[str]) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    parts = []
    if missing:
        parts.append(f"missing={missing}")
    if extra:
        parts.append(f"extra={extra}")
    return "; ".join(parts)


def validate_final_results() -> dict[str, Any]:
    ensure_phase5_dirs()
    checks: list[dict[str, Any]] = []

    algorithm_rows = read_csv_rows(ALGORITHM_EPISODE_PATH)
    algorithm_summary = read_csv_rows(ALGORITHM_SUMMARY_PATH)
    ablation_rows = read_csv_rows(ABLATION_RESULTS_PATH)
    ablation_summary = read_csv_rows(ABLATION_SUMMARY_PATH)

    for path, rows, fields in [
        (ALGORITHM_EPISODE_PATH, algorithm_rows, EPISODE_REQUIRED_FIELDS),
        (ALGORITHM_SUMMARY_PATH, algorithm_summary, ALGORITHM_SUMMARY_REQUIRED_FIELDS),
        (ABLATION_RESULTS_PATH, ablation_rows, EPISODE_REQUIRED_FIELDS),
        (ABLATION_SUMMARY_PATH, ablation_summary, ABLATION_SUMMARY_REQUIRED_FIELDS),
    ]:
        errors = required_fields_present(rows, fields)
        checks.append(_check(not errors, f"{rel(path)} required fields", "; ".join(errors)))

    for method in DRL_METHODS:
        seeds = {
            int(row["training_seed"])
            for row in algorithm_rows
            if row["algorithm"] == method
        }
        checks.append(
            _check(
                seeds == {0, 1, 2},
                f"{method} has 3 training seeds",
                f"seeds={sorted(seeds)}",
            )
        )
        scenarios = {
            row["scenario_id"]
            for row in algorithm_rows
            if row["algorithm"] == method
        }
        checks.append(
            _check(
                scenarios == set(EVAL_SCENARIO_IDS),
                f"{method} evaluated on eval_0..eval_4",
                _missing_set(scenarios, set(EVAL_SCENARIO_IDS)),
            )
        )

    for method in BASELINE_METHODS:
        scenarios = {
            row["scenario_id"]
            for row in algorithm_rows
            if row["algorithm"] == method
        }
        checks.append(
            _check(
                scenarios == set(EVAL_SCENARIO_IDS),
                f"{method} uses fixed eval scenarios",
                _missing_set(scenarios, set(EVAL_SCENARIO_IDS)),
            )
        )

    for group in ABLATION_GROUPS:
        seeds = {int(row["training_seed"]) for row in ablation_rows if row["algorithm"] == group}
        scenarios = {row["scenario_id"] for row in ablation_rows if row["algorithm"] == group}
        checks.append(_check(seeds == {0, 1, 2}, f"{group} has 3 training seeds", f"seeds={sorted(seeds)}"))
        checks.append(
            _check(
                scenarios == set(EVAL_SCENARIO_IDS),
                f"{group} evaluated on eval_0..eval_4",
                _missing_set(scenarios, set(EVAL_SCENARIO_IDS)),
            )
        )

    params_paths = all_phase4_training_param_paths()
    metadata_errors: list[str] = []
    commits: set[str] = set()
    source_hashes: set[str] = set()
    for path in params_paths:
        metadata = load_training_metadata(path.parent)
        if metadata.get("git_dirty") is not False:
            metadata_errors.append(f"{rel(path)} git_dirty={metadata.get('git_dirty')!r}")
        if metadata.get("official_result") is not True:
            metadata_errors.append(f"{rel(path)} official_result={metadata.get('official_result')!r}")
        commit = str(metadata.get("git_commit", ""))
        source_hash = str(metadata.get("source_code_hash", ""))
        if not commit:
            metadata_errors.append(f"{rel(path)} missing git_commit")
        if not source_hash:
            metadata_errors.append(f"{rel(path)} missing source_code_hash")
        commits.add(commit)
        source_hashes.add(source_hash)
    checks.append(_check(not metadata_errors, "official metadata flags are valid", "; ".join(metadata_errors[:10])))
    checks.append(_check(len(commits) == 1, "all formal results use one git_commit", f"commits={sorted(commits)}"))
    checks.append(
        _check(
            len(source_hashes) == 1,
            "all formal results use one source_code_hash",
            f"source_hashes={sorted(source_hashes)}",
        )
    )

    path_errors: list[str] = []
    for path in params_paths:
        metadata = load_training_metadata(path.parent)
        path_errors.extend(_path_errors(metadata, rel(path)))
    for row in read_csv_rows(ROOT / "results" / "phase4" / "algorithm_training_summary.csv"):
        for key, value in row.items():
            if ("path" in key.lower() or key.lower().endswith("_dir")) and not is_relative_recorded_path(value):
                path_errors.append(f"algorithm_training_summary.csv {key}={value}")
    checks.append(_check(not path_errors, "recorded result paths are relative", "; ".join(path_errors[:10])))

    csv_errors = finite_csv_report(
        [
            ALGORITHM_EPISODE_PATH,
            ALGORITHM_SUMMARY_PATH,
            ABLATION_RESULTS_PATH,
            ABLATION_SUMMARY_PATH,
        ]
    )
    json_errors = finite_json_report(params_paths)
    checks.append(_check(not csv_errors and not json_errors, "no NaN or inf in Phase 4 CSV/JSON", "; ".join((csv_errors + json_errors)[:10])))

    duplicate_errors = _duplicate_key_errors(algorithm_rows, "algorithm", "algorithm episode")
    duplicate_errors += _duplicate_key_errors(ablation_rows, "algorithm", "ablation episode")
    checks.append(_check(not duplicate_errors, "no duplicate method + seed + scenario_id", "; ".join(duplicate_errors[:10])))

    summary_errors = compare_summary_rows(
        algorithm_summary,
        recompute_algorithm_summary(),
        "algorithm",
        (
            "average_rate_e2e_mean",
            "average_rate_e2e_std",
            "average_reward_mean",
            "average_reward_std",
            "outage_count_mean",
            "constraint_violation_count_mean",
            "trajectory_length_mean",
        ),
    )
    ablation_errors = compare_summary_rows(
        ablation_summary,
        recompute_ablation_summary(),
        "ablation",
        (
            "average_rate_e2e_mean",
            "average_rate_e2e_std",
            "average_reward_mean",
            "constraint_violation_count_mean",
            "trajectory_length_mean",
            "difference_from_full_td3",
        ),
    )
    checks.append(_check(not summary_errors, "algorithm summary matches episode data", "; ".join(summary_errors[:10])))
    checks.append(_check(not ablation_errors, "ablation summary matches episode data", "; ".join(ablation_errors[:10])))

    passed = all(check["passed"] for check in checks)
    metadata = official_metadata_values()
    report = {
        "passed": passed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase4_training_params_count": len(params_paths),
        "algorithm_episode_rows": len(algorithm_rows),
        "ablation_episode_rows": len(ablation_rows),
        "git_commit": metadata.get("git_commit", ""),
        "source_code_hash": metadata.get("source_code_hash", ""),
        "checks": checks,
    }
    write_json(PHASE5_ANALYSIS / "final_validation_report.json", report)
    _write_markdown_report(report, PHASE5_ANALYSIS / "final_validation_report.md")
    return report


def _path_errors(data: Any, prefix: str) -> list[str]:
    errors: list[str] = []
    if isinstance(data, Mapping):
        for key, value in data.items():
            if ("path" in str(key).lower() or str(key).lower().endswith("_dir")) and not is_relative_recorded_path(value):
                errors.append(f"{prefix}.{key}={value}")
            errors.extend(_path_errors(value, f"{prefix}.{key}"))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            errors.extend(_path_errors(value, f"{prefix}[{index}]"))
    return errors


def _duplicate_key_errors(rows: list[dict[str, str]], method_field: str, label: str) -> list[str]:
    seen: set[tuple[str, str, str]] = set()
    errors: list[str] = []
    for row in rows:
        key = (row[method_field], row["training_seed"], row["scenario_id"])
        if key in seen:
            errors.append(f"{label}: {key}")
        seen.add(key)
    return errors


def _write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Phase 5 Final Validation Report",
        "",
        f"- passed: `{str(report['passed']).lower()}`",
        f"- git_commit: `{report.get('git_commit', '')}`",
        f"- source_code_hash: `{report.get('source_code_hash', '')}`",
        f"- phase4_training_params_count: `{report.get('phase4_training_params_count', 0)}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        status = "pass" if check["passed"] else "fail"
        details = str(check.get("details", "")).replace("|", "\\|")
        lines.append(f"| {check['name']} | {status} | {details} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    assert_clean_phase5_generation()
    result = validate_final_results()
    print(json.dumps({"passed": result["passed"], "report": "results/phase5/analysis/final_validation_report.json"}, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)
