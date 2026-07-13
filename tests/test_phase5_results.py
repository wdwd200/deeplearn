from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_paper_tables import build_paper_tables  # noqa: E402
from phase5_common import (  # noqa: E402
    ABLATION_RESULTS_PATH,
    ABLATION_SUMMARY_PATH,
    ALGORITHM_EPISODE_PATH,
    ALGORITHM_SUMMARY_PATH,
    PHASE5_SOURCE_DATA,
    PHASE5_ANALYSIS,
    compare_summary_rows,
    finite_csv_report,
    read_csv_rows,
    recompute_ablation_summary,
    recompute_algorithm_summary,
    to_float,
)


def test_phase5_final_validation_passes() -> None:
    report_path = PHASE5_ANALYSIS / "final_validation_report.json"
    assert report_path.exists()
    report = __import__("json").loads(report_path.read_text(encoding="utf-8"))

    assert report["passed"] is True


def test_algorithm_summary_can_be_recomputed_from_episode_rows() -> None:
    errors = compare_summary_rows(
        read_csv_rows(ALGORITHM_SUMMARY_PATH),
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

    assert errors == []


def test_ablation_summary_can_be_recomputed_from_episode_rows() -> None:
    errors = compare_summary_rows(
        read_csv_rows(ABLATION_SUMMARY_PATH),
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

    assert errors == []


def test_no_nan_or_inf_and_no_duplicate_keys() -> None:
    assert finite_csv_report([ALGORITHM_EPISODE_PATH, ALGORITHM_SUMMARY_PATH, ABLATION_RESULTS_PATH, ABLATION_SUMMARY_PATH]) == []

    for path, method_field in [(ALGORITHM_EPISODE_PATH, "algorithm"), (ABLATION_RESULTS_PATH, "algorithm")]:
        seen: set[tuple[str, str, str]] = set()
        for row in read_csv_rows(path):
            key = (row[method_field], row["training_seed"], row["scenario_id"])
            assert key not in seen
            seen.add(key)


def test_phase5_source_data_preserves_rate_bps_and_mbps() -> None:
    build_paper_tables()
    source_files = sorted(PHASE5_SOURCE_DATA.glob("*.csv"))

    assert source_files
    for path in source_files:
        rows = read_csv_rows(path)
        assert rows
        for row in rows:
            assert row["source_file"]
            assert row["source_stage"]
            assert row["git_commit"]
            assert row["source_code_hash"]
            if "rate_bps" in row and row["rate_bps"] != "":
                assert math.isclose(to_float(row["rate_mbps"]), to_float(row["rate_bps"]) / 1_000_000.0, rel_tol=1e-9)
