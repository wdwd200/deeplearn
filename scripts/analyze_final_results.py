from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase5_common import (  # noqa: E402
    ABLATION_RESULTS_PATH,
    ALGORITHM_EPISODE_PATH,
    BASELINE_METHODS,
    DRL_METHODS,
    PHASE5_ANALYSIS,
    ensure_phase5_dirs,
    group_rows,
    mean,
    median,
    pstdev,
    read_csv_rows,
    to_float,
    write_csv_rows,
)


def _rate_stats(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    rates = [to_float(row["average_rate_e2e"]) for row in rows]
    rewards = [to_float(row["total_reward"]) for row in rows]
    constraints = [to_float(row["constraint_violation_count"]) for row in rows]
    lengths = [to_float(row["trajectory_length"]) for row in rows]
    return {
        "sample_count": float(len(rows)),
        "mean_rate_bps": mean(rates),
        "mean_rate_mbps": mean(rates) / 1_000_000.0,
        "std_rate_bps": pstdev(rates),
        "std_rate_mbps": pstdev(rates) / 1_000_000.0,
        "median_rate_bps": median(rates),
        "median_rate_mbps": median(rates) / 1_000_000.0,
        "min_rate_bps": min(rates) if rates else 0.0,
        "min_rate_mbps": (min(rates) / 1_000_000.0) if rates else 0.0,
        "max_rate_bps": max(rates) if rates else 0.0,
        "max_rate_mbps": (max(rates) / 1_000_000.0) if rates else 0.0,
        "mean_reward": mean(rewards),
        "mean_constraint_violation_count": mean(constraints),
        "mean_trajectory_length": mean(lengths),
    }


def build_algorithm_statistics() -> list[dict[str, Any]]:
    grouped = group_rows(read_csv_rows(ALGORITHM_EPISODE_PATH), "algorithm")
    base_rows: dict[str, dict[str, Any]] = {}
    for method, rows in sorted(grouped.items()):
        stats = _rate_stats(rows)
        stats["method"] = method
        stats["sample_unit"] = "training_seed x eval_scenario" if method in DRL_METHODS else "eval_scenario"
        base_rows[method] = stats

    best_baseline = max(BASELINE_METHODS, key=lambda method: base_rows[method]["mean_rate_bps"])
    td3_mean = base_rows["TD3"]["mean_rate_bps"]
    ddpg_mean = base_rows["DDPG"]["mean_rate_bps"]
    sac_mean = base_rows["SAC"]["mean_rate_bps"]
    best_baseline_mean = base_rows[best_baseline]["mean_rate_bps"]

    rows: list[dict[str, Any]] = []
    for method in (*DRL_METHODS, *BASELINE_METHODS):
        row = dict(base_rows[method])
        row["best_baseline"] = best_baseline
        row["td3_minus_ddpg_bps"] = td3_mean - ddpg_mean
        row["td3_minus_ddpg_mbps"] = (td3_mean - ddpg_mean) / 1_000_000.0
        row["td3_minus_sac_bps"] = td3_mean - sac_mean
        row["td3_minus_sac_mbps"] = (td3_mean - sac_mean) / 1_000_000.0
        row["sac_minus_best_baseline_bps"] = sac_mean - best_baseline_mean
        row["sac_minus_best_baseline_mbps"] = (sac_mean - best_baseline_mean) / 1_000_000.0
        row["td3_minus_best_baseline_bps"] = td3_mean - best_baseline_mean
        row["td3_minus_best_baseline_mbps"] = (td3_mean - best_baseline_mean) / 1_000_000.0
        rows.append(row)
    return rows


def build_ablation_statistics() -> list[dict[str, Any]]:
    grouped = group_rows(read_csv_rows(ABLATION_RESULTS_PATH), "algorithm")
    base_stats = {group: _rate_stats(rows) for group, rows in grouped.items()}
    a0_rate = base_stats["A0_full_td3"]["mean_rate_bps"]
    rows = []
    for group in sorted(base_stats):
        row = dict(base_stats[group])
        row["ablation"] = group
        row["delta_from_A0_bps"] = row["mean_rate_bps"] - a0_rate
        row["delta_from_A0_mbps"] = (row["mean_rate_bps"] - a0_rate) / 1_000_000.0
        rows.append(row)
    return rows


def build_scenario_statistics() -> list[dict[str, Any]]:
    source_rows = read_csv_rows(ALGORITHM_EPISODE_PATH)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in source_rows:
        grouped.setdefault((row["scenario_id"], row["algorithm"]), []).append(row)

    per_scenario: dict[str, list[dict[str, Any]]] = {}
    for (scenario_id, method), rows in grouped.items():
        rates = [to_float(row["average_rate_e2e"]) for row in rows]
        per_scenario.setdefault(scenario_id, []).append(
            {
                "scenario_id": scenario_id,
                "method": method,
                "mean_rate_bps": mean(rates),
                "mean_rate_mbps": mean(rates) / 1_000_000.0,
                "std_rate_bps": pstdev(rates),
                "std_rate_mbps": pstdev(rates) / 1_000_000.0,
                "sample_count": len(rates),
            }
        )

    output: list[dict[str, Any]] = []
    for scenario_id, rows in sorted(per_scenario.items()):
        ranked = sorted(rows, key=lambda row: row["mean_rate_bps"], reverse=True)
        for rank, row in enumerate(ranked, start=1):
            new_row = dict(row)
            new_row["rank"] = rank
            output.append(new_row)
    return output


def build_constraint_statistics() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, rows in sorted(group_rows(read_csv_rows(ALGORITHM_EPISODE_PATH), "algorithm").items()):
        values = [to_float(row["constraint_violation_count"]) for row in rows]
        lengths = [to_float(row["trajectory_length"]) for row in rows]
        mean_length = mean(lengths)
        output.append(
            {
                "method": method,
                "sample_count": len(values),
                "mean_constraint_violation_count": mean(values),
                "std_constraint_violation_count": pstdev(values),
                "median_constraint_violation_count": median(values),
                "min_constraint_violation_count": min(values) if values else 0.0,
                "max_constraint_violation_count": max(values) if values else 0.0,
                "total_constraint_violation_count": sum(values),
                "mean_trajectory_length": mean_length,
                "constraint_count_per_100_steps": mean(values) / mean_length * 100.0 if mean_length else 0.0,
            }
        )
    return output


def analyze_final_results() -> list[Path]:
    ensure_phase5_dirs()
    outputs = [
        (PHASE5_ANALYSIS / "algorithm_statistics.csv", build_algorithm_statistics()),
        (PHASE5_ANALYSIS / "ablation_statistics.csv", build_ablation_statistics()),
        (PHASE5_ANALYSIS / "scenario_statistics.csv", build_scenario_statistics()),
        (PHASE5_ANALYSIS / "constraint_statistics.csv", build_constraint_statistics()),
    ]
    paths: list[Path] = []
    for path, rows in outputs:
        write_csv_rows(path, rows)
        paths.append(path)
        print(f"saved analysis: {path}")
    return paths


if __name__ == "__main__":
    analyze_final_results()
