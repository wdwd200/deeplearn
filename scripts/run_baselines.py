from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_relay_env import EpisodeMetrics, UAVRelayCommEnv, load_config, make_default_policies

RESULTS_DIR = ROOT / "results" / "phase1"
SUMMARY_PATH = RESULTS_DIR / "baseline_results.csv"
STEPS_PATH = RESULTS_DIR / "baseline_step_results.csv"

STEP_FIELDS = [
    "policy",
    "step",
    "q_R_x",
    "q_R_y",
    "q_R_z",
    "d_HR",
    "d_RL",
    "theta_HR",
    "theta_RL",
    "snr_HR",
    "snr_RL",
    "rate_HR",
    "rate_RL",
    "rate_e2e",
    "reward",
    "constraint_violation",
]

SUMMARY_FIELDS = [
    "policy",
    "episode_length",
    "total_reward",
    "average_rate_e2e",
    "average_rate_HR",
    "average_rate_RL",
    "average_snr_HR",
    "average_snr_RL",
    "outage_count",
    "constraint_violation_count",
    "trajectory_length",
    "final_q_R_x",
    "final_q_R_y",
    "final_q_R_z",
]


def _step_row(policy_name: str, info: dict[str, Any], reward: float) -> dict[str, Any]:
    q_r = info["q_R"]
    return {
        "policy": policy_name,
        "step": info["step"],
        "q_R_x": q_r[0],
        "q_R_y": q_r[1],
        "q_R_z": q_r[2],
        "d_HR": info["d_HR"],
        "d_RL": info["d_RL"],
        "theta_HR": info["theta_HR"],
        "theta_RL": info["theta_RL"],
        "snr_HR": info["snr_HR"],
        "snr_RL": info["snr_RL"],
        "rate_HR": info["rate_HR"],
        "rate_RL": info["rate_RL"],
        "rate_e2e": info["rate_e2e"],
        "reward": reward,
        "constraint_violation": int(bool(info["constraint_violation"])),
    }


def _summary_row(policy_name: str, metrics: EpisodeMetrics) -> dict[str, Any]:
    summary = metrics.summary()
    final_q_r = summary["final_q_R"] if summary["final_q_R"] else [0.0, 0.0, 0.0]
    return {
        "policy": policy_name,
        "episode_length": summary["episode_length"],
        "total_reward": summary["total_reward"],
        "average_rate_e2e": summary["avg_rate_e2e_bps"],
        "average_rate_HR": summary["avg_rate_HR_bps"],
        "average_rate_RL": summary["avg_rate_RL_bps"],
        "average_snr_HR": summary["avg_snr_HR"],
        "average_snr_RL": summary["avg_snr_RL"],
        "outage_count": summary["outage_count"],
        "constraint_violation_count": summary["constraint_violation_count"],
        "trajectory_length": summary["trajectory_length"],
        "final_q_R_x": final_q_r[0],
        "final_q_R_y": final_q_r[1],
        "final_q_R_z": final_q_r[2],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_policy(policy: Any, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = UAVRelayCommEnv(config=policy.config)
    observation, info = env.reset(seed=seed)
    metrics = EpisodeMetrics()
    step_rows: list[dict[str, Any]] = []

    while True:
        action = policy.select_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        metrics.record(info, reward)
        step_rows.append(_step_row(policy.name, info, reward))
        if terminated or truncated:
            break

    return _summary_row(policy.name, metrics), step_rows


def run_all_baselines(seed: int = 7) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = load_config(ROOT / "configs" / "comm_env_default.yaml")
    summary_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for policy in make_default_policies(config, seed=seed):
        summary, step_rows = run_policy(policy, seed=seed)
        summary_rows.append(summary)
        all_step_rows.extend(step_rows)
        _write_csv(RESULTS_DIR / f"trajectory_{policy.name}.csv", step_rows, STEP_FIELDS)
        print(
            "{policy}: episode_length={episode_length} "
            "average_rate_e2e={rate:.6f}Mbps constraints={constraints}".format(
                policy=policy.name,
                episode_length=summary["episode_length"],
                rate=summary["average_rate_e2e"] / 1e6,
                constraints=summary["constraint_violation_count"],
            )
        )

    _write_csv(SUMMARY_PATH, summary_rows, SUMMARY_FIELDS)
    _write_csv(STEPS_PATH, all_step_rows, STEP_FIELDS)
    print(f"saved summary: {SUMMARY_PATH}")
    print(f"saved step results: {STEPS_PATH}")
    return summary_rows, all_step_rows


if __name__ == "__main__":
    run_all_baselines()
