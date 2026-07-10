from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_relay_env import EpisodeMetrics, UAVRelayCommEnv, load_config
from uav_relay_env.drl import ObservationNormalizer, TD3Agent

RESULTS_DIR = ROOT / "results" / "phase2"
CONFIG_PATH = ROOT / "configs" / "comm_env_default.yaml"
BASELINE_PATH = ROOT / "results" / "phase1" / "baseline_results.csv"
STEP_PATH = RESULTS_DIR / "td3_eval_step_results.csv"
SUMMARY_PATH = RESULTS_DIR / "td3_eval_summary.csv"

STEP_FIELDS = [
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
    "eval_reward",
    "eval_average_rate_e2e",
    "eval_average_rate_HR",
    "eval_average_rate_RL",
    "eval_constraint_violation_count",
    "best_baseline_policy",
    "best_baseline_average_rate_e2e",
    "td3_minus_best_baseline_rate_e2e",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _step_row(info: dict[str, Any], reward: float) -> dict[str, Any]:
    q_r = info["q_R"]
    return {
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


def _best_baseline() -> tuple[str, float]:
    if not BASELINE_PATH.exists():
        return "", 0.0
    with BASELINE_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return "", 0.0
    best = max(rows, key=lambda row: float(row["average_rate_e2e"]))
    return best["policy"], float(best["average_rate_e2e"])


def evaluate_td3() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG_PATH)
    env = UAVRelayCommEnv(config=config)
    agent = TD3Agent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        max_action=config.mobility.max_speed_mps,
        normalizer=ObservationNormalizer(env.observation_dim, enabled=True),
    )
    agent.load(RESULTS_DIR, prefer_best=True)

    observation, info = env.reset(seed=101)
    metrics = EpisodeMetrics()
    step_rows: list[dict[str, Any]] = []
    while True:
        action = agent.select_action(observation, noise=False)
        observation, reward, terminated, truncated, info = env.step(action)
        metrics.record(info, reward)
        step_rows.append(_step_row(info, reward))
        if terminated or truncated:
            break

    summary = metrics.summary()
    baseline_policy, baseline_rate = _best_baseline()
    summary_row = {
        "policy": "td3",
        "episode_length": summary["episode_length"],
        "eval_reward": summary["total_reward"],
        "eval_average_rate_e2e": summary["avg_rate_e2e_bps"],
        "eval_average_rate_HR": summary["avg_rate_HR_bps"],
        "eval_average_rate_RL": summary["avg_rate_RL_bps"],
        "eval_constraint_violation_count": summary["constraint_violation_count"],
        "best_baseline_policy": baseline_policy,
        "best_baseline_average_rate_e2e": baseline_rate,
        "td3_minus_best_baseline_rate_e2e": summary["avg_rate_e2e_bps"] - baseline_rate,
    }
    _write_csv(STEP_PATH, step_rows, STEP_FIELDS)
    _write_csv(SUMMARY_PATH, [summary_row], SUMMARY_FIELDS)
    print(
        "td3 eval: reward={reward:.6f} average_rate_e2e={rate:.6f}Mbps "
        "best_baseline={baseline}({baseline_rate:.6f}Mbps)".format(
            reward=summary_row["eval_reward"],
            rate=summary_row["eval_average_rate_e2e"] / 1e6,
            baseline=baseline_policy or "n/a",
            baseline_rate=baseline_rate / 1e6,
        )
    )
    print(f"saved eval steps: {STEP_PATH}")
    print(f"saved eval summary: {SUMMARY_PATH}")
    return summary_row


if __name__ == "__main__":
    evaluate_td3()
