from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_relay_env import EpisodeMetrics, UAVRelayCommEnv, load_config
from uav_relay_env.drl import ObservationNormalizer, TD3Agent
from uav_relay_env.drl.utils import load_yaml

RESULTS_DIR = ROOT / "results" / "phase3"
COMM_CONFIG_PATH = ROOT / "configs" / "comm_env_default.yaml"
BASELINE_PATH = ROOT / "results" / "phase1" / "baseline_results.csv"
SUMMARY_PATH = RESULTS_DIR / "multiseed_eval_summary.csv"
MEAN_STD_PATH = RESULTS_DIR / "td3_multiseed_mean_std.csv"
BASELINE_COMPARE_PATH = RESULTS_DIR / "td3_vs_baseline_summary.csv"

SUMMARY_FIELDS = [
    "seed",
    "eval_average_rate_e2e",
    "eval_average_rate_HR",
    "eval_average_rate_RL",
    "eval_average_snr_HR",
    "eval_average_snr_RL",
    "eval_outage_count",
    "eval_constraint_violation_count",
    "eval_trajectory_length",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "rate_HR": info["rate_HR"],
        "rate_RL": info["rate_RL"],
        "rate_e2e": info["rate_e2e"],
        "snr_HR": info["snr_HR"],
        "snr_RL": info["snr_RL"],
        "reward": reward,
        "constraint_violation": int(bool(info["constraint_violation"])),
    }


def _agent_for_seed(model_dir: Path) -> TD3Agent:
    config = load_config(COMM_CONFIG_PATH)
    env = UAVRelayCommEnv(config=config)
    params_path = model_dir / "training_params.json"
    params = load_yaml(params_path) if params_path.exists() else {}
    network = params.get("network", {})
    normalizer_cfg = params.get("normalizer", {})
    agent = TD3Agent(
        env.observation_dim,
        env.action_dim,
        config.mobility.max_speed_mps,
        hidden_sizes=network.get("hidden_sizes", [256, 256]),
        activation=network.get("activation", "relu"),
        normalizer=ObservationNormalizer(
            env.observation_dim,
            enabled=bool(normalizer_cfg.get("enabled", True)),
            clip=float(normalizer_cfg.get("clip_value", 5.0)),
        ),
    )
    agent.load(model_dir, prefer_best=True)
    return agent


def _evaluate_seed(seed: int, model_dir: Path) -> dict[str, Any]:
    config = load_config(COMM_CONFIG_PATH)
    env = UAVRelayCommEnv(config=config)
    agent = _agent_for_seed(model_dir)
    observation, info = env.reset(seed=20_000 + seed)
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
    _write_csv(model_dir / "td3_eval_step_results.csv", step_rows, list(step_rows[0].keys()))
    return {
        "seed": seed,
        "eval_average_rate_e2e": summary["avg_rate_e2e_bps"],
        "eval_average_rate_HR": summary["avg_rate_HR_bps"],
        "eval_average_rate_RL": summary["avg_rate_RL_bps"],
        "eval_average_snr_HR": summary["avg_snr_HR"],
        "eval_average_snr_RL": summary["avg_snr_RL"],
        "eval_outage_count": summary["outage_count"],
        "eval_constraint_violation_count": summary["constraint_violation_count"],
        "eval_trajectory_length": summary["trajectory_length"],
    }


def _mean_std(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [key for key in SUMMARY_FIELDS if key != "seed"]
    output: list[dict[str, Any]] = []
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        output.append(
            {
                "metric": metric,
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )
    return output


def _baseline_compare(td3_stats: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compare_rows: list[dict[str, Any]] = []
    if BASELINE_PATH.exists():
        with BASELINE_PATH.open("r", newline="", encoding="utf-8") as file:
            baseline_rows = list(csv.DictReader(file))
        for row in baseline_rows:
            compare_rows.append(
                {
                    "method": row["policy"],
                    "average_rate_e2e_mean": row["average_rate_e2e"],
                    "average_rate_e2e_std": 0.0,
                    "outage_count_mean": row["outage_count"],
                    "constraint_violation_count_mean": row["constraint_violation_count"],
                    "source": "phase1_baseline",
                }
            )
    rate_stat = next(item for item in td3_stats if item["metric"] == "eval_average_rate_e2e")
    outage_stat = next(item for item in td3_stats if item["metric"] == "eval_outage_count")
    constraint_stat = next(item for item in td3_stats if item["metric"] == "eval_constraint_violation_count")
    compare_rows.append(
        {
            "method": "td3_multiseed",
            "average_rate_e2e_mean": rate_stat["mean"],
            "average_rate_e2e_std": rate_stat["std"],
            "outage_count_mean": outage_stat["mean"],
            "constraint_violation_count_mean": constraint_stat["mean"],
            "source": "phase3_td3",
        }
    )
    if compare_rows:
        best_baseline = max(
            [row for row in compare_rows if row["source"] == "phase1_baseline"],
            key=lambda row: float(row["average_rate_e2e_mean"]),
            default=None,
        )
        if best_baseline is not None:
            compare_rows.append(
                {
                    "method": "td3_minus_best_baseline",
                    "average_rate_e2e_mean": rate_stat["mean"] - float(best_baseline["average_rate_e2e_mean"]),
                    "average_rate_e2e_std": rate_stat["std"],
                    "outage_count_mean": "",
                    "constraint_violation_count_mean": "",
                    "source": f"best_baseline={best_baseline['method']}",
                }
            )
    return compare_rows


def evaluate_td3_multiseed() -> list[dict[str, Any]]:
    seed_dirs = sorted(RESULTS_DIR.glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError("no seed directories found under results/phase3")
    rows: list[dict[str, Any]] = []
    for model_dir in seed_dirs:
        seed = int(model_dir.name.split("_", 1)[1])
        rows.append(_evaluate_seed(seed, model_dir))
    _write_csv(SUMMARY_PATH, rows, SUMMARY_FIELDS)
    stats = _mean_std(rows)
    _write_csv(MEAN_STD_PATH, stats, ["metric", "mean", "std", "min", "max"])
    compare_rows = _baseline_compare(stats, rows)
    _write_csv(
        BASELINE_COMPARE_PATH,
        compare_rows,
        ["method", "average_rate_e2e_mean", "average_rate_e2e_std", "outage_count_mean", "constraint_violation_count_mean", "source"],
    )
    print(f"saved multiseed eval summary: {SUMMARY_PATH}")
    print(f"saved TD3 mean/std: {MEAN_STD_PATH}")
    print(f"saved baseline comparison: {BASELINE_COMPARE_PATH}")
    return rows


if __name__ == "__main__":
    evaluate_td3_multiseed()
