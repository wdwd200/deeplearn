from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase4_common import (
    EPISODE_RESULT_FIELDS,
    PHASE4_CONFIG_PATH,
    env_config_from_overrides,
    evaluate_saved_agent,
    load_phase4_config,
    phase4_eval_scenarios,
    phase4_training_metadata,
    phase4_training_overrides,
    resolve_path,
    summarize_episode_results,
    train_algorithm_seed,
    validate_phase4_reuse,
    write_csv,
)

ABLATION_SUMMARY_FIELDS = [
    "ablation",
    "average_rate_e2e_mean",
    "average_rate_e2e_std",
    "average_reward_mean",
    "constraint_violation_count_mean",
    "trajectory_length_mean",
    "difference_from_full_td3",
]


def zero_z_action(action: np.ndarray) -> np.ndarray:
    action[2] = 0.0
    return action


def action_transform_for_ablation(ablation_config: Mapping[str, Any]):
    return zero_z_action if bool(ablation_config.get("fixed_z_action", False)) else None


def ablation_env_config(ablation_config: Mapping[str, Any]):
    return env_config_from_overrides(ablation_config.get("env_overrides", {}) or {})


def _ablation_episode_row(ablation: str, row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["algorithm"] = ablation
    return row


def run_phase4_ablations(config_path: str | Path = PHASE4_CONFIG_PATH, force: bool = False) -> tuple[list[dict], list[dict]]:
    config = load_phase4_config(config_path)
    output_root = resolve_path(config["output"]["root_dir"])
    ablation_root = output_root / "ablations"
    seeds = [int(seed) for seed in config["experiment"]["training_seeds"]]
    max_steps = int(config["experiment"]["max_steps"])
    eval_scenarios = phase4_eval_scenarios(config)
    if int(config["experiment"]["evaluation_episodes"]) != len(eval_scenarios):
        raise ValueError("experiment.evaluation_episodes must match the fixed eval scenario count")
    all_rows: list[dict[str, Any]] = []

    for ablation_name, ablation_config in config["ablations"].items():
        env_config = ablation_env_config(ablation_config)
        action_transform = action_transform_for_ablation(ablation_config)
        for seed in seeds:
            model_dir = ablation_root / ablation_name / f"seed_{seed}"
            overrides = phase4_training_overrides(config, seed, "td3", model_dir)
            expected_metadata = phase4_training_metadata("td3", overrides, env_config, eval_scenarios, action_transform)
            if model_dir.exists() and not force:
                validate_phase4_reuse(model_dir, expected_metadata)
                print(
                    "reuse existing ablation {ablation} seed={seed}: {path} config_hash={config_hash}".format(
                        ablation=ablation_name,
                        seed=seed,
                        path=model_dir,
                        config_hash=expected_metadata["config_hash"],
                    )
                )
            else:
                train_algorithm_seed(
                    "td3",
                    overrides,
                    model_dir,
                    env_config=env_config,
                    eval_scenarios=eval_scenarios,
                    action_transform=action_transform,
                )
            for eval_episode, scenario in enumerate(eval_scenarios):
                row = evaluate_saved_agent(
                    "td3",
                    model_dir,
                    training_seed=seed,
                    evaluation_episode=eval_episode,
                    max_steps=max_steps,
                    env_config=env_config,
                    scenario=scenario,
                    action_transform=action_transform,
                )
                all_rows.append(_ablation_episode_row(ablation_name, row))

    summary_rows = summarize_episode_results(all_rows)
    full_row = next(row for row in summary_rows if row["algorithm"] == "A0_full_td3")
    full_rate = float(full_row["average_rate_e2e_mean"])
    ablation_summary = [
        {
            "ablation": row["algorithm"],
            "average_rate_e2e_mean": row["average_rate_e2e_mean"],
            "average_rate_e2e_std": row["average_rate_e2e_std"],
            "average_reward_mean": row["average_reward_mean"],
            "constraint_violation_count_mean": row["constraint_violation_count_mean"],
            "trajectory_length_mean": row["trajectory_length_mean"],
            "difference_from_full_td3": float(row["average_rate_e2e_mean"]) - full_rate,
        }
        for row in summary_rows
    ]

    results_path = ablation_root / "ablation_results.csv"
    summary_path = ablation_root / "ablation_summary.csv"
    write_csv(results_path, all_rows, EPISODE_RESULT_FIELDS)
    write_csv(summary_path, ablation_summary, ABLATION_SUMMARY_FIELDS)
    print(f"saved Phase 4 ablation results: {results_path}")
    print(f"saved Phase 4 ablation summary: {summary_path}")
    return all_rows, ablation_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 TD3 ablations.")
    parser.add_argument("--config", default=str(PHASE4_CONFIG_PATH), help="Path to Phase 4 experiment YAML.")
    parser.add_argument("--force", action="store_true", help="Retrain and overwrite Phase 4 ablation outputs.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_phase4_ablations(args.config, force=args.force)
