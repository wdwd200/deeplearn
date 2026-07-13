from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase4_common import (
    ALGORITHMS,
    PHASE4_CONFIG_PATH,
    assert_clean_git_worktree,
    compute_source_code_hash,
    env_config_from_overrides,
    phase4_eval_scenarios,
    phase4_training_metadata,
    load_phase4_config,
    phase4_training_overrides,
    read_csv,
    relative_path,
    resolve_path,
    train_algorithm_seed,
    validate_phase4_reuse,
    write_csv,
)

SUMMARY_FIELDS = [
    "algorithm",
    "seed",
    "best_eval_average_rate_e2e",
    "final_eval_average_rate_e2e",
    "best_eval_reward",
    "final_eval_reward",
    "best_episode",
    "final_episode",
    "model_dir",
]


def _summary_row(algorithm: str, seed: int, eval_rows: list[dict[str, Any]], model_dir: Path) -> dict[str, Any]:
    best = max(eval_rows, key=lambda row: float(row["eval_average_rate_e2e"]))
    final = eval_rows[-1]
    return {
        "algorithm": algorithm.upper(),
        "seed": seed,
        "best_eval_average_rate_e2e": best["eval_average_rate_e2e"],
        "final_eval_average_rate_e2e": final["eval_average_rate_e2e"],
        "best_eval_reward": best["eval_reward"],
        "final_eval_reward": final["eval_reward"],
        "best_episode": best["episode"],
        "final_episode": final["episode"],
        "model_dir": relative_path(model_dir),
    }


def _load_existing_eval(model_dir: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in read_csv(model_dir / "eval_log.csv")]


def train_phase4_algorithms(
    config_path: str | Path = PHASE4_CONFIG_PATH,
    force: bool = False,
    allow_dirty: bool = False,
) -> list[dict[str, Any]]:
    git_dirty = assert_clean_git_worktree(allow_dirty=allow_dirty)
    source_code_hash = compute_source_code_hash()
    config = load_phase4_config(config_path)
    output_root = resolve_path(config["output"]["root_dir"])
    seeds = [int(seed) for seed in config["experiment"]["training_seeds"]]
    eval_scenarios = phase4_eval_scenarios(config)
    env_config = env_config_from_overrides()
    rows: list[dict[str, Any]] = []

    for algorithm in ALGORITHMS:
        for seed in seeds:
            model_dir = output_root / "algorithms" / algorithm / f"seed_{seed}"
            overrides = phase4_training_overrides(config, seed, algorithm, model_dir)
            expected_metadata = phase4_training_metadata(
                algorithm,
                overrides,
                env_config,
                eval_scenarios,
                git_dirty=git_dirty,
                source_code_hash=source_code_hash,
            )
            if model_dir.exists() and not force:
                validate_phase4_reuse(model_dir, expected_metadata)
                print(
                    "reuse existing {algorithm} seed={seed}: {path} config_hash={config_hash}".format(
                        algorithm=algorithm,
                        seed=seed,
                        path=model_dir,
                        config_hash=expected_metadata["config_hash"],
                    )
                )
                eval_rows = _load_existing_eval(model_dir)
            else:
                _, eval_rows, model_dir = train_algorithm_seed(
                    algorithm,
                    overrides,
                    model_dir,
                    env_config=env_config,
                    eval_scenarios=eval_scenarios,
                    git_dirty=git_dirty,
                    source_code_hash=source_code_hash,
                )
            rows.append(_summary_row(algorithm, seed, eval_rows, model_dir))

    summary_path = output_root / "algorithm_training_summary.csv"
    write_csv(summary_path, rows, SUMMARY_FIELDS)
    print(f"saved Phase 4 training summary: {summary_path}")
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 4 TD3/DDPG/SAC algorithms.")
    parser.add_argument("--config", default=str(PHASE4_CONFIG_PATH), help="Path to Phase 4 experiment YAML.")
    parser.add_argument("--force", action="store_true", help="Retrain and overwrite Phase 4 algorithm outputs.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow debug training from a dirty worktree; marks outputs unofficial.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train_phase4_algorithms(args.config, force=args.force, allow_dirty=args.allow_dirty)
