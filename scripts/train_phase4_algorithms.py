from __future__ import annotations

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
    load_phase4_config,
    phase4_training_overrides,
    read_csv,
    relative_path,
    resolve_path,
    train_algorithm_seed,
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


def train_phase4_algorithms(config_path: str | Path = PHASE4_CONFIG_PATH, force: bool = False) -> list[dict[str, Any]]:
    config = load_phase4_config(config_path)
    output_root = resolve_path(config["output"]["root_dir"])
    seeds = [int(seed) for seed in config["experiment"]["training_seeds"]]
    rows: list[dict[str, Any]] = []

    for algorithm in ALGORITHMS:
        for seed in seeds:
            model_dir = output_root / "algorithms" / algorithm / f"seed_{seed}"
            overrides = phase4_training_overrides(config, seed, algorithm, model_dir)
            if not force and (model_dir / "eval_log.csv").exists() and (model_dir / "best_actor.pt").exists():
                print(f"reuse existing {algorithm} seed={seed}: {model_dir}")
                eval_rows = _load_existing_eval(model_dir)
            else:
                _, eval_rows, model_dir = train_algorithm_seed(algorithm, overrides, model_dir)
            rows.append(_summary_row(algorithm, seed, eval_rows, model_dir))

    summary_path = output_root / "algorithm_training_summary.csv"
    write_csv(summary_path, rows, SUMMARY_FIELDS)
    print(f"saved Phase 4 training summary: {summary_path}")
    return rows


if __name__ == "__main__":
    train_phase4_algorithms()
