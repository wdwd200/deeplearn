from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_td3 import train_td3
from uav_relay_env.drl.utils import deep_update, load_yaml

SWEEP_CONFIG_PATH = ROOT / "configs" / "td3_sweep.yaml"
RESULTS_DIR = ROOT / "results" / "phase3"
SWEEP_RESULTS_PATH = RESULTS_DIR / "sweep_results.csv"
BEST_CONFIG_PATH = RESULTS_DIR / "best_config.yaml"
FIELDS = [
    "run_id",
    "actor_lr",
    "critic_lr",
    "exploration_noise",
    "policy_noise",
    "best_eval_average_rate_e2e",
    "final_eval_average_rate_e2e",
    "best_episode",
    "output_dir",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_best_config(row: dict[str, Any]) -> None:
    text = "\n".join(
        [
            f"run_id: {row['run_id']}",
            "td3:",
            f"  actor_lr: {row['actor_lr']}",
            f"  critic_lr: {row['critic_lr']}",
            f"  exploration_noise: {row['exploration_noise']}",
            f"  policy_noise: {row['policy_noise']}",
            f"best_eval_average_rate_e2e: {row['best_eval_average_rate_e2e']}",
            f"output_dir: {row['output_dir']}",
            "",
        ]
    )
    BEST_CONFIG_PATH.write_text(text, encoding="utf-8")


def sweep_td3() -> list[dict[str, Any]]:
    config = load_yaml(SWEEP_CONFIG_PATH)
    sweep = config["sweep"]
    keys = ["actor_lr", "critic_lr", "exploration_noise", "policy_noise"]
    combinations = list(itertools.product(*(sweep[key] for key in keys)))[: int(config.get("max_runs", 6))]
    rows: list[dict[str, Any]] = []
    for run_index, values in enumerate(combinations):
        params = dict(zip(keys, values))
        output_dir = RESULTS_DIR / "sweep" / f"run_{run_index:02d}"
        overrides = {
            "seed": int(config.get("seed", 0)),
            "training": config.get("training", {}),
            "td3": params,
            "output": {"root_dir": str(output_dir.relative_to(ROOT))},
        }
        _, eval_rows, model_dir = train_td3(ROOT / config["base_config"], overrides=overrides, output_dir=output_dir)
        best = max(eval_rows, key=lambda row: float(row["eval_average_rate_e2e"]))
        final = eval_rows[-1]
        row = {
            "run_id": run_index,
            **params,
            "best_eval_average_rate_e2e": best["eval_average_rate_e2e"],
            "final_eval_average_rate_e2e": final["eval_average_rate_e2e"],
            "best_episode": best["episode"],
            "output_dir": str(model_dir),
        }
        rows.append(row)
    _write_csv(SWEEP_RESULTS_PATH, rows, FIELDS)
    best_row = max(rows, key=lambda row: float(row["best_eval_average_rate_e2e"]))
    _write_best_config(best_row)
    print(f"saved sweep results: {SWEEP_RESULTS_PATH}")
    print(f"saved best config: {BEST_CONFIG_PATH}")
    return rows


if __name__ == "__main__":
    sweep_td3()
