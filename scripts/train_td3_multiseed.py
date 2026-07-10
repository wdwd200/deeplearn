from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_td3 import TD3_CONFIG_PATH, train_td3

RESULTS_DIR = ROOT / "results" / "phase3"
SUMMARY_PATH = RESULTS_DIR / "multiseed_summary.csv"
SUMMARY_FIELDS = [
    "seed",
    "best_eval_average_rate_e2e",
    "final_eval_average_rate_e2e",
    "best_eval_reward",
    "final_eval_reward",
    "best_episode",
    "final_episode",
    "constraint_violation_count",
    "outage_count",
    "model_dir",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_for_seed(seed: int, eval_rows: list[dict[str, Any]], model_dir: Path) -> dict[str, Any]:
    if not eval_rows:
        raise ValueError(f"seed {seed} did not produce eval rows")
    best = max(eval_rows, key=lambda row: float(row["eval_average_rate_e2e"]))
    final = eval_rows[-1]
    return {
        "seed": seed,
        "best_eval_average_rate_e2e": best["eval_average_rate_e2e"],
        "final_eval_average_rate_e2e": final["eval_average_rate_e2e"],
        "best_eval_reward": best["eval_reward"],
        "final_eval_reward": final["eval_reward"],
        "best_episode": best["episode"],
        "final_episode": final["episode"],
        "constraint_violation_count": final["eval_constraint_violation_count"],
        "outage_count": final["eval_outage_count"],
        "model_dir": str(model_dir),
    }


def train_td3_multiseed(seeds: list[int] | None = None) -> list[dict[str, Any]]:
    seeds = [0, 1, 2] if seeds is None else seeds
    summary_rows: list[dict[str, Any]] = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        output_dir = RESULTS_DIR / f"seed_{seed}"
        overrides = {"seed": seed, "output": {"root_dir": str(output_dir.relative_to(ROOT))}}
        _, eval_rows, model_dir = train_td3(TD3_CONFIG_PATH, overrides=overrides, output_dir=output_dir)
        summary_rows.append(_summary_for_seed(seed, eval_rows, model_dir))
    _write_csv(SUMMARY_PATH, summary_rows, SUMMARY_FIELDS)
    print(f"saved multiseed summary: {SUMMARY_PATH}")
    return summary_rows


if __name__ == "__main__":
    train_td3_multiseed()
