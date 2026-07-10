from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = ROOT / "results" / "phase4"
FIGURES_DIR = RESULTS_DIR / "figures"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _save_method_comparison() -> Path:
    rows = _read_csv(RESULTS_DIR / "phase4_method_comparison.csv")
    rows = [row for row in rows if row["method"] != "td3_minus_best_baseline"]
    labels = [row["method"] for row in rows]
    means = [float(row["average_rate_e2e_mean"]) / 1e6 for row in rows]
    stds = [float(row["average_rate_e2e_std"] or 0.0) / 1e6 for row in rows]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, means, yerr=stds, capsize=4)
    plt.title("Phase 4 TD3 vs Baseline Comparison")
    plt.xlabel("method")
    plt.ylabel("average rate_e2e (Mbps)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "phase4_td3_vs_baselines.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _save_ablation_delta() -> Path:
    rows = _read_csv(RESULTS_DIR / "phase4_ablation_vs_reference.csv")
    labels = [row["variant"] for row in rows]
    deltas = [float(row["rate_delta_vs_phase3_td3_mean"]) / 1e6 for row in rows]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, deltas)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.title("Phase 4 Ablation Rate Delta vs Phase 3 TD3 Mean")
    plt.xlabel("ablation variant")
    plt.ylabel("rate delta (Mbps)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "phase4_ablation_rate_delta.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _save_ablation_constraints() -> Path:
    rows = _read_csv(RESULTS_DIR / "phase4_ablation_summary.csv")
    labels = [row["variant"] for row in rows]
    constraints = [float(row["eval_constraint_violation_count"]) for row in rows]
    outages = [float(row["eval_outage_count"]) for row in rows]
    positions = list(range(len(rows)))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar([item - width / 2 for item in positions], constraints, width=width, label="constraint violations")
    plt.bar([item + width / 2 for item in positions], outages, width=width, label="outages")
    plt.title("Phase 4 Ablation Constraint and Outage Counts")
    plt.xlabel("ablation variant")
    plt.ylabel("count")
    plt.xticks(positions, labels, rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "phase4_ablation_constraints_outages.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_phase4_results() -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        _save_method_comparison(),
        _save_ablation_delta(),
        _save_ablation_constraints(),
    ]
    for path in paths:
        print(f"saved figure: {path}")
    return paths


if __name__ == "__main__":
    plot_phase4_results()
