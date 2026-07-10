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

RESULTS_DIR = ROOT / "results" / "phase2"
FIGURES_DIR = RESULTS_DIR / "figures"
BASELINE_PATH = ROOT / "results" / "phase1" / "baseline_results.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _save_training_reward_curve() -> None:
    rows = _read_csv(RESULTS_DIR / "training_log.csv")
    episodes = [int(row["episode"]) for row in rows]
    rewards = [float(row["episode_reward"]) for row in rows]
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, rewards, marker="o")
    plt.title("TD3 Training Reward")
    plt.xlabel("episode")
    plt.ylabel("episode reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_reward_curve.png", dpi=150)
    plt.close()


def _save_eval_rate_curve() -> None:
    rows = _read_csv(RESULTS_DIR / "eval_log.csv")
    episodes = [int(row["episode"]) for row in rows]
    rates = [float(row["eval_average_rate_e2e"]) / 1e6 for row in rows]
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, rates, marker="o")
    plt.title("TD3 Evaluation End-to-End Rate")
    plt.xlabel("episode")
    plt.ylabel("average rate_e2e (Mbps)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eval_rate_curve.png", dpi=150)
    plt.close()


def _save_td3_vs_baseline_bar() -> None:
    baseline_rows = _read_csv(BASELINE_PATH)
    td3_rows = _read_csv(RESULTS_DIR / "td3_eval_summary.csv")
    labels = [row["policy"] for row in baseline_rows] + ["td3"]
    values = [float(row["average_rate_e2e"]) / 1e6 for row in baseline_rows]
    values.append(float(td3_rows[0]["eval_average_rate_e2e"]) / 1e6)
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title("TD3 vs Phase 1 Baseline Rate")
    plt.xlabel("policy")
    plt.ylabel("average rate_e2e (Mbps)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "td3_vs_baseline_rate_bar.png", dpi=150)
    plt.close()


def plot_training_curves() -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_training_reward_curve()
    _save_eval_rate_curve()
    _save_td3_vs_baseline_bar()
    paths = sorted(FIGURES_DIR.glob("*.png"))
    for path in paths:
        print(f"saved figure: {path}")
    return paths


if __name__ == "__main__":
    plot_training_curves()
