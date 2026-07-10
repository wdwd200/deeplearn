from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = ROOT / "results" / "phase3"
FIGURES_DIR = RESULTS_DIR / "figures"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _seed_dirs() -> list[Path]:
    return sorted(RESULTS_DIR.glob("seed_*"))


def _save_training_reward_multiseed() -> None:
    plt.figure(figsize=(9, 5))
    for seed_dir in _seed_dirs():
        rows = _read_csv(seed_dir / "training_log.csv")
        plt.plot(
            [int(row["episode"]) for row in rows],
            [float(row["episode_reward"]) for row in rows],
            label=seed_dir.name,
            alpha=0.9,
        )
    plt.title("TD3 Training Reward Across Seeds")
    plt.xlabel("episode")
    plt.ylabel("episode reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "td3_training_reward_multiseed.png", dpi=150)
    plt.close()


def _save_eval_rate_curve() -> None:
    plt.figure(figsize=(9, 5))
    for seed_dir in _seed_dirs():
        rows = _read_csv(seed_dir / "eval_log.csv")
        plt.plot(
            [int(row["episode"]) for row in rows],
            [float(row["eval_average_rate_e2e"]) / 1e6 for row in rows],
            marker="o",
            label=seed_dir.name,
        )
    plt.title("TD3 Multiseed Eval End-to-End Rate")
    plt.xlabel("episode")
    plt.ylabel("average rate_e2e (Mbps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "td3_multiseed_eval_rate_curve.png", dpi=150)
    plt.close()


def _save_loss_curve() -> None:
    plt.figure(figsize=(9, 5))
    for seed_dir in _seed_dirs():
        rows = _read_csv(seed_dir / "training_log.csv")
        episodes = [int(row["episode"]) for row in rows]
        critic = [float(row["critic_loss"]) if row["critic_loss"] else 0.0 for row in rows]
        actor = [float(row["actor_loss"]) if row["actor_loss"] else 0.0 for row in rows]
        plt.plot(episodes, critic, label=f"{seed_dir.name} critic", alpha=0.8)
        plt.plot(episodes, actor, label=f"{seed_dir.name} actor", alpha=0.8, linestyle="--")
    plt.title("TD3 Actor/Critic Loss")
    plt.xlabel("episode")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "td3_actor_critic_loss_curve.png", dpi=150)
    plt.close()


def _save_vs_baseline_bar() -> None:
    rows = _read_csv(RESULTS_DIR / "td3_vs_baseline_summary.csv")
    rows = [row for row in rows if row["method"] != "td3_minus_best_baseline"]
    labels = [row["method"] for row in rows]
    means = [float(row["average_rate_e2e_mean"]) / 1e6 for row in rows]
    stds = [float(row["average_rate_e2e_std"] or 0.0) / 1e6 for row in rows]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, means, yerr=stds, capsize=4)
    plt.title("TD3 Mean/Std vs Phase 1 Baselines")
    plt.xlabel("method")
    plt.ylabel("average rate_e2e (Mbps)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "td3_vs_baseline_mean_std_bar.png", dpi=150)
    plt.close()


def _best_seed_dir() -> Path:
    rows = _read_csv(RESULTS_DIR / "multiseed_eval_summary.csv")
    best = max(rows, key=lambda row: float(row["eval_average_rate_e2e"]))
    return RESULTS_DIR / f"seed_{best['seed']}"


def _save_best_seed_trajectory() -> None:
    rows = _read_csv(_best_seed_dir() / "td3_eval_step_results.csv")
    fig = plt.figure(figsize=(9, 6))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(
        [float(row["q_R_x"]) for row in rows],
        [float(row["q_R_y"]) for row in rows],
        [float(row["q_R_z"]) for row in rows],
        label="TD3 best seed",
    )
    axis.set_title("TD3 Best Seed Relay Trajectory")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "td3_best_seed_trajectory_3d.png", dpi=150)
    plt.close()


def _save_best_seed_rate_snr() -> None:
    rows = _read_csv(_best_seed_dir() / "td3_eval_step_results.csv")
    steps = [int(row["step"]) for row in rows]
    plt.figure(figsize=(9, 5))
    plt.plot(steps, [float(row["rate_HR"]) / 1e6 for row in rows], label="H-R")
    plt.plot(steps, [float(row["rate_RL"]) / 1e6 for row in rows], label="R-L")
    plt.plot(steps, [float(row["rate_e2e"]) / 1e6 for row in rows], label="e2e")
    plt.title("TD3 Best Seed Rate Curve")
    plt.xlabel("step")
    plt.ylabel("rate (Mbps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "td3_best_seed_rate_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(steps, [float(row["snr_HR"]) for row in rows], label="H-R")
    plt.plot(steps, [float(row["snr_RL"]) for row in rows], label="R-L")
    plt.title("TD3 Best Seed SNR Curve")
    plt.xlabel("step")
    plt.ylabel("SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "td3_best_seed_snr_curve.png", dpi=150)
    plt.close()


def plot_phase3_results() -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_eval_rate_curve()
    _save_training_reward_multiseed()
    _save_loss_curve()
    _save_vs_baseline_bar()
    _save_best_seed_trajectory()
    _save_best_seed_rate_snr()
    paths = sorted(FIGURES_DIR.glob("*.png"))
    for path in paths:
        print(f"saved figure: {path}")
    return paths


if __name__ == "__main__":
    plot_phase3_results()
