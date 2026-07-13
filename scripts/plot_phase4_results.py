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

from phase4_common import load_phase4_config, resolve_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _save_bar(
    labels: list[str],
    means: list[float],
    stds: list[float],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(labels, means, yerr=stds, capsize=4)
    plt.title(title)
    plt.xlabel("method")
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_algorithm_rate(summary_rows: list[dict[str, str]], figures_dir: Path) -> None:
    labels = [row["algorithm"] for row in summary_rows]
    means = [float(row["average_rate_e2e_mean"]) / 1e6 for row in summary_rows]
    stds = [float(row["average_rate_e2e_std"]) / 1e6 for row in summary_rows]
    _save_bar(labels, means, stds, "Phase 4 Algorithm End-to-End Rate", "average rate_e2e (Mbps)", figures_dir / "algorithm_rate_mean_std.png")


def _plot_algorithm_reward(summary_rows: list[dict[str, str]], figures_dir: Path) -> None:
    labels = [row["algorithm"] for row in summary_rows]
    means = [float(row["average_reward_mean"]) for row in summary_rows]
    stds = [float(row["average_reward_std"]) for row in summary_rows]
    _save_bar(labels, means, stds, "Phase 4 Algorithm Reward", "average reward", figures_dir / "algorithm_reward_mean_std.png")


def _plot_training_curves(output_root: Path, figures_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for algorithm in ("td3", "ddpg", "sac"):
        rows = _read_csv(output_root / "algorithms" / algorithm / "seed_0" / "training_log.csv")
        plt.plot(
            [int(row["episode"]) for row in rows],
            [float(row["episode_reward"]) for row in rows],
            label=algorithm.upper(),
        )
    plt.title("TD3/DDPG/SAC Training Reward Curve")
    plt.xlabel("episode")
    plt.ylabel("episode reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "td3_ddpg_sac_training_curve.png", dpi=150)
    plt.close()


def _plot_eval_curves(output_root: Path, figures_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for algorithm in ("td3", "ddpg", "sac"):
        rows = _read_csv(output_root / "algorithms" / algorithm / "seed_0" / "eval_log.csv")
        plt.plot(
            [int(row["episode"]) for row in rows],
            [float(row["eval_average_rate_e2e"]) / 1e6 for row in rows],
            marker="o",
            label=algorithm.upper(),
        )
    plt.title("TD3/DDPG/SAC Evaluation Rate Curve")
    plt.xlabel("episode")
    plt.ylabel("average rate_e2e (Mbps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "td3_ddpg_sac_eval_curve.png", dpi=150)
    plt.close()


def _plot_td3_vs_baselines(summary_rows: list[dict[str, str]], figures_dir: Path) -> None:
    rows = [row for row in summary_rows if row["algorithm"] in {"TD3", "RandomPolicy", "StaticRelayPolicy", "MidpointPolicy", "HorizontalMidpointPolicy", "GreedyRatePolicy", "BalancedLinkPolicy"}]
    labels = [row["algorithm"] for row in rows]
    means = [float(row["average_rate_e2e_mean"]) / 1e6 for row in rows]
    stds = [float(row["average_rate_e2e_std"]) / 1e6 for row in rows]
    _save_bar(labels, means, stds, "TD3 vs Phase 1 Baselines", "average rate_e2e (Mbps)", figures_dir / "td3_vs_baselines_rate_bar.png")


def _plot_ablation_rate(ablation_rows: list[dict[str, str]], figures_dir: Path) -> None:
    labels = [row["ablation"] for row in ablation_rows]
    means = [float(row["average_rate_e2e_mean"]) / 1e6 for row in ablation_rows]
    stds = [float(row["average_rate_e2e_std"]) / 1e6 for row in ablation_rows]
    _save_bar(labels, means, stds, "Phase 4 TD3 Ablation Rate", "average rate_e2e (Mbps)", figures_dir / "ablation_rate_mean_std.png")


def _plot_ablation_metric(ablation_rows: list[dict[str, str]], metric: str, title: str, ylabel: str, filename: str, figures_dir: Path) -> None:
    labels = [row["ablation"] for row in ablation_rows]
    values = [float(row[metric]) for row in ablation_rows]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("ablation")
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=150)
    plt.close()


def plot_phase4_results(config_path: str | Path = "configs/phase4_experiments.yaml") -> list[Path]:
    config = load_phase4_config(config_path)
    output_root = resolve_path(config["output"]["root_dir"])
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = _read_csv(output_root / "algorithm_summary.csv")
    ablation_rows = _read_csv(output_root / "ablations" / "ablation_summary.csv")

    _plot_algorithm_rate(summary_rows, figures_dir)
    _plot_algorithm_reward(summary_rows, figures_dir)
    _plot_training_curves(output_root, figures_dir)
    _plot_eval_curves(output_root, figures_dir)
    _plot_td3_vs_baselines(summary_rows, figures_dir)
    _plot_ablation_rate(ablation_rows, figures_dir)
    _plot_ablation_metric(ablation_rows, "constraint_violation_count_mean", "Phase 4 Ablation Constraint Count", "mean count", "ablation_constraint_count.png", figures_dir)
    _plot_ablation_metric(ablation_rows, "trajectory_length_mean", "Phase 4 Ablation Trajectory Length", "mean steps", "ablation_trajectory_length.png", figures_dir)

    paths = sorted(figures_dir.glob("*.png"))
    for path in paths:
        print(f"saved figure: {path}")
    return paths


if __name__ == "__main__":
    plot_phase4_results()
