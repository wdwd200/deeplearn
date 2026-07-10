from __future__ import annotations

import csv
import math
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

RESULTS_DIR = ROOT / "results" / "phase1"
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARY_PATH = RESULTS_DIR / "baseline_results.csv"
STEPS_PATH = RESULTS_DIR / "baseline_step_results.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing input file: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _group_steps(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    for policy_rows in grouped.values():
        policy_rows.sort(key=lambda item: int(item["step"]))
    return dict(grouped)


def _save_line_plot(
    grouped: dict[str, list[dict[str, str]]],
    filename: str,
    title: str,
    ylabel: str,
    series: list[tuple[str, str, float]],
) -> None:
    plt.figure(figsize=(10, 6))
    for policy, rows in grouped.items():
        steps = [int(row["step"]) for row in rows]
        for key, label, scale in series:
            values = [_float(row, key) * scale for row in rows]
            plt.plot(steps, values, label=f"{policy} {label}")
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()


def _save_summary_bar(rows: list[dict[str, str]], key: str, filename: str, title: str, ylabel: str, scale: float) -> None:
    policies = [row["policy"] for row in rows]
    values = [_float(row, key) * scale for row in rows]
    plt.figure(figsize=(10, 6))
    plt.bar(policies, values)
    plt.title(title)
    plt.xlabel("policy")
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()


def _save_trajectory_3d(grouped: dict[str, list[dict[str, str]]]) -> None:
    fig = plt.figure(figsize=(10, 7))
    axis = fig.add_subplot(111, projection="3d")
    for policy, rows in grouped.items():
        xs = [_float(row, "q_R_x") for row in rows]
        ys = [_float(row, "q_R_y") for row in rows]
        zs = [_float(row, "q_R_z") for row in rows]
        axis.plot(xs, ys, zs, label=policy)
    axis.set_title("Relay UAV 3D Trajectory")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "trajectory_3d.png", dpi=150)
    plt.close()


def plot_baseline_results() -> list[Path]:
    summary_rows = _read_csv(SUMMARY_PATH)
    step_rows = _read_csv(STEPS_PATH)
    grouped = _group_steps(step_rows)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    _save_line_plot(
        grouped,
        "rate_e2e_curve.png",
        "End-to-End Rate",
        "rate_e2e (Mbps)",
        [("rate_e2e", "", 1e-6)],
    )
    _save_line_plot(
        grouped,
        "rate_hr_rl_curve.png",
        "Per-Hop Rate",
        "rate (Mbps)",
        [("rate_HR", "H-R", 1e-6), ("rate_RL", "R-L", 1e-6)],
    )
    _save_line_plot(
        grouped,
        "snr_curve.png",
        "Per-Hop SNR",
        "SNR",
        [("snr_HR", "H-R", 1.0), ("snr_RL", "R-L", 1.0)],
    )
    _save_line_plot(
        grouped,
        "distance_curve.png",
        "Link Distance",
        "distance (m)",
        [("d_HR", "H-R", 1.0), ("d_RL", "R-L", 1.0)],
    )
    _save_line_plot(
        grouped,
        "elevation_angle_curve.png",
        "Elevation Angle",
        "angle (degree)",
        [("theta_HR", "H-R", 180.0 / math.pi), ("theta_RL", "R-L", 180.0 / math.pi)],
    )
    _save_trajectory_3d(grouped)
    _save_summary_bar(
        summary_rows,
        "average_rate_e2e",
        "baseline_average_rate_bar.png",
        "Average End-to-End Rate",
        "average rate_e2e (Mbps)",
        1e-6,
    )
    _save_summary_bar(
        summary_rows,
        "constraint_violation_count",
        "constraint_violation_bar.png",
        "Constraint Violations",
        "count",
        1.0,
    )

    figure_paths = sorted(FIGURES_DIR.glob("*.png"))
    for path in figure_paths:
        print(f"saved figure: {path}")
    return figure_paths


if __name__ == "__main__":
    plot_baseline_results()
