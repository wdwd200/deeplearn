from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

from phase5_common import (  # noqa: E402
    ABLATION_GROUPS,
    ABLATION_SUMMARY_PATH,
    ALGORITHM_EPISODE_PATH,
    ALGORITHM_SUMMARY_PATH,
    BASELINE_METHODS,
    COMM_CONFIG_PATH,
    DRL_METHODS,
    PHASE4_ROOT,
    PHASE5_FIGURES,
    PHASE5_SOURCE_DATA,
    ensure_phase5_dirs,
    load_yaml,
    official_metadata_values,
    read_csv_rows,
    run_episode_trace,
    to_float,
    write_csv_rows,
)

DPI = 600
METHOD_ORDER = (*DRL_METHODS, *BASELINE_METHODS)
COLORS = {
    "TD3": "#1b9e77",
    "DDPG": "#d95f02",
    "SAC": "#7570b3",
    "RandomPolicy": "#666666",
    "StaticRelayPolicy": "#a6761d",
    "MidpointPolicy": "#e7298a",
    "HorizontalMidpointPolicy": "#66a61e",
    "GreedyRatePolicy": "#e6ab02",
    "BalancedLinkPolicy": "#1f78b4",
}


def _with_trace_metadata(rows: list[dict[str, Any]], source_file: str) -> list[dict[str, Any]]:
    metadata = official_metadata_values()
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["rate_bps"] = row["rate_e2e"]
        enriched["rate_mbps"] = row["rate_e2e"] / 1_000_000.0
        enriched["source_file"] = source_file
        enriched["source_stage"] = "phase5"
        enriched["git_commit"] = metadata.get("git_commit", "")
        enriched["source_code_hash"] = metadata.get("source_code_hash", "")
        output.append(enriched)
    return output


def _save(fig: plt.Figure, name: str) -> list[Path]:
    PHASE5_FIGURES.mkdir(parents=True, exist_ok=True)
    png_path = PHASE5_FIGURES / f"{name}.png"
    pdf_path = PHASE5_FIGURES / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _summary_by_method() -> dict[str, dict[str, str]]:
    return {row["algorithm"]: row for row in read_csv_rows(ALGORITHM_SUMMARY_PATH)}


def _ablation_by_group() -> dict[str, dict[str, str]]:
    return {row["ablation"]: row for row in read_csv_rows(ABLATION_SUMMARY_PATH)}


def _figure_1_system_model() -> list[Path]:
    config = load_yaml(COMM_CONFIG_PATH)
    q_h = config["uavs"]["q_H_m"]
    q_r = config["uavs"]["q_R_initial_m"]
    q_l = config["uavs"]["q_L_m"]
    bounds = config["area"]["bounds_m"]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*q_h, marker="^", s=70, color="#377eb8", label="High UAV H")
    ax.scatter(*q_r, marker="o", s=70, color="#4daf4a", label="Relay UAV R")
    ax.scatter(*q_l, marker="s", s=70, color="#e41a1c", label="Low UAV L")
    ax.plot([q_h[0], q_r[0]], [q_h[1], q_r[1]], [q_h[2], q_r[2]], color="#377eb8", label="H-R link")
    ax.plot([q_r[0], q_l[0]], [q_r[1], q_l[1]], [q_r[2], q_l[2]], color="#e41a1c", label="R-L link")
    _draw_bounds(ax, bounds)
    ax.text(*q_h, " H")
    ax.text(*q_r, " R")
    ax.text(*q_l, " L")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("System model and relay flight region")
    ax.legend(loc="upper left", fontsize=7)
    return _save(fig, "figure_1_system_model")


def _draw_bounds(ax: Any, bounds: Mapping[str, list[float]]) -> None:
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    z0, z1 = bounds["z"]
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    segments = [[corners[i], corners[j]] for i, j in edges]
    ax.add_collection3d(Line3DCollection(segments, colors="#444444", linestyles="dashed", linewidths=0.8, alpha=0.5))


def _figure_2_pipeline() -> list[Path]:
    labels = [
        "3D positions",
        "distance\nand elevation",
        "antenna gain",
        "channel gain",
        "SNR",
        "hop rates",
        "end-to-end rate",
        "reward",
        "DRL action",
        "relay update",
    ]
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.axis("off")
    y = 0.5
    width = 0.085
    gap = 0.014
    for index, label in enumerate(labels):
        x = 0.02 + index * (width + gap)
        rect = plt.Rectangle((x, y - 0.14), width, 0.28, facecolor="#f0f0f0", edgecolor="#333333", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x + width / 2, y, label, ha="center", va="center", fontsize=8)
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x + width + gap * 0.75, y),
                xytext=(x + width, y),
                arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "#333333"},
            )
    ax.set_title("Modeling and training pipeline", fontsize=11)
    return _save(fig, "figure_2_modeling_pipeline")


def _figure_3_training_curves() -> list[Path]:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algorithm in ("td3", "ddpg", "sac"):
        rows_by_episode: dict[int, list[float]] = defaultdict(list)
        for seed_dir in sorted((PHASE4_ROOT / "algorithms" / algorithm).glob("seed_*")):
            for row in read_csv_rows(seed_dir / "training_log.csv"):
                rows_by_episode[int(row["episode"])].append(to_float(row["episode_reward"]))
        episodes = sorted(rows_by_episode)
        means = [sum(rows_by_episode[episode]) / len(rows_by_episode[episode]) for episode in episodes]
        stds = [
            (sum((value - means[index]) ** 2 for value in rows_by_episode[episode]) / len(rows_by_episode[episode])) ** 0.5
            for index, episode in enumerate(episodes)
        ]
        color = COLORS[algorithm.upper()]
        ax.plot(episodes, means, label=algorithm.upper(), color=color, linewidth=1.5)
        ax.fill_between(
            episodes,
            [mean_value - std for mean_value, std in zip(means, stds)],
            [mean_value + std for mean_value, std in zip(means, stds)],
            color=color,
            alpha=0.18,
        )
    ax.set_xlabel("episode")
    ax.set_ylabel("episode reward")
    ax.set_title("Training reward curves, mean +/- std across seeds (no smoothing)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save(fig, "figure_3_training_curves")


def _figure_4_algorithm_rate() -> list[Path]:
    summary = _summary_by_method()
    labels = list(METHOD_ORDER)
    means = [to_float(summary[label]["average_rate_e2e_mean"]) / 1_000_000.0 for label in labels]
    stds = [to_float(summary[label]["average_rate_e2e_std"]) / 1_000_000.0 for label in labels]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(labels, means, yerr=stds, capsize=3, color=[COLORS[label] for label in labels])
    ax.set_ylabel("average end-to-end rate (Mbps)")
    ax.set_title("Algorithm and baseline rate comparison")
    ax.text(0.01, 0.96, "Error bars: DRL seed x scenario std; baselines scenario std", transform=ax.transAxes, fontsize=8, va="top")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    return _save(fig, "figure_4_algorithm_rate_comparison")


def _figure_5_scenario_comparison() -> list[Path]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in read_csv_rows(ALGORITHM_EPISODE_PATH):
        grouped[row["scenario_id"]][row["algorithm"]].append(to_float(row["average_rate_e2e"]) / 1_000_000.0)
    scenarios = sorted(grouped)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for method in METHOD_ORDER:
        values = [sum(grouped[scenario][method]) / len(grouped[scenario][method]) for scenario in scenarios]
        ax.plot(scenarios, values, marker="o", linewidth=1.3, label=method, color=COLORS[method])
    ax.set_xlabel("evaluation scenario")
    ax.set_ylabel("average end-to-end rate (Mbps)")
    ax.set_title("Scenario-wise algorithm performance")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=3)
    return _save(fig, "figure_5_scenario_comparison")


def _figure_6_ablation_rate() -> list[Path]:
    summary = _ablation_by_group()
    labels = list(ABLATION_GROUPS)
    means = [to_float(summary[label]["average_rate_e2e_mean"]) / 1_000_000.0 for label in labels]
    stds = [to_float(summary[label]["average_rate_e2e_std"]) / 1_000_000.0 for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, means, yerr=stds, capsize=4, color="#4c78a8")
    ax.set_ylabel("average end-to-end rate (Mbps)")
    ax.set_title("TD3 ablation rate comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    return _save(fig, "figure_6_ablation_rate")


def _figure_7_constraints() -> list[Path]:
    summary = _summary_by_method()
    labels = list(METHOD_ORDER)
    values = [to_float(summary[label]["constraint_violation_count_mean"]) for label in labels]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(labels, values, color=[COLORS[label] for label in labels])
    ax.set_ylabel("mean constraint violation count")
    ax.set_title("Constraint violations")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    return _save(fig, "figure_7_constraint_violations")


def _figure_8_trajectories() -> list[Path]:
    methods = ("TD3", "DDPG", "SAC", "BalancedLinkPolicy")
    traces: list[dict[str, Any]] = []
    for method in methods:
        traces.extend(run_episode_trace(method, scenario_id="eval_0", seed=0))
    traces = _with_trace_metadata(traces, "results/phase4/algorithms and fixed Phase 4 eval_0 scenario")
    write_csv_rows(PHASE5_SOURCE_DATA / "representative_trajectories.csv", traces)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    for method in methods:
        rows = [row for row in traces if row["method"] == method]
        ax.plot(
            [row["q_R_x"] for row in rows],
            [row["q_R_y"] for row in rows],
            [row["q_R_z"] for row in rows],
            label=method,
            color=COLORS[method],
            linewidth=1.4,
        )
        ax.scatter(rows[0]["q_R_x"], rows[0]["q_R_y"], rows[0]["q_R_z"], color=COLORS[method], s=18)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Representative relay trajectories on eval_0")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(50, 500)
    ax.legend(fontsize=8)
    return _save(fig, "figure_8_representative_trajectories")


def _figure_9_rate_over_time() -> list[Path]:
    summary = _summary_by_method()
    best_drl = max(DRL_METHODS, key=lambda method: to_float(summary[method]["average_rate_e2e_mean"]))
    best_baseline = max(BASELINE_METHODS, key=lambda method: to_float(summary[method]["average_rate_e2e_mean"]))
    methods = (best_drl, best_baseline)
    traces: list[dict[str, Any]] = []
    for method in methods:
        traces.extend(run_episode_trace(method, scenario_id="eval_0", seed=0))
    traces = _with_trace_metadata(traces, "results/phase4/algorithms and fixed Phase 4 eval_0 scenario")
    write_csv_rows(PHASE5_SOURCE_DATA / "representative_rate_trace.csv", traces)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True)
    for ax, method in zip(axes, methods):
        rows = [row for row in traces if row["method"] == method]
        steps = [row["step"] for row in rows]
        ax.plot(steps, [row["rate_HR"] / 1_000_000.0 for row in rows], label="R_HR", color="#1b9e77")
        ax.plot(steps, [row["rate_RL"] / 1_000_000.0 for row in rows], label="R_RL", color="#d95f02")
        ax.plot(steps, [row["rate_e2e"] / 1_000_000.0 for row in rows], label="R_e2e", color="#7570b3", linewidth=1.6)
        ax.set_ylabel("rate (Mbps)")
        ax.set_title(f"{method} on eval_0")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("step")
    return _save(fig, "figure_9_rate_over_time")


def plot_paper_figures() -> list[Path]:
    ensure_phase5_dirs()
    outputs: list[Path] = []
    outputs.extend(_figure_1_system_model())
    outputs.extend(_figure_2_pipeline())
    outputs.extend(_figure_3_training_curves())
    outputs.extend(_figure_4_algorithm_rate())
    outputs.extend(_figure_5_scenario_comparison())
    outputs.extend(_figure_6_ablation_rate())
    outputs.extend(_figure_7_constraints())
    outputs.extend(_figure_8_trajectories())
    outputs.extend(_figure_9_rate_over_time())
    for path in outputs:
        print(f"saved figure: {path}")
    return outputs


if __name__ == "__main__":
    plot_paper_figures()
