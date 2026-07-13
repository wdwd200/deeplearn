from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase5_common import (  # noqa: E402
    ABLATION_GROUPS,
    ABLATION_SUMMARY_PATH,
    ALGORITHM_EPISODE_PATH,
    ALGORITHM_SUMMARY_PATH,
    BASELINE_METHODS,
    COMM_CONFIG_PATH,
    DRL_METHODS,
    PHASE4_CONFIG_PATH,
    PHASE5_TABLES,
    assert_clean_phase5_generation,
    ensure_phase5_dirs,
    load_yaml,
    read_csv_rows,
    to_float,
    write_csv_rows,
    write_markdown_table,
    write_source_data,
)


def _json_compact(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _format_range(values: Mapping[str, Any]) -> str:
    return f"x={values['x']}, y={values['y']}, z={values['z']}"


def build_system_parameter_rows() -> list[dict[str, Any]]:
    config = load_yaml(COMM_CONFIG_PATH)
    simulation = config["simulation"]
    area = config["area"]
    uavs = config["uavs"]
    communication = config["communication"]
    antenna = config["antenna"]
    reward = config["reward"]
    environment = config["environment"]
    max_steps = int(simulation["max_steps"])
    delta_t = float(simulation["delta_t_s"])
    return [
        {"parameter": "task_duration", "description": "任务周期", "value": max_steps * delta_t, "unit": "s", "source": "configs/comm_env_default.yaml"},
        {"parameter": "slot_duration", "description": "时隙长度", "value": delta_t, "unit": "s", "source": "configs/comm_env_default.yaml"},
        {"parameter": "episode_steps", "description": "每个 episode 步数", "value": max_steps, "unit": "step", "source": "configs/comm_env_default.yaml"},
        {"parameter": "flight_area", "description": "飞行区域", "value": _format_range(area["bounds_m"]), "unit": "m", "source": "configs/comm_env_default.yaml"},
        {"parameter": "max_speed", "description": "最大速度", "value": environment["max_speed_mps"], "unit": "m/s", "source": "configs/comm_env_default.yaml"},
        {"parameter": "q_H_height", "description": "H 初始高度", "value": uavs["q_H_m"][2], "unit": "m", "source": "configs/comm_env_default.yaml"},
        {"parameter": "q_R_height", "description": "R 初始高度", "value": uavs["q_R_initial_m"][2], "unit": "m", "source": "configs/comm_env_default.yaml"},
        {"parameter": "q_L_height", "description": "L 初始高度", "value": uavs["q_L_m"][2], "unit": "m", "source": "configs/comm_env_default.yaml"},
        {"parameter": "power_HR", "description": "H-R 发射功率", "value": communication["power_HR_w"], "unit": "W", "source": "configs/comm_env_default.yaml"},
        {"parameter": "power_RL", "description": "R-L 发射功率", "value": communication["power_RL_w"], "unit": "W", "source": "configs/comm_env_default.yaml"},
        {"parameter": "bandwidth", "description": "带宽", "value": communication["bandwidth_hz"], "unit": "Hz", "source": "configs/comm_env_default.yaml"},
        {"parameter": "noise_power", "description": "噪声功率", "value": communication["noise_power_w"], "unit": "W", "source": "configs/comm_env_default.yaml"},
        {"parameter": "beta0", "description": "参考信道增益", "value": communication["beta0"], "unit": "-", "source": "configs/comm_env_default.yaml"},
        {"parameter": "path_loss_exponent", "description": "路径损耗指数", "value": communication["path_loss_exponent"], "unit": "-", "source": "configs/comm_env_default.yaml"},
        {"parameter": "antenna_model", "description": "天线模型", "value": _json_compact(antenna), "unit": "-", "source": "configs/comm_env_default.yaml"},
        {"parameter": "r_min", "description": "最低速率阈值", "value": communication["r_min_bps"], "unit": "bps", "source": "configs/comm_env_default.yaml"},
        {"parameter": "reward_weights", "description": "奖励函数权重", "value": _json_compact(reward), "unit": "-", "source": "configs/comm_env_default.yaml"},
    ]


def build_training_parameter_rows() -> list[dict[str, Any]]:
    config = load_yaml(PHASE4_CONFIG_PATH)
    experiment = config["experiment"]
    training = config["training"]
    rows: list[dict[str, Any]] = []
    for algorithm in ("td3", "ddpg", "sac"):
        algo = dict(config["algorithms"][algorithm])
        rows.append(
            {
                "algorithm": algorithm.upper(),
                "episodes": experiment["episodes"],
                "training_seeds": _json_compact(experiment["training_seeds"]),
                "batch_size": training["batch_size"],
                "replay_buffer_size": training["replay_size"],
                "hidden_sizes": _json_compact(training["hidden_sizes"]),
                "actor_lr": algo.get("actor_lr", ""),
                "critic_lr": algo.get("critic_lr", ""),
                "gamma": algo.get("gamma", ""),
                "tau": algo.get("tau", ""),
                "exploration_noise": algo.get("exploration_noise", ""),
                "algorithm_unique_parameters": _json_compact(
                    {
                        key: value
                        for key, value in algo.items()
                        if key
                        not in {
                            "actor_lr",
                            "critic_lr",
                            "gamma",
                            "tau",
                            "exploration_noise",
                        }
                    }
                ),
            }
        )
    return rows


def build_algorithm_comparison_rows() -> list[dict[str, Any]]:
    rows = []
    summary_rows = {row["algorithm"]: row for row in read_csv_rows(ALGORITHM_SUMMARY_PATH)}
    for method in (*DRL_METHODS, *BASELINE_METHODS):
        row = summary_rows[method]
        sample_unit = "training_seed x eval_scenario" if method in DRL_METHODS else "eval_scenario"
        rows.append(
            {
                "method": method,
                "mean_rate_bps": to_float(row["average_rate_e2e_mean"]),
                "mean_rate_mbps": to_float(row["average_rate_e2e_mean"]) / 1_000_000.0,
                "std_rate_bps": to_float(row["average_rate_e2e_std"]),
                "std_rate_mbps": to_float(row["average_rate_e2e_std"]) / 1_000_000.0,
                "mean_reward": to_float(row["average_reward_mean"]),
                "outage_count_mean": to_float(row["outage_count_mean"]),
                "constraint_violation_count_mean": to_float(row["constraint_violation_count_mean"]),
                "trajectory_length_mean": to_float(row["trajectory_length_mean"]),
                "std_source": sample_unit,
            }
        )
    return rows


def build_ablation_rows() -> list[dict[str, Any]]:
    summary_rows = {row["ablation"]: row for row in read_csv_rows(ABLATION_SUMMARY_PATH)}
    rows = []
    for group in ABLATION_GROUPS:
        row = summary_rows[group]
        rows.append(
            {
                "ablation": group,
                "mean_rate_bps": to_float(row["average_rate_e2e_mean"]),
                "mean_rate_mbps": to_float(row["average_rate_e2e_mean"]) / 1_000_000.0,
                "std_rate_bps": to_float(row["average_rate_e2e_std"]),
                "std_rate_mbps": to_float(row["average_rate_e2e_std"]) / 1_000_000.0,
                "delta_from_A0_bps": to_float(row["difference_from_full_td3"]),
                "delta_from_A0_mbps": to_float(row["difference_from_full_td3"]) / 1_000_000.0,
                "mean_reward": to_float(row["average_reward_mean"]),
                "constraint_violation_count_mean": to_float(row["constraint_violation_count_mean"]),
                "trajectory_length_mean": to_float(row["trajectory_length_mean"]),
            }
        )
    return rows


def build_scenario_rows() -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in read_csv_rows(ALGORITHM_EPISODE_PATH):
        grouped.setdefault((row["scenario_id"], row["algorithm"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for scenario_id, method in sorted(grouped):
        method_rows = grouped[(scenario_id, method)]
        rates = [to_float(row["average_rate_e2e"]) for row in method_rows]
        rows.append(
            {
                "scenario_id": scenario_id,
                "method": method,
                "mean_rate_bps": sum(rates) / len(rates),
                "mean_rate_mbps": sum(rates) / len(rates) / 1_000_000.0,
                "sample_count": len(rates),
                "sample_unit": "training_seed" if method in DRL_METHODS else "policy_run",
            }
        )
    return rows


def _write_table(name: str, rows: list[dict[str, Any]]) -> list[Path]:
    csv_path = PHASE5_TABLES / f"{name}.csv"
    md_path = PHASE5_TABLES / f"{name}.md"
    write_csv_rows(csv_path, rows)
    write_markdown_table(md_path, rows)
    return [csv_path, md_path]


def build_paper_tables() -> list[Path]:
    ensure_phase5_dirs()
    write_source_data()
    outputs: list[Path] = []
    outputs.extend(_write_table("table_1_system_parameters", build_system_parameter_rows()))
    outputs.extend(_write_table("table_2_training_parameters", build_training_parameter_rows()))
    outputs.extend(_write_table("table_3_algorithm_comparison", build_algorithm_comparison_rows()))
    outputs.extend(_write_table("table_4_ablation_results", build_ablation_rows()))
    outputs.extend(_write_table("table_5_scenario_results", build_scenario_rows()))
    for path in outputs:
        print(f"saved table: {path}")
    return outputs


if __name__ == "__main__":
    assert_clean_phase5_generation()
    build_paper_tables()
