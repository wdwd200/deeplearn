from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_relay_env import EnvConfig, EpisodeMetrics, UAVRelayCommEnv, load_config, make_default_policies
from uav_relay_env.config import ScenarioConfig
from uav_relay_env.drl import DDPGAgent, ObservationNormalizer, ReplayBuffer, SACAgent, TD3Agent
from uav_relay_env.drl.utils import deep_update, load_yaml, make_output_dir, sample_random_action, save_json, set_seed

COMM_CONFIG_PATH = ROOT / "configs" / "comm_env_default.yaml"
EVAL_SCENARIOS_PATH = ROOT / "configs" / "phase4_eval_scenarios.yaml"
PHASE4_CONFIG_PATH = ROOT / "configs" / "phase4_experiments.yaml"
ALGORITHMS = ("td3", "ddpg", "sac")

TRAINING_FIELDS = [
    "episode",
    "episode_reward",
    "average_rate_e2e",
    "average_rate_HR",
    "average_rate_RL",
    "average_snr_HR",
    "average_snr_RL",
    "outage_count",
    "constraint_violation_count",
    "trajectory_length",
    "actor_loss",
    "critic_loss",
    "q1_mean",
    "q2_mean",
    "replay_buffer_size",
    "exploration_noise",
]

EVAL_FIELDS = [
    "episode",
    "eval_reward",
    "eval_average_rate_e2e",
    "eval_average_rate_HR",
    "eval_average_rate_RL",
    "eval_average_snr_HR",
    "eval_average_snr_RL",
    "eval_outage_count",
    "eval_constraint_violation_count",
    "eval_trajectory_length",
]

EPISODE_RESULT_FIELDS = [
    "algorithm",
    "training_seed",
    "evaluation_episode",
    "scenario_id",
    "average_rate_e2e",
    "average_rate_HR",
    "average_rate_RL",
    "average_snr_HR",
    "average_snr_RL",
    "total_reward",
    "outage_count",
    "constraint_violation_count",
    "trajectory_length",
]

SUMMARY_FIELDS = [
    "algorithm",
    "average_rate_e2e_mean",
    "average_rate_e2e_std",
    "average_reward_mean",
    "average_reward_std",
    "outage_count_mean",
    "constraint_violation_count_mean",
    "trajectory_length_mean",
]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(candidate)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def to_yaml(data: Mapping[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in data.items():
        if isinstance(value, Mapping):
            lines.append(f"{prefix}{key}:")
            lines.append(to_yaml(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, Mapping):
                    lines.append(f"{prefix}  -")
                    lines.append(to_yaml(item, indent + 4))
                else:
                    lines.append(f"{prefix}  - {yaml_scalar(item)}")
        else:
            lines.append(f"{prefix}{key}: {yaml_scalar(value)}")
    return "\n".join(lines)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_sha256(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def eval_scenarios_path_from_config(config: Mapping[str, Any]) -> Path:
    experiment = config.get("experiment", {})
    raw_path = experiment.get("eval_scenarios_path", config.get("eval_scenarios_path", relative_path(EVAL_SCENARIOS_PATH)))
    return resolve_path(raw_path)


def load_eval_scenarios(path: str | Path = EVAL_SCENARIOS_PATH) -> list[dict[str, Any]]:
    data = load_eval_scenarios_yaml(path)
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("phase4 eval scenarios must define a non-empty scenarios list")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_signatures: set[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            raise ValueError(f"scenario {index} must be a mapping")
        scenario_id = str(scenario.get("id", f"eval_{index}"))
        q_h = _vector3(scenario.get("q_H"), "scenario.q_H")
        q_r = _vector3(scenario.get("q_R"), "scenario.q_R")
        q_l = _vector3(scenario.get("q_L"), "scenario.q_L")
        signature = (q_h, q_r, q_l)
        if scenario_id in seen_ids:
            raise ValueError(f"duplicate scenario id: {scenario_id}")
        if signature in seen_signatures:
            raise ValueError(f"duplicate scenario positions for {scenario_id}")
        seen_ids.add(scenario_id)
        seen_signatures.add(signature)
        normalized.append({"id": scenario_id, "q_H": list(q_h), "q_R": list(q_r), "q_L": list(q_l)})

    if len(normalized) != 5:
        raise ValueError("phase4 eval scenarios must contain exactly 5 scenarios")
    return normalized


def load_eval_scenarios_yaml(path: str | Path) -> dict[str, Any]:
    try:
        return load_yaml(path)
    except ValueError:
        return parse_eval_scenarios_yaml(Path(path).read_text(encoding="utf-8"))


def parse_eval_scenarios_yaml(text: str) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    list_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped == "scenarios:":
            continue
        if indent == 2 and stripped.startswith("-"):
            if current is not None:
                scenarios.append(current)
            current = {}
            list_key = None
            tail = stripped[1:].strip()
            if tail:
                key, separator, value = tail.partition(":")
                if not separator:
                    raise ValueError(f"invalid scenario entry: {stripped}")
                current[key.strip()] = _parse_eval_scalar(value.strip())
            continue
        if current is None:
            raise ValueError("scenario item must start with '-'")
        if indent == 4:
            key, separator, value = stripped.partition(":")
            if not separator:
                raise ValueError(f"invalid scenario mapping entry: {stripped}")
            key = key.strip()
            value = value.strip()
            if value:
                current[key] = _parse_eval_scalar(value)
                list_key = None
            else:
                current[key] = []
                list_key = key
            continue
        if indent == 6 and stripped.startswith("- ") and list_key is not None:
            current[list_key].append(float(stripped[2:].strip()))
            continue
        raise ValueError(f"unsupported eval scenario YAML line: {raw_line}")

    if current is not None:
        scenarios.append(current)
    return {"scenarios": scenarios}


def _parse_eval_scalar(value: str) -> Any:
    if value == "":
        return ""
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def env_config_with_scenario(base_config: EnvConfig, scenario: Mapping[str, Any]) -> EnvConfig:
    return replace(
        base_config,
        scenario=ScenarioConfig(
            q_H_m=tuple(float(value) for value in scenario["q_H"]),
            q_R_initial_m=tuple(float(value) for value in scenario["q_R"]),
            q_L_m=tuple(float(value) for value in scenario["q_L"]),
        ),
    )


def env_config_for_scenario(
    scenario: Mapping[str, Any],
    env_overrides: Mapping[str, Any] | None = None,
    base_config_path: str | Path = COMM_CONFIG_PATH,
) -> EnvConfig:
    base_mapping = load_yaml(base_config_path)
    merged = deep_update(
        base_mapping,
        {
            "uavs": {
                "q_H_m": list(scenario["q_H"]),
                "q_R_initial_m": list(scenario["q_R"]),
                "q_L_m": list(scenario["q_L"]),
            }
        },
    )
    if env_overrides:
        merged = deep_update(merged, env_overrides)
    return EnvConfig.from_mapping(merged)


def _vector3(values: Sequence[float] | Sequence[Any], name: str) -> tuple[float, float, float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError(f"{name} must be a sequence of three numbers")
    return float(values[0]), float(values[1]), float(values[2])


def clean_value(value: float | int | None) -> float | int | str:
    if value is None:
        return ""
    numeric = float(value)
    return value if math.isfinite(numeric) else ""


def mean_metric(metrics: list[dict[str, float | None]], key: str) -> float | str:
    values = [float(metric[key]) for metric in metrics if metric.get(key) is not None and math.isfinite(float(metric[key]))]
    return float(np.mean(values)) if values else ""


def load_phase4_config(path: str | Path = PHASE4_CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(path)
    validate_phase4_config(config)
    return config


def validate_phase4_config(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    training = config["training"]
    experiment.setdefault("eval_scenarios_path", relative_path(EVAL_SCENARIOS_PATH))
    seeds = [int(seed) for seed in experiment["training_seeds"]]
    if seeds != [0, 1, 2]:
        raise ValueError("Phase 4 default training_seeds must be [0, 1, 2]")
    if Path(str(experiment["eval_scenarios_path"])).is_absolute():
        raise ValueError("experiment.eval_scenarios_path must be relative")
    for key in ["episodes", "max_steps", "eval_interval", "evaluation_episodes"]:
        if int(experiment[key]) <= 0:
            raise ValueError(f"experiment.{key} must be positive")
    for key in ["batch_size", "replay_size", "start_steps"]:
        if int(training[key]) < 0:
            raise ValueError(f"training.{key} must be non-negative")
    if int(training["replay_size"]) < int(training["batch_size"]):
        raise ValueError("training.replay_size must be >= training.batch_size")
    if not list(training["hidden_sizes"]):
        raise ValueError("training.hidden_sizes must not be empty")
    for algorithm in ALGORITHMS:
        if algorithm not in config["algorithms"]:
            raise ValueError(f"missing algorithm config: {algorithm}")


def phase4_training_overrides(config: Mapping[str, Any], seed: int, algorithm: str, output_dir: Path) -> dict[str, Any]:
    experiment = config["experiment"]
    training = config["training"]
    algo = config["algorithms"][algorithm]
    overrides = {
        "seed": int(seed),
        "training_seed": int(seed),
        "algorithm": algorithm,
        "eval_scenarios_path": str(experiment.get("eval_scenarios_path", relative_path(EVAL_SCENARIOS_PATH))),
        "training": {
            "episodes": int(experiment["episodes"]),
            "max_steps": int(experiment["max_steps"]),
            "start_steps": int(training["start_steps"]),
            "batch_size": int(training["batch_size"]),
            "replay_size": int(training["replay_size"]),
            "eval_interval": int(experiment["eval_interval"]),
            "save_interval": int(training.get("save_interval", 50)),
            "train_every": int(training.get("train_every", 100)),
            "updates_per_train": int(training.get("updates_per_train", 1)),
            "reward_scale": float(training.get("reward_scale", 1.0)),
        },
        "network": {
            "hidden_sizes": [int(size) for size in training["hidden_sizes"]],
            "activation": str(training.get("activation", "relu")),
        },
        "normalizer": {
            "enabled": bool(training.get("normalizer_enabled", True)),
            "clip_value": float(training.get("normalizer_clip", 5.0)),
        },
        "output": {"root_dir": relative_path(output_dir)},
        algorithm: dict(algo),
    }
    return overrides


def env_config_from_overrides(env_overrides: Mapping[str, Any] | None = None) -> EnvConfig:
    if not env_overrides:
        return load_config(COMM_CONFIG_PATH)
    base_mapping = load_yaml(COMM_CONFIG_PATH)
    merged = deep_update(base_mapping, env_overrides)
    return EnvConfig.from_mapping(merged)


def build_agent(algorithm: str, config: Mapping[str, Any], env: UAVRelayCommEnv, device: str | None = None) -> Any:
    network = config["network"]
    normalizer_cfg = config["normalizer"]
    normalizer = ObservationNormalizer(
        env.observation_dim,
        enabled=bool(normalizer_cfg["enabled"]),
        clip=float(normalizer_cfg["clip_value"]),
    )
    common = {
        "obs_dim": env.observation_dim,
        "action_dim": env.action_dim,
        "max_action": env.config.mobility.max_speed_mps,
        "hidden_sizes": [int(size) for size in network["hidden_sizes"]],
        "activation": str(network.get("activation", "relu")),
        "reward_scale": float(config["training"].get("reward_scale", 1.0)),
        "normalizer": normalizer,
    }
    if device is not None:
        common["device"] = device
    if algorithm == "td3":
        algo = config["td3"]
        return TD3Agent(
            **common,
            actor_lr=float(algo["actor_lr"]),
            critic_lr=float(algo["critic_lr"]),
            gamma=float(algo["gamma"]),
            tau=float(algo["tau"]),
            policy_noise=float(algo["policy_noise"]),
            noise_clip=float(algo["noise_clip"]),
            policy_delay=int(algo["policy_delay"]),
            exploration_noise=float(algo["exploration_noise"]),
        )
    if algorithm == "ddpg":
        algo = config["ddpg"]
        return DDPGAgent(
            **common,
            actor_lr=float(algo["actor_lr"]),
            critic_lr=float(algo["critic_lr"]),
            gamma=float(algo["gamma"]),
            tau=float(algo["tau"]),
            exploration_noise=float(algo["exploration_noise"]),
        )
    if algorithm == "sac":
        algo = config["sac"]
        return SACAgent(
            **common,
            actor_lr=float(algo["actor_lr"]),
            critic_lr=float(algo["critic_lr"]),
            gamma=float(algo["gamma"]),
            tau=float(algo["tau"]),
            entropy_coef=float(algo["entropy_coef"]),
        )
    raise ValueError(f"unsupported algorithm: {algorithm}")


def select_agent_action(agent: Any, observation: Sequence[float], algorithm: str, explore: bool) -> np.ndarray:
    if algorithm == "sac":
        return agent.select_action(observation, deterministic=not explore)
    return agent.select_action(observation, noise=explore)


def apply_action_transform(
    action: Sequence[float],
    action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    action_array = np.asarray(action, dtype=np.float32)
    if action_transform is not None:
        action_array = np.asarray(action_transform(action_array.copy()), dtype=np.float32)
    return action_array


def action_transform_name(action_transform: Callable[[np.ndarray], np.ndarray] | None) -> str:
    if action_transform is None:
        return "none"
    return str(getattr(action_transform, "__name__", action_transform.__class__.__name__))


def phase4_eval_scenarios(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return load_eval_scenarios(eval_scenarios_path_from_config(config))


def phase4_config_payload(
    algorithm: str,
    config: Mapping[str, Any],
    env_config: EnvConfig,
    eval_scenarios: Sequence[Mapping[str, Any]],
    action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    training = config["training"]
    network = config["network"]
    algo_config = dict(config[algorithm])
    environment_config = asdict(env_config)
    evaluation_scenarios = [
        {
            "id": str(scenario["id"]),
            "q_H": [float(value) for value in scenario["q_H"]],
            "q_R": [float(value) for value in scenario["q_R"]],
            "q_L": [float(value) for value in scenario["q_L"]],
        }
        for scenario in eval_scenarios
    ]
    payload = {
        "algorithm": algorithm,
        "training_seed": int(config["seed"]),
        "episodes": int(training["episodes"]),
        "max_steps": int(training["max_steps"]),
        "batch_size": int(training["batch_size"]),
        "replay_size": int(training["replay_size"]),
        "hidden_sizes": [int(size) for size in network["hidden_sizes"]],
        "reward_parameters": environment_config["reward"],
        "environment_config": environment_config,
        "algorithm_config": algo_config,
        "training_config": {
            "start_steps": int(training.get("start_steps", 0)),
            "eval_interval": int(training.get("eval_interval", 0)),
            "train_every": int(training.get("train_every", 0)),
            "updates_per_train": int(training.get("updates_per_train", 0)),
            "reward_scale": float(training.get("reward_scale", 1.0)),
            "save_interval": int(training.get("save_interval", 0)),
            "activation": str(network.get("activation", "relu")),
            "normalizer_enabled": bool(config["normalizer"]["enabled"]),
            "normalizer_clip": float(config["normalizer"]["clip_value"]),
        },
        "eval_scenarios_path": relative_path(eval_scenarios_path_from_config(config)),
        "eval_scenarios": evaluation_scenarios,
        "action_transform": action_transform_name(action_transform),
    }
    return payload


def phase4_config_hash(
    algorithm: str,
    config: Mapping[str, Any],
    env_config: EnvConfig,
    eval_scenarios: Sequence[Mapping[str, Any]],
    action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> str:
    return stable_sha256(phase4_config_payload(algorithm, config, env_config, eval_scenarios, action_transform))


def phase4_training_metadata(
    algorithm: str,
    config: Mapping[str, Any],
    env_config: EnvConfig,
    eval_scenarios: Sequence[Mapping[str, Any]],
    action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    payload = phase4_config_payload(algorithm, config, env_config, eval_scenarios, action_transform)
    return {
        **payload,
        "config_hash": stable_sha256(payload),
        "git_commit": current_git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


PHASE4_REUSE_FIELDS = [
    "algorithm",
    "training_seed",
    "episodes",
    "max_steps",
    "batch_size",
    "replay_size",
    "hidden_sizes",
    "reward_parameters",
    "environment_config",
    "algorithm_config",
    "training_config",
    "eval_scenarios_path",
    "eval_scenarios",
    "action_transform",
    "config_hash",
]


def validate_phase4_reuse(
    model_dir: Path,
    expected_metadata: Mapping[str, Any],
) -> None:
    required_files = ["eval_log.csv", "best_actor.pt", "best_critic.pt"]
    missing_files = [filename for filename in required_files if not (model_dir / filename).exists()]
    if missing_files:
        raise RuntimeError(
            f"existing result incomplete in {relative_path(model_dir)}: {', '.join(missing_files)}; use --force"
        )
    params_path = model_dir / "training_params.json"
    if not params_path.exists():
        raise RuntimeError(f"existing result missing training_params.json: {relative_path(model_dir)}; use --force")
    saved = load_yaml(params_path)
    if not isinstance(saved, Mapping):
        raise RuntimeError(f"training_params.json in {relative_path(model_dir)} is invalid; use --force")
    mismatches: list[str] = []
    for field in PHASE4_REUSE_FIELDS:
        if field not in saved:
            mismatches.append(f"missing {field}")
            continue
        if canonical_json(saved[field]) != canonical_json(expected_metadata[field]):
            mismatches.append(field)
    if mismatches:
        raise RuntimeError(
            f"existing result config mismatch in {relative_path(model_dir)}: {', '.join(mismatches)}; use --force"
        )


def phase4_eval_metrics_from_rows(rows: Sequence[dict[str, Any]], episode: int, scenario_ids: Sequence[str]) -> dict[str, Any]:
    if len(rows) != len(scenario_ids):
        raise ValueError("rows and scenario_ids must have the same length")
    if not rows:
        raise ValueError("rows must not be empty")
    return {
        "episode": episode,
        "eval_reward": float(np.mean([float(row["total_reward"]) for row in rows])),
        "eval_average_rate_e2e": float(np.mean([float(row["average_rate_e2e"]) for row in rows])),
        "eval_average_rate_HR": float(np.mean([float(row["average_rate_HR"]) for row in rows])),
        "eval_average_rate_RL": float(np.mean([float(row["average_rate_RL"]) for row in rows])),
        "eval_average_snr_HR": float(np.mean([float(row["average_snr_HR"]) for row in rows])),
        "eval_average_snr_RL": float(np.mean([float(row["average_snr_RL"]) for row in rows])),
        "eval_outage_count": float(np.sum([float(row["outage_count"]) for row in rows])),
        "eval_constraint_violation_count": float(np.sum([float(row["constraint_violation_count"]) for row in rows])),
        "eval_trajectory_length": float(np.mean([float(row["trajectory_length"]) for row in rows])),
    }


def episode_summary(metrics: EpisodeMetrics) -> dict[str, Any]:
    summary = metrics.summary()
    return {
        "average_rate_e2e": summary["avg_rate_e2e_bps"],
        "average_rate_HR": summary["avg_rate_HR_bps"],
        "average_rate_RL": summary["avg_rate_RL_bps"],
        "average_snr_HR": summary["avg_snr_HR"],
        "average_snr_RL": summary["avg_snr_RL"],
        "total_reward": summary["total_reward"],
        "outage_count": summary["outage_count"],
        "constraint_violation_count": summary["constraint_violation_count"],
        "trajectory_length": summary["trajectory_length"],
    }


def training_row(
    episode: int,
    metrics: EpisodeMetrics,
    train_metrics: list[dict[str, float | None]],
    replay_buffer_size: int,
    exploration_noise: float | None,
) -> dict[str, Any]:
    summary = metrics.summary()
    return {
        "episode": episode,
        "episode_reward": clean_value(summary["total_reward"]),
        "average_rate_e2e": clean_value(summary["avg_rate_e2e_bps"]),
        "average_rate_HR": clean_value(summary["avg_rate_HR_bps"]),
        "average_rate_RL": clean_value(summary["avg_rate_RL_bps"]),
        "average_snr_HR": clean_value(summary["avg_snr_HR"]),
        "average_snr_RL": clean_value(summary["avg_snr_RL"]),
        "outage_count": summary["outage_count"],
        "constraint_violation_count": summary["constraint_violation_count"],
        "trajectory_length": summary["trajectory_length"],
        "actor_loss": mean_metric(train_metrics, "actor_loss"),
        "critic_loss": mean_metric(train_metrics, "critic_loss"),
        "q1_mean": mean_metric(train_metrics, "q1_mean"),
        "q2_mean": mean_metric(train_metrics, "q2_mean"),
        "replay_buffer_size": replay_buffer_size,
        "exploration_noise": clean_value(exploration_noise) if exploration_noise is not None else "",
    }


def eval_row(episode: int, metrics: EpisodeMetrics) -> dict[str, Any]:
    summary = metrics.summary()
    return {
        "episode": episode,
        "eval_reward": clean_value(summary["total_reward"]),
        "eval_average_rate_e2e": clean_value(summary["avg_rate_e2e_bps"]),
        "eval_average_rate_HR": clean_value(summary["avg_rate_HR_bps"]),
        "eval_average_rate_RL": clean_value(summary["avg_rate_RL_bps"]),
        "eval_average_snr_HR": clean_value(summary["avg_snr_HR"]),
        "eval_average_snr_RL": clean_value(summary["avg_snr_RL"]),
        "eval_outage_count": summary["outage_count"],
        "eval_constraint_violation_count": summary["constraint_violation_count"],
        "eval_trajectory_length": summary["trajectory_length"],
    }


def run_agent_episode(
    agent: Any,
    algorithm: str,
    env_config: EnvConfig,
    seed: int,
    max_steps: int,
    explore: bool = False,
    action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> EpisodeMetrics:
    env = UAVRelayCommEnv(config=env_config)
    observation, _ = env.reset(seed=seed)
    metrics = EpisodeMetrics()
    steps = 0
    while True:
        action = select_agent_action(agent, observation, algorithm, explore=explore)
        action = apply_action_transform(action, action_transform)
        observation, reward, terminated, truncated, info = env.step(action)
        metrics.record(info, reward)
        steps += 1
        if terminated or truncated or steps >= max_steps:
            break
    return metrics


def run_agent_on_scenarios(
    agent: Any,
    algorithm: str,
    base_env_config: EnvConfig,
    scenarios: Sequence[Mapping[str, Any]],
    max_steps: int,
    action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        scenario_env = env_config_with_scenario(base_env_config, scenario)
        metrics = run_agent_episode(
            agent,
            algorithm,
            scenario_env,
            seed=10_000 + index,
            max_steps=max_steps,
            explore=False,
            action_transform=action_transform,
        )
        summary = episode_summary(metrics)
        rows.append(
            {
                "scenario_id": str(scenario["id"]),
                "average_rate_e2e": summary["average_rate_e2e"],
                "average_rate_HR": summary["average_rate_HR"],
                "average_rate_RL": summary["average_rate_RL"],
                "average_snr_HR": summary["average_snr_HR"],
                "average_snr_RL": summary["average_snr_RL"],
                "total_reward": summary["total_reward"],
                "outage_count": summary["outage_count"],
                "constraint_violation_count": summary["constraint_violation_count"],
                "trajectory_length": summary["trajectory_length"],
            }
        )
    return rows


def train_algorithm_seed(
    algorithm: str,
    config: Mapping[str, Any],
    output_dir: Path,
    env_config: EnvConfig | None = None,
    eval_scenarios: Sequence[Mapping[str, Any]] | None = None,
    action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    seed = int(config["seed"])
    training = config["training"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    output_path = make_output_dir(output_dir)
    env_config = env_config if env_config is not None else load_config(COMM_CONFIG_PATH)
    eval_scenarios = list(eval_scenarios) if eval_scenarios is not None else phase4_eval_scenarios(config)
    metadata = phase4_training_metadata(algorithm, config, env_config, eval_scenarios, action_transform)
    env = UAVRelayCommEnv(config=env_config)
    agent = build_agent(algorithm, config, env)
    replay_buffer = ReplayBuffer(env.observation_dim, env.action_dim, capacity=int(training["replay_size"]), seed=seed)
    training_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    total_steps = 0
    best_eval_rate = -math.inf

    for episode in range(1, int(training["episodes"]) + 1):
        observation, _ = env.reset(seed=seed + episode)
        agent.normalizer.update(observation)
        metrics = EpisodeMetrics()
        train_metrics: list[dict[str, float | None]] = []
        for _ in range(int(training["max_steps"])):
            if total_steps < int(training["start_steps"]):
                action = sample_random_action(env_config.mobility.max_speed_mps, env.action_dim, rng)
            else:
                action = select_agent_action(agent, observation, algorithm, explore=True)
            action = apply_action_transform(action, action_transform)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.add(observation, action, reward, next_observation, done)
            agent.normalizer.update(next_observation)
            metrics.record(info, reward)
            observation = next_observation
            total_steps += 1

            should_train = (
                len(replay_buffer) >= int(training["batch_size"])
                and total_steps >= int(training["start_steps"])
                and total_steps % int(training.get("train_every", 1)) == 0
            )
            if should_train:
                for _ in range(int(training.get("updates_per_train", 1))):
                    train_metrics.append(agent.train(replay_buffer, batch_size=int(training["batch_size"])))
            if done:
                break

        if hasattr(agent, "decay_exploration_noise") and algorithm in {"td3", "ddpg"}:
            decay = float(config[algorithm]["exploration_noise_decay"])
            minimum = float(config[algorithm]["min_exploration_noise"])
            exploration_noise = float(agent.decay_exploration_noise(decay, minimum))
        else:
            exploration_noise = None
        row = training_row(episode, metrics, train_metrics, len(replay_buffer), exploration_noise)
        training_rows.append(row)

        if episode % int(training["eval_interval"]) == 0 or episode == int(training["episodes"]):
            scenario_rows = run_agent_on_scenarios(
                agent,
                algorithm,
                env_config,
                eval_scenarios,
                max_steps=int(training["max_steps"]),
                action_transform=action_transform,
            )
            current_eval_row = phase4_eval_metrics_from_rows(
                scenario_rows,
                episode,
                [str(scenario["id"]) for scenario in eval_scenarios],
            )
            eval_rows.append(current_eval_row)
            eval_rate = float(current_eval_row["eval_average_rate_e2e"])
            if eval_rate > best_eval_rate:
                best_eval_rate = eval_rate
                agent.save(output_path, prefix="best")

        if episode % int(training.get("save_interval", 50)) == 0:
            agent.save(output_path, prefix="final")

        if episode == 1 or episode % int(training["eval_interval"]) == 0 or episode == int(training["episodes"]):
            print(
                "{algorithm} seed={seed} episode={episode:03d} reward={reward:.6f} rate={rate:.6f}Mbps".format(
                    algorithm=algorithm,
                    seed=seed,
                    episode=episode,
                    reward=float(row["episode_reward"]),
                    rate=float(row["average_rate_e2e"]) / 1e6,
                )
            )

    agent.save(output_path, prefix="final")
    if not (output_path / "best_actor.pt").exists():
        agent.save(output_path, prefix="best")
    write_csv(output_path / "training_log.csv", training_rows, TRAINING_FIELDS)
    write_csv(output_path / "eval_log.csv", eval_rows, EVAL_FIELDS)
    (output_path / "config_used.yaml").write_text(to_yaml(config) + "\n", encoding="utf-8")
    training_params = dict(config)
    training_params.update(metadata)
    save_json(output_path / "training_params.json", training_params)
    return training_rows, eval_rows, output_path


def load_agent_from_dir(algorithm: str, model_dir: Path, env_config: EnvConfig | None = None, prefer_best: bool = True) -> Any:
    params = load_yaml(model_dir / "training_params.json")
    env = UAVRelayCommEnv(config=env_config if env_config is not None else load_config(COMM_CONFIG_PATH))
    agent = build_agent(algorithm, params, env)
    if prefer_best and not (model_dir / "best_actor.pt").exists():
        prefer_best = False
    agent.load(model_dir, prefix="best" if prefer_best else "final", prefer_best=prefer_best)
    return agent


def evaluate_saved_agent(
    algorithm: str,
    model_dir: Path,
    training_seed: int,
    evaluation_episode: int,
    max_steps: int,
    env_config: EnvConfig | None = None,
    scenario: Mapping[str, Any] | None = None,
    action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    base_env_config = env_config if env_config is not None else load_config(COMM_CONFIG_PATH)
    active_env_config = env_config_with_scenario(base_env_config, scenario) if scenario is not None else base_env_config
    agent = load_agent_from_dir(algorithm, model_dir, env_config=active_env_config, prefer_best=True)
    metrics = run_agent_episode(
        agent,
        algorithm,
        active_env_config,
        seed=10_000 + int(training_seed) * 100 + int(evaluation_episode),
        max_steps=max_steps,
        explore=False,
        action_transform=action_transform,
    )
    return {
        "algorithm": algorithm.upper(),
        "training_seed": training_seed,
        "evaluation_episode": evaluation_episode,
        "scenario_id": str(scenario["id"]) if scenario is not None else "",
        **episode_summary(metrics),
    }


def evaluate_policy_episode(
    policy: Any,
    evaluation_episode: int,
    max_steps: int,
    scenario_id: str = "",
) -> dict[str, Any]:
    env = UAVRelayCommEnv(config=policy.config)
    observation, info = env.reset(seed=20_000 + evaluation_episode)
    metrics = EpisodeMetrics()
    steps = 0
    while True:
        action = policy.select_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        metrics.record(info, reward)
        steps += 1
        if terminated or truncated or steps >= max_steps:
            break
    return {
        "algorithm": policy.__class__.__name__,
        "training_seed": -1,
        "evaluation_episode": evaluation_episode,
        "scenario_id": scenario_id,
        **episode_summary(metrics),
    }


def summarize_episode_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["algorithm"])].append(row)
    summaries: list[dict[str, Any]] = []
    for algorithm, algorithm_rows in sorted(grouped.items()):
        rates = [float(row["average_rate_e2e"]) for row in algorithm_rows]
        rewards = [float(row["total_reward"]) for row in algorithm_rows]
        outages = [float(row["outage_count"]) for row in algorithm_rows]
        constraints = [float(row["constraint_violation_count"]) for row in algorithm_rows]
        lengths = [float(row["trajectory_length"]) for row in algorithm_rows]
        summaries.append(
            {
                "algorithm": algorithm,
                "average_rate_e2e_mean": statistics.fmean(rates),
                "average_rate_e2e_std": statistics.pstdev(rates) if len(rates) > 1 else 0.0,
                "average_reward_mean": statistics.fmean(rewards),
                "average_reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
                "outage_count_mean": statistics.fmean(outages),
                "constraint_violation_count_mean": statistics.fmean(constraints),
                "trajectory_length_mean": statistics.fmean(lengths),
            }
        )
    return summaries


def baseline_episode_rows(
    config: Mapping[str, Any],
    env_config: EnvConfig | None = None,
    eval_scenarios: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    experiment = config["experiment"]
    base_env_config = env_config if env_config is not None else load_config(COMM_CONFIG_PATH)
    scenarios = list(eval_scenarios) if eval_scenarios is not None else phase4_eval_scenarios(config)
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        scenario_env = env_config_with_scenario(base_env_config, scenario)
        for policy in make_default_policies(scenario_env, seed=0):
            rows.append(
                evaluate_policy_episode(
                    policy,
                    scenario_index,
                    max_steps=int(experiment["max_steps"]),
                    scenario_id=str(scenario["id"]),
                )
            )
    return rows
