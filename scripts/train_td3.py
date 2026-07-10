from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_relay_env import EpisodeMetrics, UAVRelayCommEnv, load_config
from uav_relay_env.drl import ObservationNormalizer, ReplayBuffer, TD3Agent
from uav_relay_env.drl.utils import deep_update, load_yaml, make_output_dir, sample_random_action, save_json, set_seed

COMM_CONFIG_PATH = ROOT / "configs" / "comm_env_default.yaml"
TD3_CONFIG_PATH = ROOT / "configs" / "td3_default.yaml"

DEFAULT_TD3_CONFIG: dict[str, Any] = {
    "seed": 0,
    "training": {
        "episodes": 20,
        "max_steps": 100,
        "start_steps": 200,
        "batch_size": 64,
        "replay_size": 50_000,
        "eval_interval": 5,
        "save_interval": 10,
        "train_every": 1,
        "updates_per_train": 1,
        "reward_scale": 1.0,
    },
    "td3": {
        "gamma": 0.99,
        "tau": 0.005,
        "actor_lr": 1.0e-3,
        "critic_lr": 1.0e-3,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_delay": 2,
        "exploration_noise": 0.1,
        "exploration_noise_decay": 0.995,
        "min_exploration_noise": 0.02,
    },
    "network": {
        "hidden_sizes": [64, 64],
        "activation": "relu",
    },
    "normalizer": {
        "enabled": True,
        "clip_value": 5.0,
    },
    "output": {
        "root_dir": "results/phase2",
    },
}

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


def _warn_missing(config: Mapping[str, Any], defaults: Mapping[str, Any], prefix: str = "") -> None:
    for key, default_value in defaults.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if key not in config:
            print(f"missing TD3 config value '{dotted}', using default {default_value!r}")
        elif isinstance(default_value, Mapping) and isinstance(config[key], Mapping):
            _warn_missing(config[key], default_value, dotted)


def load_td3_training_config(
    config_path: str | Path = TD3_CONFIG_PATH,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(config_path)
    file_config = load_yaml(path) if path.exists() else {}
    _warn_missing(file_config, DEFAULT_TD3_CONFIG)
    config = deep_update(DEFAULT_TD3_CONFIG, file_config)
    if overrides:
        config = deep_update(config, overrides)
    validate_td3_training_config(config)
    return config


def validate_td3_training_config(config: Mapping[str, Any]) -> None:
    training = config["training"]
    td3 = config["td3"]
    network = config["network"]
    normalizer = config["normalizer"]
    output = config["output"]
    positive_ints = ["episodes", "max_steps", "batch_size", "replay_size", "eval_interval", "save_interval"]
    for key in positive_ints:
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if int(training["start_steps"]) < 0:
        raise ValueError("training.start_steps must be non-negative")
    if int(training.get("train_every", 1)) <= 0:
        raise ValueError("training.train_every must be positive")
    if int(training.get("updates_per_train", 1)) <= 0:
        raise ValueError("training.updates_per_train must be positive")
    for key in ["gamma", "tau", "actor_lr", "critic_lr", "policy_noise", "noise_clip", "exploration_noise"]:
        if float(td3[key]) < 0.0:
            raise ValueError(f"td3.{key} must be non-negative")
    if not list(network["hidden_sizes"]):
        raise ValueError("network.hidden_sizes must not be empty")
    if float(normalizer["clip_value"]) <= 0.0:
        raise ValueError("normalizer.clip_value must be positive")
    if not str(output["root_dir"]):
        raise ValueError("output.root_dir must not be empty")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _to_yaml(data: Mapping[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in data.items():
        if isinstance(value, Mapping):
            lines.append(f"{prefix}{key}:")
            lines.append(_to_yaml(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                lines.append(f"{prefix}  - {_yaml_scalar(item)}")
        else:
            lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")
    return "\n".join(lines)


def _clean(value: float | int | None) -> float | int | str:
    if value is None:
        return ""
    numeric = float(value)
    if not math.isfinite(numeric):
        return ""
    return value


def _mean_metric(metrics: list[dict[str, float | None]], key: str) -> float | str:
    values = [float(metric[key]) for metric in metrics if metric.get(key) is not None and math.isfinite(float(metric[key]))]
    return float(np.mean(values)) if values else ""


def _training_row(
    episode: int,
    episode_metrics: EpisodeMetrics,
    train_metrics: list[dict[str, float | None]],
    replay_buffer_size: int,
    exploration_noise: float,
) -> dict[str, Any]:
    summary = episode_metrics.summary()
    return {
        "episode": episode,
        "episode_reward": _clean(summary["total_reward"]),
        "average_rate_e2e": _clean(summary["avg_rate_e2e_bps"]),
        "average_rate_HR": _clean(summary["avg_rate_HR_bps"]),
        "average_rate_RL": _clean(summary["avg_rate_RL_bps"]),
        "average_snr_HR": _clean(summary["avg_snr_HR"]),
        "average_snr_RL": _clean(summary["avg_snr_RL"]),
        "outage_count": summary["outage_count"],
        "constraint_violation_count": summary["constraint_violation_count"],
        "trajectory_length": summary["trajectory_length"],
        "actor_loss": _mean_metric(train_metrics, "actor_loss"),
        "critic_loss": _mean_metric(train_metrics, "critic_loss"),
        "q1_mean": _mean_metric(train_metrics, "q1_mean"),
        "q2_mean": _mean_metric(train_metrics, "q2_mean"),
        "replay_buffer_size": replay_buffer_size,
        "exploration_noise": _clean(exploration_noise),
    }


def _eval_row(episode: int, eval_metrics: EpisodeMetrics) -> dict[str, Any]:
    summary = eval_metrics.summary()
    return {
        "episode": episode,
        "eval_reward": _clean(summary["total_reward"]),
        "eval_average_rate_e2e": _clean(summary["avg_rate_e2e_bps"]),
        "eval_average_rate_HR": _clean(summary["avg_rate_HR_bps"]),
        "eval_average_rate_RL": _clean(summary["avg_rate_RL_bps"]),
        "eval_average_snr_HR": _clean(summary["avg_snr_HR"]),
        "eval_average_snr_RL": _clean(summary["avg_snr_RL"]),
        "eval_outage_count": summary["outage_count"],
        "eval_constraint_violation_count": summary["constraint_violation_count"],
        "eval_trajectory_length": summary["trajectory_length"],
    }


def build_agent(config: Mapping[str, Any], env: UAVRelayCommEnv) -> TD3Agent:
    td3 = config["td3"]
    network = config["network"]
    normalizer_config = config["normalizer"]
    normalizer = ObservationNormalizer(
        env.observation_dim,
        enabled=bool(normalizer_config["enabled"]),
        clip=float(normalizer_config["clip_value"]),
    )
    return TD3Agent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        max_action=env.config.mobility.max_speed_mps,
        hidden_sizes=[int(size) for size in network["hidden_sizes"]],
        activation=str(network.get("activation", "relu")),
        actor_lr=float(td3["actor_lr"]),
        critic_lr=float(td3["critic_lr"]),
        gamma=float(td3["gamma"]),
        tau=float(td3["tau"]),
        policy_noise=float(td3["policy_noise"]),
        noise_clip=float(td3["noise_clip"]),
        policy_delay=int(td3["policy_delay"]),
        exploration_noise=float(td3["exploration_noise"]),
        reward_scale=float(config["training"].get("reward_scale", 1.0)),
        normalizer=normalizer,
    )


def evaluate_agent(agent: TD3Agent, episode: int, seed: int, max_steps: int | None = None) -> dict[str, Any]:
    config = load_config(COMM_CONFIG_PATH)
    env = UAVRelayCommEnv(config=config)
    observation, info = env.reset(seed=seed)
    metrics = EpisodeMetrics()
    steps = 0
    while True:
        action = agent.select_action(observation, noise=False)
        observation, reward, terminated, truncated, info = env.step(action)
        metrics.record(info, reward)
        steps += 1
        if terminated or truncated or (max_steps is not None and steps >= max_steps):
            break
    return _eval_row(episode, metrics)


def train_td3(
    config_path: str | Path = TD3_CONFIG_PATH,
    overrides: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    config = load_td3_training_config(config_path, overrides)
    seed = int(config["seed"])
    training = config["training"]
    td3 = config["td3"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    root_output = Path(output_dir) if output_dir is not None else ROOT / str(config["output"]["root_dir"])
    output_path = make_output_dir(root_output)

    env_config = load_config(COMM_CONFIG_PATH)
    env = UAVRelayCommEnv(config=env_config)
    agent = build_agent(config, env)
    replay_buffer = ReplayBuffer(
        env.observation_dim,
        env.action_dim,
        capacity=int(training["replay_size"]),
        seed=seed,
    )
    training_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    total_steps = 0
    best_eval_rate = -math.inf

    for episode in range(1, int(training["episodes"]) + 1):
        observation, info = env.reset(seed=seed + episode)
        agent.normalizer.update(observation)
        metrics = EpisodeMetrics()
        train_metrics: list[dict[str, float | None]] = []
        for _ in range(int(training["max_steps"])):
            if total_steps < int(training["start_steps"]):
                action = sample_random_action(env_config.mobility.max_speed_mps, env.action_dim, rng)
            else:
                action = agent.select_action(observation, noise=True)
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

        agent.decay_exploration_noise(td3["exploration_noise_decay"], td3["min_exploration_noise"])
        row = _training_row(episode, metrics, train_metrics, len(replay_buffer), agent.exploration_noise)
        training_rows.append(row)

        if episode % int(training["eval_interval"]) == 0 or episode == int(training["episodes"]):
            eval_row = evaluate_agent(agent, episode=episode, seed=seed + 10_000 + episode, max_steps=int(training["max_steps"]))
            eval_rows.append(eval_row)
            eval_rate = float(eval_row["eval_average_rate_e2e"])
            if eval_rate > best_eval_rate:
                best_eval_rate = eval_rate
                agent.save(output_path, prefix="best_td3")

        if episode % int(training["save_interval"]) == 0:
            agent.save(output_path)

        print(
            "episode={episode:03d} reward={reward:.6f} avg_rate_e2e={rate:.6f}Mbps noise={noise:.4f}".format(
                episode=episode,
                reward=float(row["episode_reward"]),
                rate=float(row["average_rate_e2e"]) / 1e6,
                noise=float(row["exploration_noise"]),
            )
        )

    agent.save(output_path)
    if not (output_path / "best_td3_actor.pt").exists():
        agent.save(output_path, prefix="best_td3")
    _write_csv(output_path / "training_log.csv", training_rows, TRAINING_FIELDS)
    _write_csv(output_path / "eval_log.csv", eval_rows, EVAL_FIELDS)
    (output_path / "config_used.yaml").write_text(_to_yaml(config) + "\n", encoding="utf-8")
    save_json(output_path / "training_params.json", config)
    print(f"saved training log: {output_path / 'training_log.csv'}")
    print(f"saved eval log: {output_path / 'eval_log.csv'}")
    print(f"saved model dir: {output_path}")
    return training_rows, eval_rows, output_path


if __name__ == "__main__":
    train_td3()
