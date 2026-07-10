from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_relay_env import EpisodeMetrics, UAVRelayCommEnv, load_config
from uav_relay_env.drl import ObservationNormalizer, ReplayBuffer, TD3Agent
from uav_relay_env.drl.utils import make_output_dir, sample_random_action, save_json, set_seed

RESULTS_DIR = ROOT / "results" / "phase2"
CONFIG_PATH = ROOT / "configs" / "comm_env_default.yaml"

TRAINING_FIELDS = [
    "episode",
    "episode_reward",
    "average_rate_e2e",
    "average_rate_HR",
    "average_rate_RL",
    "outage_count",
    "constraint_violation_count",
    "trajectory_length",
]

EVAL_FIELDS = [
    "episode",
    "eval_reward",
    "eval_average_rate_e2e",
    "eval_constraint_violation_count",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _training_row(episode: int, metrics: EpisodeMetrics) -> dict[str, Any]:
    summary = metrics.summary()
    return {
        "episode": episode,
        "episode_reward": summary["total_reward"],
        "average_rate_e2e": summary["avg_rate_e2e_bps"],
        "average_rate_HR": summary["avg_rate_HR_bps"],
        "average_rate_RL": summary["avg_rate_RL_bps"],
        "outage_count": summary["outage_count"],
        "constraint_violation_count": summary["constraint_violation_count"],
        "trajectory_length": summary["trajectory_length"],
    }


def evaluate_agent(agent: TD3Agent, config_path: Path = CONFIG_PATH, seed: int = 0) -> dict[str, Any]:
    config = load_config(config_path)
    env = UAVRelayCommEnv(config=config)
    observation, info = env.reset(seed=seed)
    metrics = EpisodeMetrics()
    while True:
        action = agent.select_action(observation, noise=False)
        observation, reward, terminated, truncated, info = env.step(action)
        metrics.record(info, reward)
        if terminated or truncated:
            break
    summary = metrics.summary()
    return {
        "eval_reward": summary["total_reward"],
        "eval_average_rate_e2e": summary["avg_rate_e2e_bps"],
        "eval_constraint_violation_count": summary["constraint_violation_count"],
    }


def train_td3() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    params = {
        "seed": 11,
        "episodes": 20,
        "batch_size": 64,
        "buffer_capacity": 50_000,
        "start_steps": 200,
        "eval_interval": 5,
        "reward_scale": 1.0,
    }
    set_seed(params["seed"])
    rng = np.random.default_rng(params["seed"])
    output_dir = make_output_dir(RESULTS_DIR)
    config = load_config(CONFIG_PATH)
    env = UAVRelayCommEnv(config=config)
    obs_dim = env.observation_dim
    action_dim = env.action_dim
    max_action = config.mobility.max_speed_mps

    normalizer = ObservationNormalizer(obs_dim, enabled=True)
    agent = TD3Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_action=max_action,
        normalizer=normalizer,
        reward_scale=params["reward_scale"],
    )
    replay_buffer = ReplayBuffer(obs_dim, action_dim, capacity=params["buffer_capacity"], seed=params["seed"])
    training_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    total_steps = 0

    for episode in range(1, params["episodes"] + 1):
        observation, info = env.reset(seed=params["seed"] + episode)
        normalizer.update(observation)
        metrics = EpisodeMetrics()
        while True:
            if total_steps < params["start_steps"]:
                action = sample_random_action(max_action, action_dim, rng)
            else:
                action = agent.select_action(observation, noise=True)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.add(observation, action, reward, next_observation, done)
            normalizer.update(next_observation)
            metrics.record(info, reward)
            observation = next_observation
            total_steps += 1

            if len(replay_buffer) >= params["batch_size"] and total_steps >= params["start_steps"]:
                agent.train(replay_buffer, batch_size=params["batch_size"])

            if done:
                break

        row = _training_row(episode, metrics)
        training_rows.append(row)
        print(
            "episode={episode:03d} reward={reward:.6f} avg_rate_e2e={rate:.6f}Mbps".format(
                episode=episode,
                reward=row["episode_reward"],
                rate=row["average_rate_e2e"] / 1e6,
            )
        )

        if episode % params["eval_interval"] == 0 or episode == params["episodes"]:
            eval_summary = evaluate_agent(agent, seed=params["seed"] + 10_000 + episode)
            eval_row = {"episode": episode, **eval_summary}
            eval_rows.append(eval_row)
            print(
                "eval episode={episode:03d} reward={reward:.6f} avg_rate_e2e={rate:.6f}Mbps".format(
                    episode=episode,
                    reward=eval_row["eval_reward"],
                    rate=eval_row["eval_average_rate_e2e"] / 1e6,
                )
            )

    _write_csv(output_dir / "training_log.csv", training_rows, TRAINING_FIELDS)
    _write_csv(output_dir / "eval_log.csv", eval_rows, EVAL_FIELDS)
    (output_dir / "config_used.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    save_json(output_dir / "training_params.json", params)
    agent.save(output_dir)
    print(f"saved training log: {output_dir / 'training_log.csv'}")
    print(f"saved eval log: {output_dir / 'eval_log.csv'}")
    print(f"saved model dir: {output_dir}")
    return training_rows, eval_rows


if __name__ == "__main__":
    train_td3()
