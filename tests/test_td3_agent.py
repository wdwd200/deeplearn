import numpy as np
import torch

from uav_relay_env import UAVRelayCommEnv
from uav_relay_env.drl import ReplayBuffer, TD3Agent


def _filled_buffer(obs_dim: int, action_dim: int, max_action: float) -> ReplayBuffer:
    buffer = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, capacity=128, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(80):
        obs = rng.normal(size=obs_dim).astype(np.float32)
        action = rng.uniform(-max_action, max_action, size=action_dim).astype(np.float32)
        next_obs = rng.normal(size=obs_dim).astype(np.float32)
        buffer.add(obs, action, float(rng.normal()), next_obs, False)
    return buffer


def test_td3_agent_initializes_and_selects_valid_action():
    env = UAVRelayCommEnv()
    agent = TD3Agent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    obs, _ = env.reset()

    action = agent.select_action(obs)

    assert action.shape == (env.action_dim,)
    assert np.isfinite(action).all()
    assert np.linalg.norm(action) <= env.config.mobility.max_speed_mps * np.sqrt(env.action_dim)


def test_td3_agent_train_runs_once_with_enough_samples():
    env = UAVRelayCommEnv()
    agent = TD3Agent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    buffer = _filled_buffer(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps)

    metrics = agent.train(buffer, batch_size=16)

    assert "critic_loss_1" in metrics
    assert np.isfinite(metrics["critic_loss_1"])


def test_td3_agent_soft_update_changes_target_parameters():
    env = UAVRelayCommEnv()
    agent = TD3Agent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, tau=0.5, device="cpu")
    before = [parameter.clone() for parameter in agent.actor_target.parameters()]
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.add_(1.0)

    agent.soft_update_targets()

    assert any(not torch.equal(old, new) for old, new in zip(before, agent.actor_target.parameters()))


def test_td3_agent_save_and_load(tmp_path):
    env = UAVRelayCommEnv()
    agent = TD3Agent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    agent.save(tmp_path)

    loaded = TD3Agent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    loaded.load(tmp_path)
    obs, _ = env.reset()

    assert np.isfinite(loaded.select_action(obs)).all()


def test_td3_agent_does_not_break_env_reset_step_interface():
    env = UAVRelayCommEnv()
    agent = TD3Agent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    obs, info = env.reset()
    action = agent.select_action(obs)

    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert len(next_obs) == env.observation_dim
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "rate_e2e" in next_info
