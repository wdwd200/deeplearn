import numpy as np

from uav_relay_env import UAVRelayCommEnv
from uav_relay_env.drl import DDPGAgent, ReplayBuffer


def _filled_buffer(obs_dim: int, action_dim: int, max_action: float) -> ReplayBuffer:
    buffer = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, capacity=128, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(80):
        obs = rng.normal(size=obs_dim).astype(np.float32)
        action = rng.uniform(-max_action, max_action, size=action_dim).astype(np.float32)
        norm = np.linalg.norm(action)
        if norm > max_action:
            action = action * (max_action / norm)
        next_obs = rng.normal(size=obs_dim).astype(np.float32)
        buffer.add(obs, action, float(rng.normal()), next_obs, False)
    return buffer


def test_ddpg_agent_initializes():
    env = UAVRelayCommEnv()
    agent = DDPGAgent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")

    assert agent.obs_dim == env.observation_dim
    assert agent.action_dim == env.action_dim


def test_ddpg_action_dimension_and_range():
    env = UAVRelayCommEnv()
    agent = DDPGAgent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    obs, _ = env.reset()

    action = agent.select_action(obs, noise=True)

    assert action.shape == (env.action_dim,)
    assert np.isfinite(action).all()
    assert np.linalg.norm(action) <= env.config.mobility.max_speed_mps + 1.0e-5


def test_ddpg_train_once_runs():
    env = UAVRelayCommEnv()
    agent = DDPGAgent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    buffer = _filled_buffer(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps)

    metrics = agent.train(buffer, batch_size=16)

    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["q1_mean"])


def test_ddpg_save_and_load(tmp_path):
    env = UAVRelayCommEnv()
    agent = DDPGAgent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    agent.save(tmp_path, prefix="best")

    loaded = DDPGAgent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    loaded.load(tmp_path, prefer_best=True)
    obs, _ = env.reset()

    assert np.isfinite(loaded.select_action(obs)).all()
