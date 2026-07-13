import numpy as np

from uav_relay_env import UAVRelayCommEnv
from uav_relay_env.drl import ReplayBuffer, SACAgent


def _filled_buffer(obs_dim: int, action_dim: int, max_action: float) -> ReplayBuffer:
    buffer = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, capacity=128, seed=0)
    rng = np.random.default_rng(1)
    for _ in range(80):
        obs = rng.normal(size=obs_dim).astype(np.float32)
        action = rng.uniform(-max_action, max_action, size=action_dim).astype(np.float32)
        norm = np.linalg.norm(action)
        if norm > max_action:
            action = action * (max_action / norm)
        next_obs = rng.normal(size=obs_dim).astype(np.float32)
        buffer.add(obs, action, float(rng.normal()), next_obs, False)
    return buffer


def test_sac_agent_initializes():
    env = UAVRelayCommEnv()
    agent = SACAgent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")

    assert agent.obs_dim == env.observation_dim
    assert agent.action_dim == env.action_dim


def test_sac_random_and_deterministic_actions_are_valid():
    env = UAVRelayCommEnv()
    agent = SACAgent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    obs, _ = env.reset()

    random_action = agent.select_action(obs, deterministic=False)
    deterministic_action = agent.select_action(obs, deterministic=True)

    assert random_action.shape == (env.action_dim,)
    assert deterministic_action.shape == (env.action_dim,)
    assert np.isfinite(random_action).all()
    assert np.isfinite(deterministic_action).all()
    assert np.linalg.norm(random_action) <= env.config.mobility.max_speed_mps + 1.0e-5
    assert np.linalg.norm(deterministic_action) <= env.config.mobility.max_speed_mps + 1.0e-5


def test_sac_train_once_runs():
    env = UAVRelayCommEnv()
    agent = SACAgent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    buffer = _filled_buffer(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps)

    metrics = agent.train(buffer, batch_size=16)

    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["q1_mean"])
    assert np.isfinite(metrics["q2_mean"])


def test_sac_save_and_load(tmp_path):
    env = UAVRelayCommEnv()
    agent = SACAgent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    agent.save(tmp_path, prefix="best")

    loaded = SACAgent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    loaded.load(tmp_path, prefer_best=True)
    obs, _ = env.reset()

    assert np.isfinite(loaded.select_action(obs)).all()
