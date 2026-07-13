import inspect

import numpy as np
import pytest
import torch

from uav_relay_env import UAVRelayCommEnv
from uav_relay_env.drl import ReplayBuffer, SACAgent
from uav_relay_env.drl.sac_agent import velocity_ball_transform


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
    obs_tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
    stochastic_tensor_action, stochastic_log_prob = agent.actor.sample(obs_tensor, deterministic=False)
    deterministic_tensor_action, deterministic_log_prob = agent.actor.sample(obs_tensor, deterministic=True)

    assert random_action.shape == (env.action_dim,)
    assert deterministic_action.shape == (env.action_dim,)
    assert np.isfinite(random_action).all()
    assert np.isfinite(deterministic_action).all()
    assert torch.isfinite(stochastic_tensor_action).all()
    assert torch.isfinite(deterministic_tensor_action).all()
    assert torch.isfinite(stochastic_log_prob).all()
    assert torch.isfinite(deterministic_log_prob).all()
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


def test_sac_velocity_ball_transform_formula():
    z = torch.tensor([[3.0, 4.0, 0.0]])
    max_action = 10.0

    action, log_det = velocity_ball_transform(z, max_action)

    assert action.detach().numpy().reshape(-1).tolist() == pytest.approx([5.0, 20.0 / 3.0, 0.0])
    assert torch.linalg.vector_norm(action, dim=-1).item() < max_action
    assert log_det.item() == pytest.approx(3.0 * np.log(max_action) - 4.0 * np.log(6.0))


def test_sac_critic_receives_policy_returned_action(monkeypatch):
    env = UAVRelayCommEnv()
    agent = SACAgent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    buffer = _filled_buffer(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps)
    policy_action_value = env.config.mobility.max_speed_mps * 2.0
    recorded_actions: list[torch.Tensor] = []

    def fixed_sample(obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        del deterministic
        action = torch.zeros((obs.shape[0], env.action_dim), dtype=obs.dtype, device=obs.device)
        action[:, 0] = policy_action_value
        log_prob = torch.zeros((obs.shape[0], 1), dtype=obs.dtype, device=obs.device)
        return action, log_prob

    original_forward = agent.critic_1.forward

    def record_forward(obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        recorded_actions.append(action.detach().cpu())
        return original_forward(obs, action)

    monkeypatch.setattr(agent.actor, "sample", fixed_sample)
    monkeypatch.setattr(agent.critic_1, "forward", record_forward)

    metrics = agent.train(buffer, batch_size=16)

    assert np.isfinite(metrics["actor_loss"])
    assert any(torch.allclose(actions[:, 0], torch.full_like(actions[:, 0], policy_action_value)) for actions in recorded_actions)


def test_sac_has_no_post_policy_radial_clip():
    source = inspect.getsource(SACAgent.select_action) + inspect.getsource(SACAgent.train)

    assert "_clip_action" not in source


def test_sac_save_and_load(tmp_path):
    env = UAVRelayCommEnv()
    agent = SACAgent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    agent.save(tmp_path, prefix="best")

    loaded = SACAgent(env.observation_dim, env.action_dim, env.config.mobility.max_speed_mps, device="cpu")
    loaded.load(tmp_path, prefer_best=True)
    obs, _ = env.reset()

    assert np.isfinite(loaded.select_action(obs)).all()
