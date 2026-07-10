import torch

from uav_relay_env.drl import Actor, Critic


def test_actor_output_dimension_and_action_bounds_on_cpu():
    actor = Actor(obs_dim=5, action_dim=3, max_action=2.0)
    obs = torch.zeros((4, 5))

    action = actor(obs)

    assert action.shape == (4, 3)
    assert torch.all(action <= 2.0)
    assert torch.all(action >= -2.0)
    assert torch.isfinite(action).all()


def test_critic_output_dimension_and_finite_forward_on_cpu():
    critic = Critic(obs_dim=5, action_dim=3)
    obs = torch.zeros((4, 5))
    action = torch.zeros((4, 3))

    q_value = critic(obs, action)

    assert q_value.shape == (4, 1)
    assert torch.isfinite(q_value).all()
