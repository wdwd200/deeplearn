import random

import pytest

from uav_relay_env import UAVRelayCommEnv
from uav_relay_env.config import Bounds3D, EnvConfig, MobilityConfig, RateConfig, ScenarioConfig


REQUIRED_INFO_KEYS = {
    "q_H",
    "q_R",
    "q_L",
    "d_HR",
    "d_RL",
    "theta_HR",
    "theta_RL",
    "h_HR",
    "h_RL",
    "snr_HR",
    "snr_RL",
    "rate_HR",
    "rate_RL",
    "rate_e2e",
    "reward_terms",
    "constraint_violation",
}


def test_reset_output_dimension_and_info_keys():
    env = UAVRelayCommEnv()
    observation, info = env.reset()

    assert len(observation) == env.observation_dim
    assert REQUIRED_INFO_KEYS.issubset(info.keys())


def test_step_returns_gymnasium_style_tuple():
    env = UAVRelayCommEnv()
    env.reset()

    observation, reward, terminated, truncated, info = env.step((0.0, 0.0, 0.0))

    assert len(observation) == env.observation_dim
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert REQUIRED_INFO_KEYS.issubset(info.keys())


def test_step_updates_relay_position():
    config = EnvConfig(
        scenario=ScenarioConfig(q_R_initial_m=(100.0, 100.0, 100.0)),
        mobility=MobilityConfig(delta_t_s=1.0, max_speed_mps=20.0, max_steps=10),
    )
    env = UAVRelayCommEnv(config=config)
    env.reset()

    _, _, _, _, info = env.step((10.0, 0.0, 0.0))

    assert info["q_R"] == pytest.approx([110.0, 100.0, 100.0])


def test_overspeed_action_is_clipped_by_environment():
    config = EnvConfig(
        scenario=ScenarioConfig(q_R_initial_m=(100.0, 100.0, 100.0)),
        mobility=MobilityConfig(delta_t_s=1.0, max_speed_mps=10.0, max_steps=10),
    )
    env = UAVRelayCommEnv(config=config)
    env.reset()

    _, _, _, _, info = env.step((100.0, 0.0, 0.0))

    assert info["q_R"] == pytest.approx([110.0, 100.0, 100.0])
    assert info["constraint_violation"]
    assert info["constraint_info"]["velocity_clipped"]


def test_out_of_bounds_action_is_clipped_by_environment():
    config = EnvConfig(
        scenario=ScenarioConfig(q_R_initial_m=(100.0, 100.0, 100.0)),
        mobility=MobilityConfig(
            delta_t_s=1.0,
            max_speed_mps=30.0,
            max_steps=10,
            bounds_m=Bounds3D(x_min=0.0, x_max=105.0, y_min=0.0, y_max=105.0, z_min=50.0, z_max=105.0),
        ),
    )
    env = UAVRelayCommEnv(config=config)
    env.reset()

    _, _, _, _, info = env.step((10.0, 10.0, 10.0))

    assert info["q_R"] == pytest.approx([105.0, 105.0, 105.0])
    assert info["constraint_violation"]
    assert info["constraint_info"]["boundary_clipped"]


def test_episode_truncates_after_configured_steps():
    config = EnvConfig(mobility=MobilityConfig(max_steps=3))
    env = UAVRelayCommEnv(config=config)
    env.reset()

    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step((0.0, 0.0, 0.0))

    assert not terminated
    assert truncated


def test_rate_e2e_matches_two_hop_bottleneck():
    config = EnvConfig(rate=RateConfig(half_duplex=False))
    env = UAVRelayCommEnv(config=config)
    env.reset()

    _, _, _, _, info = env.step((0.0, 0.0, 0.0))

    assert info["rate_e2e"] == pytest.approx(min(info["rate_HR"], info["rate_RL"]))


def test_reward_terms_are_numeric():
    env = UAVRelayCommEnv()
    env.reset()

    _, _, _, _, info = env.step((0.0, 0.0, 0.0))

    for value in info["reward_terms"].values():
        assert isinstance(value, (int, float))


def test_random_actions_can_run_full_episode():
    config = EnvConfig(mobility=MobilityConfig(max_steps=5, max_speed_mps=10.0))
    env = UAVRelayCommEnv(config=config)
    rng = random.Random(0)
    env.reset()

    truncated = False
    for _ in range(config.mobility.max_steps):
        action = tuple(rng.uniform(-10.0, 10.0) for _ in range(3))
        _, _, terminated, truncated, info = env.step(action)
        assert not terminated
        assert REQUIRED_INFO_KEYS.issubset(info.keys())

    assert truncated
