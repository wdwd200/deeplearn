import pytest

from uav_relay_env import UAVRelayCommEnv
from uav_relay_env.config import EnvConfig, MobilityConfig, ScenarioConfig


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


def test_step_updates_relay_position():
    config = EnvConfig(
        scenario=ScenarioConfig(q_R_initial_m=(100.0, 100.0, 100.0)),
        mobility=MobilityConfig(delta_t_s=1.0, max_speed_mps=20.0, max_steps=10),
    )
    env = UAVRelayCommEnv(config=config)
    env.reset()

    _, _, _, _, info = env.step((10.0, 0.0, 0.0))

    assert info["q_R"] == pytest.approx([110.0, 100.0, 100.0])


def test_episode_truncates_after_configured_steps():
    config = EnvConfig(mobility=MobilityConfig(max_steps=3))
    env = UAVRelayCommEnv(config=config)
    env.reset()

    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step((0.0, 0.0, 0.0))

    assert not terminated
    assert truncated
