import math

import pytest

from uav_relay_env import (
    BalancedLinkPolicy,
    GreedyRatePolicy,
    HorizontalMidpointPolicy,
    MidpointPolicy,
    RandomPolicy,
    StaticRelayPolicy,
    UAVRelayCommEnv,
    make_default_policies,
)
from uav_relay_env.config import EnvConfig, MobilityConfig, ScenarioConfig


def _small_config() -> EnvConfig:
    return EnvConfig(mobility=MobilityConfig(max_steps=5, max_speed_mps=10.0))


def _reset_env(config: EnvConfig) -> tuple[UAVRelayCommEnv, list[float], dict]:
    env = UAVRelayCommEnv(config=config)
    observation, info = env.reset()
    return env, observation, info


def test_all_policies_output_legal_action_dimensions_and_finite_values():
    config = _small_config()
    _, observation, info = _reset_env(config)

    for policy in make_default_policies(config, seed=1):
        action = policy.select_action(observation, info)
        assert len(action) == 3
        assert all(math.isfinite(value) for value in action)
        assert math.sqrt(sum(value * value for value in action)) <= config.mobility.max_speed_mps + 1.0e-9


def test_static_relay_policy_outputs_zero_action():
    config = _small_config()
    _, observation, info = _reset_env(config)

    assert StaticRelayPolicy(config).select_action(observation, info) == pytest.approx([0.0, 0.0, 0.0])


def test_midpoint_policy_points_toward_h_l_midpoint():
    config = EnvConfig(
        scenario=ScenarioConfig(
            q_H_m=(10.0, 0.0, 0.0),
            q_R_initial_m=(0.0, 0.0, 0.0),
            q_L_m=(10.0, 0.0, 0.0),
        ),
        mobility=MobilityConfig(max_speed_mps=5.0),
    )
    _, observation, info = _reset_env(config)

    action = MidpointPolicy(config).select_action(observation, info)

    assert action[0] > 0.0
    assert action[1] == pytest.approx(0.0)
    assert action[2] == pytest.approx(0.0)


def test_horizontal_midpoint_policy_does_not_change_altitude_by_default():
    config = _small_config()
    _, observation, info = _reset_env(config)

    action = HorizontalMidpointPolicy(config).select_action(observation, info)

    assert action[2] == pytest.approx(0.0)


def test_random_policy_outputs_finite_values():
    config = _small_config()
    _, observation, info = _reset_env(config)
    policy = RandomPolicy(config, seed=2)

    for _ in range(20):
        action = policy.select_action(observation, info)
        assert all(math.isfinite(value) for value in action)
        assert math.sqrt(sum(value * value for value in action)) <= config.mobility.max_speed_mps + 1.0e-9


@pytest.mark.parametrize("policy_cls", [GreedyRatePolicy, BalancedLinkPolicy])
def test_search_policies_do_not_mutate_environment_state(policy_cls):
    config = _small_config()
    env, observation, info = _reset_env(config)
    initial_position = env.q_R.position_m
    initial_step_count = env.step_count

    policy_cls(config).select_action(observation, info)

    assert env.q_R.position_m == initial_position
    assert env.step_count == initial_step_count


def test_all_policies_can_run_for_at_least_five_steps():
    config = _small_config()

    for policy in make_default_policies(config, seed=3):
        env, observation, info = _reset_env(config)
        for _ in range(5):
            action = policy.select_action(observation, info)
            observation, _, terminated, truncated, info = env.step(action)
            assert not terminated
            assert len(action) == 3
        assert truncated
