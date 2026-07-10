import pytest

from uav_relay_env.config import EnvConfig, RateConfig, RewardConfig
from uav_relay_env.entities import CommunicationState, LinkState
from uav_relay_env.reward import compute_reward


def _link(rate_bps: float) -> LinkState:
    return LinkState(
        name="test",
        distance_3d_m=1.0,
        horizontal_distance_m=1.0,
        elevation_angle_rad=0.0,
        gain_tx=1.0,
        gain_rx=1.0,
        channel_gain=1.0,
        snr=1.0,
        rate_bps=rate_bps,
    )


def _communication(rate_hr_bps: float, rate_rl_bps: float, rate_e2e_bps: float | None = None) -> CommunicationState:
    return CommunicationState(
        hr=_link(rate_hr_bps),
        rl=_link(rate_rl_bps),
        rate_e2e_bps=min(rate_hr_bps, rate_rl_bps) if rate_e2e_bps is None else rate_e2e_bps,
    )


def test_reward_increases_when_e2e_rate_increases():
    config = EnvConfig(rate=RateConfig(r_min_bps=0.0), reward=RewardConfig(omega_R=1.0e-6))

    low_reward, _ = compute_reward(_communication(1.0e6, 1.0e6), (0.0, 0.0, 0.0), config, False)
    high_reward, _ = compute_reward(_communication(2.0e6, 2.0e6), (0.0, 0.0, 0.0), config, False)

    assert high_reward > low_reward


def test_outage_penalty_is_applied_below_r_min():
    config = EnvConfig(rate=RateConfig(r_min_bps=1.0e6), reward=RewardConfig(omega_O=3.0))

    _, terms = compute_reward(_communication(1.0e5, 1.0e5), (0.0, 0.0, 0.0), config, False)

    assert terms["raw_outage_indicator"] == pytest.approx(1.0)
    assert terms["outage_penalty"] == pytest.approx(3.0)


def test_balance_penalty_increases_with_rate_gap():
    config = EnvConfig(rate=RateConfig(r_min_bps=0.0), reward=RewardConfig(omega_B=1.0))

    _, balanced = compute_reward(_communication(1.0e6, 1.0e6), (0.0, 0.0, 0.0), config, False)
    _, imbalanced = compute_reward(_communication(1.0e6, 3.0e6), (0.0, 0.0, 0.0), config, False)

    assert imbalanced["raw_balance"] > balanced["raw_balance"]
    assert imbalanced["balance_penalty"] > balanced["balance_penalty"]


def test_energy_penalty_increases_with_speed():
    config = EnvConfig(rate=RateConfig(r_min_bps=0.0), reward=RewardConfig(omega_E=1.0, kappa=0.5))

    _, slow = compute_reward(_communication(1.0e6, 1.0e6), (1.0, 0.0, 0.0), config, False)
    _, fast = compute_reward(_communication(1.0e6, 1.0e6), (2.0, 0.0, 0.0), config, False)

    assert fast["raw_E_fly"] > slow["raw_E_fly"]
    assert fast["energy_penalty"] > slow["energy_penalty"]


def test_constraint_penalty_is_applied_on_violation():
    config = EnvConfig(rate=RateConfig(r_min_bps=0.0), reward=RewardConfig(omega_C=2.0))

    _, terms = compute_reward(_communication(1.0e6, 1.0e6), (0.0, 0.0, 0.0), config, True)

    assert terms["constraint_penalty"] == pytest.approx(2.0)


def test_reward_terms_include_all_components():
    config = EnvConfig(rate=RateConfig(r_min_bps=0.0))

    _, terms = compute_reward(_communication(1.0e6, 2.0e6), (1.0, 0.0, 0.0), config, True)

    assert {
        "rate_reward",
        "energy_penalty",
        "outage_penalty",
        "balance_penalty",
        "constraint_penalty",
        "raw_rate_e2e_bps",
        "raw_E_fly",
        "raw_outage_indicator",
        "raw_balance",
    }.issubset(terms)
