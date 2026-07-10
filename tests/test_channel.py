import math

import pytest

from uav_relay_env.channel import channel_gain, compute_channel_components
from uav_relay_env.config import ChannelConfig


def test_channel_gain_formula_is_exact():
    config = ChannelConfig(beta0=2.0, path_loss_exponent=2.0, min_distance_m=1.0)

    assert channel_gain(10.0, gain_tx=3.0, gain_rx=4.0, config=config) == pytest.approx(
        2.0 * (10.0**-2.0) * 3.0 * 4.0
    )


def test_channel_gain_decreases_when_distance_increases():
    config = ChannelConfig(antenna_model="isotropic", beta0=1.0, path_loss_exponent=2.0)
    near = compute_channel_components((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), config)
    far = compute_channel_components((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), config)

    assert near.channel_gain > far.channel_gain


def test_channel_gain_decreases_when_path_loss_exponent_increases():
    low_alpha = ChannelConfig(antenna_model="isotropic", beta0=1.0, path_loss_exponent=2.0)
    high_alpha = ChannelConfig(antenna_model="isotropic", beta0=1.0, path_loss_exponent=3.0)

    assert channel_gain(10.0, 1.0, 1.0, high_alpha) < channel_gain(10.0, 1.0, 1.0, low_alpha)


def test_channel_gain_decreases_when_antenna_gain_decreases():
    config = ChannelConfig(beta0=1.0, path_loss_exponent=2.0)

    assert channel_gain(10.0, 0.5, 0.5, config) < channel_gain(10.0, 1.0, 1.0, config)


def test_channel_gain_uses_minimum_distance_protection():
    config = ChannelConfig(beta0=1.0, path_loss_exponent=2.0, min_distance_m=5.0)

    assert channel_gain(1.0, 1.0, 1.0, config) == pytest.approx(5.0**-2.0)


def test_channel_components_are_finite_for_coincident_positions():
    config = ChannelConfig(antenna_model="dipole", beta0=1.0, path_loss_exponent=2.0, min_distance_m=1.0)

    components = compute_channel_components((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), config)

    assert math.isfinite(components.channel_gain)
    assert math.isfinite(components.elevation_angle_rad)
    assert math.isfinite(components.gain_tx)
    assert math.isfinite(components.gain_rx)
