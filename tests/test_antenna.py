import math

import pytest

from uav_relay_env.antenna import antenna_gain, dipole_gain, isotropic_gain
from uav_relay_env.config import ChannelConfig


def test_dipole_gain_at_zero_angle_is_maximum():
    assert dipole_gain(0.0, g_max=2.0, g_min=0.1) == pytest.approx(2.0)


def test_dipole_gain_at_vertical_angle_respects_minimum():
    assert dipole_gain(math.pi / 2.0, g_max=2.0, g_min=0.1) == pytest.approx(0.1)


def test_dipole_gain_decreases_with_elevation_angle():
    horizontal_gain = dipole_gain(0.0, g_max=2.0, g_min=0.1)
    diagonal_gain = dipole_gain(math.pi / 4.0, g_max=2.0, g_min=0.1)
    vertical_gain = dipole_gain(math.pi / 2.0, g_max=2.0, g_min=0.1)

    assert horizontal_gain == pytest.approx(2.0)
    assert horizontal_gain > diagonal_gain > vertical_gain
    assert vertical_gain == pytest.approx(0.1)


def test_isotropic_gain_is_constant():
    config = ChannelConfig(antenna_model="isotropic")

    assert isotropic_gain() == pytest.approx(1.0)
    assert antenna_gain(0.0, config) == pytest.approx(1.0)
    assert antenna_gain(math.pi / 2.0, config) == pytest.approx(1.0)


def test_unsupported_antenna_model_raises_clear_error():
    config = ChannelConfig(antenna_model="unknown")

    with pytest.raises(ValueError, match="unsupported antenna model"):
        antenna_gain(0.0, config)
