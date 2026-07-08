import math

import pytest

from uav_relay_env.antenna import dipole_gain


def test_dipole_gain_decreases_with_elevation_angle():
    horizontal_gain = dipole_gain(0.0, g_max=2.0, g_min=0.1)
    diagonal_gain = dipole_gain(math.pi / 4.0, g_max=2.0, g_min=0.1)
    vertical_gain = dipole_gain(math.pi / 2.0, g_max=2.0, g_min=0.1)

    assert horizontal_gain == pytest.approx(2.0)
    assert horizontal_gain > diagonal_gain > vertical_gain
    assert vertical_gain == pytest.approx(0.1)
