import math

import pytest

from uav_relay_env.geometry import distance_3d, elevation_angle, horizontal_distance, relative_position


def test_distance_and_horizontal_distance_are_exact():
    q_a = (0.0, 0.0, 0.0)
    q_b = (3.0, 4.0, 12.0)

    assert distance_3d(q_a, q_b) == pytest.approx(13.0)
    assert horizontal_distance(q_a, q_b) == pytest.approx(5.0)


def test_elevation_angle_is_correct():
    q_a = (0.0, 0.0, 0.0)
    q_b = (3.0, 4.0, 5.0)

    assert elevation_angle(q_a, q_b) == pytest.approx(math.atan(1.0))


def test_horizontal_link_elevation_is_zero():
    assert elevation_angle((0.0, 0.0, 100.0), (10.0, 10.0, 100.0)) == pytest.approx(0.0)


def test_vertical_link_elevation_is_pi_over_two():
    assert elevation_angle((1.0, 1.0, 0.0), (1.0, 1.0, 10.0)) == pytest.approx(math.pi / 2.0)


def test_zero_horizontal_distance_without_altitude_difference_is_safe():
    assert elevation_angle((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == pytest.approx(0.0)


def test_relative_position_direction_is_from_b_to_a():
    assert relative_position((5.0, 7.0, 11.0), (2.0, 3.0, 5.0)) == pytest.approx((3.0, 4.0, 6.0))
