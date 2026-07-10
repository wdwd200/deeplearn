import pytest

from uav_relay_env.config import Bounds3D
from uav_relay_env.mobility import update_relay_position, vector_norm


def test_velocity_above_vmax_is_clipped():
    result = update_relay_position(
        current_position_m=(100.0, 100.0, 100.0),
        action_velocity_mps=(30.0, 40.0, 0.0),
        delta_t_s=1.0,
        max_speed_mps=10.0,
        bounds_m=Bounds3D(),
    )

    assert result.velocity_clipped
    assert vector_norm(result.applied_velocity_mps) <= 10.0


def test_position_update_uses_applied_velocity_and_delta_t():
    result = update_relay_position(
        current_position_m=(100.0, 100.0, 100.0),
        action_velocity_mps=(1.0, -2.0, 3.0),
        delta_t_s=2.0,
        max_speed_mps=10.0,
        bounds_m=Bounds3D(),
    )

    assert result.next_position_m == pytest.approx((102.0, 96.0, 106.0))
    assert not result.constraint_violation


def test_boundary_clipping_keeps_position_inside_area():
    bounds = Bounds3D(x_min=0.0, x_max=105.0, y_min=0.0, y_max=105.0, z_min=50.0, z_max=105.0)

    result = update_relay_position(
        current_position_m=(100.0, 100.0, 100.0),
        action_velocity_mps=(10.0, 10.0, 10.0),
        delta_t_s=1.0,
        max_speed_mps=30.0,
        bounds_m=bounds,
    )

    assert result.next_position_m == pytest.approx((105.0, 105.0, 105.0))
    assert bounds.contains(result.next_position_m)
    assert result.boundary_clipped
    assert result.constraint_violation
    assert result.boundary_excess_m > 0.0


def test_legal_action_has_no_constraint_violation():
    result = update_relay_position(
        current_position_m=(100.0, 100.0, 100.0),
        action_velocity_mps=(1.0, 2.0, 3.0),
        delta_t_s=1.0,
        max_speed_mps=10.0,
        bounds_m=Bounds3D(),
    )

    assert not result.velocity_clipped
    assert not result.boundary_clipped
    assert not result.constraint_violation
