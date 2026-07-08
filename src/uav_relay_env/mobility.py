from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

from .config import Bounds3D
from .geometry import as_vector3

Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class MotionResult:
    requested_velocity_mps: Vector3
    applied_velocity_mps: Vector3
    unclipped_position_m: Vector3
    next_position_m: Vector3
    speed_mps: float
    velocity_clipped: bool
    boundary_clipped: bool
    boundary_excess_m: float

    @property
    def constraint_violation(self) -> bool:
        return self.velocity_clipped or self.boundary_clipped

    def violation_info(self) -> dict[str, float | bool | list[float]]:
        return {
            "constraint_violation": self.constraint_violation,
            "velocity_clipped": self.velocity_clipped,
            "boundary_clipped": self.boundary_clipped,
            "boundary_excess_m": self.boundary_excess_m,
            "speed_mps": self.speed_mps,
            "requested_velocity_mps": list(self.requested_velocity_mps),
            "applied_velocity_mps": list(self.applied_velocity_mps),
        }


def vector_norm(vector: Sequence[float]) -> float:
    x, y, z = as_vector3(vector)
    return math.sqrt(x * x + y * y + z * z)


def clip_velocity(velocity_mps: Sequence[float], max_speed_mps: float) -> tuple[Vector3, bool]:
    velocity = as_vector3(velocity_mps)
    speed = vector_norm(velocity)
    if speed <= max_speed_mps or speed == 0.0:
        return velocity, False
    scale = max_speed_mps / speed
    return (velocity[0] * scale, velocity[1] * scale, velocity[2] * scale), True


def boundary_excess(unclipped: Vector3, clipped: Vector3) -> float:
    dx = unclipped[0] - clipped[0]
    dy = unclipped[1] - clipped[1]
    dz = unclipped[2] - clipped[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def update_relay_position(
    current_position_m: Sequence[float],
    action_velocity_mps: Sequence[float],
    delta_t_s: float,
    max_speed_mps: float,
    bounds_m: Bounds3D,
) -> MotionResult:
    requested_velocity = as_vector3(action_velocity_mps)
    applied_velocity, velocity_clipped = clip_velocity(requested_velocity, max_speed_mps)
    current = as_vector3(current_position_m)
    unclipped_position = (
        current[0] + applied_velocity[0] * delta_t_s,
        current[1] + applied_velocity[1] * delta_t_s,
        current[2] + applied_velocity[2] * delta_t_s,
    )
    next_position = bounds_m.clamp(unclipped_position)
    boundary_clipped = next_position != unclipped_position
    return MotionResult(
        requested_velocity_mps=requested_velocity,
        applied_velocity_mps=applied_velocity,
        unclipped_position_m=unclipped_position,
        next_position_m=next_position,
        speed_mps=vector_norm(applied_velocity),
        velocity_clipped=velocity_clipped,
        boundary_clipped=boundary_clipped,
        boundary_excess_m=boundary_excess(unclipped_position, next_position),
    )
