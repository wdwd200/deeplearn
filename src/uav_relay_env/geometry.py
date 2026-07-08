from __future__ import annotations

import math
from typing import Sequence, Tuple

Vector3 = Tuple[float, float, float]


def as_vector3(position: Sequence[float]) -> Vector3:
    if len(position) != 3:
        raise ValueError("position must contain exactly three coordinates")
    return float(position[0]), float(position[1]), float(position[2])


def relative_position(q_a: Sequence[float], q_b: Sequence[float]) -> Vector3:
    a = as_vector3(q_a)
    b = as_vector3(q_b)
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def distance_3d(q_a: Sequence[float], q_b: Sequence[float]) -> float:
    dx, dy, dz = relative_position(q_a, q_b)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def horizontal_distance(q_a: Sequence[float], q_b: Sequence[float]) -> float:
    dx, dy, _ = relative_position(q_a, q_b)
    return math.sqrt(dx * dx + dy * dy)


def elevation_angle(q_a: Sequence[float], q_b: Sequence[float]) -> float:
    rho = horizontal_distance(q_a, q_b)
    dz = abs(as_vector3(q_a)[2] - as_vector3(q_b)[2])
    if rho == 0.0:
        return math.pi / 2.0 if dz > 0.0 else 0.0
    return math.atan(dz / rho)
