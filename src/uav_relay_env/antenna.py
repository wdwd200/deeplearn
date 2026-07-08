from __future__ import annotations

import math

from .config import ChannelConfig


def isotropic_gain() -> float:
    return 1.0


def dipole_gain(theta_rad: float, g_max: float = 1.0, g_min: float = 0.0) -> float:
    return max(g_min, g_max * (math.cos(theta_rad) ** 2))


def antenna_gain(theta_rad: float, config: ChannelConfig) -> float:
    model = config.antenna_model.lower()
    if model == "isotropic":
        return isotropic_gain()
    if model == "dipole":
        return dipole_gain(theta_rad, g_max=config.g_max, g_min=config.g_min)
    raise ValueError(f"unsupported antenna model: {config.antenna_model}")
