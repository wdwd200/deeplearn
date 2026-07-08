from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .antenna import antenna_gain
from .config import ChannelConfig
from .geometry import distance_3d, elevation_angle, horizontal_distance


@dataclass(frozen=True)
class ChannelComponents:
    distance_3d_m: float
    horizontal_distance_m: float
    elevation_angle_rad: float
    gain_tx: float
    gain_rx: float
    channel_gain: float


def channel_gain(distance_m: float, gain_tx: float, gain_rx: float, config: ChannelConfig) -> float:
    effective_distance = max(distance_m, config.min_distance_m)
    path_loss = effective_distance ** (-config.path_loss_exponent)
    return config.beta0 * path_loss * gain_tx * gain_rx


def compute_channel_components(
    tx_position_m: Sequence[float],
    rx_position_m: Sequence[float],
    config: ChannelConfig,
) -> ChannelComponents:
    d = distance_3d(tx_position_m, rx_position_m)
    rho = horizontal_distance(tx_position_m, rx_position_m)
    theta = elevation_angle(tx_position_m, rx_position_m)
    gain_tx = antenna_gain(theta, config)
    gain_rx = antenna_gain(theta, config)
    h = channel_gain(d, gain_tx, gain_rx, config)
    return ChannelComponents(
        distance_3d_m=d,
        horizontal_distance_m=rho,
        elevation_angle_rad=theta,
        gain_tx=gain_tx,
        gain_rx=gain_rx,
        channel_gain=h,
    )
