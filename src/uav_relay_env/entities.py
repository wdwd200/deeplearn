from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class UAVState:
    name: str
    position_m: Vector3

    def moved_to(self, position_m: Vector3) -> "UAVState":
        return UAVState(name=self.name, position_m=position_m)


@dataclass(frozen=True)
class LinkState:
    name: str
    distance_3d_m: float
    horizontal_distance_m: float
    elevation_angle_rad: float
    gain_tx: float
    gain_rx: float
    channel_gain: float
    snr: float
    rate_bps: float


@dataclass(frozen=True)
class CommunicationState:
    hr: LinkState
    rl: LinkState
    rate_e2e_bps: float


RewardTerms = Dict[str, float]
