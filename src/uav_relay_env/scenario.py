from __future__ import annotations

from .config import EnvConfig
from .entities import UAVState


def initial_uav_states(config: EnvConfig) -> tuple[UAVState, UAVState, UAVState]:
    return (
        UAVState(name="H", position_m=config.scenario.q_H_m),
        UAVState(name="R", position_m=config.scenario.q_R_initial_m),
        UAVState(name="L", position_m=config.scenario.q_L_m),
    )
