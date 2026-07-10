from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

from .channel import compute_channel_components
from .config import EnvConfig
from .entities import CommunicationState, LinkState
from .mobility import update_relay_position
from .rate import end_to_end_rate_bps, single_hop_rate_bps, snr

Vector3 = tuple[float, float, float]


def _vector3(values: Sequence[float]) -> Vector3:
    return float(values[0]), float(values[1]), float(values[2])


def _norm(values: Sequence[float]) -> float:
    x, y, z = _vector3(values)
    return math.sqrt(x * x + y * y + z * z)


def _positions(observation: Sequence[float], info: Mapping[str, Any]) -> tuple[Vector3, Vector3, Vector3]:
    if {"q_H", "q_R", "q_L"}.issubset(info):
        return _vector3(info["q_H"]), _vector3(info["q_R"]), _vector3(info["q_L"])
    return _vector3(observation[0:3]), _vector3(observation[3:6]), _vector3(observation[6:9])


def _velocity_toward(current: Vector3, target: Vector3, config: EnvConfig) -> list[float]:
    delta = (
        target[0] - current[0],
        target[1] - current[1],
        target[2] - current[2],
    )
    distance = _norm(delta)
    if distance == 0.0:
        return [0.0, 0.0, 0.0]
    desired_speed = distance / config.mobility.delta_t_s
    speed = min(config.mobility.max_speed_mps, desired_speed)
    scale = speed / distance
    return [delta[0] * scale, delta[1] * scale, delta[2] * scale]


def candidate_actions(max_speed_mps: float) -> list[list[float]]:
    actions = [[0.0, 0.0, 0.0]]
    for dx in (-1.0, 0.0, 1.0):
        for dy in (-1.0, 0.0, 1.0):
            for dz in (-1.0, 0.0, 1.0):
                direction = (dx, dy, dz)
                direction_norm = _norm(direction)
                if direction_norm == 0.0:
                    continue
                scale = max_speed_mps / direction_norm
                actions.append([dx * scale, dy * scale, dz * scale])
    return actions


def compute_communication_for_relay(
    q_h_m: Sequence[float],
    q_r_m: Sequence[float],
    q_l_m: Sequence[float],
    config: EnvConfig,
) -> CommunicationState:
    hr_components = compute_channel_components(q_h_m, q_r_m, config.channel)
    rl_components = compute_channel_components(q_r_m, q_l_m, config.channel)
    snr_hr = snr(config.rate.power_HR_w, hr_components.channel_gain, config.rate.noise_power_w)
    snr_rl = snr(config.rate.power_RL_w, rl_components.channel_gain, config.rate.noise_power_w)
    rate_hr = single_hop_rate_bps(config.rate.bandwidth_hz, snr_hr)
    rate_rl = single_hop_rate_bps(config.rate.bandwidth_hz, snr_rl)
    return CommunicationState(
        hr=LinkState(
            name="H-R",
            distance_3d_m=hr_components.distance_3d_m,
            horizontal_distance_m=hr_components.horizontal_distance_m,
            elevation_angle_rad=hr_components.elevation_angle_rad,
            gain_tx=hr_components.gain_tx,
            gain_rx=hr_components.gain_rx,
            channel_gain=hr_components.channel_gain,
            snr=snr_hr,
            rate_bps=rate_hr,
        ),
        rl=LinkState(
            name="R-L",
            distance_3d_m=rl_components.distance_3d_m,
            horizontal_distance_m=rl_components.horizontal_distance_m,
            elevation_angle_rad=rl_components.elevation_angle_rad,
            gain_tx=rl_components.gain_tx,
            gain_rx=rl_components.gain_rx,
            channel_gain=rl_components.channel_gain,
            snr=snr_rl,
            rate_bps=rate_rl,
        ),
        rate_e2e_bps=end_to_end_rate_bps(rate_hr, rate_rl, config.rate.half_duplex),
    )


class BasePolicy:
    name = "base"

    def __init__(self, config: EnvConfig) -> None:
        self.config = config

    def select_action(self, observation: Sequence[float], info: Mapping[str, Any]) -> list[float]:
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    name = "random"

    def __init__(self, config: EnvConfig, seed: int = 0) -> None:
        super().__init__(config)
        self.rng = random.Random(seed)

    def select_action(self, observation: Sequence[float], info: Mapping[str, Any]) -> list[float]:
        del observation, info
        vmax = self.config.mobility.max_speed_mps
        for _ in range(100):
            action = [self.rng.uniform(-vmax, vmax) for _ in range(3)]
            if _norm(action) <= vmax:
                return action
        return [0.0, 0.0, 0.0]


class StaticRelayPolicy(BasePolicy):
    name = "static"

    def select_action(self, observation: Sequence[float], info: Mapping[str, Any]) -> list[float]:
        del observation, info
        return [0.0, 0.0, 0.0]


class MidpointPolicy(BasePolicy):
    name = "midpoint"

    def select_action(self, observation: Sequence[float], info: Mapping[str, Any]) -> list[float]:
        q_h, q_r, q_l = _positions(observation, info)
        target = (
            0.5 * (q_h[0] + q_l[0]),
            0.5 * (q_h[1] + q_l[1]),
            0.5 * (q_h[2] + q_l[2]),
        )
        return _velocity_toward(q_r, target, self.config)


class HorizontalMidpointPolicy(BasePolicy):
    name = "horizontal_midpoint"

    def __init__(self, config: EnvConfig, target_altitude_m: float | None = None) -> None:
        super().__init__(config)
        self.target_altitude_m = target_altitude_m

    def select_action(self, observation: Sequence[float], info: Mapping[str, Any]) -> list[float]:
        q_h, q_r, q_l = _positions(observation, info)
        target = (
            0.5 * (q_h[0] + q_l[0]),
            0.5 * (q_h[1] + q_l[1]),
            q_r[2] if self.target_altitude_m is None else self.target_altitude_m,
        )
        return _velocity_toward(q_r, target, self.config)


class GreedyRatePolicy(BasePolicy):
    name = "greedy_rate"

    def select_action(self, observation: Sequence[float], info: Mapping[str, Any]) -> list[float]:
        q_h, q_r, q_l = _positions(observation, info)
        best_action = [0.0, 0.0, 0.0]
        best_key: tuple[float, int] | None = None
        for action in candidate_actions(self.config.mobility.max_speed_mps):
            motion = update_relay_position(
                q_r,
                action,
                self.config.mobility.delta_t_s,
                self.config.mobility.max_speed_mps,
                self.config.mobility.bounds_m,
            )
            communication = compute_communication_for_relay(q_h, motion.next_position_m, q_l, self.config)
            key = (communication.rate_e2e_bps, -int(motion.constraint_violation))
            if best_key is None or key > best_key:
                best_key = key
                best_action = action
        return best_action


class BalancedLinkPolicy(BasePolicy):
    name = "balanced_link"

    def select_action(self, observation: Sequence[float], info: Mapping[str, Any]) -> list[float]:
        q_h, q_r, q_l = _positions(observation, info)
        best_action = [0.0, 0.0, 0.0]
        best_key: tuple[float, float, int] | None = None
        for action in candidate_actions(self.config.mobility.max_speed_mps):
            motion = update_relay_position(
                q_r,
                action,
                self.config.mobility.delta_t_s,
                self.config.mobility.max_speed_mps,
                self.config.mobility.bounds_m,
            )
            communication = compute_communication_for_relay(q_h, motion.next_position_m, q_l, self.config)
            balance = abs(communication.hr.rate_bps - communication.rl.rate_bps) / (
                communication.hr.rate_bps + communication.rl.rate_bps + self.config.reward.epsilon
            )
            key = (-balance, communication.rate_e2e_bps, -int(motion.constraint_violation))
            if best_key is None or key > best_key:
                best_key = key
                best_action = action
        return best_action


def make_default_policies(config: EnvConfig, seed: int = 0) -> list[BasePolicy]:
    return [
        RandomPolicy(config, seed=seed),
        StaticRelayPolicy(config),
        MidpointPolicy(config),
        HorizontalMidpointPolicy(config),
        GreedyRatePolicy(config),
        BalancedLinkPolicy(config),
    ]
