from __future__ import annotations

from typing import Any, Mapping, Sequence

from .channel import compute_channel_components
from .config import EnvConfig, load_config
from .entities import CommunicationState, LinkState, UAVState
from .mobility import update_relay_position
from .rate import end_to_end_rate_bps, single_hop_rate_bps, snr
from .reward import compute_reward, zero_reward_terms
from .scenario import initial_uav_states


class UAVRelayCommEnv:
    """Gymnasium-style H -> R -> L two-hop communication environment."""

    action_dim = 3
    observation_dim = 15

    def __init__(self, config: EnvConfig | None = None, config_path: str | None = None) -> None:
        self.config = config if config is not None else load_config(config_path)
        self.q_H: UAVState
        self.q_R: UAVState
        self.q_L: UAVState
        self.step_count = 0
        self._last_communication: CommunicationState | None = None
        self.reset()

    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        del seed, options
        self.q_H, self.q_R, self.q_L = initial_uav_states(self.config)
        self.step_count = 0
        communication = self._compute_communication()
        self._last_communication = communication
        observation = self._build_observation(communication)
        info = self._build_info(
            communication,
            reward_terms=zero_reward_terms(),
            constraint_info=self._empty_constraint_info(),
        )
        return observation, info

    def step(self, action: Sequence[float]) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        motion = update_relay_position(
            current_position_m=self.q_R.position_m,
            action_velocity_mps=action,
            delta_t_s=self.config.mobility.delta_t_s,
            max_speed_mps=self.config.mobility.max_speed_mps,
            bounds_m=self.config.mobility.bounds_m,
        )
        self.q_R = self.q_R.moved_to(motion.next_position_m)
        communication = self._compute_communication()
        reward, reward_terms = compute_reward(
            communication=communication,
            velocity_mps=motion.applied_velocity_mps,
            config=self.config,
            constraint_violation=motion.constraint_violation,
        )
        self.step_count += 1
        self._last_communication = communication

        terminated = False
        truncated = self.step_count >= self.config.mobility.max_steps
        observation = self._build_observation(communication)
        info = self._build_info(
            communication,
            reward_terms=reward_terms,
            constraint_info=motion.violation_info(),
        )
        return observation, reward, terminated, truncated, info

    def _compute_link(
        self,
        name: str,
        tx: UAVState,
        rx: UAVState,
        transmit_power_w: float,
    ) -> LinkState:
        components = compute_channel_components(tx.position_m, rx.position_m, self.config.channel)
        snr_value = snr(transmit_power_w, components.channel_gain, self.config.rate.noise_power_w)
        rate_bps = single_hop_rate_bps(self.config.rate.bandwidth_hz, snr_value)
        return LinkState(
            name=name,
            distance_3d_m=components.distance_3d_m,
            horizontal_distance_m=components.horizontal_distance_m,
            elevation_angle_rad=components.elevation_angle_rad,
            gain_tx=components.gain_tx,
            gain_rx=components.gain_rx,
            channel_gain=components.channel_gain,
            snr=snr_value,
            rate_bps=rate_bps,
        )

    def _compute_communication(self) -> CommunicationState:
        hr = self._compute_link("H-R", self.q_H, self.q_R, self.config.rate.power_HR_w)
        rl = self._compute_link("R-L", self.q_R, self.q_L, self.config.rate.power_RL_w)
        rate_e2e = end_to_end_rate_bps(hr.rate_bps, rl.rate_bps, self.config.rate.half_duplex)
        return CommunicationState(hr=hr, rl=rl, rate_e2e_bps=rate_e2e)

    def _build_observation(self, communication: CommunicationState) -> list[float]:
        return [
            *self.q_H.position_m,
            *self.q_R.position_m,
            *self.q_L.position_m,
            communication.hr.rate_bps,
            communication.rl.rate_bps,
            communication.rate_e2e_bps,
            communication.hr.snr,
            communication.rl.snr,
            self.step_count / self.config.mobility.max_steps,
        ]

    def _build_info(
        self,
        communication: CommunicationState,
        reward_terms: Mapping[str, float],
        constraint_info: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "step": self.step_count,
            "q_H": list(self.q_H.position_m),
            "q_R": list(self.q_R.position_m),
            "q_L": list(self.q_L.position_m),
            "d_HR": communication.hr.distance_3d_m,
            "d_RL": communication.rl.distance_3d_m,
            "rho_HR": communication.hr.horizontal_distance_m,
            "rho_RL": communication.rl.horizontal_distance_m,
            "theta_HR": communication.hr.elevation_angle_rad,
            "theta_RL": communication.rl.elevation_angle_rad,
            "gain_tx_HR": communication.hr.gain_tx,
            "gain_rx_HR": communication.hr.gain_rx,
            "gain_tx_RL": communication.rl.gain_tx,
            "gain_rx_RL": communication.rl.gain_rx,
            "h_HR": communication.hr.channel_gain,
            "h_RL": communication.rl.channel_gain,
            "snr_HR": communication.hr.snr,
            "snr_RL": communication.rl.snr,
            "rate_HR": communication.hr.rate_bps,
            "rate_RL": communication.rl.rate_bps,
            "rate_e2e": communication.rate_e2e_bps,
            "reward_terms": dict(reward_terms),
            "constraint_violation": bool(constraint_info.get("constraint_violation", False)),
            "constraint_info": dict(constraint_info),
        }

    def _empty_constraint_info(self) -> dict[str, Any]:
        return {
            "constraint_violation": False,
            "velocity_clipped": False,
            "boundary_clipped": False,
            "boundary_excess_m": 0.0,
            "speed_mps": 0.0,
            "requested_velocity_mps": [0.0, 0.0, 0.0],
            "applied_velocity_mps": [0.0, 0.0, 0.0],
        }
