from __future__ import annotations

from typing import Mapping

from .config import EnvConfig
from .entities import CommunicationState, RewardTerms
from .mobility import vector_norm


def compute_reward(
    communication: CommunicationState,
    velocity_mps: tuple[float, float, float],
    config: EnvConfig,
    constraint_violation: bool,
) -> tuple[float, RewardTerms]:
    rate_e2e = communication.rate_e2e_bps
    rate_hr = communication.hr.rate_bps
    rate_rl = communication.rl.rate_bps

    outage_penalty = 1.0 if rate_e2e < config.rate.r_min_bps else 0.0
    balance_penalty = abs(rate_hr - rate_rl) / (rate_hr + rate_rl + config.reward.epsilon)
    speed = vector_norm(velocity_mps)
    e_fly = config.reward.kappa * speed * speed * config.mobility.delta_t_s
    constraint_penalty = 1.0 if constraint_violation else 0.0

    rate_reward = config.reward.omega_R * rate_e2e
    energy_cost = config.reward.omega_E * e_fly
    outage_cost = config.reward.omega_O * outage_penalty
    balance_cost = config.reward.omega_B * balance_penalty
    constraint_cost = config.reward.omega_C * constraint_penalty
    reward = rate_reward - energy_cost - outage_cost - balance_cost - constraint_cost

    terms: RewardTerms = {
        "rate_reward": rate_reward,
        "energy_penalty": energy_cost,
        "outage_penalty": outage_cost,
        "balance_penalty": balance_cost,
        "constraint_penalty": constraint_cost,
        "raw_rate_e2e_bps": rate_e2e,
        "raw_E_fly": e_fly,
        "raw_outage_indicator": outage_penalty,
        "raw_balance": balance_penalty,
    }
    return reward, terms


def zero_reward_terms() -> Mapping[str, float]:
    return {
        "rate_reward": 0.0,
        "energy_penalty": 0.0,
        "outage_penalty": 0.0,
        "balance_penalty": 0.0,
        "constraint_penalty": 0.0,
        "raw_rate_e2e_bps": 0.0,
        "raw_E_fly": 0.0,
        "raw_outage_indicator": 0.0,
        "raw_balance": 0.0,
    }
