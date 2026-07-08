from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class EpisodeMetrics:
    rewards: list[float] = field(default_factory=list)
    rate_hr_bps: list[float] = field(default_factory=list)
    rate_rl_bps: list[float] = field(default_factory=list)
    rate_e2e_bps: list[float] = field(default_factory=list)
    snr_hr: list[float] = field(default_factory=list)
    snr_rl: list[float] = field(default_factory=list)
    constraint_violations: int = 0

    def record(self, info: Mapping[str, Any], reward: float) -> None:
        self.rewards.append(float(reward))
        self.rate_hr_bps.append(float(info["rate_HR"]))
        self.rate_rl_bps.append(float(info["rate_RL"]))
        self.rate_e2e_bps.append(float(info["rate_e2e"]))
        self.snr_hr.append(float(info["snr_HR"]))
        self.snr_rl.append(float(info["snr_RL"]))
        if bool(info["constraint_violation"]):
            self.constraint_violations += 1

    def summary(self) -> dict[str, float | int]:
        steps = len(self.rate_e2e_bps)
        if steps == 0:
            return {
                "steps": 0,
                "avg_reward": 0.0,
                "avg_rate_e2e_bps": 0.0,
                "min_rate_e2e_bps": 0.0,
                "max_rate_e2e_bps": 0.0,
                "constraint_violations": 0,
            }
        return {
            "steps": steps,
            "avg_reward": sum(self.rewards) / steps,
            "avg_rate_e2e_bps": sum(self.rate_e2e_bps) / steps,
            "min_rate_e2e_bps": min(self.rate_e2e_bps),
            "max_rate_e2e_bps": max(self.rate_e2e_bps),
            "constraint_violations": self.constraint_violations,
        }
