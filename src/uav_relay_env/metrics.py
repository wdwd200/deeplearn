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
    outage_count: int = 0
    trajectory_q_r: list[list[float]] = field(default_factory=list)

    def record(self, info: Mapping[str, Any], reward: float) -> None:
        self.rewards.append(float(reward))
        self.rate_hr_bps.append(float(info["rate_HR"]))
        self.rate_rl_bps.append(float(info["rate_RL"]))
        self.rate_e2e_bps.append(float(info["rate_e2e"]))
        self.snr_hr.append(float(info["snr_HR"]))
        self.snr_rl.append(float(info["snr_RL"]))
        if bool(info["constraint_violation"]):
            self.constraint_violations += 1
        if float(info.get("reward_terms", {}).get("raw_outage_indicator", 0.0)) > 0.0:
            self.outage_count += 1
        self.trajectory_q_r.append([float(value) for value in info["q_R"]])

    @staticmethod
    def _average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> dict[str, float | int | list[float]]:
        steps = len(self.rate_e2e_bps)
        if steps == 0:
            return {
                "steps": 0,
                "episode_length": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0,
                "avg_rate_e2e_bps": 0.0,
                "avg_rate_HR_bps": 0.0,
                "avg_rate_RL_bps": 0.0,
                "avg_snr_HR": 0.0,
                "avg_snr_RL": 0.0,
                "min_rate_e2e_bps": 0.0,
                "max_rate_e2e_bps": 0.0,
                "constraint_violations": 0,
                "constraint_violation_count": 0,
                "outage_count": 0,
                "trajectory_length": 0,
                "final_q_R": [],
            }
        return {
            "steps": steps,
            "episode_length": steps,
            "total_reward": sum(self.rewards),
            "avg_reward": self._average(self.rewards),
            "avg_rate_e2e_bps": self._average(self.rate_e2e_bps),
            "avg_rate_HR_bps": self._average(self.rate_hr_bps),
            "avg_rate_RL_bps": self._average(self.rate_rl_bps),
            "avg_snr_HR": self._average(self.snr_hr),
            "avg_snr_RL": self._average(self.snr_rl),
            "min_rate_e2e_bps": min(self.rate_e2e_bps),
            "max_rate_e2e_bps": max(self.rate_e2e_bps),
            "constraint_violations": self.constraint_violations,
            "constraint_violation_count": self.constraint_violations,
            "outage_count": self.outage_count,
            "trajectory_length": len(self.trajectory_q_r),
            "final_q_R": self.trajectory_q_r[-1],
        }
