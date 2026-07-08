from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_relay_env import EpisodeMetrics, UAVRelayCommEnv, load_config

REQUIRED_INFO_KEYS = {
    "q_H",
    "q_R",
    "q_L",
    "d_HR",
    "d_RL",
    "theta_HR",
    "theta_RL",
    "h_HR",
    "h_RL",
    "snr_HR",
    "snr_RL",
    "rate_HR",
    "rate_RL",
    "rate_e2e",
    "reward_terms",
    "constraint_violation",
}


def _check_info(info: dict[str, Any]) -> None:
    missing = REQUIRED_INFO_KEYS.difference(info)
    if missing:
        raise RuntimeError(f"missing info keys: {sorted(missing)}")


def run_random_episode(seed: int = 7, verbose: bool = True) -> dict[str, float | int]:
    config = load_config(ROOT / "configs" / "comm_env_default.yaml")
    rng = random.Random(seed)
    env = UAVRelayCommEnv(config=config)
    _, info = env.reset(seed=seed)
    _check_info(info)

    metrics = EpisodeMetrics()
    for _ in range(config.mobility.max_steps):
        vmax = config.mobility.max_speed_mps
        action = [rng.uniform(-vmax, vmax), rng.uniform(-vmax, vmax), rng.uniform(-vmax, vmax)]
        _, reward, terminated, truncated, info = env.step(action)
        _check_info(info)
        metrics.record(info, reward)
        if verbose:
            print(
                "step={step:03d} rate_e2e={rate:.3f}Mbps "
                "rate_HR={rate_hr:.3f}Mbps rate_RL={rate_rl:.3f}Mbps "
                "constraint={constraint}".format(
                    step=info["step"],
                    rate=info["rate_e2e"] / 1e6,
                    rate_hr=info["rate_HR"] / 1e6,
                    rate_rl=info["rate_RL"] / 1e6,
                    constraint=info["constraint_violation"],
                )
            )
        if terminated or truncated:
            break

    summary = metrics.summary()
    if verbose:
        print(
            "summary steps={steps} avg_rate_e2e={avg:.3f}Mbps "
            "min_rate_e2e={min_rate:.3f}Mbps max_rate_e2e={max_rate:.3f}Mbps "
            "constraint_violations={violations}".format(
                steps=summary["steps"],
                avg=summary["avg_rate_e2e_bps"] / 1e6,
                min_rate=summary["min_rate_e2e_bps"] / 1e6,
                max_rate=summary["max_rate_e2e_bps"] / 1e6,
                violations=summary["constraint_violations"],
            )
        )
    return summary


if __name__ == "__main__":
    run_random_episode()
