from __future__ import annotations

import math

def snr(transmit_power_w: float, channel_gain: float, noise_power_w: float) -> float:
    if noise_power_w <= 0.0:
        raise ValueError("noise_power_w must be positive")
    return transmit_power_w * channel_gain / noise_power_w


def single_hop_rate_bps(bandwidth_hz: float, snr_value: float) -> float:
    if bandwidth_hz <= 0.0:
        raise ValueError("bandwidth_hz must be positive")
    if snr_value < 0.0:
        raise ValueError("snr_value must be non-negative")
    return bandwidth_hz * math.log2(1.0 + snr_value)


def end_to_end_rate_bps(rate_hr_bps: float, rate_rl_bps: float, half_duplex: bool = False) -> float:
    bottleneck = min(rate_hr_bps, rate_rl_bps)
    return 0.5 * bottleneck if half_duplex else bottleneck
