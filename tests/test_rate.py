import math

import pytest

from uav_relay_env.rate import end_to_end_rate_bps, single_hop_rate_bps, snr


def test_snr_formula_is_exact():
    assert snr(transmit_power_w=2.0, channel_gain=3.0e-9, noise_power_w=1.0e-12) == pytest.approx(6000.0)


def test_single_hop_rate_formula_is_exact():
    bandwidth_hz = 1_000_000.0
    snr_value = 3.0

    assert single_hop_rate_bps(bandwidth_hz, snr_value) == pytest.approx(
        bandwidth_hz * math.log2(1.0 + snr_value)
    )


def test_snr_and_rate_increase_together():
    snr_low = snr(transmit_power_w=1.0, channel_gain=1e-10, noise_power_w=1e-9)
    snr_high = snr(transmit_power_w=2.0, channel_gain=1e-10, noise_power_w=1e-9)

    assert snr_high > snr_low
    assert single_hop_rate_bps(1_000_000.0, snr_high) > single_hop_rate_bps(1_000_000.0, snr_low)


def test_end_to_end_rate_is_bottleneck_rate():
    assert end_to_end_rate_bps(10.0, 5.0, half_duplex=False) == pytest.approx(5.0)
    assert end_to_end_rate_bps(10.0, 5.0, half_duplex=True) == pytest.approx(2.5)


def test_extremely_low_snr_rate_is_near_zero():
    assert single_hop_rate_bps(1_000_000.0, 1.0e-15) == pytest.approx(0.0, abs=1.0e-8)


def test_rate_outputs_are_finite():
    snr_value = snr(transmit_power_w=1.0, channel_gain=1.0e-12, noise_power_w=1.0e-9)
    rate = single_hop_rate_bps(1_000_000.0, snr_value)
    e2e = end_to_end_rate_bps(rate, rate * 2.0, half_duplex=True)

    assert math.isfinite(snr_value)
    assert math.isfinite(rate)
    assert math.isfinite(e2e)
