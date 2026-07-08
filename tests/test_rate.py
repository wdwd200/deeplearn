import pytest

from uav_relay_env.rate import end_to_end_rate_bps, single_hop_rate_bps, snr


def test_snr_and_rate_increase_together():
    snr_low = snr(transmit_power_w=1.0, channel_gain=1e-10, noise_power_w=1e-9)
    snr_high = snr(transmit_power_w=2.0, channel_gain=1e-10, noise_power_w=1e-9)

    assert snr_high > snr_low
    assert single_hop_rate_bps(1_000_000.0, snr_high) > single_hop_rate_bps(1_000_000.0, snr_low)


def test_end_to_end_rate_is_bottleneck_rate():
    assert end_to_end_rate_bps(10.0, 5.0, half_duplex=False) == pytest.approx(5.0)
    assert end_to_end_rate_bps(10.0, 5.0, half_duplex=True) == pytest.approx(2.5)
