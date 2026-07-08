from uav_relay_env.channel import compute_channel_components
from uav_relay_env.config import ChannelConfig


def test_channel_gain_decreases_when_distance_increases():
    config = ChannelConfig(antenna_model="isotropic", beta0=1.0, path_loss_exponent=2.0)
    near = compute_channel_components((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), config)
    far = compute_channel_components((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), config)

    assert near.channel_gain > far.channel_gain
