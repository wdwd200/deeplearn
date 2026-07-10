import numpy as np

from uav_relay_env.drl import ReplayBuffer


def test_replay_buffer_add_and_len():
    buffer = ReplayBuffer(obs_dim=4, action_dim=2, capacity=10, seed=0)

    buffer.add([1, 2, 3, 4], [0.1, -0.1], 1.0, [2, 3, 4, 5], False)

    assert len(buffer) == 1


def test_replay_buffer_sample_shapes_when_not_full():
    buffer = ReplayBuffer(obs_dim=4, action_dim=2, capacity=10, seed=0)
    buffer.add([1, 2, 3, 4], [0.1, -0.1], 1.0, [2, 3, 4, 5], True)

    batch = buffer.sample(batch_size=4)

    assert batch.obs.shape == (4, 4)
    assert batch.actions.shape == (4, 2)
    assert batch.rewards.shape == (4, 1)
    assert batch.next_obs.shape == (4, 4)
    assert batch.dones.shape == (4, 1)
    assert batch.dones.dtype == np.float32


def test_replay_buffer_length_caps_at_capacity():
    buffer = ReplayBuffer(obs_dim=2, action_dim=1, capacity=3, seed=0)

    for index in range(5):
        buffer.add([index, index + 1], [0.0], 0.0, [index + 1, index + 2], False)

    assert len(buffer) == 3
