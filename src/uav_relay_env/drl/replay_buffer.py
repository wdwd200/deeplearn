from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _finite_array(values: Any, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or inf")
    return array


@dataclass
class ReplayBatch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 100_000, seed: int | None = None) -> None:
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool | float,
    ) -> None:
        obs_array = _finite_array(obs, (self.obs_dim,), "obs")
        action_array = _finite_array(action, (self.action_dim,), "action")
        next_obs_array = _finite_array(next_obs, (self.obs_dim,), "next_obs")
        reward_value = float(reward)
        done_value = float(done)
        if not np.isfinite(reward_value):
            raise ValueError("reward contains NaN or inf")
        if not np.isfinite(done_value):
            raise ValueError("done contains NaN or inf")

        self.obs[self.index] = obs_array
        self.actions[self.index] = action_array
        self.rewards[self.index, 0] = reward_value
        self.next_obs[self.index] = next_obs_array
        self.dones[self.index, 0] = 1.0 if done_value else 0.0
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        replace = self.size < batch_size
        indices = self.rng.choice(self.size, size=batch_size, replace=replace)
        return ReplayBatch(
            obs=self.obs[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_obs=self.next_obs[indices].copy(),
            dones=self.dones[indices].copy(),
        )

    def __len__(self) -> int:
        return self.size
