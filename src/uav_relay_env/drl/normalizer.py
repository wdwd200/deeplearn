from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ObservationNormalizer:
    obs_dim: int
    enabled: bool = False
    epsilon: float = 1.0e-8
    clip: float = 10.0

    def __post_init__(self) -> None:
        self.count = 0
        self.mean = np.zeros(self.obs_dim, dtype=np.float64)
        self.m2 = np.zeros(self.obs_dim, dtype=np.float64)

    def update(self, obs: Any) -> None:
        if not self.enabled:
            return
        array = np.asarray(obs, dtype=np.float64)
        if array.ndim == 1:
            rows = array.reshape(1, -1)
        else:
            rows = array
        for row in rows:
            if row.shape != (self.obs_dim,):
                raise ValueError(f"obs must have shape ({self.obs_dim},)")
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.m2 += delta * delta2

    @property
    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.obs_dim, dtype=np.float64)
        return np.maximum(self.m2 / (self.count - 1), self.epsilon)

    def normalize(self, obs: Any) -> np.ndarray:
        array = np.asarray(obs, dtype=np.float32)
        if not self.enabled:
            return array
        normalized = (array - self.mean.astype(np.float32)) / np.sqrt(self.variance.astype(np.float32) + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "enabled": self.enabled,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "count": self.count,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.enabled = bool(state.get("enabled", self.enabled))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.clip = float(state.get("clip", self.clip))
        self.count = int(state.get("count", 0))
        self.mean = np.asarray(state.get("mean", np.zeros(self.obs_dim)), dtype=np.float64)
        self.m2 = np.asarray(state.get("m2", np.zeros(self.obs_dim)), dtype=np.float64)
