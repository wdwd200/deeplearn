from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    if name.lower() == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported activation: {name}")


def _hidden_sizes(hidden_sizes: int | Sequence[int]) -> list[int]:
    if isinstance(hidden_sizes, int):
        return [hidden_sizes, hidden_sizes]
    sizes = [int(size) for size in hidden_sizes]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("hidden_sizes must contain positive integers")
    return sizes


def _mlp(input_dim: int, hidden_sizes: int | Sequence[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in _hidden_sizes(hidden_sizes):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(_activation(activation))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int | None = None,
        hidden_sizes: int | Sequence[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.max_action = float(max_action)
        sizes = hidden_sizes if hidden_sizes is not None else (64 if hidden_dim is None else hidden_dim)
        self.net = _mlp(obs_dim, sizes, action_dim, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.max_action * torch.tanh(self.net(obs))
        norm = torch.linalg.vector_norm(action, dim=-1, keepdim=True).clamp_min(1.0e-8)
        scale = torch.clamp(self.max_action / norm, max=1.0)
        return action * scale


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int | None = None,
        hidden_sizes: int | Sequence[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        sizes = hidden_sizes if hidden_sizes is not None else (64 if hidden_dim is None else hidden_dim)
        self.net = _mlp(obs_dim + action_dim, sizes, 1, activation)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))
