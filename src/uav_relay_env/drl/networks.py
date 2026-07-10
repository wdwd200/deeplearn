from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, max_action: float, hidden_dim: int = 64) -> None:
        super().__init__()
        self.max_action = float(max_action)
        self.net = _mlp(obs_dim, hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.max_action * torch.tanh(self.net(obs))
        norm = torch.linalg.vector_norm(action, dim=-1, keepdim=True).clamp_min(1.0e-8)
        scale = torch.clamp(self.max_action / norm, max=1.0)
        return action * scale


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))
