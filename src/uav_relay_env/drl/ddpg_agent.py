from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .networks import Actor, Critic
from .normalizer import ObservationNormalizer
from .replay_buffer import ReplayBuffer


class DDPGAgent:
    """Deterministic policy gradient agent with one critic."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_action: float,
        hidden_sizes: int | Sequence[int] = (256, 256),
        activation: str = "relu",
        actor_lr: float = 3.0e-4,
        critic_lr: float = 3.0e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        exploration_noise: float = 0.1,
        reward_scale: float = 1.0,
        device: str | torch.device | None = None,
        normalizer: ObservationNormalizer | None = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.exploration_noise = float(exploration_noise)
        self.reward_scale = float(reward_scale)
        self.hidden_sizes = list(hidden_sizes) if not isinstance(hidden_sizes, int) else hidden_sizes
        self.activation = activation
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer = normalizer if normalizer is not None else ObservationNormalizer(self.obs_dim, enabled=False)
        self.total_it = 0

        self.actor = Actor(obs_dim, action_dim, max_action, hidden_sizes=hidden_sizes, activation=activation).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim, max_action, hidden_sizes=hidden_sizes, activation=activation).to(self.device)
        self.critic = Critic(obs_dim, action_dim, hidden_sizes=hidden_sizes, activation=activation).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, hidden_sizes=hidden_sizes, activation=activation).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _normalize_obs(self, obs: Any) -> np.ndarray:
        return self.normalizer.normalize(obs).astype(np.float32)

    def _to_tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _clip_action_array(self, action: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(action))
        if norm > self.max_action and norm > 0.0:
            action = action * (self.max_action / norm)
        return np.clip(action, -self.max_action, self.max_action).astype(np.float32)

    def _clip_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(action, dim=-1, keepdim=True).clamp_min(1.0e-8)
        scale = torch.clamp(self.max_action / norm, max=1.0)
        return (action * scale).clamp(-self.max_action, self.max_action)

    def select_action(self, obs: Any, noise: bool = False) -> np.ndarray:
        obs_array = self._normalize_obs(obs).reshape(1, -1)
        with torch.no_grad():
            action = self.actor(self._to_tensor(obs_array)).cpu().numpy()[0]
        if noise:
            action = action + np.random.normal(0.0, self.exploration_noise * self.max_action, size=self.action_dim)
        return self._clip_action_array(action)

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 128) -> dict[str, float | None]:
        if len(replay_buffer) == 0:
            raise ValueError("replay buffer is empty")
        self.total_it += 1
        batch = replay_buffer.sample(batch_size)
        obs = self._to_tensor(self._normalize_obs(batch.obs))
        actions = self._to_tensor(batch.actions)
        rewards = self._to_tensor(batch.rewards * self.reward_scale)
        next_obs = self._to_tensor(self._normalize_obs(batch.next_obs))
        dones = self._to_tensor(batch.dones)

        with torch.no_grad():
            next_actions = self._clip_action_tensor(self.actor_target(next_obs))
            target_q = rewards + (1.0 - dones) * self.gamma * self.critic_target(next_obs, next_actions)

        current_q = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()
        self.soft_update_targets()

        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "q1_mean": float(current_q.detach().mean().cpu().item()),
            "q2_mean": None,
        }

    def decay_exploration_noise(self, decay: float, minimum: float) -> float:
        self.exploration_noise = max(float(minimum), self.exploration_noise * float(decay))
        return self.exploration_noise

    def soft_update_targets(self) -> None:
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str | Path, prefix: str = "final") -> None:
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "normalizer": self.normalizer.state_dict(),
                "params": self._params(),
            },
            output_dir / f"{prefix}_actor.pt",
        )
        torch.save(
            {
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_it": self.total_it,
                "params": self._params(),
            },
            output_dir / f"{prefix}_critic.pt",
        )

    def load(self, path: str | Path, prefix: str = "final", prefer_best: bool = False) -> None:
        input_dir = Path(path)
        if prefer_best and (input_dir / "best_actor.pt").exists() and (input_dir / "best_critic.pt").exists():
            prefix = "best"
        actor_state = torch.load(input_dir / f"{prefix}_actor.pt", map_location=self.device, weights_only=False)
        critic_state = torch.load(input_dir / f"{prefix}_critic.pt", map_location=self.device, weights_only=False)
        self.actor.load_state_dict(actor_state["actor"])
        self.actor_target.load_state_dict(actor_state["actor_target"])
        self.actor_optimizer.load_state_dict(actor_state["actor_optimizer"])
        if "normalizer" in actor_state:
            self.normalizer.load_state_dict(actor_state["normalizer"])
        self.critic.load_state_dict(critic_state["critic"])
        self.critic_target.load_state_dict(critic_state["critic_target"])
        self.critic_optimizer.load_state_dict(critic_state["critic_optimizer"])
        self.total_it = int(critic_state.get("total_it", self.total_it))

    def _params(self) -> dict[str, float | int | str]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "max_action": self.max_action,
            "gamma": self.gamma,
            "tau": self.tau,
            "exploration_noise": self.exploration_noise,
            "reward_scale": self.reward_scale,
            "hidden_sizes": str(self.hidden_sizes),
            "activation": self.activation,
            "device": str(self.device),
        }
