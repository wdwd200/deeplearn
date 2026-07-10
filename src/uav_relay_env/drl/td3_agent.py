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


class TD3Agent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int | None = None,
        hidden_sizes: int | Sequence[int] | None = None,
        activation: str = "relu",
        actor_lr: float = 1.0e-3,
        critic_lr: float = 1.0e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
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
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = int(policy_delay)
        self.exploration_noise = float(exploration_noise)
        self.reward_scale = float(reward_scale)
        self.hidden_sizes = list(hidden_sizes) if hidden_sizes is not None and not isinstance(hidden_sizes, int) else hidden_sizes
        self.activation = activation
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer = normalizer if normalizer is not None else ObservationNormalizer(self.obs_dim, enabled=False)
        self.total_it = 0

        sizes = hidden_sizes if hidden_sizes is not None else (64 if hidden_dim is None else hidden_dim)
        self.actor = Actor(obs_dim, action_dim, max_action, hidden_sizes=sizes, activation=activation).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim, max_action, hidden_sizes=sizes, activation=activation).to(self.device)
        self.critic_1 = Critic(obs_dim, action_dim, hidden_sizes=sizes, activation=activation).to(self.device)
        self.critic_2 = Critic(obs_dim, action_dim, hidden_sizes=sizes, activation=activation).to(self.device)
        self.critic_target_1 = Critic(obs_dim, action_dim, hidden_sizes=sizes, activation=activation).to(self.device)
        self.critic_target_2 = Critic(obs_dim, action_dim, hidden_sizes=sizes, activation=activation).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

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

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 64) -> dict[str, float | None]:
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
            noise = torch.randn_like(actions) * (self.policy_noise * self.max_action)
            noise = noise.clamp(-self.noise_clip * self.max_action, self.noise_clip * self.max_action)
            next_actions = self._clip_action_tensor(self.actor_target(next_obs) + noise)
            target_q1 = self.critic_target_1(next_obs, next_actions)
            target_q2 = self.critic_target_2(next_obs, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * torch.minimum(target_q1, target_q2)

        current_q1 = self.critic_1(obs, actions)
        current_q2 = self.critic_2(obs, actions)
        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)

        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=10.0)
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=10.0)
        self.critic_optimizer_2.step()

        metrics = {
            "critic_loss_1": float(critic_loss_1.detach().cpu().item()),
            "critic_loss_2": float(critic_loss_2.detach().cpu().item()),
            "critic_loss": float((critic_loss_1 + critic_loss_2).detach().cpu().item()),
            "q1_mean": float(current_q1.detach().mean().cpu().item()),
            "q2_mean": float(current_q2.detach().mean().cpu().item()),
            "actor_loss": None,
        }

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic_1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_optimizer.step()
            self.soft_update_targets()
            metrics["actor_loss"] = float(actor_loss.detach().cpu().item())

        return metrics

    def decay_exploration_noise(self, decay: float, minimum: float) -> float:
        self.exploration_noise = max(float(minimum), self.exploration_noise * float(decay))
        return self.exploration_noise

    def soft_update_targets(self) -> None:
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic_1, self.critic_target_1)
        self._soft_update(self.critic_2, self.critic_target_2)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str | Path, prefix: str = "td3") -> None:
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
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "critic_target_1": self.critic_target_1.state_dict(),
                "critic_target_2": self.critic_target_2.state_dict(),
                "critic_optimizer_1": self.critic_optimizer_1.state_dict(),
                "critic_optimizer_2": self.critic_optimizer_2.state_dict(),
                "total_it": self.total_it,
                "params": self._params(),
            },
            output_dir / f"{prefix}_critic.pt",
        )

    def load(self, path: str | Path, prefix: str = "td3", prefer_best: bool = False) -> None:
        input_dir = Path(path)
        if prefer_best and (input_dir / "best_td3_actor.pt").exists() and (input_dir / "best_td3_critic.pt").exists():
            prefix = "best_td3"
        actor_state = torch.load(input_dir / f"{prefix}_actor.pt", map_location=self.device, weights_only=False)
        critic_state = torch.load(input_dir / f"{prefix}_critic.pt", map_location=self.device, weights_only=False)
        self.actor.load_state_dict(actor_state["actor"])
        self.actor_target.load_state_dict(actor_state["actor_target"])
        self.actor_optimizer.load_state_dict(actor_state["actor_optimizer"])
        if "normalizer" in actor_state:
            self.normalizer.load_state_dict(actor_state["normalizer"])
        self.critic_1.load_state_dict(critic_state["critic_1"])
        self.critic_2.load_state_dict(critic_state["critic_2"])
        self.critic_target_1.load_state_dict(critic_state["critic_target_1"])
        self.critic_target_2.load_state_dict(critic_state["critic_target_2"])
        self.critic_optimizer_1.load_state_dict(critic_state["critic_optimizer_1"])
        self.critic_optimizer_2.load_state_dict(critic_state["critic_optimizer_2"])
        self.total_it = int(critic_state.get("total_it", self.total_it))

    def _params(self) -> dict[str, float | int | str]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "max_action": self.max_action,
            "gamma": self.gamma,
            "tau": self.tau,
            "policy_noise": self.policy_noise,
            "noise_clip": self.noise_clip,
            "policy_delay": self.policy_delay,
            "exploration_noise": self.exploration_noise,
            "reward_scale": self.reward_scale,
            "hidden_sizes": str(self.hidden_sizes),
            "activation": self.activation,
            "device": str(self.device),
        }
