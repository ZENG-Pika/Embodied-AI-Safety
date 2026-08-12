"""Small randomly initialized diffusion policy for SimBox smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RandomDiffusionConfig:
    observation_dim: int = 12
    action_dim: int = 12
    action_horizon: int = 8
    diffusion_steps: int = 10
    hidden_dim: int = 128
    max_joint_delta: float | Sequence[float] = 0.02
    seed: int = 42


class _NoisePredictor(nn.Module):
    def __init__(self, cfg: RandomDiffusionConfig):
        super().__init__()
        input_dim = cfg.observation_dim + cfg.action_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((observation, action, time), dim=-1))


class RandomDiffusionPolicy:
    """Generate bounded joint deltas with an untrained denoising network.

    This is intentionally not a useful controller. It verifies the policy to
    simulator to safety-report path without requiring a trained checkpoint.
    """

    def __init__(self, cfg: RandomDiffusionConfig):
        if cfg.action_horizon < 1 or cfg.diffusion_steps < 1:
            raise ValueError("action_horizon and diffusion_steps must be positive")
        if cfg.observation_dim != cfg.action_dim:
            raise ValueError("the SimBox smoke policy requires equal observation/action dimensions")
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.model = _NoisePredictor(cfg).cpu().eval()
        self.generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
        self._chunk = np.empty((0, cfg.action_dim), dtype=np.float64)

    @torch.inference_mode()
    def _sample_chunk(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        obs = obs.repeat(self.cfg.action_horizon, 1)
        action = torch.randn(
            self.cfg.action_horizon,
            self.cfg.action_dim,
            generator=self.generator,
        )
        for step in reversed(range(self.cfg.diffusion_steps)):
            time = torch.full(
                (self.cfg.action_horizon, 1),
                step / max(self.cfg.diffusion_steps - 1, 1),
            )
            predicted_noise = self.model(obs, action, time)
            action = action - predicted_noise / self.cfg.diffusion_steps
        scale = torch.as_tensor(self.cfg.max_joint_delta, dtype=action.dtype)
        if scale.ndim == 0:
            scale = scale.repeat(self.cfg.action_dim)
        if scale.numel() != self.cfg.action_dim:
            raise ValueError("max_joint_delta must be scalar or match action_dim")
        delta = torch.tanh(action) * scale.reshape(1, -1)
        result = delta.numpy().astype(np.float64, copy=False)
        if not np.isfinite(result).all():
            raise FloatingPointError("random diffusion policy produced NaN or Inf")
        return result

    def predict_joint_target(
        self,
        joint_position: np.ndarray,
        lower_limit: np.ndarray | None = None,
        upper_limit: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(joint_position, dtype=np.float64).reshape(-1)
        if position.size != self.cfg.observation_dim:
            raise ValueError(f"expected {self.cfg.observation_dim} joints, got {position.size}")
        if not len(self._chunk):
            self._chunk = self._sample_chunk(position)
        delta, self._chunk = self._chunk[0], self._chunk[1:]
        target = position + delta
        if lower_limit is not None and upper_limit is not None:
            target = np.clip(target, lower_limit, upper_limit)
        if not np.isfinite(target).all():
            raise FloatingPointError("random diffusion target produced NaN or Inf")
        return target, delta

    def manifest(self) -> dict:
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        return {
            "policy_type": "diffusion_policy",
            "policy_name": "random-diffusion-policy",
            "initialization": "random",
            "seed": self.cfg.seed,
            "checkpoint": None,
            "observation_dim": self.cfg.observation_dim,
            "action_dim": self.cfg.action_dim,
            "action_horizon": self.cfg.action_horizon,
            "diffusion_steps": self.cfg.diffusion_steps,
            "max_action_delta": (
                list(self.cfg.max_joint_delta)
                if not isinstance(self.cfg.max_joint_delta, (int, float))
                else self.cfg.max_joint_delta
            ),
            "parameter_count": parameter_count,
        }
