"""Bounded continuous-action distributions for PPO."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch
from torch.distributions import Normal


@dataclass(frozen=True)
class SquashedActionSample:
    """A tanh-squashed action sample.

    Attributes:
        action: Bounded action tensor with shape ``(..., action_dim)``.
        pre_tanh_action: Unbounded raw Normal sample with the same shape.
        log_prob: Corrected log probability with shape ``(...)``.
        entropy: Sample estimate ``-log_prob`` with shape ``(...)``. The exact
            analytic entropy of the tanh-transformed distribution is not used.
    """

    action: torch.Tensor
    pre_tanh_action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


class TanhDiagNormal:
    """Diagonal Normal transformed by tanh and affine action bounds.

    The log probability includes the tanh Jacobian and the action-scale
    Jacobian. Bounds must be finite and have positive width. Actions are never
    clipped; finite boundedness comes solely from the tanh transform.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        *,
        device: torch.device | str,
        eps: float = 1e-6,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
    ) -> None:
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("action_space must be a gymnasium.spaces.Box.")
        if len(action_space.shape) != 1:
            raise ValueError("action_space must be one-dimensional.")
        low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
        if not torch.all(torch.isfinite(low)) or not torch.all(torch.isfinite(high)):
            raise ValueError("action_space bounds must be finite.")
        scale = (high - low) / 2.0
        if not torch.all(torch.isfinite(scale)) or not torch.all(scale > 0):
            raise ValueError("action_space must have positive finite width.")
        bias = (high + low) / 2.0
        self.low = low
        self.high = high
        self.action_scale = scale
        self.action_bias = bias
        self.eps = float(eps)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

    @property
    def action_dim(self) -> int:
        """Number of scalar action coordinates."""

        return int(self.action_scale.numel())

    def _normal(self, mean: torch.Tensor, log_std: torch.Tensor) -> Normal:
        log_std_clamped = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std_clamped)
        return Normal(mean, std)

    def action_from_pre_tanh(self, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        """Map raw Normal samples to bounded actions without clipping."""

        return self.action_bias + self.action_scale * torch.tanh(pre_tanh_action)

    def log_prob_from_pre_tanh(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        pre_tanh_action: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate corrected log probability of stored raw actions."""

        normal = self._normal(mean, log_std)
        squashed = torch.tanh(pre_tanh_action)
        log_prob = normal.log_prob(pre_tanh_action)
        log_prob = log_prob - torch.log(1.0 - squashed.square() + self.eps)
        log_prob = log_prob - torch.log(self.action_scale)
        corrected = log_prob.sum(dim=-1)
        if not torch.all(torch.isfinite(corrected)):
            raise ValueError("log probability must be finite.")
        return corrected

    def sample(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> SquashedActionSample:
        """Sample or deterministically transform a mean action."""

        if deterministic:
            pre_tanh = mean.clone()
        else:
            pre_tanh = self._normal(mean, log_std).rsample()
        action = self.action_from_pre_tanh(pre_tanh)
        log_prob = self.log_prob_from_pre_tanh(mean, log_std, pre_tanh)
        return SquashedActionSample(
            action=action,
            pre_tanh_action=pre_tanh,
            log_prob=log_prob,
            entropy=-log_prob,
        )


__all__ = ["SquashedActionSample", "TanhDiagNormal"]
