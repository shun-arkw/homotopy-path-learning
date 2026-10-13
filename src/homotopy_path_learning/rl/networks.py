"""Environment-independent actor-critic networks for PPO."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from homotopy_path_learning.rl.attention import AttentionActor
from homotopy_path_learning.rl.sequence import SequenceActor
from homotopy_path_learning.rl.distributions import SquashedActionSample, TanhDiagNormal
from homotopy_path_learning.systems.spec import PolynomialSystemSpec


def layer_init(layer: nn.Linear, *, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    """Apply orthogonal initialization used by PPO baselines."""

    nn.init.orthogonal_(layer.weight, float(std))
    nn.init.constant_(layer.bias, float(bias_const))
    return layer


def _activation(name: str) -> type[nn.Module]:
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    raise ValueError("activation must be 'tanh' or 'relu'.")


def _mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    *,
    activation: str,
    output_std: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    activation_cls = _activation(activation)
    for hidden in hidden_sizes:
        if int(hidden) <= 0:
            raise ValueError("hidden_sizes entries must be positive.")
        layers.append(layer_init(nn.Linear(previous, int(hidden))))
        layers.append(activation_cls())
        previous = int(hidden)
    layers.append(layer_init(nn.Linear(previous, output_dim), std=output_std))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Actor-critic network for one-dimensional Box observations and actions.

    ``actor_type='mlp'`` (default) builds the MLP actor. ``'attention'`` builds
    an :class:`AttentionActor` and ``'sequence'`` a :class:`SequenceActor` (both need ``system_spec``, a square ``basis`` of
    shape ``(2 D_free, 2 D_free)`` and ``bezier_degree``; ``attention_kwargs``
    may contain ``d_model``, ``n_heads``, ``n_layers``, ``ff_dim``). The critic
    is always an MLP.
    """

    def __init__(
        self,
        *,
        observation_shape: tuple[int, ...],
        action_space: gym.spaces.Box,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        actor_logstd_init: float = -0.5,
        device: torch.device | str = "cpu",
        actor_type: str = "mlp",
        attention_kwargs: dict[str, int] | None = None,
        system_spec: PolynomialSystemSpec | None = None,
        basis: npt.ArrayLike | None = None,
        bezier_degree: int | None = None,
    ) -> None:
        super().__init__()
        if actor_type not in ("mlp", "attention", "sequence"):
            raise ValueError("actor_type must be 'mlp', 'attention' or 'sequence'.")
        if len(observation_shape) != 1:
            raise ValueError("observation_shape must be one-dimensional.")
        if not isinstance(action_space, gym.spaces.Box) or len(action_space.shape) != 1:
            raise TypeError("action_space must be a one-dimensional Box.")
        obs_dim = int(np.prod(observation_shape))
        action_dim = int(np.prod(action_space.shape))
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("observation and action dimensions must be positive.")
        self.observation_shape = tuple(int(v) for v in observation_shape)
        self.action_shape = tuple(int(v) for v in action_space.shape)
        self.actor_type = actor_type
        self.attention_kwargs = dict(attention_kwargs or {})
        if actor_type == "mlp":
            self.actor_mean: nn.Module = _mlp(
                obs_dim,
                hidden_sizes,
                action_dim,
                activation=activation,
                output_std=0.01,
            )
        else:
            if system_spec is None or basis is None or bezier_degree is None:
                raise ValueError(f"actor_type={actor_type!r} requires system_spec, basis and bezier_degree.")
            actor_cls = AttentionActor if actor_type == "attention" else SequenceActor
            attention = actor_cls(system_spec, basis, int(bezier_degree), **self.attention_kwargs)
            if attention.action_dim != action_dim or 2 * attention.free_dim != obs_dim:
                raise ValueError(
                    f"{actor_type} actor expects observation/action dimensions ({2 * attention.free_dim}, "
                    f"{attention.action_dim}); got ({obs_dim}, {action_dim})."
                )
            self.actor_mean = attention
        self.critic = _mlp(
            obs_dim,
            hidden_sizes,
            1,
            activation=activation,
            output_std=1.0,
        )
        self.actor_logstd = nn.Parameter(
            torch.full((1, action_dim), float(actor_logstd_init), dtype=torch.float32)
        )
        self.distribution = TanhDiagNormal(action_space, device=device)
        self.to(device)

    def _as_batch(self, observation: torch.Tensor) -> torch.Tensor:
        values = observation.to(dtype=torch.float32)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        return values.reshape(values.shape[0], -1)

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Return value estimates with shape ``(batch,)``."""

        return self.critic(self._as_batch(observation)).squeeze(-1)

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return bounded action, raw action, log-probability, entropy, and value."""

        batch = self._as_batch(observation)
        mean = self.actor_mean(batch)
        log_std = self.actor_logstd.expand_as(mean)
        sample = self.distribution.sample(mean, log_std, deterministic=deterministic)
        value = self.critic(batch).squeeze(-1)
        return sample.action, sample.pre_tanh_action, sample.log_prob, sample.entropy, value

    def evaluate_pre_tanh_actions(
        self,
        observation: torch.Tensor,
        pre_tanh_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute log probability, sample-entropy estimate, and value."""

        batch = self._as_batch(observation)
        mean = self.actor_mean(batch)
        log_std = self.actor_logstd.expand_as(mean)
        log_prob = self.distribution.log_prob_from_pre_tanh(mean, log_std, pre_tanh_action)
        value = self.critic(batch).squeeze(-1)
        return log_prob, -log_prob, value

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the bounded action obtained from ``tanh(actor_mean)``."""

        action, _, _, _, _ = self.get_action_and_value(observation, deterministic=True)
        return action


__all__ = ["ActorCritic", "SquashedActionSample", "layer_init"]
