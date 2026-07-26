"""Rollout collection and GAE for PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from homotopy_path_learning.rl.networks import ActorCritic


@dataclass(frozen=True)
class RolloutBatch:
    """A fixed-length rollout converted to torch tensors.

    Tensor leading dimension is ``rollout_steps``. Observations have shape
    ``(rollout_steps, observation_dim)`` and actions/pre-tanh actions have shape
    ``(rollout_steps, action_dim)``.
    """

    observations: torch.Tensor
    actions: torch.Tensor
    pre_tanh_actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    env_metrics: list[dict[str, Any]]
    next_observation: np.ndarray

    @property
    def rollout_steps(self) -> int:
        return int(self.rewards.shape[0])


def _finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.all(torch.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values.")


def compute_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantage estimates.

    ``terminated`` transitions stop bootstrap. ``truncated`` transitions
    bootstrap from ``next_values`` when supplied, matching Gymnasium semantics.
    """

    for name, tensor in (
        ("rewards", rewards),
        ("values", values),
        ("next_values", next_values),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
    if not (
        rewards.shape == values.shape == next_values.shape == terminated.shape == truncated.shape
    ):
        raise ValueError("GAE tensors must have identical shape.")
    gamma_value = float(gamma)
    lambda_value = float(gae_lambda)
    if not np.isfinite(gamma_value) or gamma_value < 0.0:
        raise ValueError("gamma must be finite and nonnegative.")
    if not np.isfinite(lambda_value) or lambda_value < 0.0:
        raise ValueError("gae_lambda must be finite and nonnegative.")

    rewards = rewards.to(dtype=torch.float32)
    values = values.to(dtype=torch.float32)
    next_values = next_values.to(dtype=torch.float32)
    terminated_f = terminated.to(dtype=torch.float32)
    truncated_f = truncated.to(dtype=torch.float32)
    for name, tensor in (
        ("rewards", rewards),
        ("values", values),
        ("next_values", next_values),
    ):
        _finite_tensor(name, tensor)

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((), dtype=torch.float32, device=rewards.device)
    for index in range(rewards.shape[0] - 1, -1, -1):
        bootstrap = torch.where(
            terminated_f[index] > 0.0,
            torch.zeros((), dtype=torch.float32, device=rewards.device),
            torch.ones((), dtype=torch.float32, device=rewards.device),
        )
        delta = rewards[index] + gamma_value * next_values[index] * bootstrap - values[index]
        continuation = torch.where(
            terminated_f[index] > 0.0,
            torch.zeros((), dtype=torch.float32, device=rewards.device),
            torch.ones((), dtype=torch.float32, device=rewards.device),
        )
        # Truncation bootstraps through next_values but starts a fresh GAE chain.
        continuation = torch.where(truncated_f[index] > 0.0, torch.zeros_like(continuation), continuation)
        last_gae = delta + gamma_value * lambda_value * continuation * last_gae
        advantages[index] = last_gae
    returns = advantages + values
    _finite_tensor("advantages", advantages)
    _finite_tensor("returns", returns)
    return advantages, returns


def _extract_metrics(info: dict[str, Any], reward: float) -> dict[str, Any]:
    keys = (
        "linear_cost",
        "bezier_cost",
        "cost_improvement",
        "linear_success",
        "bezier_success",
        "linear_n_success",
        "bezier_n_success",
        "linear_accepted_steps",
        "bezier_accepted_steps",
        "linear_rejected_steps",
        "bezier_rejected_steps",
    )
    metrics: dict[str, Any] = {"reward": float(reward)}
    for key in keys:
        if key in info:
            value = info[key]
            if isinstance(value, (bool, np.bool_)):
                metrics[key] = bool(value)
            elif isinstance(value, (int, np.integer)):
                metrics[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                metrics[key] = float(value)
    return metrics


def collect_rollout(
    env: gym.Env,
    agent: ActorCritic,
    *,
    rollout_steps: int,
    device: torch.device | str,
    initial_observation: np.ndarray | None = None,
    initial_seed: int | None = None,
) -> RolloutBatch:
    """Collect ``rollout_steps`` transitions from one sequential Gymnasium env."""

    steps = int(rollout_steps)
    if steps <= 0:
        raise ValueError("rollout_steps must be positive.")
    if initial_observation is None:
        observation, _ = env.reset(seed=initial_seed)
    else:
        observation = np.array(initial_observation, dtype=np.float32, copy=True)

    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    pre_tanh_actions: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    terminated_values: list[bool] = []
    truncated_values: list[bool] = []
    values: list[torch.Tensor] = []
    next_values: list[torch.Tensor] = []
    env_metrics: list[dict[str, Any]] = []

    for _ in range(steps):
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t, raw_t, log_prob_t, _, value_t = agent.get_action_and_value(obs_tensor)
        action = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        next_observation, reward, terminated, truncated, info = env.step(action)
        with torch.no_grad():
            if bool(terminated):
                next_value_t = torch.zeros(1, dtype=torch.float32, device=device)
            else:
                next_obs_tensor = torch.as_tensor(
                    next_observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                next_value_t = agent.get_value(next_obs_tensor)

        observations.append(obs_tensor.squeeze(0).detach().cpu())
        actions.append(action_t.squeeze(0).detach().cpu())
        pre_tanh_actions.append(raw_t.squeeze(0).detach().cpu())
        log_probs.append(log_prob_t.squeeze(0).detach().cpu())
        values.append(value_t.squeeze(0).detach().cpu())
        next_values.append(next_value_t.squeeze(0).detach().cpu())
        rewards.append(float(reward))
        terminated_values.append(bool(terminated))
        truncated_values.append(bool(truncated))
        env_metrics.append(_extract_metrics(info, float(reward)))

        if bool(terminated) or bool(truncated):
            observation, _ = env.reset()
        else:
            observation = np.array(next_observation, dtype=np.float32, copy=True)

    batch = RolloutBatch(
        observations=torch.stack(observations).to(device),
        actions=torch.stack(actions).to(device),
        pre_tanh_actions=torch.stack(pre_tanh_actions).to(device),
        log_probs=torch.stack(log_probs).to(device),
        rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
        terminated=torch.as_tensor(terminated_values, dtype=torch.bool, device=device),
        truncated=torch.as_tensor(truncated_values, dtype=torch.bool, device=device),
        values=torch.stack(values).to(device),
        next_values=torch.stack(next_values).to(device),
        env_metrics=env_metrics,
        next_observation=np.array(observation, dtype=np.float32, copy=True),
    )
    for name in (
        "observations",
        "actions",
        "pre_tanh_actions",
        "log_probs",
        "rewards",
        "values",
        "next_values",
    ):
        _finite_tensor(name, getattr(batch, name))
    return batch


__all__ = ["RolloutBatch", "collect_rollout", "compute_gae"]
