"""Environment-independent PPO update logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from homotopy_path_learning.config import PPOSettings
from homotopy_path_learning.rl.networks import ActorCritic
from homotopy_path_learning.rl.rollout import RolloutBatch, compute_gae


@dataclass(frozen=True)
class PPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    learning_rate: float
    gradient_norm: float
    early_stop_epoch: int | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "approx_kl": self.approx_kl,
            "clip_fraction": self.clip_fraction,
            "explained_variance": self.explained_variance,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "early_stop_epoch": self.early_stop_epoch,
        }


def clipped_policy_loss(
    *,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> torch.Tensor:
    """Return the PPO clipped policy loss."""

    ratio = torch.exp(new_log_probs - old_log_probs)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    return torch.max(pg_loss1, pg_loss2).mean()


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """Return explained variance as a finite Python float."""

    values_np = values.detach().cpu().numpy()
    returns_np = returns.detach().cpu().numpy()
    variance = float(np.var(returns_np))
    if variance == 0.0:
        return 0.0
    return float(1.0 - np.var(returns_np - values_np) / variance)


def validate_ppo_settings(settings: PPOSettings) -> None:
    """Validate PPO batch constraints used by the updater."""

    if settings.rollout_steps % settings.minibatch_size != 0:
        raise ValueError("rollout_steps must be divisible by minibatch_size.")
    if settings.minibatch_size > settings.rollout_steps:
        raise ValueError("minibatch_size must be <= rollout_steps.")


def ppo_update(
    *,
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    settings: PPOSettings,
    update_index: int,
    total_updates: int,
) -> PPOMetrics:
    """Perform one PPO update and return finite scalar diagnostics."""

    validate_ppo_settings(settings)
    if settings.anneal_learning_rate:
        frac = 1.0 - (float(update_index) - 1.0) / float(max(total_updates, 1))
        learning_rate = frac * settings.learning_rate
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    else:
        learning_rate = float(optimizer.param_groups[0]["lr"])

    advantages, returns = compute_gae(
        rewards=batch.rewards,
        values=batch.values,
        next_values=batch.next_values,
        terminated=batch.terminated,
        truncated=batch.truncated,
        gamma=settings.gamma,
        gae_lambda=settings.gae_lambda,
    )
    batch_size = int(batch.rollout_steps)
    indices = torch.randperm(batch_size, device=batch.observations.device)
    clip_fractions: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    gradient_norms: list[float] = []
    early_stop_epoch: int | None = None

    for epoch in range(settings.update_epochs):
        for start in range(0, batch_size, settings.minibatch_size):
            minibatch_indices = indices[start : start + settings.minibatch_size]
            mb_advantages = advantages[minibatch_indices]
            if settings.normalize_advantage and mb_advantages.numel() > 1:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std(unbiased=False) + 1e-8
                )

            new_log_probs, entropy, new_values = agent.evaluate_pre_tanh_actions(
                batch.observations[minibatch_indices],
                batch.pre_tanh_actions[minibatch_indices],
            )
            old_log_probs = batch.log_probs[minibatch_indices]
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)

            policy_loss = clipped_policy_loss(
                new_log_probs=new_log_probs,
                old_log_probs=old_log_probs,
                advantages=mb_advantages,
                clip_coef=settings.clip_coef,
            )
            new_values = new_values.view(-1)
            old_values = batch.values[minibatch_indices]
            mb_returns = returns[minibatch_indices]
            if settings.clip_value_loss:
                value_unclipped = (new_values - mb_returns).square()
                value_clipped = old_values + torch.clamp(
                    new_values - old_values,
                    -settings.clip_coef,
                    settings.clip_coef,
                )
                value_clipped_loss = (value_clipped - mb_returns).square()
                value_loss = 0.5 * torch.max(value_unclipped, value_clipped_loss).mean()
            else:
                value_loss = 0.5 * (new_values - mb_returns).square().mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + settings.value_coef * value_loss - settings.entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(agent.parameters(), settings.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > settings.clip_coef).to(torch.float32).mean()
                )
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy_loss.detach().cpu()))
            approx_kls.append(float(approx_kl.detach().cpu()))
            clip_fractions.append(float(clip_fraction.detach().cpu()))
            gradient_norms.append(float(torch.as_tensor(gradient_norm).detach().cpu()))

        if settings.target_kl is not None and approx_kls and approx_kls[-1] > settings.target_kl:
            early_stop_epoch = epoch + 1
            break

    metrics = PPOMetrics(
        policy_loss=float(np.mean(policy_losses)),
        value_loss=float(np.mean(value_losses)),
        entropy=float(np.mean(entropies)),
        approx_kl=float(np.mean(approx_kls)),
        clip_fraction=float(np.mean(clip_fractions)),
        explained_variance=explained_variance(batch.values, returns),
        learning_rate=float(learning_rate),
        gradient_norm=float(np.mean(gradient_norms)),
        early_stop_epoch=early_stop_epoch,
    )
    for key, value in metrics.as_dict().items():
        if value is not None and not np.isfinite(float(value)):
            raise ValueError(f"PPO metric {key} must be finite, got {value!r}.")
    return metrics


__all__ = [
    "PPOMetrics",
    "clipped_policy_loss",
    "explained_variance",
    "ppo_update",
    "validate_ppo_settings",
]
