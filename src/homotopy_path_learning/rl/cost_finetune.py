"""Cost-based fine-tuning of a one-step policy with antithetic finite differences.

For each problem the gradient of the cost ratio with respect to the action is
estimated from antithetic Gaussian perturbations around the policy mean, and is
back-propagated through ``agent.actor_mean`` as a constant upstream gradient.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch.nn import functional as F

from homotopy_path_learning.rl.behavior_cloning import predicted_actions
from homotopy_path_learning.rl.networks import ActorCritic


def antithetic_action_gradients(
    cost_fn: Callable[[int, NDArray[np.float64]], float],
    means: ArrayLike,
    *,
    sigma: float,
    n_pairs: int,
    rng: np.random.Generator,
    low: ArrayLike,
    high: ArrayLike,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Estimate ``d cost / d action`` per problem by antithetic finite differences.

    ``means`` has shape ``(B, A)``. For problem ``b`` and pair ``p`` a noise
    ``eps ~ N(0, I_A)`` is drawn and ``cost_fn(b, a)`` is evaluated at
    ``a+- = clip(mean_b +- sigma * eps, low, high)`` (``low``/``high`` have shape
    ``(A,)``). The estimate is
    ``g_b = 1/(2 sigma P) sum_p (cost(a+) - cost(a-)) eps_p``. Returns the
    ``(B, A)`` float64 gradients and an info dict with ``mean_cost_ratio`` (over
    all ``2 P B`` evaluations), ``mean_grad_norm`` and ``n_evaluations``.
    Inputs are not modified.
    """

    mu = np.array(means, dtype=np.float64, copy=True)
    if mu.ndim != 2:
        raise ValueError(f"means must have shape (B, A), got {mu.shape}.")
    lo = np.array(low, dtype=np.float64, copy=True)
    hi = np.array(high, dtype=np.float64, copy=True)
    if lo.shape != (mu.shape[1],) or hi.shape != (mu.shape[1],):
        raise ValueError(f"low and high must have shape ({mu.shape[1]},), got {lo.shape} and {hi.shape}.")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("means, low and high must be finite.")
    if np.any(lo >= hi):
        raise ValueError("low must be smaller than high.")
    if not np.isfinite(sigma) or float(sigma) <= 0.0:
        raise ValueError(f"sigma must be positive and finite, got {sigma!r}.")
    if isinstance(n_pairs, bool) or int(n_pairs) <= 0:
        raise ValueError(f"n_pairs must be a positive integer, got {n_pairs!r}.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    if not callable(cost_fn):
        raise TypeError("cost_fn must be callable.")
    sigma = float(sigma)
    pairs = int(n_pairs)

    gradients = np.zeros_like(mu)
    total = 0.0
    count = 0
    for b in range(mu.shape[0]):
        for _ in range(pairs):
            eps = rng.standard_normal(mu.shape[1])
            plus = float(cost_fn(b, np.clip(mu[b] + sigma * eps, lo, hi)))
            minus = float(cost_fn(b, np.clip(mu[b] - sigma * eps, lo, hi)))
            if not (np.isfinite(plus) and np.isfinite(minus)):
                raise ValueError("cost_fn returned a non-finite value.")
            gradients[b] += (plus - minus) * eps
            total += plus + minus
            count += 2
        gradients[b] /= 2.0 * sigma * pairs
    info = {
        "mean_cost_ratio": total / count if count else float("nan"),
        "mean_grad_norm": float(np.mean(np.linalg.norm(gradients, axis=1))) if mu.shape[0] else 0.0,
        "n_evaluations": float(count),
    }
    return gradients, info


def finetune_step(
    agent: ActorCritic,
    observations: ArrayLike,
    gradients: ArrayLike,
    optimizer: torch.optim.Optimizer,
    *,
    bc_observations: ArrayLike | None = None,
    bc_actions: ArrayLike | None = None,
    bc_weight: float = 0.0,
    augmentations: Sequence[tuple[ArrayLike, ArrayLike]] | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """One optimizer step on ``mean_b g_b . mu_theta(s_b) + bc_weight * BC-MSE``.

    ``observations`` has shape ``(B, obs_dim)`` and ``gradients`` ``(B, A)``
    (constants). The optional BC term uses ``bc_observations`` ``(N, obs_dim)``
    and ``bc_actions`` ``(N, A)``; if ``augmentations`` (pairs of
    ``(obs_dim, obs_dim)`` and ``(A, A)`` matrices) are given, each sample draws
    an element from ``generator`` and becomes ``obs' = M_obs obs``,
    ``act' = clip(M_act act, low, high)`` as in behavior cloning. Returns the
    scalar losses ``surrogate_loss``, ``bc_loss`` and ``total_loss``.
    """

    obs_dim = int(np.prod(agent.observation_shape))
    action_dim = int(np.prod(agent.action_shape))
    obs = np.array(observations, dtype=np.float32, copy=True)
    grad = np.array(gradients, dtype=np.float32, copy=True)
    if obs.ndim != 2 or obs.shape[1] != obs_dim:
        raise ValueError(f"observations must have shape (B, {obs_dim}), got {obs.shape}.")
    if grad.shape != (obs.shape[0], action_dim):
        raise ValueError(f"gradients must have shape ({obs.shape[0]}, {action_dim}), got {grad.shape}.")
    if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(grad))):
        raise ValueError("observations and gradients must be finite.")
    weight = float(bc_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(f"bc_weight must be nonnegative and finite, got {bc_weight!r}.")
    device = agent.distribution.low.device

    mu = predicted_actions(agent, torch.as_tensor(obs, device=device))
    surrogate = (mu * torch.as_tensor(grad, device=device)).sum(dim=1).mean()
    total = surrogate
    bc_loss = torch.zeros((), device=device)
    if weight > 0.0:
        if bc_observations is None or bc_actions is None:
            raise ValueError("bc_observations and bc_actions are required when bc_weight > 0.")
        bo = torch.as_tensor(np.array(bc_observations, dtype=np.float32, copy=True), device=device)
        ba = torch.as_tensor(np.array(bc_actions, dtype=np.float32, copy=True), device=device)
        if bo.ndim != 2 or bo.shape[1] != obs_dim or ba.shape != (bo.shape[0], action_dim):
            raise ValueError("bc_observations/bc_actions must have shapes (N, obs_dim) and (N, A).")
        if not (torch.all(torch.isfinite(bo)) and torch.all(torch.isfinite(ba))):
            raise ValueError("bc_observations and bc_actions must be finite.")
        if augmentations is not None:
            if len(augmentations) == 0:
                raise ValueError("augmentations must not be empty when given.")
            m_obs = torch.as_tensor(np.stack([np.asarray(m, dtype=np.float32) for m, _ in augmentations]), device=device)
            m_act = torch.as_tensor(np.stack([np.asarray(m, dtype=np.float32) for _, m in augmentations]), device=device)
            if m_obs.shape[1:] != (obs_dim, obs_dim) or m_act.shape[1:] != (action_dim, action_dim):
                raise ValueError("augmentation matrices must have shapes (obs_dim, obs_dim) and (A, A).")
            choice = torch.randint(m_obs.shape[0], (bo.shape[0],), generator=generator).to(device)
            bo = torch.einsum("bij,bj->bi", m_obs[choice], bo)
            ba = torch.einsum("bij,bj->bi", m_act[choice], ba)
            ba = torch.maximum(torch.minimum(ba, agent.distribution.high), agent.distribution.low)
        bc_loss = F.mse_loss(predicted_actions(agent, bo), ba)
        total = total + weight * bc_loss

    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return {
        "surrogate_loss": float(surrogate.item()),
        "bc_loss": float(bc_loss.item()),
        "total_loss": float(total.item()),
    }


__all__ = ["antithetic_action_gradients", "finetune_step"]
