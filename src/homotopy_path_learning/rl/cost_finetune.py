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


def sigma_at(iteration: int, *, iterations: int, sigma_start: float, sigma_end: float | None) -> float:
    """Return the perturbation scale of ``iteration`` (1-based) under a linear schedule.

    ``sigma_k = sigma_start + (sigma_end - sigma_start) * (k - 1) / (T - 1)`` with
    ``T = iterations`` (``sigma_start`` if ``T == 1``). ``sigma_end=None`` returns
    ``sigma_start``. Requires ``1 <= iteration <= iterations``, ``sigma_start > 0``
    and, when given, ``0 < sigma_end <= sigma_start``.
    """

    if isinstance(iterations, bool) or int(iterations) < 1:
        raise ValueError(f"iterations must be a positive integer, got {iterations!r}.")
    if isinstance(iteration, bool) or not 1 <= int(iteration) <= int(iterations):
        raise ValueError(f"iteration must lie in [1, {iterations}], got {iteration!r}.")
    if not np.isfinite(sigma_start) or float(sigma_start) <= 0.0:
        raise ValueError(f"sigma_start must be positive and finite, got {sigma_start!r}.")
    if sigma_end is None:
        return float(sigma_start)
    if not np.isfinite(sigma_end) or float(sigma_end) <= 0.0 or float(sigma_end) > float(sigma_start):
        raise ValueError(f"sigma_end must satisfy 0 < sigma_end <= sigma_start, got {sigma_end!r}.")
    if int(iterations) == 1:
        return float(sigma_start)
    fraction = (int(iteration) - 1) / (int(iterations) - 1)
    return float(sigma_start) + (float(sigma_end) - float(sigma_start)) * fraction


def antithetic_candidates(
    means: ArrayLike,
    *,
    sigma: float,
    n_pairs: int,
    rng: np.random.Generator,
    low: ArrayLike,
    high: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Draw the antithetic candidate actions for every problem.

    ``means`` has shape ``(B, A)``; ``low``/``high`` have shape ``(A,)``. The
    generator is consumed in the order problem ``b``, pair ``p``
    (``eps = rng.standard_normal(A)``), exactly as the sequential estimator.
    Returns ``(eps, plus, minus)``, each float64 of shape ``(B, P, A)``, with
    ``plus = clip(mean + sigma * eps, low, high)`` and
    ``minus = clip(mean - sigma * eps, low, high)``. Inputs are not modified.
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
    sigma = float(sigma)
    pairs = int(n_pairs)

    batch, dim = mu.shape
    eps = np.zeros((batch, pairs, dim), dtype=np.float64)
    plus = np.zeros_like(eps)
    minus = np.zeros_like(eps)
    for b in range(batch):
        for p in range(pairs):
            eps[b, p] = rng.standard_normal(dim)
            plus[b, p] = np.clip(mu[b] + sigma * eps[b, p], lo, hi)
            minus[b, p] = np.clip(mu[b] - sigma * eps[b, p], lo, hi)
    return eps, plus, minus


def antithetic_gradients_from_costs(
    eps: ArrayLike,
    plus_costs: ArrayLike,
    minus_costs: ArrayLike,
    *,
    sigma: float,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Compute the antithetic gradient estimate from candidate costs.

    ``eps`` has shape ``(B, P, A)``; ``plus_costs`` and ``minus_costs`` have
    shape ``(B, P)``. Returns the same ``(gradients, info)`` as
    :func:`antithetic_action_gradients`, accumulating in the same order.
    """

    e = np.asarray(eps, dtype=np.float64)
    plus = np.asarray(plus_costs, dtype=np.float64)
    minus = np.asarray(minus_costs, dtype=np.float64)
    if e.ndim != 3 or plus.shape != e.shape[:2] or minus.shape != e.shape[:2]:
        raise ValueError("eps must have shape (B, P, A) and the costs shape (B, P).")
    sigma = float(sigma)
    batch, pairs, dim = e.shape
    gradients = np.zeros((batch, dim), dtype=np.float64)
    total = 0.0
    count = 0
    for b in range(batch):
        for p in range(pairs):
            plus_cost = float(plus[b, p])
            minus_cost = float(minus[b, p])
            if not (np.isfinite(plus_cost) and np.isfinite(minus_cost)):
                raise ValueError("cost_fn returned a non-finite value.")
            gradients[b] += (plus_cost - minus_cost) * e[b, p]
            total += plus_cost + minus_cost
            count += 2
        gradients[b] /= 2.0 * sigma * pairs
    info = {
        "mean_cost_ratio": total / count if count else float("nan"),
        "mean_grad_norm": float(np.mean(np.linalg.norm(gradients, axis=1))) if batch else 0.0,
        "n_evaluations": float(count),
    }
    return gradients, info


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
    Inputs are not modified. This is :func:`antithetic_candidates` followed by
    sequential ``cost_fn`` calls and :func:`antithetic_gradients_from_costs`.
    """

    if not callable(cost_fn):
        raise TypeError("cost_fn must be callable.")
    eps, plus, minus = antithetic_candidates(
        means, sigma=sigma, n_pairs=n_pairs, rng=rng, low=low, high=high
    )
    batch, pairs, _ = eps.shape
    plus_costs = np.zeros((batch, pairs), dtype=np.float64)
    minus_costs = np.zeros((batch, pairs), dtype=np.float64)
    for b in range(batch):
        for p in range(pairs):
            plus_costs[b, p] = float(cost_fn(b, plus[b, p].copy()))
            minus_costs[b, p] = float(cost_fn(b, minus[b, p].copy()))
    return antithetic_gradients_from_costs(eps, plus_costs, minus_costs, sigma=sigma)


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


__all__ = [
    "antithetic_action_gradients",
    "antithetic_candidates",
    "antithetic_gradients_from_costs",
    "finetune_step",
    "sigma_at",
]
