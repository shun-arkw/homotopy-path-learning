"""Supervised behavior cloning of oracle actions into the PPO actor network."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
import torch
from torch.nn import functional as F

from homotopy_path_learning.rl.networks import ActorCritic


@dataclass(frozen=True)
class BehaviorCloningResult:
    """Outcome of :func:`train_behavior_cloning`.

    ``history`` has one dict per epoch with keys ``epoch`` (1-based),
    ``train_loss`` and ``val_loss`` (MSE over all action coordinates).
    ``best_epoch`` is the 1-based epoch with the lowest validation loss; its
    weights are restored into the agent.
    """

    history: list[dict[str, float]]
    best_epoch: int
    best_val_loss: float
    n_train: int
    n_val: int


def predicted_actions(agent: ActorCritic, observations: torch.Tensor) -> torch.Tensor:
    """Return the differentiable bounded action ``squash(actor_mean(obs))``.

    ``observations`` has shape ``(batch, obs_dim)`` (or ``(obs_dim,)``); the
    result has shape ``(batch, action_dim)`` and dtype ``float32``.
    """

    mean = agent.actor_mean(agent._as_batch(observations))
    return agent.distribution.action_from_pre_tanh(mean)


def _positive(name: str, value: object, kind: type) -> None:
    if isinstance(value, bool) or not isinstance(value, kind) and not (
        kind is float and isinstance(value, int)
    ):
        raise TypeError(f"{name} must be of type {kind.__name__}, got {type(value).__name__}.")
    if value <= 0:  # type: ignore[operator]
        raise ValueError(f"{name} must be positive, got {value!r}.")


def train_behavior_cloning(
    agent: ActorCritic,
    observations: ArrayLike,
    actions: ArrayLike,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    val_fraction: float,
    seed: int,
    device: str | torch.device = "cpu",
) -> tuple[BehaviorCloningResult, torch.optim.Optimizer]:
    """Fit ``agent.actor_mean`` to oracle actions with an MSE loss.

    ``observations`` has shape ``(N, obs_dim)`` and ``actions`` has shape
    ``(N, action_dim)`` (floating point; copied, never mutated). Labels must lie
    inside the agent's action bounds. The data are split into train/validation
    sets with ``default_rng(seed).permutation``; minibatches are shuffled with a
    ``torch.Generator`` seeded by ``seed``. Only ``actor_mean`` parameters are
    optimized (Adam). The weights of the best validation epoch are restored at
    the end. Returns the result and the optimizer (for ``save_checkpoint``).
    """

    _positive("epochs", epochs, int)
    _positive("batch_size", batch_size, int)
    _positive("learning_rate", learning_rate, float)
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError(f"val_fraction must lie in (0, 1), got {val_fraction!r}.")
    obs = np.array(observations, dtype=np.float32, copy=True)
    act = np.array(actions, dtype=np.float32, copy=True)
    if obs.ndim != 2 or act.ndim != 2:
        raise ValueError(f"observations and actions must be 2-D, got {obs.shape} and {act.shape}.")
    if obs.shape[0] != act.shape[0]:
        raise ValueError(f"observations and actions differ in length: {obs.shape[0]} vs {act.shape[0]}.")
    n = int(obs.shape[0])
    if n < 2:
        raise ValueError(f"at least 2 samples are required, got {n}.")
    obs_dim = int(np.prod(agent.observation_shape))
    action_dim = int(np.prod(agent.action_shape))
    if obs.shape[1] != obs_dim:
        raise ValueError(f"observation dimension must be {obs_dim}, got {obs.shape[1]}.")
    if act.shape[1] != action_dim:
        raise ValueError(f"action dimension must be {action_dim}, got {act.shape[1]}.")
    if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(act))):
        raise ValueError("observations and actions must be finite.")
    low = agent.distribution.low.detach().cpu().numpy()
    high = agent.distribution.high.detach().cpu().numpy()
    if np.any(act < low - 1.0e-6) or np.any(act > high + 1.0e-6):
        raise ValueError("action labels lie outside the action bounds of the agent.")
    n_val = int(round(n * float(val_fraction)))
    n_val = min(max(n_val, 1), n - 1)
    n_train = n - n_val
    if n_val < 1 or n_train < 1:
        raise ValueError("val_fraction leaves an empty train or validation split.")

    order = np.random.default_rng(seed).permutation(n)
    train_idx, val_idx = order[:n_train], order[n_train:]
    obs_t = torch.as_tensor(obs, device=device)
    act_t = torch.as_tensor(act, device=device)
    train_obs, train_act = obs_t[train_idx], act_t[train_idx]
    val_obs, val_act = obs_t[val_idx], act_t[val_idx]

    optimizer = torch.optim.Adam(agent.actor_mean.parameters(), lr=float(learning_rate))
    generator = torch.Generator().manual_seed(int(seed))
    history: list[dict[str, float]] = []
    best_state = deepcopy(agent.actor_mean.state_dict())
    best_val = float("inf")
    best_epoch = 0

    def full_loss(x: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            return float(F.mse_loss(predicted_actions(agent, x), y).item())

    for epoch in range(1, int(epochs) + 1):
        agent.train()
        perm = torch.randperm(n_train, generator=generator)
        for start in range(0, n_train, int(batch_size)):
            idx = perm[start : start + int(batch_size)].to(train_obs.device)
            loss = F.mse_loss(predicted_actions(agent, train_obs[idx]), train_act[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        agent.eval()
        train_loss = full_loss(train_obs, train_act)
        val_loss = full_loss(val_obs, val_act)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = deepcopy(agent.actor_mean.state_dict())

    agent.actor_mean.load_state_dict(best_state)
    agent.eval()
    result = BehaviorCloningResult(
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        n_train=n_train,
        n_val=n_val,
    )
    return result, optimizer


__all__ = ["BehaviorCloningResult", "predicted_actions", "train_behavior_cloning"]
