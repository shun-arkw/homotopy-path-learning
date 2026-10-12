"""Supervised behavior cloning of oracle actions into the PPO actor network."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence
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
    weights are restored into the agent. ``n_augmentations`` is the number of
    symmetry elements used for augmentation (0: none) and
    ``clipped_label_fraction`` is the fraction of augmented label entries that
    had to be clipped into the action bounds over the whole run.
    """

    history: list[dict[str, float]]
    best_epoch: int
    best_val_loss: float
    n_train: int
    n_val: int
    n_augmentations: int = 0
    clipped_label_fraction: float = 0.0


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
    augmentations: Sequence[tuple[ArrayLike, ArrayLike]] | None = None,
) -> tuple[BehaviorCloningResult, torch.optim.Optimizer]:
    """Fit ``agent.actor_mean`` to oracle actions with an MSE loss.

    ``observations`` has shape ``(N, obs_dim)`` and ``actions`` has shape
    ``(N, action_dim)`` (floating point; copied, never mutated). Labels must lie
    inside the agent's action bounds. The data are split into train/validation
    sets with ``default_rng(seed).permutation``; minibatches are shuffled with a
    ``torch.Generator`` seeded by ``seed``. Only ``actor_mean`` parameters are
    optimized (Adam). The weights of the best validation epoch are restored at
    the end. Returns the result and the optimizer (for ``save_checkpoint``).

    ``augmentations`` is an optional sequence of pairs ``(M_obs, M_act)`` with
    shapes ``(obs_dim, obs_dim)`` and ``(action_dim, action_dim)`` (symmetry
    group elements, identity included). Every training sample of every
    minibatch draws an element independently (from the seeded torch generator)
    and becomes ``obs' = M_obs @ obs``, ``act' = clip(M_act @ act, low, high)``.
    Validation and the reported full train loss use the original data.
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

    aug_obs = aug_act = None
    if augmentations is not None:
        if len(augmentations) == 0:
            raise ValueError("augmentations must not be empty when given.")
        obs_mats, act_mats = [], []
        for index, (m_obs, m_act) in enumerate(augmentations):
            m_obs = np.array(m_obs, dtype=np.float32, copy=True)
            m_act = np.array(m_act, dtype=np.float32, copy=True)
            if m_obs.shape != (obs_dim, obs_dim):
                raise ValueError(f"augmentations[{index}] observation matrix must have shape ({obs_dim}, {obs_dim}), got {m_obs.shape}.")
            if m_act.shape != (action_dim, action_dim):
                raise ValueError(f"augmentations[{index}] action matrix must have shape ({action_dim}, {action_dim}), got {m_act.shape}.")
            if not (np.all(np.isfinite(m_obs)) and np.all(np.isfinite(m_act))):
                raise ValueError(f"augmentations[{index}] must be finite.")
            obs_mats.append(m_obs)
            act_mats.append(m_act)
        aug_obs = torch.as_tensor(np.stack(obs_mats), device=device)
        aug_act = torch.as_tensor(np.stack(act_mats), device=device)
    n_clipped = 0
    n_entries = 0

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
            batch_obs, batch_act = train_obs[idx], train_act[idx]
            if aug_obs is not None:
                choice = torch.randint(aug_obs.shape[0], (batch_obs.shape[0],), generator=generator).to(batch_obs.device)
                batch_obs = torch.einsum("bij,bj->bi", aug_obs[choice], batch_obs)
                raw = torch.einsum("bij,bj->bi", aug_act[choice], batch_act)
                low_t, high_t = agent.distribution.low, agent.distribution.high
                n_clipped += int(((raw < low_t - 1e-6) | (raw > high_t + 1e-6)).sum().item())
                n_entries += int(raw.numel())
                batch_act = torch.maximum(torch.minimum(raw, high_t), low_t)
            loss = F.mse_loss(predicted_actions(agent, batch_obs), batch_act)
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
        n_augmentations=0 if aug_obs is None else int(aug_obs.shape[0]),
        clipped_label_fraction=float(n_clipped / n_entries) if n_entries else 0.0,
    )
    return result, optimizer


__all__ = ["BehaviorCloningResult", "predicted_actions", "train_behavior_cloning"]
