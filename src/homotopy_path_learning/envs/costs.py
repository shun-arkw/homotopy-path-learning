"""Tracking cost helpers for one-step Gymnasium environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from homotopy_path_learning.backends.types import TrackingResult


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite float, got {value!r}.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite float, got {value!r}.") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return normalized


@dataclass(frozen=True)
class TrackingCost:
    """Mean and per-path tracking costs.

    Attributes:
        mean: Mean cost over all paths as a finite Python ``float``.
        per_path: Python-owned array with shape ``(n_paths,)`` and dtype
            ``float64``.
    """

    mean: float
    per_path: npt.NDArray[np.float64]


def tracking_cost(
    result: TrackingResult,
    *,
    reject_weight: float,
    failure_penalty: float,
) -> TrackingCost:
    """Compute the mean tracking cost from path-level tracker diagnostics.

    Successful path ``l`` receives cost
    ``accepted_l + reject_weight * rejected_l``. Failed paths receive
    ``failure_penalty``. The returned per-path array is independent of the
    input ``TrackingResult`` arrays.
    """

    if not isinstance(result, TrackingResult):
        raise TypeError("result must be a TrackingResult.")
    rho = _finite_float("reject_weight", reject_weight)
    penalty = _finite_float("failure_penalty", failure_penalty)
    if rho < 0.0:
        raise ValueError(f"reject_weight must be nonnegative, got {rho}.")
    if penalty <= 0.0:
        raise ValueError(f"failure_penalty must be positive, got {penalty}.")

    accepted = np.array(result.per_path_accepted_steps, dtype=np.float64, copy=True)
    rejected = np.array(result.per_path_rejected_steps, dtype=np.float64, copy=True)
    path_success = np.array(result.path_success, dtype=np.bool_, copy=True)
    per_path = np.where(path_success, accepted + rho * rejected, penalty).astype(
        np.float64,
        copy=False,
    )

    if not np.all(np.isfinite(per_path)):
        raise ValueError("per-path tracking costs must be finite.")
    mean = float(np.mean(per_path))
    if not np.isfinite(mean):
        raise ValueError("mean tracking cost must be finite.")

    return TrackingCost(mean=mean, per_path=np.array(per_path, dtype=np.float64, copy=True))


def reward_from_costs(
    *,
    linear_cost: float,
    bezier_cost: float,
    reward_scale: float,
) -> float:
    """Return ``reward_scale * (linear_cost - bezier_cost)`` as a finite float."""

    linear = _finite_float("linear_cost", linear_cost)
    bezier = _finite_float("bezier_cost", bezier_cost)
    scale = _finite_float("reward_scale", reward_scale)
    if scale <= 0.0:
        raise ValueError(f"reward_scale must be positive, got {scale}.")
    reward = float(scale * (linear - bezier))
    if not np.isfinite(reward):
        raise ValueError("reward must be finite.")
    return reward


__all__ = ["TrackingCost", "reward_from_costs", "tracking_cost"]
