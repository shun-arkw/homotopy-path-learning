"""Pure helpers to compare policies on a common problem set.

All inputs are one-dimensional ``float64`` arrays of per-problem mean tracking
costs, aligned by problem (same target seeds, same order). ``linear`` holds the
linear-path costs, ``oracle`` the oracle costs, and a method's array its costs.

Definitions (per problem set):

* ``delta_j_percent = (mean(cost) - mean(linear)) / mean(linear) * 100``
* ``positive_rate = mean(cost < linear)``
* ``captured_fraction = (mean(linear) - mean(cost)) / (mean(linear) - mean(oracle))``
  (NaN when the oracle does not improve on the linear path on average)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]

METRICS = ("delta_j_percent", "positive_rate", "captured_fraction")


def _vector(name: str, values: npt.ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty 1-D array, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _same_length(**arrays: FloatArray | None) -> int:
    sizes = {name: a.shape[-1] for name, a in arrays.items() if a is not None}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"arrays must have the same length, got {sizes}.")
    return next(iter(sizes.values()))


def delta_j_percent(cost: npt.ArrayLike, linear: npt.ArrayLike) -> float:
    """Relative change of the mean cost against the linear path, in percent."""

    c, lin = _vector("cost", cost), _vector("linear", linear)
    _same_length(cost=c, linear=lin)
    lin_mean = float(lin.mean())
    if lin_mean <= 0.0:
        raise ValueError("mean linear cost must be positive.")
    return (float(c.mean()) - lin_mean) / lin_mean * 100.0


def positive_rate(cost: npt.ArrayLike, linear: npt.ArrayLike) -> float:
    """Fraction of problems with ``cost < linear``."""

    c, lin = _vector("cost", cost), _vector("linear", linear)
    _same_length(cost=c, linear=lin)
    return float(np.mean(c < lin))


def captured_fraction(cost: npt.ArrayLike, linear: npt.ArrayLike, oracle: npt.ArrayLike) -> float:
    """Share of the oracle's mean improvement obtained by ``cost`` (NaN if no headroom)."""

    c, lin, orc = _vector("cost", cost), _vector("linear", linear), _vector("oracle", oracle)
    _same_length(cost=c, linear=lin, oracle=orc)
    headroom = float(lin.mean()) - float(orc.mean())
    if headroom <= 0.0:
        return float("nan")
    return (float(lin.mean()) - float(c.mean())) / headroom


def metrics(
    cost: npt.ArrayLike, linear: npt.ArrayLike, oracle: npt.ArrayLike | None = None
) -> dict[str, float]:
    """Return ``delta_j_percent``, ``positive_rate`` and ``captured_fraction`` (NaN without oracle)."""

    return {
        "delta_j_percent": delta_j_percent(cost, linear),
        "positive_rate": positive_rate(cost, linear),
        "captured_fraction": float("nan") if oracle is None else captured_fraction(cost, linear, oracle),
    }


def average_group(costs: Sequence[npt.ArrayLike]) -> FloatArray:
    """Average per-problem costs over group members (e.g. seeds); result has shape ``(N,)``."""

    if len(costs) == 0:
        raise ValueError("a group must contain at least one run.")
    stacked = np.stack([_vector("group member", c) for c in costs])
    return stacked.mean(axis=0)


def _resampled_metrics(
    cost: FloatArray, linear: FloatArray, oracle: FloatArray | None, indices: npt.NDArray[np.int64]
) -> dict[str, FloatArray]:
    """Metrics for each row of ``indices`` (shape ``(B, N)``), vectorized."""

    c, lin = cost[indices], linear[indices]
    lin_mean = lin.mean(axis=1)
    c_mean = c.mean(axis=1)
    out = {
        "delta_j_percent": (c_mean - lin_mean) / lin_mean * 100.0,
        "positive_rate": (c < lin).mean(axis=1),
    }
    if oracle is None:
        out["captured_fraction"] = np.full(indices.shape[0], np.nan)
    else:
        headroom = lin_mean - oracle[indices].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["captured_fraction"] = np.where(headroom > 0.0, (lin_mean - c_mean) / headroom, np.nan)
    return out


def paired_bootstrap_difference(
    cost_a: npt.ArrayLike,
    cost_b: npt.ArrayLike,
    linear: npt.ArrayLike,
    oracle: npt.ArrayLike | None = None,
    *,
    n_boot: int = 2000,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Paired percentile-bootstrap CI of ``metric(B) - metric(A)`` over problems.

    Problems are resampled with replacement (same indices for A, B, linear and
    oracle). Returns ``{metric: {"a", "b", "diff", "ci_low", "ci_high"}}`` for
    every name in :data:`METRICS`; ``delta_j_percent`` differences are in
    percentage points. Metrics that are NaN on some resample (no oracle
    headroom) use ``nanpercentile``; without an oracle the captured fraction
    entries are NaN.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    if isinstance(n_boot, bool) or int(n_boot) <= 0:
        raise ValueError("n_boot must be a positive integer.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    a, b, lin = _vector("cost_a", cost_a), _vector("cost_b", cost_b), _vector("linear", linear)
    orc = None if oracle is None else _vector("oracle", oracle)
    n = _same_length(cost_a=a, cost_b=b, linear=lin, oracle=orc)
    indices = rng.integers(0, n, size=(int(n_boot), n))
    boot_a = _resampled_metrics(a, lin, orc, indices)
    boot_b = _resampled_metrics(b, lin, orc, indices)
    point_a, point_b = metrics(a, lin, orc), metrics(b, lin, orc)
    tail = (1.0 - confidence) / 2.0 * 100.0
    result: dict[str, dict[str, float]] = {}
    for name in METRICS:
        diffs = boot_b[name] - boot_a[name]
        if np.all(np.isnan(diffs)):
            low = high = float("nan")
        else:
            low, high = (float(v) for v in np.nanpercentile(diffs, [tail, 100.0 - tail]))
        result[name] = {
            "a": point_a[name],
            "b": point_b[name],
            "diff": point_b[name] - point_a[name],
            "ci_low": low,
            "ci_high": high,
        }
    return result


__all__ = [
    "METRICS",
    "average_group",
    "captured_fraction",
    "delta_j_percent",
    "metrics",
    "paired_bootstrap_difference",
    "positive_rate",
]
