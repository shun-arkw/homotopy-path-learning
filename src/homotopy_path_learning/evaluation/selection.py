"""Selection of hard target seeds from linear-path screening results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class HardSeedSelection:
    """Result of :func:`select_hard_seeds`.

    ``splits`` maps each split name (in the requested order) to its ordered list
    of target seeds. ``threshold`` is the calibration quantile of the linear cost.
    """

    threshold: float
    calibration_count: int
    calibration_seed_range: tuple[int, int]
    n_candidates: int
    n_eligible: int
    splits: dict[str, list[int]]


def parse_split(text: str) -> tuple[str, int]:
    """Parse ``name:count`` (name is a plain file stem, count a positive integer)."""

    name, separator, count = text.partition(":")
    if not separator or not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"split must have the form name:count, got {text!r}.")
    try:
        number = int(count)
    except ValueError:
        raise ValueError(f"split count must be an integer, got {text!r}.") from None
    if number <= 0:
        raise ValueError(f"split count must be positive, got {text!r}.")
    return name, number


def select_hard_seeds(
    candidates: Sequence[Mapping[str, Any]],
    *,
    calibration: int,
    quantile: float,
    splits: Sequence[tuple[str, int]],
) -> HardSeedSelection:
    """Pick hard seeds by the linear-path cost.

    ``candidates`` are rows with ``target_seed``, ``linear_cost`` and
    ``linear_success`` (values may be strings from a CSV). They are sorted by
    seed (duplicate seeds are rejected). The threshold is the ``quantile`` of
    ``linear_cost`` over the first ``calibration`` candidates (NumPy's default
    linear interpolation). Of the remaining candidates, those with
    ``linear_success`` and ``linear_cost >= threshold`` are kept in ascending
    seed order and assigned to ``splits`` in the given order from the front.
    Raises ``ValueError`` when there are not enough eligible candidates.
    """

    if int(calibration) <= 0:
        raise ValueError("calibration must be positive.")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0, 1].")
    if not splits:
        raise ValueError("at least one split is required.")
    names = [name for name, _ in splits]
    if len(set(names)) != len(names):
        raise ValueError("split names must be unique.")
    parsed = sorted(
        (
            (int(row["target_seed"]), float(row["linear_cost"]), bool(int(row["linear_success"])))
            for row in candidates
        ),
        key=lambda item: item[0],
    )
    seeds = [item[0] for item in parsed]
    if len(set(seeds)) != len(seeds):
        raise ValueError("candidate seeds must be unique.")
    if len(parsed) <= int(calibration):
        raise ValueError(f"need more than {calibration} candidates, got {len(parsed)}.")
    head = parsed[: int(calibration)]
    threshold = float(np.quantile(np.array([item[1] for item in head], dtype=np.float64), float(quantile)))
    eligible = [seed for seed, cost, success in parsed[int(calibration) :] if success and cost >= threshold]
    needed = sum(count for _, count in splits)
    if len(eligible) < needed:
        raise ValueError(f"only {len(eligible)} eligible seeds after calibration; {needed} are needed.")
    assigned: dict[str, list[int]] = {}
    position = 0
    for name, count in splits:
        assigned[name] = eligible[position : position + count]
        position += count
    return HardSeedSelection(
        threshold=threshold,
        calibration_count=int(calibration),
        calibration_seed_range=(head[0][0], head[-1][0]),
        n_candidates=len(parsed),
        n_eligible=len(eligible),
        splits=assigned,
    )


__all__ = ["HardSeedSelection", "parse_split", "select_hard_seeds"]
