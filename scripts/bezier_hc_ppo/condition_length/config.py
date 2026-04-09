"""Configuration for condition length evaluation (Bezier / linear paths)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConditionLengthConfig:
    """Config for condition length numeric evaluation.

    - samples_per_segment: Number of sample points M per path (or per segment).
    - eps_soft, delta_soft: Softabs and epsilon for stable 1/|Disc|^{1/n} weight.
    - disc_eps, lead_eps: Passed to discriminant_univariate_logabs.
    """

    samples_per_segment: int = 16
    eps_soft: float = 1e-12
    delta_soft: float = 1e-12
    disc_eps: float = 0.0
    lead_eps: float = 1e-24
    disc_backend: str = "complex"
