"""Coefficient convention helpers: full (ascending power) <-> discriminant (descending)."""
from __future__ import annotations

import torch


def full_coeffs_ascending_to_descending_ri(a_ri: torch.Tensor) -> torch.Tensor:
    """Convert full polynomial coefficients from ascending to descending power order.

    Bezier env uses ascending power: [a_0, a_1, ..., a_degree] (index 0 = constant).
    discriminant_univariate_logabs expects descending: [a_degree, ..., a_0] (index 0 = leading).

    Args:
        a_ri: (..., degree+1, 2) in (Re, Im), ascending power order.

    Returns:
        (..., degree+1, 2) in descending power order (index 0 = leading coefficient).
    """
    return a_ri.flip(dims=(-2,))


def full_coeffs_descending_to_ascending_ri(a_ri: torch.Tensor) -> torch.Tensor:
    """Convert full polynomial coefficients from descending to ascending power order."""
    return a_ri.flip(dims=(-2,))
