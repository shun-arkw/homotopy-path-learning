"""Coefficient generation for fixed polynomial system supports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from homotopy_path_learning.systems.spec import PolynomialSystemSpec


ComplexVector = npt.NDArray[np.complex128]


class CoefficientSampler(Protocol):
    """Protocol for reproducible target coefficient samplers."""

    def sample(self, rng: np.random.Generator, spec: PolynomialSystemSpec) -> ComplexVector:
        """Sample a full complex coefficient vector with shape ``(M,)``."""


@dataclass(frozen=True)
class ComplexUniformSampler:
    """Sample target coefficients from independent complex uniform entries.

    For each non-leading coefficient, the real and imaginary parts are sampled
    independently from ``Uniform[-bound, bound]`` using the provided
    ``numpy.random.Generator``. Leading coefficients are then fixed to exactly
    ``1 + 0j``. No global NumPy random state is used.
    """

    bound: float = 1.0

    def __post_init__(self) -> None:
        bound = float(self.bound)
        if not np.isfinite(bound):
            raise ValueError(f"bound must be finite, got {self.bound!r}.")
        if bound < 0.0:
            raise ValueError(f"bound must be nonnegative, got {bound}.")
        object.__setattr__(self, "bound", bound)

    def sample(self, rng: np.random.Generator, spec: PolynomialSystemSpec) -> ComplexVector:
        """Return a target coefficient vector with leading entries fixed to one."""

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be an instance of numpy.random.Generator.")

        coeffs = np.empty(spec.n_coeffs, dtype=np.complex128)
        free_mask = spec.free_coefficient_mask
        n_free = int(np.count_nonzero(free_mask))
        real = rng.uniform(-self.bound, self.bound, size=n_free)
        imag = rng.uniform(-self.bound, self.bound, size=n_free)
        coeffs[free_mask] = real + 1j * imag
        coeffs[spec.leading_indices] = 1.0 + 0.0j
        return coeffs


def start_system_coefficients(spec: PolynomialSystemSpec) -> ComplexVector:
    """Return coefficients for the Pham start system ``G_i = x_i ** d_i - 1``.

    The vector has shape ``(M,)`` and dtype ``complex128``. Leading indices are
    ``1 + 0j``, constant indices are ``-1 + 0j``, and all other coefficients are
    zero.
    """

    coeffs = np.zeros(spec.n_coeffs, dtype=np.complex128)
    coeffs[spec.leading_indices] = 1.0 + 0.0j
    coeffs[spec.constant_indices] = -1.0 + 0.0j
    return coeffs


pham_start_coefficients = start_system_coefficients


def sample_target_coefficients(
    spec: PolynomialSystemSpec,
    rng: np.random.Generator,
    *,
    bound: float = 1.0,
) -> ComplexVector:
    """Sample target coefficients using ``ComplexUniformSampler(bound)``."""

    return ComplexUniformSampler(bound=bound).sample(rng, spec)


__all__ = [
    "CoefficientSampler",
    "ComplexUniformSampler",
    "ComplexVector",
    "pham_start_coefficients",
    "sample_target_coefficients",
    "start_system_coefficients",
]
