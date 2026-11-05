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


def make_complex_subspace_basis(
    rng: np.random.Generator,
    *,
    n_free: int,
    subspace_dim: int,
) -> npt.NDArray[np.complex128]:
    """Return a random complex matrix ``W`` with orthonormal columns, shape ``(n_free, subspace_dim)``.

    ``W`` is the Q factor of the reduced QR decomposition of a complex Gaussian matrix
    (real and imaginary parts drawn from ``rng.standard_normal`` in that order). Column ``l``
    is multiplied by the phase that makes ``R[l, l]`` real and positive, which fixes ``W``
    uniquely for a given ``rng`` state. ``W^H W = I``; with ``subspace_dim == n_free`` it is a
    random unitary matrix. Rows follow the order of ``spec.free_coefficient_mask``.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator.")
    for name, value in (("n_free", n_free), ("subspace_dim", subspace_dim)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    n, k = int(n_free), int(subspace_dim)
    if k > n:
        raise ValueError(f"subspace_dim={k} cannot exceed n_free={n}.")
    real = rng.standard_normal(size=(n, k))
    imag = rng.standard_normal(size=(n, k))
    q, r = np.linalg.qr(real + 1j * imag, mode="reduced")
    diag = np.diag(r)
    phases = np.where(np.abs(diag) > 0.0, diag / np.where(np.abs(diag) > 0.0, np.abs(diag), 1.0), 1.0)
    return (q * phases[None, :]).astype(np.complex128, copy=False)


@dataclass(frozen=True, eq=False)
class GaussianSubspaceSampler:
    """Sample target coefficients from a linear family ``c_free = W u``.

    ``basis`` is ``W``, complex array of shape ``(n_free, k)`` with orthonormal columns (rows in
    ``spec.free_coefficient_mask`` order). ``u`` is complex Gaussian ``CN(0, s^2 I_k)`` with
    ``s^2 = (n_free / k) * (2 bound^2 / 3)``: the real and imaginary parts of each entry are
    independent normals with variance ``s^2 / 2`` drawn from ``rng.standard_normal`` (real parts
    first). The mean of ``E|c_j|^2`` over the free coefficients is then ``2 bound^2 / 3``, the
    value of :class:`ComplexUniformSampler`; individual coefficients have variance proportional to
    the squared norm of their row of ``W``. With ``k == n_free`` and a unitary ``W`` the free
    coefficients are independent ``CN(0, 2 bound^2 / 3)``. Leading coefficients are ``1 + 0j``.
    """

    basis: npt.NDArray[np.complex128]
    bound: float = 1.0

    def __post_init__(self) -> None:
        basis = np.array(self.basis, dtype=np.complex128, copy=True)
        if basis.ndim != 2 or basis.shape[1] < 1 or basis.shape[1] > basis.shape[0]:
            raise ValueError(f"basis must have shape (n_free, k) with 1 <= k <= n_free, got {basis.shape}.")
        if not np.all(np.isfinite(basis.real)) or not np.all(np.isfinite(basis.imag)):
            raise ValueError("basis must not contain NaN or Inf.")
        if not np.allclose(basis.conj().T @ basis, np.eye(basis.shape[1]), rtol=0.0, atol=1e-10):
            raise ValueError("basis columns must be orthonormal.")
        bound = float(self.bound)
        if not np.isfinite(bound) or bound < 0.0:
            raise ValueError(f"bound must be finite and nonnegative, got {self.bound!r}.")
        basis.setflags(write=False)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "bound", bound)

    @classmethod
    def from_spec(
        cls,
        spec: PolynomialSystemSpec,
        *,
        subspace_dim: int,
        family_seed: int,
        bound: float = 1.0,
    ) -> "GaussianSubspaceSampler":
        """Build ``W`` with :func:`make_complex_subspace_basis` from ``default_rng(family_seed)``."""

        n_free = int(np.count_nonzero(spec.free_coefficient_mask))
        basis = make_complex_subspace_basis(
            np.random.default_rng(int(family_seed)), n_free=n_free, subspace_dim=subspace_dim
        )
        return cls(basis=basis, bound=bound)

    @property
    def subspace_dim(self) -> int:
        """Complex dimension ``k`` of the family."""

        return int(self.basis.shape[1])

    def sample(self, rng: np.random.Generator, spec: PolynomialSystemSpec) -> ComplexVector:
        """Return a target coefficient vector of shape ``(M,)`` with leading entries fixed to one."""

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be an instance of numpy.random.Generator.")
        free_mask = spec.free_coefficient_mask
        n_free = int(np.count_nonzero(free_mask))
        if n_free != self.basis.shape[0]:
            raise ValueError(f"basis has {self.basis.shape[0]} rows but the spec has {n_free} free coefficients.")
        k = self.subspace_dim
        std = np.sqrt((n_free / k) * (2.0 * self.bound**2 / 3.0) / 2.0)
        real = rng.standard_normal(size=k)
        imag = rng.standard_normal(size=k)
        coeffs = np.empty(spec.n_coeffs, dtype=np.complex128)
        coeffs[free_mask] = self.basis @ (std * (real + 1j * imag))
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
    "GaussianSubspaceSampler",
    "make_complex_subspace_basis",
    "pham_start_coefficients",
    "sample_target_coefficients",
    "start_system_coefficients",
]
