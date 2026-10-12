"""Finite symmetry group of Pham-type polynomial system families.

A symmetry element ``g = (sigma, k, conj)`` acts on coefficient vectors of a
fixed support (see ``docs/design/mathematics/symmetry.md``). The action is
applied in the order: complex conjugation, scaling ``x_j -> omega_j x_j`` with
``omega_j = exp(2 pi i k_j / d_j)``, then simultaneous permutation of variables
and equations (old index ``j`` becomes ``sigma[j]``).

Coefficient vectors follow :class:`PolynomialSystemSpec`: shape ``(M,)`` (or
``(K, M)`` for stacks such as Bezier control points), dtype ``complex128``.
The free real layout ``[Re c_free, Im c_free]`` in
``np.flatnonzero(spec.free_coefficient_mask)`` order is also the layout of the
environment observation and of the rows of the latent basis ``U``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools

import numpy as np
import numpy.typing as npt

from homotopy_path_learning.systems.spec import PolynomialSystemSpec


@dataclass(frozen=True)
class SymmetryElement:
    """Element ``(sigma, k, conj)`` of the symmetry group of a system family.

    ``permutation[j]`` is the new index of old variable/equation ``j``,
    ``root_exponents[j]`` is ``k_j`` (``0 <= k_j < d_j``) and ``conjugate``
    selects complex conjugation of all coefficients.
    """

    permutation: tuple[int, ...]
    root_exponents: tuple[int, ...]
    conjugate: bool

    @property
    def is_identity(self) -> bool:
        """True for the identity element."""

        return (
            self.permutation == tuple(range(len(self.permutation)))
            and not any(self.root_exponents)
            and not self.conjugate
        )

    @property
    def label(self) -> str:
        """Short human-readable description."""

        text = f"perm={self.permutation} k={self.root_exponents}"
        return text + (" conj" if self.conjugate else "")

    @property
    def kind(self) -> str:
        """One of ``identity``, ``perm-only``, ``scale-only``, ``conj-only``, ``mixed``."""

        parts = (
            self.permutation != tuple(range(len(self.permutation))),
            any(self.root_exponents),
            self.conjugate,
        )
        if sum(parts) == 0:
            return "identity"
        if sum(parts) > 1:
            return "mixed"
        return ("perm-only", "scale-only", "conj-only")[parts.index(True)]


def _exact_unit_phase(fraction: Fraction) -> complex:
    """Return ``exp(2 pi i fraction)``, exact for multiples of a quarter turn."""

    frac = fraction % 1
    exact = {Fraction(0): 1.0 + 0.0j, Fraction(1, 4): 1j, Fraction(1, 2): -1.0 + 0.0j, Fraction(3, 4): -1j}
    if frac in exact:
        return exact[frac]
    return complex(np.exp(2j * np.pi * float(frac)))


def _check_element(spec: PolynomialSystemSpec, element: SymmetryElement) -> None:
    n = spec.n_vars
    if not isinstance(element, SymmetryElement):
        raise TypeError("element must be a SymmetryElement.")
    if sorted(element.permutation) != list(range(n)):
        raise ValueError(f"permutation must be a permutation of range({n}), got {element.permutation}.")
    if len(element.root_exponents) != n:
        raise ValueError(f"root_exponents must have length {n}, got {len(element.root_exponents)}.")
    for j, (k, d) in enumerate(zip(element.root_exponents, spec.degrees)):
        if not 0 <= int(k) < d:
            raise ValueError(f"root_exponents[{j}]={k} must satisfy 0 <= k < d_j={d}.")


def _target_indices(spec: PolynomialSystemSpec, permutation: tuple[int, ...]) -> npt.NDArray[np.int64] | None:
    """Return the coefficient index map of a variable/equation permutation, or None if incompatible."""

    n = spec.n_vars
    if any(spec.degrees[permutation[i]] != spec.degrees[i] for i in range(n)):
        return None
    lookup = [
        {tuple(int(v) for v in row): int(spec.offsets[i]) + q for q, row in enumerate(spec.exponents[spec.offsets[i] : spec.offsets[i + 1]])}
        for i in range(n)
    ]
    target = np.zeros(spec.n_coeffs, dtype=np.int64)
    for i in range(n):
        for q in range(int(spec.offsets[i]), int(spec.offsets[i + 1])):
            alpha = spec.exponents[q]
            beta = [0] * n
            for j in range(n):
                beta[permutation[j]] = int(alpha[j])
            index = lookup[permutation[i]].get(tuple(beta))
            if index is None:
                return None
            target[q] = index
    return target


def enumerate_symmetries(spec: PolynomialSystemSpec) -> list[SymmetryElement]:
    """Return all compatible symmetry elements, identity first (deterministic order).

    Order: permutations in ``itertools.permutations`` order, then ``k`` in
    lexicographic order, then ``conj`` False before True.
    """

    n = spec.n_vars
    elements: list[SymmetryElement] = []
    for permutation in itertools.permutations(range(n)):
        if _target_indices(spec, permutation) is None:
            continue
        for k in itertools.product(*(range(d) for d in spec.degrees)):
            for conjugate in (False, True):
                elements.append(SymmetryElement(tuple(permutation), tuple(k), conjugate))
    return elements


def coefficient_map(
    spec: PolynomialSystemSpec, element: SymmetryElement
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.complex128]]:
    """Return ``(target, phase)`` with ``out[target[m]] = phase[m] * c'[m]``.

    ``c' = conj(c)`` if ``element.conjugate`` else ``c``. ``target`` has shape
    ``(M,)`` and dtype ``int64`` (a permutation); ``phase`` has shape ``(M,)``
    and dtype ``complex128``. Leading entries have phase exactly ``1 + 0j``.
    """

    _check_element(spec, element)
    target = _target_indices(spec, tuple(element.permutation))
    if target is None:
        raise ValueError(f"element {element.label} is not compatible with the support of the system.")
    phase = np.empty(spec.n_coeffs, dtype=np.complex128)
    for q in range(spec.n_coeffs):
        total = sum(
            Fraction(int(k) * int(a), int(d))
            for k, a, d in zip(element.root_exponents, spec.exponents[q], spec.degrees)
        )
        phase[q] = _exact_unit_phase(total)
    return target, phase


def transform_coefficients(
    spec: PolynomialSystemSpec,
    element: SymmetryElement,
    coefficients: npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    """Apply ``element`` to coefficients of shape ``(M,)`` or ``(K, M)``.

    Returns a new ``complex128`` array of the same shape; the input is not modified.
    """

    values = np.asarray(coefficients, dtype=np.complex128)
    if values.ndim not in (1, 2) or values.shape[-1] != spec.n_coeffs:
        raise ValueError(f"coefficients must have shape (M,) or (K, M) with M={spec.n_coeffs}, got {values.shape}.")
    target, phase = coefficient_map(spec, element)
    moved = np.conj(values) if element.conjugate else values
    out = np.empty_like(values, dtype=np.complex128)
    out[..., target] = moved * phase
    return out


def free_real_matrix(spec: PolynomialSystemSpec, element: SymmetryElement) -> npt.NDArray[np.float64]:
    """Real-linear map on the free real vector ``[Re c_free, Im c_free]``.

    Returns an orthogonal ``(2D, 2D)`` ``float64`` matrix ``R`` such that, for
    coefficient vectors with fixed leading entries, the free real vector of
    ``transform_coefficients(c)`` equals ``R`` times that of ``c`` (up to the
    fixed leading entries, which are invariant).
    """

    free = np.flatnonzero(spec.free_coefficient_mask)
    d = free.size
    matrix = np.zeros((2 * d, 2 * d), dtype=np.float64)
    for col in range(2 * d):
        vector = np.zeros(spec.n_coeffs, dtype=np.complex128)
        if col < d:
            vector[free[col]] = 1.0
        else:
            vector[free[col - d]] = 1j
        image = transform_coefficients(spec, element, vector)[free]
        matrix[:, col] = np.concatenate([image.real, image.imag])
    return matrix


def latent_action_matrix(
    spec: PolynomialSystemSpec,
    element: SymmetryElement,
    basis: npt.ArrayLike,
    bezier_degree: int,
) -> npt.NDArray[np.float64]:
    """Induced map on flattened latent actions, shape ``(A, A)``, ``A=(d_b-1)m``.

    Block diagonal with ``bezier_degree - 1`` copies of ``U^T R U``. The basis
    must be square (``m == 2D``): for a smaller subspace the image of the
    symmetry generally leaves the span of ``U``, so the action is not closed.
    """

    u = np.asarray(basis, dtype=np.float64)
    real_dim = 2 * int(np.count_nonzero(spec.free_coefficient_mask))
    if u.ndim != 2 or u.shape[0] != real_dim:
        raise ValueError(f"basis must have shape ({real_dim}, m), got {u.shape}.")
    if u.shape[1] != real_dim:
        raise ValueError(
            f"latent dimension m={u.shape[1]} must equal 2D={real_dim}; the span of a smaller "
            "basis is not closed under the symmetry action."
        )
    if isinstance(bezier_degree, bool) or int(bezier_degree) < 2:
        raise ValueError(f"bezier_degree must be an integer >= 2, got {bezier_degree!r}.")
    block = u.T @ free_real_matrix(spec, element) @ u
    return np.kron(np.eye(int(bezier_degree) - 1), block)


__all__ = [
    "SymmetryElement",
    "coefficient_map",
    "enumerate_symmetries",
    "free_real_matrix",
    "latent_action_matrix",
    "transform_coefficients",
]
