"""Typed data model for fixed multivariate polynomial system supports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt


IntArray = npt.NDArray[np.int64]


Support = tuple[tuple[int, ...], ...]
Supports = tuple[Support, ...]


def _is_integer_value(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _normalize_degrees(degrees: Iterable[int], *, n_vars: int) -> tuple[int, ...]:
    degree_tuple = tuple(degrees)
    if len(degree_tuple) != n_vars:
        raise ValueError(
            f"degrees must have length n_vars={n_vars}, got {len(degree_tuple)}."
        )
    normalized: list[int] = []
    for i, degree in enumerate(degree_tuple):
        if not _is_integer_value(degree):
            raise TypeError(f"degrees[{i}] must be an integer, got {degree!r}.")
        value = int(degree)
        if value <= 0:
            raise ValueError(f"degrees[{i}] must be positive, got {value}.")
        normalized.append(value)
    return tuple(normalized)


def _normalize_supports(supports: Iterable[Iterable[Iterable[int]]], *, n_vars: int) -> Supports:
    support_blocks = tuple(supports)
    if len(support_blocks) != n_vars:
        raise ValueError(
            f"supports must contain one exponent block per equation; "
            f"expected {n_vars}, got {len(support_blocks)}."
        )

    normalized_blocks: list[Support] = []
    for equation_index, block in enumerate(support_blocks):
        normalized_block: list[tuple[int, ...]] = []
        for exponent_index, exponent in enumerate(block):
            try:
                exponent_tuple = tuple(exponent)
            except TypeError as exc:
                raise TypeError(
                    f"supports[{equation_index}][{exponent_index}] must be an iterable "
                    "of integer exponents."
                ) from exc
            if len(exponent_tuple) != n_vars:
                raise ValueError(
                    f"supports[{equation_index}][{exponent_index}] must have length "
                    f"n_vars={n_vars}, got {len(exponent_tuple)}."
                )
            normalized_exponent: list[int] = []
            for component_index, component in enumerate(exponent_tuple):
                if not _is_integer_value(component):
                    raise TypeError(
                        f"supports[{equation_index}][{exponent_index}] component "
                        f"{component_index} must be an integer, got {component!r}."
                    )
                value = int(component)
                if value < 0:
                    raise ValueError(
                        f"supports[{equation_index}][{exponent_index}] component "
                        f"{component_index} must be nonnegative, got {value}."
                    )
                normalized_exponent.append(value)
            normalized_block.append(tuple(normalized_exponent))
        normalized_blocks.append(tuple(normalized_block))
    return tuple(normalized_blocks)


@dataclass(frozen=True)
class PolynomialSystemSpec:
    """Fixed support for a square multivariate Pham-type polynomial system.

    The first implementation targets ``n_vars`` equations in ``n_vars`` complex
    variables. Coefficients are stored equation block by equation block. Within
    each block, the leading exponent ``d_i e_i`` is first, followed by the
    non-leading exponents in the deterministic order supplied to the builder.

    Attributes:
        n_vars: Number of variables and equations, denoted by ``n``.
        degrees: Tuple ``(d_1, ..., d_n)`` of positive integer leading degrees.
        exponents: Flattened exponent matrix with shape ``(M, n_vars)`` and
            dtype ``int64``. Row ``q`` contains the multi-index alpha for
            coefficient ``c_q``.
        offsets: Equation block offsets with shape ``(n_vars + 1,)`` and dtype
            ``int64``. Equation ``i`` uses flattened coefficient indices
            ``offsets[i] <= q < offsets[i + 1]``. Indices are zero-based on the
            Python side.
        leading_indices: Shape ``(n_vars,)``, dtype ``int64``. Entry ``i`` is
            the flattened index of the leading monomial ``x_i ** d_i``.
        constant_indices: Shape ``(n_vars,)``, dtype ``int64``. Entry ``i`` is
            the flattened index of the constant term in equation ``i``.
        supports: Per-equation candidate exponent sets, including the leading
            exponent, in coefficient order. If omitted, it is derived from
            ``exponents`` and ``offsets``.

    Coefficient vector convention:
        A coefficient vector has shape ``(M,)`` and dtype ``complex128``. Its
        entry ``q`` multiplies ``x ** exponents[q]``. Blocks are concatenated in
        equation order; inside each block the first coefficient is the fixed
        leading coefficient, which is always ``1`` for Pham-type systems.
    """

    n_vars: int
    degrees: tuple[int, ...]
    exponents: IntArray
    offsets: IntArray
    leading_indices: IntArray
    constant_indices: IntArray
    supports: Supports | None = None

    def __post_init__(self) -> None:
        if not _is_integer_value(self.n_vars):
            raise TypeError(f"n_vars must be an integer, got {self.n_vars!r}.")
        n_vars = int(self.n_vars)
        if n_vars <= 0:
            raise ValueError(f"n_vars must be positive, got {n_vars}.")
        object.__setattr__(self, "n_vars", n_vars)

        degrees = _normalize_degrees(self.degrees, n_vars=n_vars)
        object.__setattr__(self, "degrees", degrees)

        self._validate_int_array("exponents", self.exponents, ndim=2)
        self._validate_int_array("offsets", self.offsets, ndim=1)
        self._validate_int_array("leading_indices", self.leading_indices, ndim=1)
        self._validate_int_array("constant_indices", self.constant_indices, ndim=1)

        if self.exponents.shape[1] != n_vars:
            raise ValueError(
                f"exponents must have shape (M, {n_vars}), got {self.exponents.shape}."
            )
        if np.any(self.exponents < 0):
            raise ValueError("exponents must contain only nonnegative integers.")

        n_coeffs = int(self.exponents.shape[0])
        if self.offsets.shape != (n_vars + 1,):
            raise ValueError(
                f"offsets must have shape ({n_vars + 1},), got {self.offsets.shape}."
            )
        if self.leading_indices.shape != (n_vars,):
            raise ValueError(
                f"leading_indices must have shape ({n_vars},), got "
                f"{self.leading_indices.shape}."
            )
        if self.constant_indices.shape != (n_vars,):
            raise ValueError(
                f"constant_indices must have shape ({n_vars},), got "
                f"{self.constant_indices.shape}."
            )

        if int(self.offsets[0]) != 0:
            raise ValueError(f"offsets[0] must be 0, got {int(self.offsets[0])}.")
        if int(self.offsets[-1]) != n_coeffs:
            raise ValueError(
                f"offsets[-1] must equal M={n_coeffs}, got {int(self.offsets[-1])}."
            )
        if np.any(np.diff(self.offsets) <= 0):
            raise ValueError("offsets must be strictly increasing.")

        derived_supports = self._validate_blocks()
        if self.supports is None:
            object.__setattr__(self, "supports", derived_supports)
        else:
            normalized_supports = _normalize_supports(self.supports, n_vars=n_vars)
            if normalized_supports != derived_supports:
                raise ValueError("supports must match exponents and offsets exactly.")
            object.__setattr__(self, "supports", normalized_supports)

    @staticmethod
    def _validate_int_array(name: str, array: object, *, ndim: int) -> None:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray.")
        if array.dtype != np.int64:
            raise TypeError(f"{name} must have dtype int64, got {array.dtype}.")
        if array.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}-dimensional, got ndim={array.ndim}.")

    def _validate_blocks(self) -> Supports:
        support_blocks: list[Support] = []
        zero = np.zeros(self.n_vars, dtype=np.int64)

        for equation_index, degree in enumerate(self.degrees):
            start = int(self.offsets[equation_index])
            stop = int(self.offsets[equation_index + 1])
            block = self.exponents[start:stop]

            leading_exponent = np.zeros(self.n_vars, dtype=np.int64)
            leading_exponent[equation_index] = degree
            leading_matches = np.flatnonzero(np.all(block == leading_exponent, axis=1))
            if leading_matches.size != 1:
                raise ValueError(
                    f"equation {equation_index} must contain leading monomial "
                    f"{tuple(int(v) for v in leading_exponent)} exactly once; "
                    f"found {leading_matches.size}."
                )
            leading_index = start + int(leading_matches[0])
            if int(self.leading_indices[equation_index]) != leading_index:
                raise ValueError(
                    f"leading_indices[{equation_index}] must be {leading_index}, "
                    f"got {int(self.leading_indices[equation_index])}."
                )

            constant_matches = np.flatnonzero(np.all(block == zero, axis=1))
            if constant_matches.size != 1:
                raise ValueError(
                    f"equation {equation_index} must contain the constant term exactly "
                    f"once; found {constant_matches.size}."
                )
            constant_index = start + int(constant_matches[0])
            if int(self.constant_indices[equation_index]) != constant_index:
                raise ValueError(
                    f"constant_indices[{equation_index}] must be {constant_index}, "
                    f"got {int(self.constant_indices[equation_index])}."
                )

            seen: set[tuple[int, ...]] = set()
            normalized_block: list[tuple[int, ...]] = []
            for local_index, exponent in enumerate(block):
                exponent_tuple = tuple(int(v) for v in exponent)
                if exponent_tuple in seen:
                    raise ValueError(
                        f"equation {equation_index} contains duplicate exponent "
                        f"{exponent_tuple}."
                    )
                seen.add(exponent_tuple)
                normalized_block.append(exponent_tuple)

                if local_index == int(leading_matches[0]):
                    continue
                total_degree = int(np.sum(exponent))
                if total_degree >= degree:
                    raise ValueError(
                        f"equation {equation_index} non-leading exponent "
                        f"{exponent_tuple} has total degree {total_degree}; expected "
                        f"total degree < d_i={degree}."
                    )

            support_blocks.append(tuple(normalized_block))

        return tuple(support_blocks)

    @property
    def n_equations(self) -> int:
        """Number of equations; equal to ``n_vars`` in the initial square case."""

        return self.n_vars

    @property
    def n_coeffs(self) -> int:
        """Total number of coefficients ``M``."""

        return int(self.exponents.shape[0])

    @property
    def M(self) -> int:
        """Alias for the total number of coefficients."""

        return self.n_coeffs

    @property
    def total_coeffs(self) -> int:
        """Alias for the total number of coefficients."""

        return self.n_coeffs

    @property
    def equation_exponents(self) -> Supports:
        """Per-equation exponent blocks in coefficient order."""

        assert self.supports is not None
        return self.supports

    @property
    def free_coefficient_mask(self) -> npt.NDArray[np.bool_]:
        """Boolean mask selecting all non-leading coefficients."""

        mask = np.ones(self.n_coeffs, dtype=np.bool_)
        mask[self.leading_indices] = False
        return mask
