"""Construction helpers for fixed Pham-type polynomial supports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from homotopy_path_learning.systems.spec import (
    PolynomialSystemSpec,
    Supports,
    _normalize_degrees,
    _normalize_supports,
)


def build_pham_spec(
    degrees: Sequence[int],
    supports: Sequence[Sequence[Sequence[int]]],
) -> PolynomialSystemSpec:
    """Build and validate a Pham-type polynomial system specification.

    Args:
        degrees: Positive leading degrees ``(d_1, ..., d_n)``.
        supports: Per-equation non-leading exponent lists. ``supports[i]`` must
            include the constant exponent ``(0, ..., 0)`` and must not include
            the leading exponent ``d_i e_i``. The function prepends that leading
            exponent internally.

    Returns:
        A ``PolynomialSystemSpec`` whose coefficient order is deterministic:
        equation blocks are concatenated in order, and each block is
        ``[d_i e_i, *supports[i]]``.

    Raises:
        TypeError: If integer-valued inputs are not represented by integer
            types.
        ValueError: If the support violates the Pham-type assumptions.
    """

    degree_values = tuple(degrees)
    n_vars = len(degree_values)
    if n_vars <= 0:
        raise ValueError("degrees must contain at least one positive degree.")

    degree_tuple = _normalize_degrees(degree_values, n_vars=n_vars)
    nonleading_supports = _normalize_supports(supports, n_vars=n_vars)

    exponent_rows: list[tuple[int, ...]] = []
    offsets = [0]
    leading_indices: list[int] = []
    constant_indices: list[int] = []
    candidate_supports: list[tuple[tuple[int, ...], ...]] = []
    zero = tuple(0 for _ in range(n_vars))

    for equation_index, degree in enumerate(degree_tuple):
        leading_exponent = tuple(
            degree if variable_index == equation_index else 0
            for variable_index in range(n_vars)
        )
        support_block = nonleading_supports[equation_index]
        if leading_exponent in support_block:
            raise ValueError(
                f"supports[{equation_index}] must not include the leading exponent "
                f"{leading_exponent}; it is added exactly once by build_pham_spec."
            )
        if zero not in support_block:
            raise ValueError(
                f"supports[{equation_index}] must include the constant exponent {zero}."
            )

        candidate_block = (leading_exponent, *support_block)
        block_start = offsets[-1]
        leading_indices.append(block_start)
        constant_indices.append(block_start + candidate_block.index(zero))
        exponent_rows.extend(candidate_block)
        offsets.append(block_start + len(candidate_block))
        candidate_supports.append(candidate_block)

    exponents = np.asarray(exponent_rows, dtype=np.int64)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    leading_array = np.asarray(leading_indices, dtype=np.int64)
    constant_array = np.asarray(constant_indices, dtype=np.int64)

    return PolynomialSystemSpec(
        n_vars=n_vars,
        degrees=degree_tuple,
        exponents=exponents,
        offsets=offsets_array,
        leading_indices=leading_array,
        constant_indices=constant_array,
        supports=tuple(candidate_supports),
    )


__all__ = ["Supports", "build_pham_spec"]
