"""Reference start solutions for Pham start systems."""

from __future__ import annotations

from itertools import product
from math import prod
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from homotopy_path_learning.systems.spec import _normalize_degrees


def pham_start_solutions(degrees: Sequence[int]) -> npt.NDArray[np.complex128]:
    """Generate all start solutions for ``G_i(x) = x_i ** d_i - 1``.

    The returned array has shape ``(prod_i d_i, n_vars)`` and dtype
    ``complex128``. Row order is deterministic: it is the Cartesian product of
    each variable's roots in variable order, with roots ordered by
    ``k = 0, ..., d_i - 1``.
    """

    degree_values = tuple(degrees)
    n_vars = len(degree_values)
    if n_vars <= 0:
        raise ValueError("degrees must contain at least one positive degree.")
    degree_tuple = _normalize_degrees(degree_values, n_vars=n_vars)

    roots_by_variable = []
    for degree in degree_tuple:
        k = np.arange(degree, dtype=np.float64)
        roots = np.exp((2.0j * np.pi * k) / float(degree)).astype(np.complex128)
        roots_by_variable.append(roots)

    n_paths = prod(degree_tuple)
    solutions = np.empty((n_paths, n_vars), dtype=np.complex128)
    for row_index, root_tuple in enumerate(product(*roots_by_variable)):
        solutions[row_index, :] = root_tuple
    return solutions


__all__ = ["pham_start_solutions"]
