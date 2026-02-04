import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence
import time 
import sympy as sp


@dataclass
class ValidationResult:
    ok: bool
    num_real_roots: int
    num_distinct_real_roots: int
    required_distinct_real_roots: int
    message: str


def _count_distinct(values: np.ndarray, tol: float) -> int:
    """Count distinct values in a 1D array with a tolerance."""
    if values.size == 0:
        return 0
    sorted_vals = np.sort(values)
    count = 1
    for i in range(1, sorted_vals.size):
        if abs(sorted_vals[i] - sorted_vals[i - 1]) > tol:
            count += 1
    return count


def build_coeffs(control_point: Sequence[float]) -> np.ndarray:
    """
    Build monic polynomial coefficients: [1, a_{n-1}, ..., a_0],
    where control_point = [a_{n-1}, ..., a_0].
    """
    return np.concatenate([[1.0], np.asarray(control_point, dtype=float)])


class ControlPointValidator:
    """
    Validate control points for a monic univariate polynomial.

    A control point of length n represents the coefficients
    [a_{n-1}, ..., a_0] of the monic polynomial
        x^n + a_{n-1} x^{n-1} + ... + a_0  (n >= 2).
    Its full coefficient array is [1, a_{n-1}, ..., a_0].
    Validation checks whether this polynomial has enough distinct real roots.
    """

    def __init__(
        self,
        imag_tol: float = 1e-8,
        distinct_tol: float = 1e-6,
        required_distinct_real_roots: Optional[int] = None,
    ) -> None:
        """
        Args:
            imag_tol: Maximum allowed absolute imaginary part to treat a root as real.
            distinct_tol: Minimum separation between roots to consider them distinct.
            required_distinct_real_roots: If None, it defaults to len(control_point).
        """
        self.imag_tol = imag_tol
        self.distinct_tol = distinct_tol
        self.required_distinct_real_roots = required_distinct_real_roots

    def validate(self, control_point: Sequence[float]) -> ValidationResult:
        """
        Check whether the monic polynomial defined by the control point
        has the required number of distinct real roots.

        The polynomial is x^n + a_{n-1} x^{n-1} + ... + a_0 where
        control_point = [a_{n-1}, ..., a_0].
        """
        control_point_array = np.asarray(control_point, dtype=float)

        # the degree of the polynomial corresponds to the length of the control point.
        degree = control_point_array.size

        required_distinct_real_roots = self.required_distinct_real_roots or degree

        coeffs = build_coeffs(control_point_array)
        roots = np.roots(coeffs)

        real_mask = np.abs(roots.imag) <= self.imag_tol
        real_roots = roots[real_mask].real
        num_real = real_roots.size

        num_distinct = _count_distinct(real_roots, tol=self.distinct_tol)

        ok = num_distinct >= required_distinct_real_roots
        msg = (
            f"{num_distinct} distinct real roots (required >= {required_distinct_real_roots}), "
            f"{num_real} real roots total, degree={degree}"
        )

        return ValidationResult(
            ok=ok,
            num_real_roots=num_real,
            num_distinct_real_roots=num_distinct,
            required_distinct_real_roots=required_distinct_real_roots,
            message=msg,
        )

    def __call__(self, control_point: Sequence[float]) -> ValidationResult:
        return self.validate(control_point)


if __name__ == "__main__":
    validator = ControlPointValidator()
    # control_point = [-2, -1, 2]
    # control_point = [-1, -4, 4]
    # control_point = np.array([-15, 85, -225, 274, -120])
    control_point = np.array([-40, 635, -5000, 19524, -30240])
    time_start = time.time()
    result = validator.validate(control_point)
    time_end = time.time()
    print(result.message)
    print(f"Time taken: {time_end - time_start} seconds")

    x = sp.symbols('x')
    time_start = time.time()
    poly = sp.Poly(build_coeffs(control_point), x)
    roots = poly.nroots()
    time_end = time.time()
    print(f"poly: {poly}")
    print(f"roots: {roots}")
    print(f"Time taken: {time_end - time_start} seconds")