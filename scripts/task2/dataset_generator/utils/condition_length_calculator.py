from typing import Callable
import mpmath as mp
import sympy as sp
import time
import numpy as np


def _build_poly_from_coeffs(coeffs: list[float] | np.ndarray, variable: sp.Symbol) -> sp.Poly:
    """Construct a polynomial in `variable` from coefficients ordered high-to-low degree."""
    degree = len(coeffs) - 1
    expr = sum(coeff * variable ** (degree - idx) for idx, coeff in enumerate(coeffs))
    return sp.Poly(expr, variable)


def _integrate_condition_length(
    derivative_vector: sp.Matrix,
    discriminant: sp.Expr,
    parameter: sp.Symbol,
    lower_bound: float,
    upper_bound: float,
    degree: int,
) -> mp.mpf:
    """Integrate ||derivative_vector||_2 / |discriminant^(1/degree)| to obtain the condition length.

    Args:
        derivative_vector: Derivative of the coefficient path (i.e., the homotopy path in the coefficient space).
        discriminant: Discriminant of the homotopy polynomial.
        parameter: Parameter of the homotopy path.
        lower_bound: Lower bound of the parameter.
        upper_bound: Upper bound of the parameter.
        degree: Degree of the homotopy polynomial with respect to the variable (not the parameter).

    Returns:
        Real value of `∫ ||derivative_vector||_2 / |discriminant^(1/degree)| d(parameter)` as `mp.mpf`.
    """
    integrand = derivative_vector.norm(2) / abs(discriminant ** sp.Rational(1, degree))
    result = sp.integrate(integrand, (parameter, lower_bound, upper_bound))
    result_eval = sp.N(result)
    return mp.mpf(sp.re(result_eval))


def _ensure_monic_coeffs(coeffs: list[float] | np.ndarray, name: str, atol: float = 1e-12) -> None:
    """Validate that the provided coefficient list/array represents a monic polynomial."""
    if len(coeffs) == 0:
        raise ValueError(f"{name} must contain at least one coefficient.")
    if not np.isclose(float(coeffs[0]), 1.0, atol=atol):
        raise ValueError(f"{name} must be monic (leading coefficient 1).")


def _ensure_monic_poly(hc: sp.Poly, name: str) -> None:
    """Validate that the provided sympy.Poly is monic."""
    if hc.LC() != 1:
        raise ValueError(f"{name} must be monic (leading coefficient 1).")


def _build_norm_callable(expressions: list[sp.Expr], parameter: sp.Symbol) -> Callable[[float], mp.mpf]:
    """Return a callable that evaluates the Euclidean norm of `expressions` at a numeric parameter value."""
    component_functions = [sp.lambdify(parameter, expr, "mpmath") for expr in expressions]

    def _norm(value: float) -> mp.mpf:
        squared_sum = mp.fsum(abs(component(value)) ** 2 for component in component_functions)
        return mp.sqrt(squared_sum)

    return _norm


def _convert_to_float(value: float | int | sp.Expr, name: str) -> float:
    """Convert value to float, raising ValueError with context on failure."""
    try:
        return float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be real-convertible.") from exc


def _discriminant_hits_zero(
    disc_expr: sp.Expr,
    parameter: sp.Symbol,
    lower_bound: float | int | sp.Expr,
    upper_bound: float | int | sp.Expr,
    zero_tol: float = 1e-12,
    num_samples: int | None = 200,
    use_root_finder: bool = False,
    root_tol: float = 1e-12,
    log_min: bool = False,
    log_fn: Callable[[float], None] | None = None,
) -> bool:
    """Rough test to detect zeros/non-finite discriminant values on an interval.

    This function first screens by sampling, sign-change detection, and minimum
    absolute value tracking, then optionally applies a slower root finder.

    Args:
        disc_expr: Discriminant expression to test.
        parameter: Symbol used as the parameter of the discriminant.
        lower_bound: Lower bound of the interval (coerced to float).
        upper_bound: Upper bound of the interval (coerced to float).
        zero_tol: Threshold below which the discriminant is treated as zero.
        num_samples: Number of evenly spaced samples for the rough screening.
        use_root_finder: If True, additionally call `Poly(...).nroots()` to find
            roots; real roots (|Im| <= root_tol) within [lower_bound, upper_bound]
            are treated as zeros.
        root_tol: Tolerance for the imaginary part and interval check in the
            root finder stage.
        log_min: If True, log the minimum absolute value encountered.
        log_fn: Logging function to use when `log_min` is True (default: print).

    Returns:
        True if a zero/non-finite value is detected; otherwise False.
    """
    disc_fn = sp.lambdify(parameter, disc_expr, "mpmath")
    lb = _convert_to_float(lower_bound, "lower_bound")
    ub = _convert_to_float(upper_bound, "upper_bound")
    min_abs = mp.inf
    num_samples = 200 if num_samples is None else num_samples
    reporter = log_fn if log_fn is not None else print
    prev_val: float | None = None
    for value in np.linspace(lb, ub, num_samples):
        try:
            disc_value = disc_fn(value)
        except Exception:
            return True
        if not mp.isfinite(disc_value):
            return True
        abs_val = abs(disc_value)
        if abs_val < min_abs:
            min_abs = abs_val
        if prev_val is not None and disc_value * prev_val < 0:
            if log_min and log_fn is not None:
                reporter(float(min_abs))
            return True
        if abs_val <= zero_tol:
            if log_min and log_fn is not None:
                reporter(float(min_abs))
            return True
        prev_val = disc_value
    if use_root_finder:
        try:
            poly_disc = sp.Poly(disc_expr, parameter)
            for root in poly_disc.nroots():
                if abs(sp.im(root)) <= root_tol:
                    real_root = float(sp.re(root))
                    if lb - root_tol <= real_root <= ub + root_tol:
                        if log_min and log_fn is not None:
                            reporter(float(min_abs))
                        return True
        except Exception:
            # if root finding fails, fall back to sampled decision
            pass
    if log_min and log_fn is not None:
        reporter(float(min_abs))
    return False


def _numeric_integral(integrand: Callable[[float], float], lower_bound: float, upper_bound: float) -> mp.mpf:
    """Integrate a numeric 1D integrand using mpmath.quad."""
    lb = _convert_to_float(lower_bound, "lower_bound")
    ub = _convert_to_float(upper_bound, "upper_bound")
    try:
        return mp.quad(integrand, [lb, ub])
    except Exception as exc:  # pragma: no cover - passthrough for diagnostics
        raise RuntimeError(
            f"Numeric integration failed on interval [{lower_bound}, {upper_bound}]: {exc}"
        ) from exc


def calculate_linear_condition_length(
    start_coeffs: list[float] | np.ndarray,
    target_coeffs: list[float] | np.ndarray,
    zero_tol: float = 1e-12,
    num_samples: int | None = 200,
    use_root_finder: bool = False,
    root_tol: float = 1e-12,
    log_min: bool = False,
    log_fn: Callable[[float], None] | None = print,
) -> mp.mpf:
    """
    Compute the condition length along the linear homotopy path between two polynomials defined by their coefficients.

    Args:
        start_coeffs: Coefficients `[1, a_{n-1}, ..., a_0]` for the start monic polynomial x^n + a_{n-1} x^{n-1} + ... + a_0 (n >= 2).
        target_coeffs: Coefficients `[1, a_{n-1}, ..., a_0]` for the target monic polynomial x^n + a_{n-1} x^{n-1} + ... + a_0 (n >= 2).
        zero_tol: Tolerance to treat the discriminant as zero when screening the path.
        num_samples: Number of samples to use for the discriminant screening.
        use_root_finder: If True, additionally run a root finder on the discriminant for robust zero detection.
        root_tol: Imaginary/real tolerance used in the root finder to accept a root as real and in-range.
        log_min: If True, log the minimum absolute value of the discriminant sampled.
        log_fn: If provided, use this function to log the minimum absolute value of the discriminant sampled.

    Returns:
        Real value of `∫ ||target_coeff_vector - start_coeff_vector||_2 / |discriminant|^(1/degree) d(parameter)` as `mp.mpf`, or `mp.inf` if the discriminant vanishes/non-finite on the path.
    """
    _ensure_monic_coeffs(start_coeffs, "start_coeffs")
    _ensure_monic_coeffs(target_coeffs, "target_coeffs")
    if len(start_coeffs) < 3 or len(target_coeffs) < 3:
        raise ValueError("start_coeffs and target_coeffs must have length >= 3 (degree >= 2).")
    if len(start_coeffs) != len(target_coeffs):
        raise ValueError("start_coeffs and target_coeffs must have the same length.")

    t = sp.symbols('t', real=True)
    x = sp.symbols('x')
    start_coeff_vector = sp.Matrix(start_coeffs)
    target_coeff_vector = sp.Matrix(target_coeffs)
    derivative_vector = target_coeff_vector - start_coeff_vector # this corresponds to the derivative of the coefficient path.

    start_poly = _build_poly_from_coeffs(start_coeffs, x)
    target_poly = _build_poly_from_coeffs(target_coeffs, x)
    linear_hc_expr = (1 - t) * start_poly + t * target_poly

    try:
        disc = sp.discriminant(linear_hc_expr, x)
    except Exception:
        try:
            linear_hc_expr_q = sp.nsimplify(linear_hc_expr, rational=True, tolerance=1e-8)
            disc = sp.discriminant(linear_hc_expr_q, x)
        except Exception:
            # If discriminant computation fails, treat as invalid path.
            return mp.inf
    degree = sp.degree(start_poly, x)
    if _discriminant_hits_zero(
        disc_expr=disc,
        parameter=t,
        lower_bound=0,
        upper_bound=1,
        zero_tol=zero_tol,
        num_samples=num_samples,
        use_root_finder=use_root_finder,
        root_tol=root_tol,
        log_min=log_min,
        log_fn=log_fn,
    ):
        return mp.inf
    return _integrate_condition_length(derivative_vector, disc, t, 0, 1, degree)
    
def calculate_condition_length(
    hc: sp.Poly,
    variable: sp.Symbol,
    parameter: sp.Symbol,
    lower_bound: float | int | sp.Expr,
    upper_bound: float | int | sp.Expr,
    zero_tol: float = 1e-12,
    num_samples: int | None = 200,
    use_root_finder: bool = False,
    root_tol: float = 1e-12,
    log_min: bool = False,
    log_fn: Callable[[float], None] | None = print,
) -> mp.mpf:
    """
    Compute the condition length along the homotopy path defined by homotopy polynomial `hc` and parameter `parameter`.

    Args:
        hc: Homotopy polynomial as a monic `sympy.Poly` with degree >= 2.
        variable: Polynomial variable (e.g., `x`).
        parameter: Homotopy parameter (e.g., `t`).
        lower_bound: Lower integration bound.
        upper_bound: Upper integration bound.
        zero_tol: Tolerance to treat the discriminant as zero when screening the path.
        num_samples: Number of samples to use for the discriminant screening.
        use_root_finder: If True, additionally run a root finder on the discriminant for robust zero detection.
        root_tol: Imaginary/real tolerance used in the root finder to accept a root as real and in-range.
        log_min: If True, log the minimum absolute value of the discriminant sampled.
        log_fn: If provided, use this function to log the minimum absolute value of the discriminant sampled.

    Returns:
        Real value of `∫ ||derivative_vector||_2 / |discriminant|^(1/degree) d(parameter)` as `mp.mpf`, or `mp.inf` if the discriminant vanishes/non-finite on the path.
    """
    _ensure_monic_poly(hc, "hc")
    degree = sp.degree(hc, variable)
    if degree < 2:
        raise ValueError("hc must have degree >= 2.")
    tail_coeffs = hc.coeffs()[1:]
    coeff_path_vector = sp.Matrix(tail_coeffs) # coefficients excluding the leading 1 of the monic polynomial.

    try:
        disc = sp.discriminant(hc, variable)
    except Exception:
        try:
            hc_q = sp.nsimplify(hc, rational=True, tolerance=1e-8)
            disc = sp.discriminant(hc_q, variable)
        except Exception:
            # If discriminant computation fails, treat as invalid path.
            return mp.inf
    if _discriminant_hits_zero(
        disc,
        parameter=parameter,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        zero_tol=zero_tol,
        num_samples=num_samples,
        use_root_finder=use_root_finder,
        root_tol=root_tol,
        log_min=log_min,
        log_fn=log_fn,
    ):
        return mp.inf

    derivative_vector = coeff_path_vector.diff(parameter)
    return _integrate_condition_length(derivative_vector, disc, parameter, lower_bound, upper_bound, degree)


def calculate_linear_condition_length_numeric(
    start_coeffs: list[float] | np.ndarray,
    target_coeffs: list[float] | np.ndarray,
    regularization_eps: float | None = 1e-12,
    zero_tol: float = 1e-12,
    num_samples: int | None = 200,
    use_root_finder: bool = False,
    root_tol: float = 1e-12,
    log_min: bool = False,
    log_fn: Callable[[float], None] | None = print,
) -> mp.mpf:
    """
    Compute the condition length along the linear homotopy via numeric quadrature.

    Args:
        start_coeffs: Coefficients `[1, a_{n-1}, ..., a_0]` for the start monic polynomial x^n + a_{n-1} x^{n-1} + ... + a_0 (n >= 2).
        target_coeffs: Coefficients `[1, a_{n-1}, ..., a_0]` for the target monic polynomial x^n + a_{n-1} x^{n-1} + ... + a_0 (n >= 2).
        regularization_eps: If not None, clamp `abs(discriminant)` below this value to avoid division by ~0.
        zero_tol: Tolerance to treat the discriminant as zero when screening the path.
        num_samples: Number of samples to use for the discriminant screening.
        use_root_finder: If True, additionally run a root finder on the discriminant for robust zero detection.
        root_tol: Imaginary/real tolerance used in the root finder to accept a root as real and in-range.
        log_min: If True, log the minimum absolute value of the discriminant sampled.
        log_fn: If provided, use this function to log the minimum absolute value of the discriminant sampled.

    Returns:
        Numeric approximation of `∫ ||target_coeff_vector - start_coeff_vector||_2 / |discriminant^(1/degree)| dt`.
    """
    _ensure_monic_coeffs(start_coeffs, "start_coeffs")
    _ensure_monic_coeffs(target_coeffs, "target_coeffs")
    if len(start_coeffs) < 3 or len(target_coeffs) < 3:
        raise ValueError("start_coeffs and target_coeffs must have length >= 3 (degree >= 2).")
    if len(start_coeffs) != len(target_coeffs):
        raise ValueError("start_coeffs and target_coeffs must have the same length.")
    t = sp.symbols('t', real=True)
    x = sp.symbols('x')
    start_poly = _build_poly_from_coeffs(start_coeffs, x)
    target_poly = _build_poly_from_coeffs(target_coeffs, x)

    derivative_vector = sp.Matrix(target_coeffs) - sp.Matrix(start_coeffs)
    derivative_norm = float(sp.N(derivative_vector.norm(2)))

    linear_hc_expr = (1 - t) * start_poly + t * target_poly
    try:
        # rationalize the coefficients of the homotopy polynomial
        linear_hc_expr_q = sp.nsimplify(linear_hc_expr, rational=True, tolerance=1e-8)
        disc_expr = sp.discriminant(linear_hc_expr_q, x)
    except Exception:
        # If discriminant computation fails, treat this path as invalid.
        return mp.inf

    # try:
    #     disc_expr = sp.discriminant(linear_hc_expr, x)
    # except Exception:
    #     # Fallback: rationalize coefficients and retry once
    #     try:
    #         linear_hc_expr_q = sp.nsimplify(linear_hc_expr, rational=True, tolerance=1e-8)
    #         disc_expr = sp.discriminant(linear_hc_expr_q, x)
    #     except Exception:
    #         # If discriminant computation still fails, treat this path as invalid.
    #         return mp.inf

    if _discriminant_hits_zero(
        disc_expr=disc_expr,
        parameter=t,
        lower_bound=0.0,
        upper_bound=1.0,
        zero_tol=zero_tol,
        num_samples=num_samples,
        use_root_finder=use_root_finder,
        root_tol=root_tol,
        log_min=log_min,
        log_fn=log_fn,
    ):
        return mp.inf
    disc_fn = sp.lambdify(t, disc_expr, "mpmath")
    exponent = 1 / sp.degree(start_poly, x)

    def integrand(value: float) -> float:
        disc_value = disc_fn(value)
        if not mp.isfinite(disc_value):
            raise ValueError(f"Discriminant is non-finite at t={value}: {disc_value}")
        denom_raw = abs(disc_value ** exponent)
        denom = max(denom_raw, regularization_eps) if regularization_eps is not None else denom_raw
        if not mp.isfinite(denom):
            raise ValueError(f"Denominator is non-finite at t={value}: {denom}")
        return derivative_norm / denom

    return _numeric_integral(integrand, 0.0, 1.0)


def calculate_condition_length_numeric(
    hc: sp.Poly,
    variable: sp.Symbol,
    parameter: sp.Symbol,
    lower_bound: float | int | sp.Expr,
    upper_bound: float | int | sp.Expr,
    regularization_eps: float | None = 1e-12,
    zero_tol: float = 1e-12,
    num_samples: int | None = 200,
    use_root_finder: bool = False,
    root_tol: float = 1e-12,
    log_min: bool = False,
    log_fn: Callable[[float], None] | None = print,
) -> mp.mpf:
    """
    Compute the condition length along an arbitrary homotopy using numeric quadrature.

    Args:
        hc: Homotopy polynomial as a monic `sympy.Poly` with degree >= 2.
        variable: Polynomial variable (e.g., `x`).
        parameter: Homotopy parameter (e.g., `t`).
        lower_bound: Lower integration bound.
        upper_bound: Upper integration bound.
        regularization_eps: If not None, clamp `abs(discriminant)` below this value to avoid division by ~0.
        zero_tol: Tolerance to treat the discriminant as zero when screening the path.
        use_root_finder: If True, additionally run a root finder on the discriminant for robust zero detection.
        root_tol: Imaginary/real tolerance used in the root finder to accept a root as real and in-range.
        log_min: If True, log the minimum absolute value of the discriminant sampled.
        log_fn: If provided, use this function to log the minimum absolute value of the discriminant sampled.

    Returns:
        Numeric approximation of `∫ ||derivative_vector||_2 / |discriminant^(1/degree)| d(parameter)`.
    """
    _ensure_monic_poly(hc, "hc")
    degree = sp.degree(hc, variable)
    if degree < 2:
        raise ValueError("hc must have degree >= 2.")
    tail_coeffs = hc.coeffs()[1:]
    coeff_path_vector = sp.Matrix(tail_coeffs)

    derivative_vector = coeff_path_vector.diff(parameter)
    derivative_norm_fn = _build_norm_callable(list(derivative_vector), parameter)

    try:
        disc_expr = sp.discriminant(hc, variable)
    except Exception:
        # If discriminant computation fails, treat as invalid path.
        try:
            hc_q = sp.nsimplify(hc, rational=True, tolerance=1e-8)
            disc_expr = sp.discriminant(hc_q, variable)
        except Exception:
            # If discriminant computation fails, treat as invalid path.
            return mp.inf
    if _discriminant_hits_zero(
        disc_expr=disc_expr,
        parameter=parameter,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        zero_tol=zero_tol,
        num_samples=num_samples,
        use_root_finder=use_root_finder,
        root_tol=root_tol,
        log_min=log_min,
        log_fn=log_fn,
    ):
        return mp.inf
    disc_fn = sp.lambdify(parameter, disc_expr, "mpmath")
    exponent = 1 / degree

    def integrand(value: float) -> float:
        disc_value = disc_fn(value)
        if not mp.isfinite(disc_value):
            raise ValueError(f"Discriminant is non-finite at parameter={value}: {disc_value}")
        denom_raw = abs(disc_value ** exponent)
        denom = max(denom_raw, regularization_eps) if regularization_eps is not None else denom_raw
        if not mp.isfinite(denom):
            raise ValueError(f"Denominator is non-finite at parameter={value}: {denom}")
        return derivative_norm_fn(value) / denom

    return _numeric_integral(integrand, lower_bound, upper_bound)


def main():
    p_coeffs = np.array([1, -1, -1])
    q_coeffs = np.array([1, 1, -1])
    r_coeffs = np.array([1, 0, -2])
    s_coeffs = np.array([1, 0, -1.2])
    
    a_coeffs = np.array([1, -1, -1, 2, 3, 1, 1])
    b_coeffs = np.array([1, 1, -1, 2, 1, 5, 1])
    c_coeffs = np.array([1, 0, 0, 0, 0, 0, 0])

    # p_coeffs = [1, -1, -1]
    # q_coeffs = [1, 1, -1]
    # r_coeffs = [1, 0, -2]
    # s_coeffs = [1, 0, -1.2]

    t = sp.symbols('t')
    x = sp.symbols('x')
    p = sp.Poly(p_coeffs, x)
    q = sp.Poly(q_coeffs, x)
    r = sp.Poly(r_coeffs, x)
    s = sp.Poly(s_coeffs, x)


    # Calculate the condition length of PR + RQ using the linear homotopy path.
    print("\nLinear homotopy path (P -> R -> Q):")

    time_start = time.time()
    pr_condition_length = calculate_linear_condition_length(p_coeffs, r_coeffs) 
    rq_condition_length = calculate_linear_condition_length(r_coeffs, q_coeffs)
    total_condition_length = pr_condition_length + rq_condition_length
    time_end = time.time()
    print(f"PR + RQ = {total_condition_length}, Time taken: {time_end - time_start} seconds (by calculate_linear_condition_length)")

    time_start = time.time()
    pr_condition_length_numeric = calculate_linear_condition_length_numeric(p_coeffs, r_coeffs)
    rq_condition_length_numeric = calculate_linear_condition_length_numeric(r_coeffs, q_coeffs)
    total_condition_length_numeric = pr_condition_length_numeric + rq_condition_length_numeric
    time_end = time.time()
    print(f"PR + RQ = {total_condition_length_numeric}, Time taken: {time_end - time_start} seconds (by calculate_linear_condition_length_numeric)")

    hc_1 = (1 - 2*t) * p + 2*t * r
    hc_2 = (2 - 2*t) * r + (2*t - 1) * q
    time_start = time.time()
    pr_condition_length = calculate_condition_length(hc_1, x, t, 0, sp.Rational(1, 2))
    rq_condition_length = calculate_condition_length(hc_2, x, t, sp.Rational(1, 2), 1)
    total_condition_length = pr_condition_length + rq_condition_length
    time_end = time.time()
    print(f"PR + RQ = {total_condition_length}, Time taken: {time_end - time_start} seconds (by calculate_condition_length)")

    time_start = time.time()
    pr_condition_length_numeric = calculate_condition_length_numeric(hc_1, x, t, 0, sp.Rational(1, 2))
    rq_condition_length_numeric = calculate_condition_length_numeric(hc_2, x, t, sp.Rational(1, 2), 1)
    total_condition_length_numeric = pr_condition_length_numeric + rq_condition_length_numeric
    time_end = time.time()
    print(f"PR + RQ = {total_condition_length_numeric}, Time taken: {time_end - time_start} seconds (by calculate_condition_length_numeric)\n")


    # Calculate the condition length of PR + RQ using the non-linear homotopy path.
    print("Non-linear homotopy path (P -> R -> Q):")

    time_start = time.time()
    hc_3 = sp.Poly(x**2 + sp.cos(sp.pi * (t - 1))* x + sp.sin(sp.pi * (t - 1)) - 1, x)
    condition_length = calculate_condition_length(hc_3, x, t, 0, 1)
    total_condition_length = condition_length
    time_end = time.time()
    print(f"PR + RQ = {total_condition_length}, Time taken: {time_end - time_start} seconds (by calculate_condition_length)" )

    time_start = time.time()
    condition_length_numeric = calculate_condition_length_numeric(hc_3, x, t, 0, 1)
    time_end = time.time()
    print(f"PR + RQ = {condition_length_numeric}, Time taken: {time_end - time_start} seconds (by calculate_condition_length_numeric)\n")


    # Calculate the condition length of PQ using the linear homotopy path.
    print("Linear homotopy path (P -> Q):")

    time_start = time.time()
    pq_condition_length = calculate_linear_condition_length(p_coeffs, q_coeffs)
    time_end = time.time()
    print(f"PQ = {pq_condition_length}, Time taken: {time_end - time_start} seconds (by calculate_linear_condition_length)" )

    time_start = time.time()
    pq_condition_length_numeric = calculate_linear_condition_length_numeric(p_coeffs, q_coeffs)
    time_end = time.time()
    print(f"PQ = {pq_condition_length_numeric}, Time taken: {time_end - time_start} seconds (by calculate_linear_condition_length_numeric)" )

    hc_4 = sp.Poly((1 - t) * p + t * q, x)
    time_start = time.time()
    pq_condition_length = calculate_condition_length(hc_4, x, t, 0, 1)
    time_end = time.time()
    print(f"PQ = {pq_condition_length}, Time taken: {time_end - time_start} seconds (by calculate_condition_length)" )

    time_start = time.time()
    pq_condition_length_numeric = calculate_condition_length_numeric(hc_4, x, t, 0, 1)
    time_end = time.time()
    print(f"PQ = {pq_condition_length_numeric}, Time taken: {time_end - time_start} seconds (by calculate_condition_length_numeric)\n")


    # Calculate the condition length of PS + SQ using the linear homotopy path.
    print("Linear homotopy path (P -> S -> Q):")

    time_start = time.time()
    ps_condition_length = calculate_linear_condition_length(p_coeffs, s_coeffs)
    sq_condition_length = calculate_linear_condition_length(s_coeffs, q_coeffs)
    total_condition_length = ps_condition_length + sq_condition_length
    time_end = time.time()
    print(f"PS + SQ = {total_condition_length}, Time taken: {time_end - time_start} seconds (by calculate_linear_condition_length)" )

    time_start = time.time()
    ps_condition_length_numeric = calculate_linear_condition_length_numeric(p_coeffs, s_coeffs, regularization_eps=None)
    sq_condition_length_numeric = calculate_linear_condition_length_numeric(s_coeffs, q_coeffs, regularization_eps=None)
    total_condition_length_numeric = ps_condition_length_numeric + sq_condition_length_numeric
    time_end = time.time()
    print(f"PS + SQ = {total_condition_length_numeric}, Time taken: {time_end - time_start} seconds (by calculate_linear_condition_length_numeric)" )

    hc_5 = sp.Poly((1 - t) * p + t * s, x)
    hc_6 = sp.Poly((1 - t) * s + t * q, x)
    time_start = time.time()
    ps_condition_length = calculate_condition_length(hc_5, x, t, 0, 1)
    sq_condition_length = calculate_condition_length(hc_6, x, t, 0, 1)
    total_condition_length = ps_condition_length + sq_condition_length
    time_end = time.time()
    print(f"PS + SQ = {total_condition_length}, Time taken: {time_end - time_start} seconds (by calculate_condition_length)" )

    time_start = time.time()
    ps_condition_length_numeric = calculate_condition_length_numeric(hc_5, x, t, 0, 1)
    sq_condition_length_numeric = calculate_condition_length_numeric(hc_6, x, t, 0, 1)
    total_condition_length_numeric = ps_condition_length_numeric + sq_condition_length_numeric
    time_end = time.time()
    print(f"PS + SQ = {total_condition_length_numeric}, Time taken: {time_end - time_start} seconds (by calculate_condition_length_numeric)\n")


    # Calculate the condition length of AC + CB using the linear homotopy path.
    print("Linear homotopy path (A -> C -> B):")

    time_start = time.time()
    ac_condition_length_numeric = calculate_linear_condition_length_numeric(a_coeffs, c_coeffs, log_min=True, log_fn=print)
    cb_condition_length_numeric = calculate_linear_condition_length_numeric(c_coeffs, b_coeffs, log_min=True, log_fn=print)
    total_condition_length = ac_condition_length_numeric + cb_condition_length_numeric
    time_end = time.time()
    print(f"AC + CB = {total_condition_length}, Time taken: {time_end - time_start} seconds (by calculate_linear_condition_length_numeric)" )

    time_start = time.time()
    ac_condition_length = calculate_linear_condition_length(a_coeffs, c_coeffs, num_samples=1000, log_min=True, log_fn=print)
    cb_condition_length = calculate_linear_condition_length(c_coeffs, b_coeffs, num_samples=1000, log_min=True, log_fn=print)
    total_condition_length = ac_condition_length + cb_condition_length
    time_end = time.time()
    print(f"AC + CB = {total_condition_length}, Time taken: {time_end - time_start} seconds (by calculate_linear_condition_length)" )


    # Calculate the condition length of AB using the linear homotopy path.
    print("Linear homotopy path (A -> B):")

    time_start = time.time()
    ab_condition_length_numeric = calculate_linear_condition_length_numeric(
        a_coeffs, 
        b_coeffs,
        zero_tol=1e-12,
        num_samples=200,
        use_root_finder=True,
        root_tol=1e-12,
        log_min=False,
        log_fn=print,
    )
    time_end = time.time()
    print(f"AB = {ab_condition_length_numeric}, Time taken: {time_end - time_start} seconds (by calculate_linear_condition_length_numeric)" )

    time_start = time.time()
    ab_condition_length = calculate_linear_condition_length(
        a_coeffs,
        b_coeffs,
        zero_tol=1e-12,
        num_samples=2,
        use_root_finder=True,
        root_tol=1e-12,
        log_min=False,
        log_fn=print,
    )
    time_end = time.time()
    print(f"AB = {ab_condition_length}, Time taken: {time_end - time_start} seconds (by calculate_linear_condition_length)" )


if __name__ == "__main__":
    main()
