import time
import numpy as np
import sympy as sp
from condition_length_calculator import calculate_linear_condition_length_numeric


def build_monic_from_roots(roots: np.ndarray, x: sp.Symbol):
    """Return monic coefficients from given roots (domain=QQ for stability)."""
    roots_sym = [sp.Rational(r) for r in roots]
    poly = sp.expand(sp.prod([x - r for r in roots_sym]))  # monic polynomial
    poly_obj = sp.Poly(poly, x, domain="QQ")
    return poly_obj.all_coeffs()


def main():
    np.random.seed(42)

    zero_tol = 1e-10
    num_samples = 400
    regularization_eps = 1e-12
    x = sp.symbols("x")

    # print("Degree, Time (s), ConditionLength")
    for deg in range(2, 11):  # degrees 2..10
        # Deterministic, distinct real roots; target is a small shift to avoid collisions
        roots_start = np.linspace(1, deg, deg)
        delta = 0.5
        roots_target = roots_start + delta
        start_coeffs = build_monic_from_roots(roots_start, x)
        target_coeffs = build_monic_from_roots(roots_target, x)

        # print(f"start_coeffs: {start_coeffs}")
        # print(f"target_coeffs: {target_coeffs}")
        t0 = time.time()
        cond_len = calculate_linear_condition_length_numeric(
            start_coeffs,
            target_coeffs,
            regularization_eps=regularization_eps,
            zero_tol=zero_tol,
            num_samples=num_samples,
            use_root_finder=True,
            root_tol=1e-10,
            log_min=False,
            log_fn=print,
        )
        elapsed = time.time() - t0
        print("Degree, Time (s), ConditionLength")
        print(f"{deg}, {elapsed:.4f}, {cond_len}\n")



if __name__ == "__main__":
    main()
