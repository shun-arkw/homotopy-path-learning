import sympy as sp
import time
import numpy as np


def has_d_distinct_real_roots(coeffs, tol_imag=1e-8, tol_sep=1e-6):
    roots = np.roots(np.asarray(coeffs, dtype=float))
    real_mask = np.abs(roots.imag) < tol_imag
    real_parts = np.sort(roots.real[real_mask])
    if len(real_parts) != len(coeffs) - 1:
        return False
    # check for repeated roots by checking adjacent differences
    if np.any(np.diff(real_parts) < tol_sep):
        return False
    return True


def main():

    t = sp.symbols('t', real=True)

    # expr = 2*sp.sqrt(2) * (
    #     sp.integrate(
    #         1 / abs(sp.sqrt((2*t - 1)**2 - 4*(-1 - 2*t))),  # 0 〜 1/2
    #         (t, 0, sp.Rational(1, 2))
    #     )
    #     + sp.integrate(
    #         1 / abs(sp.sqrt((2*t - 1)**2 - 4*(2*t - 3))),  # 1/2 〜 1
    #         (t, sp.Rational(1, 2), 1)
    #     )
    # )

    # # expr_2 = sp.pi * sp.integrate(1 / sp.sqrt(sp.Abs((2*t - 1)**2 - 4*(-1 - 2*t))), (t, 0, 1))

    # print(sp.N(expr, 15))

    x = sp.symbols('x')
    time_start = time.time()
    expr_2 = sp.Poly(x**10 + x**9 + x**8 + x**7 + x**6 + x**5 + x**4 + x**3 + x**2 + x + 1, x)
    roots = sp.solve(expr_2, x)
    time_end = time.time()
    print(roots)
    print(f"Time taken: {time_end - time_start} seconds")

    time_start = time.time()
    flg = has_d_distinct_real_roots(expr_2.all_coeffs())
    time_end = time.time()
    print(flg)
    print(f"Time taken: {time_end - time_start} seconds")

    time_start = time.time()
    roots = expr_2.nroots()
    time_end = time.time()
    print(roots)
    print(f"Time taken: {time_end - time_start} seconds")

if __name__ == "__main__":
    main()