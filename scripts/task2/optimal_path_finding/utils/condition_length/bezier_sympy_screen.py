from __future__ import annotations

import sympy as sp
import torch


def _as_rational(x: float, *, zero_tol: float = 0.0, max_denominator: int | None = None) -> sp.Rational:
    """Convert float to Rational (optionally snapping tiny values to 0)."""
    xf = float(x)
    if zero_tol > 0 and abs(xf) <= zero_tol:
        return sp.Integer(0)
    r = sp.Rational(str(xf))
    if max_denominator is not None:
        r = r.limit_denominator(int(max_denominator))
    return r


def bezier_disc_abs2_has_root_in_unit_interval(
    P_ctrl_ri: torch.Tensor,
    *,
    coeff_zero_tol: float = 0.0,
    max_denominator: int | None = 10**6,
    imag_tol: float = 1e-10,
    interval_tol: float = 1e-12,
    nroots_digits: int = 50,
    nroots_maxsteps: int = 200,
    clear_denoms: bool = True,
    conservative_on_failure: bool = True,
) -> bool:
    """Return True if |Disc(T(t))|^2 has a (near-)real root in t∈[0,1] for Bezier coefficient path.

    This is intended for *evaluation-time* verification only (slow).

    Args:
        P_ctrl_ri: Bezier control points, shape (d+1, degree, 2) in (Re, Im).
        coeff_zero_tol: Snap tiny coefficients to 0 when rationalizing.
        max_denominator: Denominator bound for rationalization (or None).
        imag_tol: Tolerance for considering a numeric root real.
        interval_tol: Tolerance for considering a real root inside [0,1].
        nroots_digits/maxsteps: SymPy nroots controls.
        clear_denoms: Whether to clear denominators of |Disc|^2 polynomial (often helps).
        conservative_on_failure: If True, return True when SymPy fails.
    """
    if P_ctrl_ri.ndim != 3 or P_ctrl_ri.shape[-1] != 2:
        raise ValueError("P_ctrl_ri must have shape (d+1, degree, 2).")

    d = int(P_ctrl_ri.shape[0] - 1)     # Bezier degree
    degree = int(P_ctrl_ri.shape[1])    # polynomial degree (monic)
    if degree < 2:
        return False
    if d < 1:
        return False

    # Move to CPU python floats
    P = P_ctrl_ri.detach().cpu().tolist()  # shape (d+1, degree, 2)

    t_sym = sp.Symbol("t", real=True)
    x_sym = sp.Symbol("x")

    # trailing coefficients a_{deg-1}(t),...,a_0(t) as Bezier in t
    trailing_coeffs_t: list[sp.Expr] = []
    for coeff_idx in range(degree):
        expr = 0
        for j in range(d + 1):
            re, im = P[j][coeff_idx]
            cj = _as_rational(re, zero_tol=coeff_zero_tol, max_denominator=max_denominator) + sp.I * _as_rational(
                im, zero_tol=coeff_zero_tol, max_denominator=max_denominator
            )
            expr += sp.binomial(d, j) * (1 - t_sym) ** (d - j) * (t_sym**j) * cj
        trailing_coeffs_t.append(sp.expand(expr))

    # f(x,t) = x^degree + a_{degree-1}(t) x^{degree-1} + ... + a_0(t)
    poly_in_x = x_sym**degree
    for idx, a_t in enumerate(trailing_coeffs_t):
        power = degree - 1 - idx
        poly_in_x += a_t * (x_sym**power)

    try:
        disc_expr = sp.discriminant(poly_in_x, x_sym)
        disc_abs2_expr = sp.expand(disc_expr * sp.conjugate(disc_expr))
        disc_abs2_poly = sp.Poly(disc_abs2_expr, t_sym)

        if clear_denoms:
            try:
                a, b = disc_abs2_poly.clear_denoms()
                if isinstance(a, sp.Poly):
                    disc_abs2_poly, _den = a, b
                else:
                    _den, disc_abs2_poly = a, b
            except Exception:
                pass

        # Endpoint checks (exact if simplification succeeds)
        try:
            if sp.simplify(disc_abs2_poly.eval(0)) == 0:
                return True
            if sp.simplify(disc_abs2_poly.eval(1)) == 0:
                return True
        except Exception:
            pass

        # Deterministic real-root counting in [0,1]
        try:
            cnt = int(disc_abs2_poly.count_roots(0, 1))
            return cnt > 0
        except Exception:
            # fall back: nroots on disc(t) first
            try:
                poly_disc = sp.Poly(disc_expr, t_sym)
                roots_disc = poly_disc.nroots(n=nroots_digits, maxsteps=int(nroots_maxsteps))
                for r in roots_disc:
                    if abs(sp.im(r)) <= imag_tol:
                        rr = float(sp.re(r))
                        if -interval_tol <= rr <= 1.0 + interval_tol:
                            return True
            except Exception:
                pass

            roots = disc_abs2_poly.nroots(n=nroots_digits, maxsteps=int(nroots_maxsteps))
            for r in roots:
                if abs(sp.im(r)) <= imag_tol:
                    rr = float(sp.re(r))
                    if -interval_tol <= rr <= 1.0 + interval_tol:
                        return True
            return False
    except Exception:
        return bool(conservative_on_failure)


