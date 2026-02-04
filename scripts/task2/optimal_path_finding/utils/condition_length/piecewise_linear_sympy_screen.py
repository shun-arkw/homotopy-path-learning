from __future__ import annotations

import torch
import sympy as sp


def _as_rational(x: float, *, zero_tol: float = 0.0, max_denominator: int | None = None) -> sp.Rational:
    """Convert a Python float to a (limited) rational number."""
    xf = float(x)
    if zero_tol > 0 and abs(xf) <= zero_tol:
        return sp.Integer(0)
    r = sp.Rational(str(xf))
    if max_denominator is not None:
        r = r.limit_denominator(int(max_denominator))
    return r


def _compute_disc_and_disc_abs2_poly_sympy(
    start_coeffs_ri: torch.Tensor,
    end_coeffs_ri: torch.Tensor,
    *,
    coeff_zero_tol: float,
    max_denominator: int | None,
    clear_denoms: bool,
) -> tuple[sp.Symbol, sp.Expr, sp.Poly]:
    """Build discriminant and |discriminant|^2 polynomial in the segment parameter.

    Given two complex coefficient vectors (real/imag stacked) representing a line segment,
    this constructs:
      - disc(t) = discriminant(f(x,t), x)
      - poly_abs2(t) = Poly(|disc(t)|^2, t)

    Notes:
      - Coefficients are rationalized to improve robustness.
      - The caller is expected to handle exceptions and decide whether to be conservative.
    """
    if start_coeffs_ri.ndim != 2 or start_coeffs_ri.shape[-1] != 2:
        raise ValueError("start_coeffs_ri must have shape (degree, 2).")
    if end_coeffs_ri.ndim != 2 or end_coeffs_ri.shape[-1] != 2:
        raise ValueError("end_coeffs_ri must have shape (degree, 2).")
    if start_coeffs_ri.shape != end_coeffs_ri.shape:
        raise ValueError("start_coeffs_ri and end_coeffs_ri must have the same shape.")

    degree = int(start_coeffs_ri.shape[0])
    if degree < 2:
        # The discriminant is not meaningful for degrees < 2 in this screen.
        raise ValueError("degree must be >= 2.")

    # Move to CPU Python floats for SymPy.
    start_coeffs = start_coeffs_ri.detach().cpu().tolist()  # [[re, im], ...]
    end_coeffs = end_coeffs_ri.detach().cpu().tolist()

    t_sym = sp.Symbol("t", real=True)
    x_sym = sp.Symbol("x")

    trailing_coeffs_t: list[sp.Expr] = []
    for (re0, im0), (re1, im1) in zip(start_coeffs, end_coeffs):
        a0 = _as_rational(re0, zero_tol=coeff_zero_tol, max_denominator=max_denominator) + sp.I * _as_rational(
            im0, zero_tol=coeff_zero_tol, max_denominator=max_denominator
        )
        a1 = _as_rational(re1, zero_tol=coeff_zero_tol, max_denominator=max_denominator) + sp.I * _as_rational(
            im1, zero_tol=coeff_zero_tol, max_denominator=max_denominator
        )
        trailing_coeffs_t.append((1 - t_sym) * a0 + t_sym * a1)

    # f(x,t) = x^degree + a_{degree-1}(t) x^{degree-1} + ... + a_0(t)
    poly_in_x = x_sym**degree
    for idx, a_t in enumerate(trailing_coeffs_t):
        power = degree - 1 - idx
        poly_in_x += a_t * (x_sym**power)

    disc_expr = sp.discriminant(poly_in_x, x_sym)
    disc_abs2_expr = sp.expand(disc_expr * sp.conjugate(disc_expr))  # |Disc|^2 (real on real t)
    disc_abs2_poly = sp.Poly(disc_abs2_expr, t_sym)

    if clear_denoms:
        # Roots are invariant under scaling by a nonzero constant. Clearing denominators
        # often improves `nroots` stability for high-degree polynomials.
        try:
            a, b = disc_abs2_poly.clear_denoms()
            # SymPy version differences: some return (poly, den), others (den, poly).
            if isinstance(a, sp.Poly):
                disc_abs2_poly, _den = a, b
            else:
                _den, disc_abs2_poly = a, b
        except Exception:
            pass

    return t_sym, disc_expr, disc_abs2_poly


def _segment_disc_abs2_has_root_in_unit_interval(  # pyright: ignore[reportUnusedFunction]
    start_coeffs_ri: torch.Tensor,
    end_coeffs_ri: torch.Tensor,
    *,
    coeff_zero_tol: float = 0.0,
    max_denominator: int | None = 10**6,
    imag_tol: float = 1e-10,
    interval_tol: float = 1e-12,
    nroots_digits: int = 50,
    nroots_maxsteps: int = 50,
    clear_denoms: bool = True,
    conservative_on_failure: bool = True,
) -> bool:
    """Return True if the segment is flagged by the discriminant-zero screen.

    This is a thin wrapper around `_segment_disc_abs2_root_diagnostics` to avoid
    duplicated logic. It flags a segment if |Disc(t)|^2 has a (near-)real root in t∈[0,1].

    Notes:
      We always try deterministic `count_roots` first; if it fails (SymPy limitation),
      we fall back internally to numeric root finding via `nroots`.
    """
    diagnostics = _segment_disc_abs2_root_diagnostics(
        start_coeffs_ri,
        end_coeffs_ri,
        coeff_zero_tol=coeff_zero_tol,
        max_denominator=max_denominator,
        imag_tol=imag_tol,
        interval_tol=interval_tol,
        nroots_digits=nroots_digits,
        nroots_maxsteps=nroots_maxsteps,
        clear_denoms=clear_denoms,
        conservative_on_failure=conservative_on_failure,
    )
    return bool(diagnostics["hit"])


def diagnose_disc_zero_screen(
    P_ri: torch.Tensor,
    *,
    sympy_max_denominator: int | None = 10**6,
    sympy_nroots_digits: int = 50,
    sympy_nroots_maxsteps: int = 200,
    sympy_imag_tol: float = 1e-10,
    sympy_interval_tol: float = 1e-12,
    conservative_on_failure: bool = True,
    stop_at_first: bool = True,
) -> dict:
    """Diagnose why the SymPy Disc=0 screen flags a piecewise-linear path.

    This returns *which* segment was flagged. Useful to distinguish:
      - endpoint Disc==0
      - interior root detected by nroots
      - SymPy failure (and `conservative_on_failure=True`)

    Returns:
      dict with keys:
        - ok: bool (True iff no segments flagged)
        - issues: list[dict] with fields {segment:int, reason:str}
    """
    if P_ri.ndim != 3 or P_ri.shape[-1] != 2:
        raise ValueError("P_ri must have shape (K+1, degree, 2).")
    if P_ri.shape[0] < 2:
        raise ValueError("P_ri must contain at least two control points (K+1 >= 2).")

    control_points_ri = P_ri
    segment_start_coeffs_ri = control_points_ri[:-1]
    segment_end_coeffs_ri = control_points_ri[1:]
    num_segments = int(segment_start_coeffs_ri.shape[0])

    issues: list[dict] = []
    for segment_index in range(num_segments):
        diag = _segment_disc_abs2_root_diagnostics(
            segment_start_coeffs_ri[segment_index],
            segment_end_coeffs_ri[segment_index],
            max_denominator=sympy_max_denominator,
            nroots_digits=sympy_nroots_digits,
            nroots_maxsteps=sympy_nroots_maxsteps,
            imag_tol=sympy_imag_tol,
            interval_tol=sympy_interval_tol,
            conservative_on_failure=conservative_on_failure,
        )
        if bool(diag["hit"]):
            issues.append({"segment": segment_index, **diag})
            if stop_at_first:
                return {"ok": False, "issues": issues}

    return {"ok": len(issues) == 0, "issues": issues}


def _segment_disc_abs2_root_diagnostics(
    start_coeffs_ri: torch.Tensor,
    end_coeffs_ri: torch.Tensor,
    *,
    coeff_zero_tol: float = 0.0,
    max_denominator: int | None = 10**6,
    imag_tol: float = 1e-10,
    interval_tol: float = 1e-12,
    nroots_digits: int = 50,
    nroots_maxsteps: int = 200,
    clear_denoms: bool = True,
    conservative_on_failure: bool = True,
) -> dict:
    """Return diagnostics for the discriminant-zero screen on one segment.

    The segment is defined by two complex coefficient vectors (real/imag stacked) and the
    linear interpolation in the segment parameter t∈[0,1]. We flag a segment if the
    discriminant becomes zero (equivalently |Disc(t)|^2 has a real root) in [0,1].

    Returns a dict with at least:
      - hit: bool
      - reason: str

    Additional fields may include:
      - root: float, root_imag: float (when a root is identified numerically)
      - exception: str (when SymPy fails and conservative_on_failure is enabled)
      - count_roots_exception / disc_nroots_exception: str (when fallbacks fail)
    """
    if start_coeffs_ri.ndim != 2 or start_coeffs_ri.shape[-1] != 2:
        raise ValueError("start_coeffs_ri must have shape (degree, 2).")
    if end_coeffs_ri.ndim != 2 or end_coeffs_ri.shape[-1] != 2:
        raise ValueError("end_coeffs_ri must have shape (degree, 2).")
    if start_coeffs_ri.shape != end_coeffs_ri.shape:
        raise ValueError("start_coeffs_ri and end_coeffs_ri must have the same shape.")

    degree = int(start_coeffs_ri.shape[0])
    if degree < 2:
        return {"hit": False, "reason": "no_hit"}

    try:
        t_sym, disc_expr, disc_abs2_poly = _compute_disc_and_disc_abs2_poly_sympy(
            start_coeffs_ri,
            end_coeffs_ri,
            coeff_zero_tol=coeff_zero_tol,
            max_denominator=max_denominator,
            clear_denoms=clear_denoms,
        )
    except Exception as exc:
        hit = bool(conservative_on_failure)
        return {
            "hit": hit,
            "reason": "sympy_build_exception" if hit else "no_hit",
            "exception": f"{type(exc).__name__}: {exc}",
        }

    # Endpoint checks (exact if simplification succeeds).
    try:
        if sp.simplify(disc_abs2_poly.eval(0)) == 0:
            return {"hit": True, "reason": "endpoint_t0_disc_is_zero"}
        if sp.simplify(disc_abs2_poly.eval(1)) == 0:
            return {"hit": True, "reason": "endpoint_t1_disc_is_zero"}
    except Exception:
        # Keep going; we will rely on numeric methods below.
        pass

    # Deterministic real-root counting in [0,1] (avoids nroots convergence issues).
    # If this fails (SymPy limitation), fall back to numeric roots below.
    count_roots_exc: str | None = None
    disc_nroots_exc: str | None = None
    try:
        # disc_abs2_poly is a univariate polynomial in t with real coefficients.
        cnt = int(disc_abs2_poly.count_roots(0, 1))
        if cnt > 0:
            return {"hit": True, "reason": "count_roots_positive", "count_roots": cnt}
        return {"hit": False, "reason": "no_hit"}
    except Exception as exc:
        count_roots_exc = f"{type(exc).__name__}: {exc}"
        try:
            # Prefer nroots on disc(t) itself (lower degree) before nroots on |disc|^2.
            poly_disc = sp.Poly(disc_expr, t_sym)
            roots_disc = poly_disc.nroots(n=nroots_digits, maxsteps=int(nroots_maxsteps))
            for r in roots_disc:
                if abs(sp.im(r)) <= imag_tol:
                    rr = float(sp.re(r))
                    if -interval_tol <= rr <= 1.0 + interval_tol:
                        return {
                            "hit": True,
                            "reason": "disc_nroots_in_[0,1]",
                            "root": rr,
                            "root_imag": float(sp.im(r)),
                            "count_roots_exception": count_roots_exc,
                        }
            # Do not early-return here: even if disc_nroots found nothing, we still run
            # nroots on |disc|^2 for robustness (mirrors the original bool screen).
            disc_nroots_exc = None
        except Exception as exc2:
            disc_nroots_exc = f"{type(exc2).__name__}: {exc2}"

    try:
        roots = disc_abs2_poly.nroots(n=nroots_digits, maxsteps=int(nroots_maxsteps))
    except Exception as exc:
        hit = bool(conservative_on_failure)
        return {
            "hit": hit,
            "reason": "sympy_nroots_exception" if hit else "no_hit",
            "exception": f"{type(exc).__name__}: {exc}",
            **({"count_roots_exception": count_roots_exc} if count_roots_exc else {}),
            **({"disc_nroots_exception": disc_nroots_exc} if disc_nroots_exc else {}),
        }

    for r in roots:
        if abs(sp.im(r)) <= imag_tol:
            rr = float(sp.re(r))
            if -interval_tol <= rr <= 1.0 + interval_tol:
                return {"hit": True, "reason": "interior_root_in_[0,1]", "root": rr, "root_imag": float(sp.im(r))}

    out: dict = {"hit": False, "reason": "no_hit"}
    if count_roots_exc:
        out["count_roots_exception"] = count_roots_exc
    if disc_nroots_exc:
        out["disc_nroots_exception"] = disc_nroots_exc
    return out


