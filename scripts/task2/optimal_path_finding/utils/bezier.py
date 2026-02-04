from __future__ import annotations

"""
Bezier curve utilities.

Run as a module to keep relative imports stable:
    python3 -m scripts.task2.optimal_path_finding.utils.bezier
"""

import torch


def bezier_de_casteljau(P_ctrl_ri: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate a Bezier curve using De Casteljau (stable, autodiff-friendly).

    Args:
        P_ctrl_ri: Control points, shape (d+1, degree, 2) in (Re, Im).
        t: Parameters in [0,1], shape (M,).

    Returns:
        Curve points T(t), shape (M, degree, 2).
    """
    if P_ctrl_ri.ndim != 3 or P_ctrl_ri.shape[-1] != 2:
        raise ValueError("P_ctrl_ri must have shape (d+1, degree, 2).")
    if t.ndim != 1:
        raise ValueError("t must have shape (M,).")

    M = int(t.numel())
    # Broadcast control points to (M, d+1, degree, 2)
    Q = P_ctrl_ri.unsqueeze(0).expand(M, -1, -1, -1).contiguous()
    tt = t.view(M, 1, 1, 1)

    # Repeated linear interpolation
    for _ in range(int(P_ctrl_ri.shape[0] - 1)):
        Q = (1.0 - tt) * Q[:, :-1] + tt * Q[:, 1:]

    return Q[:, 0]


_BERNSTEIN_CACHE: dict[tuple[int, str, int | None, torch.dtype], dict[str, torch.Tensor]] = {}


def _bernstein_static_cache(d: int, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Cache tensors that only depend on (d, device, dtype) for Bernstein evaluation."""
    key = (int(d), str(device.type), device.index, dtype)
    cached = _BERNSTEIN_CACHE.get(key)
    if cached is not None:
        return cached

    dd = int(d)
    if dd < 1:
        raise ValueError("Need at least 2 control points (d>=1).")

    # i in [0..d]
    i = torch.arange(dd + 1, device=device, dtype=dtype)  # (d+1,)
    d_minus_i = torch.tensor(float(dd), device=device, dtype=dtype) - i  # (d+1,)

    # binom(d, i) computed stably via lgamma
    lg_d1 = torch.lgamma(torch.tensor(float(dd + 1), device=device, dtype=dtype))
    lg_i1 = torch.lgamma(i + 1.0)
    lg_di1 = torch.lgamma(d_minus_i + 1.0)
    binom = torch.exp(lg_d1 - lg_i1 - lg_di1)  # (d+1,)

    cached = {"i": i, "d_minus_i": d_minus_i, "binom": binom}
    _BERNSTEIN_CACHE[key] = cached
    return cached


def bezier_bernstein(P_ctrl_ri: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate a Bezier curve using Bernstein basis weights (matrix-like evaluation).

    This evaluates:
        T(t) = Σ_{i=0..d} binom(d,i) (1-t)^{d-i} t^i P_i

    Compared to De Casteljau:
      - Faster asymptotically: O(d) per t (vs O(d^2))
      - Typically faster for moderate degrees (e.g. d≈20+) on GPU/TPU
      - Potentially less numerically stable for very large d or t extremely close to 0/1

    Args:
        P_ctrl_ri: Control points, shape (d+1, degree, 2) in (Re, Im).
        t: Parameters in [0,1], shape (M,).

    Returns:
        Curve points T(t), shape (M, degree, 2).
    """
    if P_ctrl_ri.ndim != 3 or P_ctrl_ri.shape[-1] != 2:
        raise ValueError("P_ctrl_ri must have shape (d+1, degree, 2).")
    if t.ndim != 1:
        raise ValueError("t must have shape (M,).")

    d = int(P_ctrl_ri.shape[0] - 1)
    device, dtype = P_ctrl_ri.device, P_ctrl_ri.dtype
    cache = _bernstein_static_cache(d, device=device, dtype=dtype)
    i = cache["i"]  # (d+1,)
    d_minus_i = cache["d_minus_i"]  # (d+1,)
    binom = cache["binom"]  # (d+1,)

    tt = t.to(device=device, dtype=dtype).view(-1, 1)  # (M,1)
    one_minus = (1.0 - tt)
    # (M,d+1)
    w = binom.view(1, -1) * (tt**i.view(1, -1)) * (one_minus**d_minus_i.view(1, -1))

    # T[m, :, :] = sum_i w[m,i] * P_ctrl_ri[i, :, :]
    return torch.einsum("mi, ixy -> mxy", w, P_ctrl_ri)


def bezier_eval(P_ctrl_ri: torch.Tensor, t: torch.Tensor, *, method: str = "casteljau") -> torch.Tensor:
    """Evaluate Bezier curve with selectable backend.

    - method="casteljau": De Casteljau recursion (O(d^2), numerically stable; default)
    - method="bernstein": Bernstein basis linear combination (O(d), often faster for d≈20+)
    """
    m = str(method).lower()
    if m in ("casteljau", "de_casteljau", "decasteljau"):
        return bezier_de_casteljau(P_ctrl_ri, t)
    if m in ("bernstein", "matrix"):
        return bezier_bernstein(P_ctrl_ri, t)
    raise ValueError(f"Unknown Bezier eval method: {method!r}. Use 'casteljau' or 'bernstein'.")


def bezier_derivative_control_points(P_ctrl_ri: torch.Tensor) -> torch.Tensor:
    """Control points of the derivative Bezier curve.

    If T is degree-d Bezier with control points P_0..P_d, then T'(t) is degree-(d-1)
    Bezier with control points Q_i = d * (P_{i+1} - P_i).

    Args:
        P_ctrl_ri: (d+1, degree, 2)

    Returns:
        Q_ctrl_ri: (d, degree, 2)
    """
    if P_ctrl_ri.ndim != 3 or P_ctrl_ri.shape[-1] != 2:
        raise ValueError("P_ctrl_ri must have shape (d+1, degree, 2).")
    d = int(P_ctrl_ri.shape[0] - 1)
    if d < 1:
        raise ValueError("Need at least 2 control points (d>=1).")
    return float(d) * (P_ctrl_ri[1:] - P_ctrl_ri[:-1])


def bezier_sample_polyline(P_ctrl_ri: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Sample Bezier curve into (K+1) polyline points on uniform parameter grid.

    This is intended for screening/visualization, not for condition length evaluation.

    Args:
        P_ctrl_ri: (d+1, degree, 2)
        num_segments: K >= 1

    Returns:
        Q_ri: (K+1, degree, 2)
    """
    K = int(num_segments)
    if K < 1:
        raise ValueError("num_segments must be >= 1.")
    device, dtype = P_ctrl_ri.device, P_ctrl_ri.dtype
    ts = torch.linspace(0.0, 1.0, K + 1, device=device, dtype=dtype)
    return bezier_de_casteljau(P_ctrl_ri, ts)


