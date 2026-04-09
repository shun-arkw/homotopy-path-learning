"""Bezier curve evaluation. No dependency on task2."""
from __future__ import annotations

import torch


def bezier_de_casteljau(P_ctrl_ri: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate Bezier curve using De Casteljau. P_ctrl_ri: (d+1, degree, 2), t: (M,). Returns (M, degree, 2)."""
    if P_ctrl_ri.ndim != 3 or P_ctrl_ri.shape[-1] != 2:
        raise ValueError("P_ctrl_ri must have shape (d+1, degree, 2).")
    if t.ndim != 1:
        raise ValueError("t must have shape (M,).")
    M = int(t.numel())
    Q = P_ctrl_ri.unsqueeze(0).expand(M, -1, -1, -1).contiguous()
    tt = t.view(M, 1, 1, 1)
    for _ in range(int(P_ctrl_ri.shape[0] - 1)):
        Q = (1.0 - tt) * Q[:, :-1] + tt * Q[:, 1:]
    return Q[:, 0]


_BERNSTEIN_CACHE: dict[tuple[int, str, int | None, torch.dtype], dict[str, torch.Tensor]] = {}


def _bernstein_static_cache(d: int, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    key = (int(d), str(device.type), device.index, dtype)
    cached = _BERNSTEIN_CACHE.get(key)
    if cached is not None:
        return cached
    dd = int(d)
    if dd < 1:
        raise ValueError("Need at least 2 control points (d>=1).")
    i = torch.arange(dd + 1, device=device, dtype=dtype)
    d_minus_i = torch.tensor(float(dd), device=device, dtype=dtype) - i
    lg_d1 = torch.lgamma(torch.tensor(float(dd + 1), device=device, dtype=dtype))
    lg_i1 = torch.lgamma(i + 1.0)
    lg_di1 = torch.lgamma(d_minus_i + 1.0)
    binom = torch.exp(lg_d1 - lg_i1 - lg_di1)
    cached = {"i": i, "d_minus_i": d_minus_i, "binom": binom}
    _BERNSTEIN_CACHE[key] = cached
    return cached


def bezier_bernstein(P_ctrl_ri: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate Bezier via Bernstein basis. P_ctrl_ri: (d+1, degree, 2), t: (M,). Returns (M, degree, 2)."""
    if P_ctrl_ri.ndim != 3 or P_ctrl_ri.shape[-1] != 2:
        raise ValueError("P_ctrl_ri must have shape (d+1, degree, 2).")
    if t.ndim != 1:
        raise ValueError("t must have shape (M,).")
    d = int(P_ctrl_ri.shape[0] - 1)
    device, dtype = P_ctrl_ri.device, P_ctrl_ri.dtype
    cache = _bernstein_static_cache(d, device=device, dtype=dtype)
    i, d_minus_i, binom = cache["i"], cache["d_minus_i"], cache["binom"]
    tt = t.to(device=device, dtype=dtype).view(-1, 1)
    one_minus = 1.0 - tt
    w = binom.view(1, -1) * (tt**i.view(1, -1)) * (one_minus**d_minus_i.view(1, -1))
    return torch.einsum("mi, ixy -> mxy", w, P_ctrl_ri)


def bezier_eval(P_ctrl_ri: torch.Tensor, t: torch.Tensor, *, method: str = "casteljau") -> torch.Tensor:
    """Evaluate Bezier curve. method='casteljau' (default) or 'bernstein'."""
    m = str(method).lower()
    if m in ("casteljau", "de_casteljau", "decasteljau"):
        return bezier_de_casteljau(P_ctrl_ri, t)
    if m in ("bernstein", "matrix"):
        return bezier_bernstein(P_ctrl_ri, t)
    raise ValueError(f"Unknown Bezier eval method: {method!r}. Use 'casteljau' or 'bernstein'.")


def bezier_derivative_control_points(P_ctrl_ri: torch.Tensor) -> torch.Tensor:
    """Q_i = d * (P_{i+1} - P_i). Returns (d, degree, 2)."""
    if P_ctrl_ri.ndim != 3 or P_ctrl_ri.shape[-1] != 2:
        raise ValueError("P_ctrl_ri must have shape (d+1, degree, 2).")
    d = int(P_ctrl_ri.shape[0] - 1)
    if d < 1:
        raise ValueError("Need at least 2 control points (d>=1).")
    return float(d) * (P_ctrl_ri[1:] - P_ctrl_ri[:-1])
