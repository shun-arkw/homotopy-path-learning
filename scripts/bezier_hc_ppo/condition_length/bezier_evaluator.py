"""Bezier path condition length for full (non-monic) polynomial coefficient paths."""
from __future__ import annotations

import torch

from .bezier import bezier_eval
from .complex_repr import complex_norm_ri
from .config import ConditionLengthConfig
from .discriminant_calculator import discriminant_univariate_logabs
from .log_stability import (
    log_softabs_from_logabs,
    log_softabs_plus_eps,
    make_uniform_ts,
)


def calculate_bezier_condition_length_numeric(
    P_ri: torch.Tensor,
    *,
    loss_cfg: ConditionLengthConfig | None = None,
) -> torch.Tensor:
    """Condition length for a Bezier curve between control points (full coeffs).

    P_ri: shape (d+1, degree+1, 2) in (Re, Im), descending power [a_degree,...,a_0]
    per control point. Supports non-monic polynomials.
    """
    if loss_cfg is None:
        loss_cfg = ConditionLengthConfig()
    if P_ri.ndim != 3 or P_ri.shape[-1] != 2:
        raise ValueError("P_ri must have shape (d+1, degree+1, 2).")
    degree = int(P_ri.shape[1] - 1)
    if degree < 2:
        raise ValueError("degree must be >= 2.")

    device, dtype = P_ri.device, P_ri.dtype
    M = int(loss_cfg.samples_per_segment)
    if M < 1:
        raise ValueError("loss_cfg.samples_per_segment must be >= 1.")

    ts = make_uniform_ts(M, device=device, dtype=dtype)
    a_ri = bezier_eval(P_ri, ts)  # (M, degree+1, 2) descending

    # Arc length (polyline through M samples)
    seg_diffs = a_ri[1:] - a_ri[:-1]  # (M-1, degree+1, 2)
    arc_len = complex_norm_ri(seg_diffs).sum()
    disc_logabs = discriminant_univariate_logabs(
        a_ri,
        eps=loss_cfg.disc_eps,
        lead_eps=loss_cfg.lead_eps,
        backend=getattr(loss_cfg, "disc_backend", "complex"),
    )  # (M,)

    log_softabs = log_softabs_from_logabs(disc_logabs, loss_cfg.delta_soft)
    log_softabs_eps = log_softabs_plus_eps(log_softabs, loss_cfg.eps_soft)
    degree_f = torch.tensor(float(degree), device=device, dtype=dtype)
    log_denom = log_softabs_eps / degree_f
    w = torch.exp(-log_denom)
    w_mean = w.mean()
    return arc_len * w_mean
