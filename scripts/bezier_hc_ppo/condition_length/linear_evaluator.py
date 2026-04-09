"""Linear (2-point) condition length for full (non-monic) polynomial coefficient paths."""
from __future__ import annotations

import torch

from .coeffs import full_coeffs_ascending_to_descending_ri
from .complex_repr import complex_norm_ri
from .config import ConditionLengthConfig
from .discriminant_calculator import discriminant_univariate_logabs
from .log_stability import (
    log_softabs_from_logabs,
    log_softabs_plus_eps,
    make_uniform_ts,
)


def calculate_linear_condition_length_numeric(
    P_ri: torch.Tensor,
    *,
    loss_cfg: ConditionLengthConfig | None = None,
) -> torch.Tensor:
    """Condition length for a straight line between two coefficient vectors (full coeffs).

    P_ri: shape (2, degree+1, 2) in (Re, Im), ascending power [a_0,...,a_degree] per row.
    Supports non-monic polynomials.
    """
    if loss_cfg is None:
        loss_cfg = ConditionLengthConfig()
    if P_ri.ndim != 3 or P_ri.shape[0] != 2 or P_ri.shape[-1] != 2:
        raise ValueError("P_ri must have shape (2, degree+1, 2).")
    degree = int(P_ri.shape[1] - 1)
    if degree < 2:
        raise ValueError("degree must be >= 2.")

    device, dtype = P_ri.device, P_ri.dtype
    M = int(loss_cfg.samples_per_segment)
    if M < 1:
        raise ValueError("loss_cfg.samples_per_segment must be >= 1.")

    seg_len = complex_norm_ri(P_ri[1] - P_ri[0])
    ts = make_uniform_ts(M, device=device, dtype=dtype)
    t = ts.view(M, 1, 1)
    gamma = (1.0 - t) * P_ri[0:1] + t * P_ri[1:2]  # (M, degree+1, 2) ascending
    a_ri = full_coeffs_ascending_to_descending_ri(gamma)  # (M, degree+1, 2)

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
    return seg_len * w_mean
