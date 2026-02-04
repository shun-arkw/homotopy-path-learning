from __future__ import annotations

# Renamed from bezier_torch_scoring.py to match the "*_torch_screen.py" naming convention.

import torch

from ...config import BezierLossConfig
from ..bezier import bezier_eval
from ..complex_repr import p_to_monic_poly_coeffs_ri
from ..discriminant_calculator import (
    discriminant_univariate_logabs,
    poly_derivative_coeffs_ri,
    sylvester_matrix_univariate_real_block,
)


@torch.no_grad()
def min_disc_logabs_on_bezier_torch(
    P_ctrl_ri: torch.Tensor,
    *,
    loss_cfg: BezierLossConfig = BezierLossConfig(),
    coarse_samples: int = 257,
    refine_steps: int = 2,
    refine_samples: int = 257,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate min_t log|Disc(T(t))| for a Bezier coefficient path T(t), t∈[0,1]."""
    if P_ctrl_ri.ndim != 3 or P_ctrl_ri.shape[-1] != 2:
        raise ValueError("P_ctrl_ri must have shape (d+1, degree, 2).")
    if coarse_samples < 2 or refine_samples < 2:
        raise ValueError("coarse_samples/refine_samples must be >= 2.")

    device, dtype = P_ctrl_ri.device, P_ctrl_ri.dtype
    degree = int(P_ctrl_ri.shape[1])

    def eval_logabs(ts: torch.Tensor) -> torch.Tensor:
        T = bezier_eval(P_ctrl_ri, ts, method=loss_cfg.bezier_eval_method)  # (S, degree, 2)
        a_ri = p_to_monic_poly_coeffs_ri(T)  # (S, degree+1, 2)
        return discriminant_univariate_logabs(
            a_ri,
            eps=loss_cfg.disc_eps,
            lead_eps=loss_cfg.lead_eps,
            backend=getattr(loss_cfg, "disc_backend", "complex"),
        )

    ts0 = torch.linspace(0.0, 1.0, int(coarse_samples), device=device, dtype=dtype)
    log0 = eval_logabs(ts0)
    min_logabs, idx0 = log0.min(dim=0)
    t_at_min = ts0[idx0]

    S0 = int(ts0.numel())
    lo = max(0, int(idx0) - 1)
    hi = min(S0 - 1, int(idx0) + 1)
    t_lo = ts0[lo]
    t_hi = ts0[hi]

    for _ in range(int(refine_steps)):
        u = torch.linspace(0.0, 1.0, int(refine_samples), device=device, dtype=dtype)
        tsr = t_lo + (t_hi - t_lo) * u
        logr = eval_logabs(tsr)
        min_logabs, idx = logr.min(dim=0)
        t_at_min = tsr[idx]

        Sr = int(tsr.numel())
        lo = max(0, int(idx) - 1)
        hi = min(Sr - 1, int(idx) + 1)
        t_lo = tsr[lo]
        t_hi = tsr[hi]

    return min_logabs, t_at_min


@torch.no_grad()
def min_sylvester_rel_sigma_min_on_bezier_torch(
    P_ctrl_ri: torch.Tensor,
    *,
    loss_cfg: BezierLossConfig = BezierLossConfig(),
    coarse_samples: int = 65,
    refine_steps: int = 1,
    refine_samples: int = 65,
    svd_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate min_t log(rel sigma_min(M(t))) for Sylvester real-block matrix along Bezier."""
    if P_ctrl_ri.ndim != 3 or P_ctrl_ri.shape[-1] != 2:
        raise ValueError("P_ctrl_ri must have shape (d+1, degree, 2).")
    if coarse_samples < 2 or refine_samples < 2:
        raise ValueError("coarse_samples/refine_samples must be >= 2.")

    device, dtype = P_ctrl_ri.device, P_ctrl_ri.dtype
    degree = int(P_ctrl_ri.shape[1])

    def eval_log_rel(ts: torch.Tensor) -> torch.Tensor:
        T = bezier_eval(P_ctrl_ri, ts, method=loss_cfg.bezier_eval_method)  # (S, degree, 2)
        a_ri = p_to_monic_poly_coeffs_ri(T)  # (S, degree+1, 2)

        a_re, a_im = a_ri[..., 0], a_ri[..., 1]  # (S, deg+1)
        fp_re, fp_im = poly_derivative_coeffs_ri(a_re, a_im)  # (S, deg)
        M = sylvester_matrix_univariate_real_block(a_re, a_im, fp_re, fp_im)  # (S, 2k, 2k)

        MtM = M.transpose(-2, -1) @ M
        lam = torch.linalg.eigvalsh(MtM)  # ascending
        lam_min = lam[..., 0]
        sigma_min = torch.sqrt(torch.clamp(lam_min, min=0.0))
        fro = torch.linalg.norm(M, ord="fro", dim=(-2, -1))
        rel = sigma_min / (fro + float(svd_eps))
        rel = torch.clamp(rel, min=torch.finfo(dtype).tiny)
        return torch.log(rel)

    ts0 = torch.linspace(0.0, 1.0, int(coarse_samples), device=device, dtype=dtype)
    log0 = eval_log_rel(ts0)
    min_logrel, idx0 = log0.min(dim=0)
    t_at_min = ts0[idx0]

    S0 = int(ts0.numel())
    lo = max(0, int(idx0) - 1)
    hi = min(S0 - 1, int(idx0) + 1)
    t_lo = ts0[lo]
    t_hi = ts0[hi]

    for _ in range(int(refine_steps)):
        u = torch.linspace(0.0, 1.0, int(refine_samples), device=device, dtype=dtype)
        tsr = t_lo + (t_hi - t_lo) * u
        logr = eval_log_rel(tsr)
        min_logrel, idx = logr.min(dim=0)
        t_at_min = tsr[idx]

        Sr = int(tsr.numel())
        lo = max(0, int(idx) - 1)
        hi = min(Sr - 1, int(idx) + 1)
        t_lo = tsr[lo]
        t_hi = tsr[hi]

    return min_logrel, t_at_min


@torch.no_grad()
def bezier_torch_screen_summary(
    P_ctrl_ri: torch.Tensor,
    *,
    loss_cfg: BezierLossConfig = BezierLossConfig(),
    disc_coarse_samples: int = 257,
    disc_refine_steps: int = 2,
    disc_refine_samples: int = 257,
    sylvester_smin_coarse_samples: int = 33,
    sylvester_smin_refine_steps: int = 1,
    sylvester_smin_refine_samples: int = 33,
    sylvester_smin_on_cpu: bool = True,
) -> dict:
    """Compact torch-only screen summary for a Bezier coefficient path."""
    device, dtype = P_ctrl_ri.device, P_ctrl_ri.dtype

    min_logabs, t_star = min_disc_logabs_on_bezier_torch(
        P_ctrl_ri,
        loss_cfg=loss_cfg,
        coarse_samples=int(disc_coarse_samples),
        refine_steps=int(disc_refine_steps),
        refine_samples=int(disc_refine_samples),
    )
    min_logabs_disc = float(min_logabs.item())
    min_abs_disc = float(torch.exp(min_logabs).item())
    disc_t_star = float(t_star.item())

    P_smin = P_ctrl_ri.detach()
    if bool(sylvester_smin_on_cpu):
        P_smin = P_smin.to(device=torch.device("cpu"))
    min_logrel, t_smin = min_sylvester_rel_sigma_min_on_bezier_torch(
        P_smin,
        loss_cfg=loss_cfg,
        coarse_samples=int(sylvester_smin_coarse_samples),
        refine_steps=int(sylvester_smin_refine_steps),
        refine_samples=int(sylvester_smin_refine_samples),
    )
    min_logrel_smin = float(min_logrel.item())
    min_rel_smin = float(torch.exp(min_logrel).item())
    smin_t_star = float(t_smin.item())

    return {
        "min_logabs_disc": min_logabs_disc,
        "min_abs_disc": min_abs_disc,
        "disc_t_star": disc_t_star,
        "min_logrel_smin": min_logrel_smin,
        "min_rel_smin": min_rel_smin,
        "smin_t_star": smin_t_star,
        "device": str(device),
        "dtype": str(dtype),
    }


