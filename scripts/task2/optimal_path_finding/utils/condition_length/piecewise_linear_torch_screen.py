from __future__ import annotations

import torch

from ...config import PiecewiseLossConfig
from ..complex_repr import p_to_monic_poly_coeffs_ri
from ..discriminant_calculator import (
    discriminant_univariate_logabs,
    poly_derivative_coeffs_ri,
    sylvester_matrix_univariate_real_block,
)


@torch.no_grad()
def min_disc_logabs_per_segment_torch(
    P_ri: torch.Tensor,
    *,
    loss_cfg: PiecewiseLossConfig = PiecewiseLossConfig(),
    coarse_samples: int = 65,
    refine_steps: int = 3,
    refine_samples: int = 65,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-segment min log|Disc| by 1D coarse-to-fine search in t∈[0,1].

    This is a fast, SymPy-free screening heuristic to detect whether a linear segment
    in coefficient space comes close to the discriminant variety (Disc=0).

    Args:
        P_ri: Control points, shape (K+1, degree, 2) in (Re, Im) format.
        loss_cfg: Discriminant evaluation config (uses disc_eps/lead_eps).
        coarse_samples: Number of initial grid samples on [0,1] (>= 2).
        refine_steps: Number of refinement rounds around the current argmin.
        refine_samples: Samples per refinement round (>= 2).

    Returns:
        (min_logabs, t_at_min) each shape (K,).
    """
    if P_ri.ndim != 3 or P_ri.shape[-1] != 2:
        raise ValueError("P_ri must have shape (K+1, degree, 2).")
    if P_ri.shape[0] < 2:
        raise ValueError("P_ri must contain at least two control points (K+1 >= 2).")
    if coarse_samples < 2 or refine_samples < 2:
        raise ValueError("coarse_samples/refine_samples must be >= 2.")

    device, dtype = P_ri.device, P_ri.dtype
    K = int(P_ri.shape[0] - 1)
    degree = int(P_ri.shape[1])

    P0s = P_ri[:-1]  # (K, degree, 2)
    P1s = P_ri[1:]   # (K, degree, 2)

    def eval_logabs(ts: torch.Tensor) -> torch.Tensor:
        # ts shape (S,) or (K,S); output logabs shape (K,S)
        if ts.ndim == 1:
            t = ts.view(1, -1, 1, 1)  # (1,S,1,1)
        elif ts.ndim == 2 and ts.shape[0] == K:
            t = ts.view(K, -1, 1, 1)  # (K,S,1,1)
        else:
            raise ValueError("ts must have shape (S,) or (K,S).")

        gamma = (1.0 - t) * P0s.unsqueeze(1) + t * P1s.unsqueeze(1)  # (K,S,degree,2)
        gamma_flat = gamma.reshape(-1, degree, 2)  # (K*S, degree, 2)
        a_ri = p_to_monic_poly_coeffs_ri(gamma_flat)  # (K*S, degree+1, 2)
        disc_logabs = discriminant_univariate_logabs(
            a_ri,
            eps=loss_cfg.disc_eps,
            lead_eps=loss_cfg.lead_eps,
            backend=getattr(loss_cfg, "disc_backend", "complex"),
        )
        return disc_logabs.view(K, -1)

    # Coarse grid on [0,1], includes endpoints (so endpoint Disc=0 is caught)
    ts0 = torch.linspace(0.0, 1.0, int(coarse_samples), device=device, dtype=dtype)  # (S0,)
    logabs0 = eval_logabs(ts0)  # (K,S0)
    min_logabs, argmin = logabs0.min(dim=1)  # (K,), (K,)
    t_at_min = ts0[argmin]  # (K,)

    # Initial bracket per segment: neighbor interval around argmin on the coarse grid
    S0 = int(ts0.numel())
    idx0 = argmin.clamp(min=0, max=S0 - 1)
    idx0_lo = (idx0 - 1).clamp(min=0)
    idx0_hi = (idx0 + 1).clamp(max=S0 - 1)
    t_lo = ts0[idx0_lo]  # (K,)
    t_hi = ts0[idx0_hi]  # (K,)

    # Refinement: repeatedly shrink [t_lo,t_hi] around the current argmin on a local grid
    for _ in range(int(refine_steps)):
        u = torch.linspace(0.0, 1.0, int(refine_samples), device=device, dtype=dtype).view(1, -1)  # (1,Sr)
        ts_ref = t_lo.view(-1, 1) + (t_hi - t_lo).view(-1, 1) * u  # (K,Sr)
        logabs_ref = eval_logabs(ts_ref)  # (K,Sr)

        min_logabs, argmin = logabs_ref.min(dim=1)  # (K,)
        t_at_min = ts_ref[torch.arange(K, device=device), argmin]

        Sr = int(ts_ref.shape[1])
        idx = argmin.clamp(min=0, max=Sr - 1)
        idx_lo = (idx - 1).clamp(min=0)
        idx_hi = (idx + 1).clamp(max=Sr - 1)
        t_lo = ts_ref[torch.arange(K, device=device), idx_lo]
        t_hi = ts_ref[torch.arange(K, device=device), idx_hi]

    return min_logabs, t_at_min


@torch.no_grad()
def min_sylvester_rel_sigma_min_per_segment_torch(
    P_ri: torch.Tensor,
    *,
    loss_cfg: PiecewiseLossConfig = PiecewiseLossConfig(),
    coarse_samples: int = 65,
    refine_steps: int = 3,
    refine_samples: int = 65,
    svd_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-segment min relative sigma_min of Sylvester real-block matrix.

    For each segment k and t∈[0,1], we form the monic polynomial coefficients a(t),
    build the Sylvester matrix S(f,f') and its real block matrix M, then compute:

        rel_sigma_min(t) = sigma_min(M(t)) / (||M(t)||_F + svd_eps)

    Disc(f)=0 implies resultant Res(f,f')=0, which implies S is singular and hence
    sigma_min(M)=0. In practice, rel_sigma_min is often a better "near-singularity"
    proxy than log|Disc| for high degrees where slogdet-based log|det| can be unstable.

    Returns:
        (min_log_rel_sigma, t_at_min) each shape (K,).
        We return log(rel_sigma_min) to compare on a log scale.
    """
    if P_ri.ndim != 3 or P_ri.shape[-1] != 2:
        raise ValueError("P_ri must have shape (K+1, degree, 2).")
    if P_ri.shape[0] < 2:
        raise ValueError("P_ri must contain at least two control points (K+1 >= 2).")
    if coarse_samples < 2 or refine_samples < 2:
        raise ValueError("coarse_samples/refine_samples must be >= 2.")

    device, dtype = P_ri.device, P_ri.dtype
    K = int(P_ri.shape[0] - 1)
    degree = int(P_ri.shape[1])

    P0s = P_ri[:-1]  # (K, degree, 2)
    P1s = P_ri[1:]   # (K, degree, 2)

    def eval_log_rel_sigma(ts: torch.Tensor) -> torch.Tensor:
        # ts shape (S,) or (K,S); output log_rel shape (K,S)
        if ts.ndim == 1:
            t = ts.view(1, -1, 1, 1)  # (1,S,1,1)
        elif ts.ndim == 2 and ts.shape[0] == K:
            t = ts.view(K, -1, 1, 1)  # (K,S,1,1)
        else:
            raise ValueError("ts must have shape (S,) or (K,S).")

        gamma = (1.0 - t) * P0s.unsqueeze(1) + t * P1s.unsqueeze(1)  # (K,S,degree,2)
        gamma_flat = gamma.reshape(-1, degree, 2)  # (K*S, degree, 2)
        a_ri = p_to_monic_poly_coeffs_ri(gamma_flat)  # (K*S, degree+1, 2)

        # Build Sylvester real-block matrix M for (f, f')
        a_re, a_im = a_ri[..., 0], a_ri[..., 1]  # (K*S, deg+1)
        fp_re, fp_im = poly_derivative_coeffs_ri(a_re, a_im)  # (K*S, deg)
        M = sylvester_matrix_univariate_real_block(a_re, a_im, fp_re, fp_im)  # (K*S, 2k, 2k) real

        MtM = M.transpose(-2, -1) @ M
        lam = torch.linalg.eigvalsh(MtM)  # ascending
        lam_min = lam[..., 0]
        sigma_min = torch.sqrt(torch.clamp(lam_min, min=0.0))
        fro = torch.linalg.norm(M, ord="fro", dim=(-2, -1))
        rel = sigma_min / (fro + float(svd_eps))
        rel = torch.clamp(rel, min=torch.finfo(dtype).tiny)
        return torch.log(rel).view(K, -1)

    ts0 = torch.linspace(0.0, 1.0, int(coarse_samples), device=device, dtype=dtype)
    logrel0 = eval_log_rel_sigma(ts0)  # (K,S0)
    min_logrel, argmin = logrel0.min(dim=1)  # (K,)
    t_at_min = ts0[argmin]

    S0 = int(ts0.numel())
    idx0 = argmin.clamp(min=0, max=S0 - 1)
    idx0_lo = (idx0 - 1).clamp(min=0)
    idx0_hi = (idx0 + 1).clamp(max=S0 - 1)
    t_lo = ts0[idx0_lo]
    t_hi = ts0[idx0_hi]

    for _ in range(int(refine_steps)):
        u = torch.linspace(0.0, 1.0, int(refine_samples), device=device, dtype=dtype).view(1, -1)
        ts_ref = t_lo.view(-1, 1) + (t_hi - t_lo).view(-1, 1) * u  # (K,Sr)
        logrel_ref = eval_log_rel_sigma(ts_ref)  # (K,Sr)
        min_logrel, argmin = logrel_ref.min(dim=1)
        t_at_min = ts_ref[torch.arange(K, device=device), argmin]

        Sr = int(ts_ref.shape[1])
        idx = argmin.clamp(min=0, max=Sr - 1)
        idx_lo = (idx - 1).clamp(min=0)
        idx_hi = (idx + 1).clamp(max=Sr - 1)
        t_lo = ts_ref[torch.arange(K, device=device), idx_lo]
        t_hi = ts_ref[torch.arange(K, device=device), idx_hi]

    return min_logrel, t_at_min


@torch.no_grad()
def torch_screen_summary(
    P_ri: torch.Tensor,
    *,
    loss_cfg: PiecewiseLossConfig = PiecewiseLossConfig(),
    disc_coarse_samples: int = 257,
    disc_refine_steps: int = 2,
    disc_refine_samples: int = 257,
    sylvester_smin_coarse_samples: int = 33,
    sylvester_smin_refine_steps: int = 1,
    sylvester_smin_refine_samples: int = 33,
    sylvester_smin_on_cpu: bool = True,
) -> dict:
    """Compute a compact torch-only screen summary for a piecewise-linear path."""
    min_logabs, t_star = min_disc_logabs_per_segment_torch(
        P_ri,
        loss_cfg=loss_cfg,
        coarse_samples=int(disc_coarse_samples),
        refine_steps=int(disc_refine_steps),
        refine_samples=int(disc_refine_samples),
    )
    idx_disc = int(min_logabs.argmin().item())
    min_logabs_disc = float(min_logabs.min().item())
    min_abs_disc = float(torch.exp(min_logabs.min()).item())
    disc_t_star = float(t_star[idx_disc].item())

    P_smin = P_ri.detach()
    if bool(sylvester_smin_on_cpu):
        P_smin = P_smin.to(device=torch.device("cpu"))
    min_logrel, t_smin = min_sylvester_rel_sigma_min_per_segment_torch(
        P_smin,
        loss_cfg=loss_cfg,
        coarse_samples=int(sylvester_smin_coarse_samples),
        refine_steps=int(sylvester_smin_refine_steps),
        refine_samples=int(sylvester_smin_refine_samples),
    )
    idx_smin = int(min_logrel.argmin().item())
    min_logrel_smin = float(min_logrel.min().item())
    min_rel_smin = float(torch.exp(min_logrel.min()).item())
    smin_t_star = float(t_smin[idx_smin].item())

    return {
        "min_logabs_disc": min_logabs_disc,
        "min_abs_disc": min_abs_disc,
        "disc_segment": idx_disc,
        "disc_t_star": disc_t_star,
        "min_logrel_smin": min_logrel_smin,
        "min_rel_smin": min_rel_smin,
        "smin_segment": idx_smin,
        "smin_t_star": smin_t_star,
    }


