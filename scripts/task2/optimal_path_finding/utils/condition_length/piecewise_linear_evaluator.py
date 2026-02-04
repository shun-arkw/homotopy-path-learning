from __future__ import annotations

# Renamed from evaluator.py to make the target path type explicit.

import torch

from ...config import PiecewiseLossConfig
from ..complex_repr import complex_norm_ri, p_to_monic_poly_coeffs_ri
from ..discriminant_calculator import discriminant_univariate_logabs
from ..log_stability import make_uniform_ts, log_softabs_from_logabs, log_softabs_plus_eps
from .piecewise_linear_sympy_screen import _segment_disc_abs2_has_root_in_unit_interval
from .piecewise_linear_torch_screen import min_disc_logabs_per_segment_torch, min_sylvester_rel_sigma_min_per_segment_torch


def calculate_piecewise_linear_condition_length_per_segment_numeric(
    P_ri: torch.Tensor,
    *,
    loss_cfg: PiecewiseLossConfig = PiecewiseLossConfig(),
    return_inf_if_disc_hits_zero: bool = True,
    fill_all_inf_if_any_disc_hit: bool = True,
    disc_zero_threshold: float = 1e-14,
    # (1) Singular-value screen using Sylvester matrix relative s_min proxy.
    # If enabled and `sympy_verify_suspicious_segments=False`, we conservatively return +inf
    # as soon as any segment falls below `sylvester_smin_rel_threshold`.
    use_sylvester_smin_screen: bool = False,
    sylvester_smin_rel_threshold: float = 1e-5,
    # (2) Optional SymPy verification of only suspicious segments found by (1).
    sympy_verify_suspicious_segments: bool = False,
    sympy_max_segments: int = 8,
    sympy_max_denominator: int | None = 10**6,
    sympy_nroots_digits: int = 50,
    sympy_nroots_maxsteps: int = 200,
    sympy_imag_tol: float = 1e-10,
    sympy_interval_tol: float = 1e-12,
    screen_coarse_samples: int = 129,
    screen_refine_steps: int = 2,
    screen_refine_samples: int = 129,
    sylvester_smin_coarse_samples: int = 33,
    sylvester_smin_refine_steps: int = 1,
    sylvester_smin_refine_samples: int = 33,
) -> torch.Tensor:
    """Per-segment condition length estimator for a piecewise-linear coefficient path.

    This returns the per-segment contributions whose sum equals
    `calculate_piecewise_linear_condition_length_numeric`.
    """
    if P_ri.ndim != 3 or P_ri.shape[-1] != 2:
        raise ValueError("P_ri must have shape (K+1, degree, 2).")
    if P_ri.shape[0] < 2:
        raise ValueError("P_ri must contain at least two control points (K+1 >= 2).")

    K = int(P_ri.shape[0] - 1)
    degree = int(P_ri.shape[1])
    if degree < 2:
        raise ValueError("degree must be >= 2 (monic polynomial degree).")

    device, dtype = P_ri.device, P_ri.dtype
    M = int(loss_cfg.samples_per_segment)
    if M < 1:
        raise ValueError("loss_cfg.samples_per_segment must be >= 1.")

    # ------------------------------------------------------------
    # Optional Disc≈0 screen (torch-only + optional SymPy verification)
    # ------------------------------------------------------------
    inf_mask: torch.Tensor | None = None
    if return_inf_if_disc_hits_zero:
        if disc_zero_threshold <= 0:
            raise ValueError("disc_zero_threshold must be > 0.")
        if sylvester_smin_rel_threshold <= 0:
            raise ValueError("sylvester_smin_rel_threshold must be > 0.")
        if sympy_max_segments < 1:
            raise ValueError("sympy_max_segments must be >= 1.")
        if bool(sympy_verify_suspicious_segments) and not bool(use_sylvester_smin_screen):
            raise ValueError("sympy_verify_suspicious_segments requires use_sylvester_smin_screen=True.")

        min_logabs, _tmin = min_disc_logabs_per_segment_torch(
            P_ri,
            loss_cfg=loss_cfg,
            coarse_samples=screen_coarse_samples,
            refine_steps=screen_refine_steps,
            refine_samples=screen_refine_samples,
        )
        thr_log = torch.tensor(float(disc_zero_threshold), device=device, dtype=dtype).log()
        # Non-finite (e.g. -inf when Disc==0) => treat as a hit.
        inf_mask = (~torch.isfinite(min_logabs)) | (min_logabs <= thr_log)
        if bool(inf_mask.any().item()) and bool(fill_all_inf_if_any_disc_hit):
            return torch.full((K,), float("inf"), device=device, dtype=dtype)

        # (1) Singular-value screen (Sylvester relative s_min proxy).
        if use_sylvester_smin_screen:
            P_cpu = P_ri.detach().to(device=torch.device("cpu"))
            min_logrel, _tstar = min_sylvester_rel_sigma_min_per_segment_torch(
                P_cpu,
                loss_cfg=loss_cfg,
                coarse_samples=sylvester_smin_coarse_samples,
                refine_steps=sylvester_smin_refine_steps,
                refine_samples=sylvester_smin_refine_samples,
            )
            log_thr = torch.tensor(
                float(sylvester_smin_rel_threshold), device=P_cpu.device, dtype=P_cpu.dtype
            ).log()
            suspicious = min_logrel <= log_thr

            # (1) only: conservative early exit if any segment is suspicious.
            if (not bool(sympy_verify_suspicious_segments)) and bool(suspicious.any().item()):
                if bool(fill_all_inf_if_any_disc_hit):
                    return torch.full((K,), float("inf"), device=device, dtype=dtype)
                susp_mask = suspicious.to(device=device)
                inf_mask = susp_mask if inf_mask is None else (inf_mask | susp_mask)

            # (1)+(2): verify only suspicious segments using SymPy root detection.
            if bool(sympy_verify_suspicious_segments) and bool(suspicious.any().item()):
                susp_idx = torch.where(suspicious)[0]
                order = torch.argsort(min_logrel[susp_idx])
                susp_idx = susp_idx[order][: int(sympy_max_segments)]

                P0s = P_cpu[:-1]
                P1s = P_cpu[1:]
                for k in susp_idx.tolist():
                    hit = _segment_disc_abs2_has_root_in_unit_interval(
                        P0s[int(k)],
                        P1s[int(k)],
                        max_denominator=sympy_max_denominator,
                        nroots_digits=sympy_nroots_digits,
                        nroots_maxsteps=sympy_nroots_maxsteps,
                        imag_tol=sympy_imag_tol,
                        interval_tol=sympy_interval_tol,
                        conservative_on_failure=True,
                    )
                    if bool(hit):
                        if bool(fill_all_inf_if_any_disc_hit):
                            return torch.full((K,), float("inf"), device=device, dtype=dtype)
                        if inf_mask is None:
                            inf_mask = torch.zeros((K,), device=device, dtype=torch.bool)
                        inf_mask[int(k)] = True

    # ------------------------------------------------------------
    # Condition-length integral approximation (per segment)
    # ------------------------------------------------------------
    P0s = P_ri[:-1]  # (K, degree, 2)
    P1s = P_ri[1:]   # (K, degree, 2)
    dP = P1s - P0s   # (K, degree, 2)
    seg_len = complex_norm_ri(dP)  # (K,)

    ts = make_uniform_ts(M, device=device, dtype=dtype)  # (M,)
    t = ts.view(1, M, 1, 1)  # (1, M, 1, 1)
    gamma = (1.0 - t) * P0s.unsqueeze(1) + t * P1s.unsqueeze(1)  # (K, M, degree, 2)
    gamma_flat = gamma.reshape(K * M, degree, 2)  # (K*M, degree, 2)
    a_ri = p_to_monic_poly_coeffs_ri(gamma_flat)  # (K*M, degree+1, 2)

    disc_logabs = discriminant_univariate_logabs(
        a_ri,
        eps=loss_cfg.disc_eps,
        lead_eps=loss_cfg.lead_eps,
        backend=getattr(loss_cfg, "disc_backend", "complex"),
    )  # (K*M,)
    disc_logabs = disc_logabs.view(K, M)

    log_softabs = log_softabs_from_logabs(disc_logabs, loss_cfg.delta_soft)
    log_softabs_eps = log_softabs_plus_eps(log_softabs, loss_cfg.eps_soft)

    degree_f = torch.tensor(float(degree), device=device, dtype=dtype)
    log_denom = log_softabs_eps / degree_f
    w = torch.exp(-log_denom)
    w_mean = w.mean(dim=1)  # (K,)
    cl = seg_len * w_mean
    if inf_mask is not None and bool(inf_mask.any().item()):
        inf_mask = inf_mask.to(device=device)
        cl = cl.clone()
        cl[inf_mask] = float("inf")
    return cl


def calculate_piecewise_linear_condition_length_numeric(
    P_ri: torch.Tensor,
    *,
    loss_cfg: PiecewiseLossConfig = PiecewiseLossConfig(),
    return_inf_if_disc_hits_zero: bool = True,
    disc_zero_threshold: float = 1e-14,
    # (1) Singular-value screen using Sylvester matrix relative s_min proxy.
    use_sylvester_smin_screen: bool = False,
    sylvester_smin_rel_threshold: float = 1e-5,
    # (2) Optional SymPy verification of only suspicious segments found by (1).
    sympy_verify_suspicious_segments: bool = False,
    sympy_max_segments: int = 8,
    sympy_max_denominator: int | None = 10**6,
    sympy_nroots_digits: int = 50,
    sympy_nroots_maxsteps: int = 200,
    sympy_imag_tol: float = 1e-10,
    sympy_interval_tol: float = 1e-12,
    screen_coarse_samples: int = 129,
    screen_refine_steps: int = 2,
    screen_refine_samples: int = 129,
    sylvester_smin_coarse_samples: int = 33,
    sylvester_smin_refine_steps: int = 1,
    sylvester_smin_refine_samples: int = 33,
) -> torch.Tensor:
    """Condition length estimator for a piecewise-linear coefficient path (torch + optional SymPy)."""
    cl_per_seg = calculate_piecewise_linear_condition_length_per_segment_numeric(
        P_ri,
        loss_cfg=loss_cfg,
        return_inf_if_disc_hits_zero=return_inf_if_disc_hits_zero,
        disc_zero_threshold=disc_zero_threshold,
        use_sylvester_smin_screen=use_sylvester_smin_screen,
        sylvester_smin_rel_threshold=sylvester_smin_rel_threshold,
        sympy_verify_suspicious_segments=sympy_verify_suspicious_segments,
        sympy_max_segments=sympy_max_segments,
        sympy_max_denominator=sympy_max_denominator,
        sympy_nroots_digits=sympy_nroots_digits,
        sympy_nroots_maxsteps=sympy_nroots_maxsteps,
        sympy_imag_tol=sympy_imag_tol,
        sympy_interval_tol=sympy_interval_tol,
        screen_coarse_samples=screen_coarse_samples,
        screen_refine_steps=screen_refine_steps,
        screen_refine_samples=screen_refine_samples,
        sylvester_smin_coarse_samples=sylvester_smin_coarse_samples,
        sylvester_smin_refine_steps=sylvester_smin_refine_steps,
        sylvester_smin_refine_samples=sylvester_smin_refine_samples,
    )
    return cl_per_seg.sum()


