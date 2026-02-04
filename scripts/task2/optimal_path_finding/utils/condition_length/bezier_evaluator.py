from __future__ import annotations

import torch

from ...config import BezierLossConfig
from ..bezier import bezier_derivative_control_points, bezier_eval
from ..complex_repr import complex_norm_ri, p_to_monic_poly_coeffs_ri
from ..discriminant_calculator import discriminant_univariate_logabs
from ..log_stability import make_uniform_ts, log_softabs_from_logabs, log_softabs_plus_eps
from .bezier_sympy_screen import bezier_disc_abs2_has_root_in_unit_interval
from .bezier_torch_screen import min_disc_logabs_on_bezier_torch, min_sylvester_rel_sigma_min_on_bezier_torch


"""
Run as a module to keep relative imports stable:
    python3 -m scripts.task2.optimal_path_finding.utils.condition_length.bezier_evaluator
"""

def calculate_bezier_condition_length_numeric(
    P_ctrl_ri: torch.Tensor,
    *,
    loss_cfg: BezierLossConfig = BezierLossConfig(),
    bezier_derivative_factor: str = "degree",
    return_inf_if_disc_hits_zero: bool = True,
    disc_zero_threshold: float = 1e-14,
    # (1) Singular-value screen using Sylvester matrix relative s_min proxy.
    use_sylvester_smin_screen: bool = False,
    sylvester_smin_rel_threshold: float = 1e-5,
    # (2) Optional SymPy verification when (1) flags suspicious.
    sympy_verify_suspicious_segments: bool = False,
    sympy_max_segments: int = 8,  # kept for API parity (Bezier is a single curve)
    sympy_max_denominator: int | None = 10**6,
    sympy_nroots_digits: int = 50,
    sympy_nroots_maxsteps: int = 200,
    sympy_imag_tol: float = 1e-10,
    sympy_interval_tol: float = 1e-12,
    # Torch screen settings (coarse-to-fine on t∈[0,1])
    screen_coarse_samples: int = 257,
    screen_refine_steps: int = 2,
    screen_refine_samples: int = 257,
    sylvester_smin_coarse_samples: int = 33,
    sylvester_smin_refine_steps: int = 1,
    sylvester_smin_refine_samples: int = 33,
) -> torch.Tensor:
    """Bezier condition length estimator using the "true" discrete approximation.

    We approximate the integral:
        ∫_0^1 ||T'(t)|| / cond_cn(T(t)) dt
    by midpoint rule with M samples, where M = loss_cfg.samples_per_segment.

    cond_cn is the same discriminant-based n-th-root weight used in the piecewise-linear
    evaluator (computed stably in the log domain).

    Disc=0 crossing checks:
      - During evaluation, if return_inf_if_disc_hits_zero=True, we first run fast torch
        screens. If enabled, suspicious cases can be verified by SymPy (slow).
      - This mimics the piecewise-linear evaluator's design philosophy.
    """
    if P_ctrl_ri.ndim != 3 or P_ctrl_ri.shape[-1] != 2:
        raise ValueError("P_ctrl_ri must have shape (d+1, degree, 2).")
    if P_ctrl_ri.shape[0] < 2:
        raise ValueError("P_ctrl_ri must contain at least two control points (d+1 >= 2).")

    d = int(P_ctrl_ri.shape[0] - 1)      # Bezier degree
    degree = int(P_ctrl_ri.shape[1])     # monic polynomial degree
    if degree < 2:
        raise ValueError("degree must be >= 2 (monic polynomial degree).")

    device, dtype = P_ctrl_ri.device, P_ctrl_ri.dtype
    M = int(loss_cfg.samples_per_segment)
    if M < 1:
        raise ValueError("loss_cfg.samples_per_segment must be >= 1.")

    if sympy_max_segments < 1:
        raise ValueError("sympy_max_segments must be >= 1.")
    if disc_zero_threshold <= 0:
        raise ValueError("disc_zero_threshold must be > 0.")
    if sylvester_smin_rel_threshold <= 0:
        raise ValueError("sylvester_smin_rel_threshold must be > 0.")
    if bool(sympy_verify_suspicious_segments) and not bool(use_sylvester_smin_screen):
        raise ValueError("sympy_verify_suspicious_segments requires use_sylvester_smin_screen=True.")

    # ------------------------------------------------------------
    # Optional Disc≈0 screen (torch-only + optional SymPy verification)
    # ------------------------------------------------------------
    if bool(return_inf_if_disc_hits_zero):
        min_logabs, _tmin = min_disc_logabs_on_bezier_torch(
            P_ctrl_ri,
            loss_cfg=loss_cfg,
            coarse_samples=int(screen_coarse_samples),
            refine_steps=int(screen_refine_steps),
            refine_samples=int(screen_refine_samples),
        )
        thr_log = torch.tensor(float(disc_zero_threshold), device=device, dtype=dtype).log()
        disc_suspicious = (~torch.isfinite(min_logabs)) | (min_logabs <= thr_log)
        if bool(disc_suspicious.item()):
            # Match piecewise behavior: if Disc screen hits and we're not in verification mode, return +inf.
            return torch.tensor(float("inf"), device=device, dtype=dtype)

        if bool(use_sylvester_smin_screen):
            # As in piecewise evaluator, do this on CPU for small matrices / SymPy adjacency.
            P_cpu = P_ctrl_ri.detach().to(device=torch.device("cpu"))
            min_logrel, _tstar = min_sylvester_rel_sigma_min_on_bezier_torch(
                P_cpu,
                loss_cfg=loss_cfg,
                coarse_samples=int(sylvester_smin_coarse_samples),
                refine_steps=int(sylvester_smin_refine_steps),
                refine_samples=int(sylvester_smin_refine_samples),
            )
            log_thr = torch.tensor(float(sylvester_smin_rel_threshold), device=P_cpu.device, dtype=P_cpu.dtype).log()
            suspicious = min_logrel <= log_thr
            if bool(suspicious.item()):
                if not bool(sympy_verify_suspicious_segments):
                    return torch.tensor(float("inf"), device=device, dtype=dtype)

                # SymPy verification (slow): direct Bezier Disc(t)=0 check on [0,1].
                hit = bezier_disc_abs2_has_root_in_unit_interval(
                    P_cpu,
                    max_denominator=sympy_max_denominator,
                    imag_tol=sympy_imag_tol,
                    interval_tol=sympy_interval_tol,
                    nroots_digits=sympy_nroots_digits,
                    nroots_maxsteps=sympy_nroots_maxsteps,
                    clear_denoms=True,
                    conservative_on_failure=True,
                )
                if bool(hit):
                    return torch.tensor(float("inf"), device=device, dtype=dtype)

    # ------------------------------------------------------------
    # True discrete approximation of the Bezier condition length
    # ------------------------------------------------------------
    ts = make_uniform_ts(M, device=device, dtype=dtype)  # midpoints in (0,1)
    T = bezier_eval(P_ctrl_ri, ts, method=loss_cfg.bezier_eval_method)  # (M, degree, 2)
    # NOTE on derivative scaling.
    # Reference: "Optimal Path Homotopy For Univariate Polynomials" (Bao Duy Tran, 2022).
    # Standard Bezier calculus (and the paper's Remark 3) uses:
    #   T'(t) = d * sum_i (P_{i+1}-P_i) B_i^{d-1}(t).
    # However, the paper's *reported numeric examples* are reproduced by scaling with (d+1)
    # instead of d (i.e., a constant (d+1)/d factor in ||T'(t)||). We expose this switch
    # purely for reproducibility/consistency checks against the paper's example values.
    if bezier_derivative_factor == "degree":
        Q_ctrl = bezier_derivative_control_points(P_ctrl_ri)  # (d, degree, 2), scaled by d
    elif bezier_derivative_factor in ("d_plus_1", "degree_plus_1"):
        Q_ctrl = float(d + 1) * (P_ctrl_ri[1:] - P_ctrl_ri[:-1])  # (d, degree, 2), scaled by d+1
    else:
        raise ValueError("bezier_derivative_factor must be 'degree' or 'd_plus_1'.")
    Tp = bezier_eval(Q_ctrl, ts, method=loss_cfg.bezier_eval_method)  # (M, degree, 2)

    speed = complex_norm_ri(Tp)  # ||T'(t_m)||, (M,)

    a_ri = p_to_monic_poly_coeffs_ri(T)  # (M, degree+1, 2)
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

    # inv-only weight: w = 1 / (softabs(Disc) + eps)^{1/degree}
    w = torch.exp(-log_denom)  # 1/cond_cn

    return (speed * w).mean()


def main() -> None:
    loss_cfg = BezierLossConfig(
        samples_per_segment=100,
        eps_soft=1e-12,
        delta_soft=1e-12,
    )
    # Example (d=3 -> 4 control points), monic polynomial degree=2 (so coefficient vector length is 2).
    # Shape must be (d+1, degree, 2) where the last dim is (Re, Im).
    P1_ctrl_ri = torch.tensor(
        [
            [[-1.0, 0.0], [-1.0, 0.0]],
            [[0.0, 0.0], [-2.0, 0.0]],
            [[1.0, 0.0], [-1.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    P2_ctrl_ri = torch.tensor(
        [
            [[-1.0, 0.0], [-1.0, 0.0]],
            [[0.0, 0.0], [-1.5, 0.0]],
            [[1.0, 0.0], [-1.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    P3_ctrl_ri = torch.tensor(
        [
            [[-1.0, 0.0], [-1.0, 0.0]],
            [[-0.5, 0.0], [-2.0, 0.0]],
            [[0.5, 0.0], [-2.5, 0.0]],
            [[1.0, 0.0], [-1.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    P4_ctrl_ri = torch.tensor(
        [
            [[-1.0, 0.0], [-1.0, 0.0]],
            [[-0.3753, 0.0], [-1.2910, 0.0]],
            [[0.3753, 0.0], [-1.2909, 0.0]],
            [[1.0, 0.0], [-1.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    cl = calculate_bezier_condition_length_numeric(
        P4_ctrl_ri,
        loss_cfg=loss_cfg,
        bezier_derivative_factor="d_plus_1", # in order to correspond to the example values (Tran, 2022)
        return_inf_if_disc_hits_zero=True,
        disc_zero_threshold=1e-14,
        use_sylvester_smin_screen=True,
        sylvester_smin_rel_threshold=1e-5,
        sympy_verify_suspicious_segments=True,
        sympy_max_segments=8,
        sympy_max_denominator=10**6,
        sympy_nroots_digits=50,
        sympy_nroots_maxsteps=200,
        sympy_imag_tol=1e-10,
        sympy_interval_tol=1e-12,
    )
    print(cl)


if __name__ == "__main__":
    main()