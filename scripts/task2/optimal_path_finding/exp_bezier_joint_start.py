from __future__ import annotations

import os
import time

"""
Run as a module to keep relative imports stable:
    python3 -m scripts.task2.optimal_path_finding.exp_bezier_joint_start
"""

# ------------------------------------------------------------
# Determinism (IMPORTANT: set env vars BEFORE importing torch)
# ------------------------------------------------------------
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import torch

from .config import BezierLossConfig, OptimConfig
from .optimizer import optimize_bezier_path_joint_start
from .utils.condition_length import calculate_bezier_condition_length_numeric


def main() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Example target endpoint (complex coefficient vector p of length "degree")
    # p corresponds to monic poly: x^n + a_{n-1}x^{n-1} + ... + a_0
    p_target = torch.tensor([1, -1, 2, 1, 5, 1, -3, -1, 1, 2], dtype=torch.complex128)

    loss_cfg = BezierLossConfig(
        samples_per_segment=16,
        lambda_smooth=0.0,
        eps_soft=1e-12,
        delta_soft=1e-12,
        disc_eps=0.0,
        lead_eps=1e-24,
        disc_backend="complex",
        bezier_eval_method="casteljau",
        alpha=0.0,
        beta=0.0,
    )

    optim_cfg = OptimConfig(
        lr=1e-2,
        steps=1500,
        print_every=50,
        grad_clip=1.0,
    )

    start_time = time.time()
    P_ctrl_best, info = optimize_bezier_path_joint_start(
        p_target=p_target,
        bezier_degree=3,
        init_imag_noise_scale=1e-1,
        init_seed=0,
        loss_cfg=loss_cfg,
        optim_cfg=optim_cfg,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float64,
        # Start polynomial init (G(x) = (x-c)^n - s^n,  s=rho*exp(i*theta))
        c_re_init=0.0,
        c_im_init=0.0,
        rho_init=1.0,
        theta_init=0.0,
        rho_eps=1e-8,
    )
    end_time = time.time()

    print(f"[bezier joint] Time taken: {end_time - start_time} seconds")
    print("[bezier joint] info:", info)
    print("[bezier joint] P_ctrl_best shape:", tuple(P_ctrl_best.shape))  # (d+1, degree, 2)

    cl_bez = calculate_bezier_condition_length_numeric(
        P_ctrl_best,
        loss_cfg=loss_cfg,
        bezier_derivative_factor="degree",
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
    print(f"[condition length] optimized Bezier path (joint start, M={loss_cfg.samples_per_segment}): {cl_bez.item():.16e}")


if __name__ == "__main__":
    main()


