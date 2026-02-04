from __future__ import annotations

import os
import time

"""
Run as a module to keep relative imports stable:
    python3 -m scripts.task2.optimal_path_finding.exp_piecewise
"""

# ------------------------------------------------------------
# Determinism (IMPORTANT: set env vars BEFORE importing torch)
# ------------------------------------------------------------
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import torch

from .config import OptimConfig, PiecewiseLossConfig
from .optimizer import optimize_piecewise_linear_path, optimize_piecewise_linear_paths_batched
from .utils.condition_length import build_linear_control_points, calculate_piecewise_linear_condition_length_numeric


def main() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Example endpoints (complex coefficient vector p of length "degree")
    p_start = torch.tensor([-1, -1, 2, 3, 1, 1], dtype=torch.complex128)
    p_target = torch.tensor([1, -1, 2, 1, 5, 1], dtype=torch.complex128)

    loss_cfg = PiecewiseLossConfig(
        samples_per_segment=32,
        lambda_smooth=0.0,
        eps_soft=1e-12,
        delta_soft=1e-12,
        disc_eps=0.0,
        lead_eps=1e-24,
        disc_backend="complex",
    )

    optim_cfg = OptimConfig(
        lr=1e-2,
        steps=400,
        print_every=50,
        grad_clip=1.0,
    )

    # ------------------------------------------------------------
    # Single-path piecewise-linear optimization
    # ------------------------------------------------------------
    start_time = time.time()
    P_best, info = optimize_piecewise_linear_path(
        p_start=p_start,
        p_target=p_target,
        num_segments=10,
        init="linear",
        init_imag_noise_scale=1e-1,
        init_seed=0,
        loss_cfg=loss_cfg,
        optim_cfg=optim_cfg,
        device=torch.device("cuda"),
        dtype=torch.float64,
    )
    end_time = time.time()
    print(f"[piecewise] Time taken: {end_time - start_time} seconds")
    print("[piecewise] info:", info)
    print("[piecewise] P_best shape:", tuple(P_best.shape))  # (K+1, degree, 2)

    # ------------------------------------------------------------
    # Condition length evaluation for piecewise-linear paths
    # ------------------------------------------------------------
    K = int(info["K"])
    P_linear = build_linear_control_points(
        p_start,
        p_target,
        K,
        device=P_best.device,
        dtype=P_best.dtype,
    )

    cl_linear = calculate_piecewise_linear_condition_length_numeric(
        P_linear,
        loss_cfg=loss_cfg,
        disc_zero_threshold=1e-14,
        use_sylvester_smin_screen=True,
        sylvester_smin_rel_threshold=1e-5,
        sympy_verify_suspicious_segments=True,
        sylvester_smin_coarse_samples=33,
        sylvester_smin_refine_steps=1,
        sylvester_smin_refine_samples=33,
        sympy_max_segments=8,
        screen_coarse_samples=257,
        screen_refine_steps=2,
        screen_refine_samples=257,
    )
    cl_opt = calculate_piecewise_linear_condition_length_numeric(
        P_best,
        loss_cfg=loss_cfg,
        disc_zero_threshold=1e-14,
        use_sylvester_smin_screen=True,
        sylvester_smin_rel_threshold=1e-5,
        sympy_verify_suspicious_segments=True,
        sylvester_smin_coarse_samples=33,
        sylvester_smin_refine_steps=1,
        sylvester_smin_refine_samples=33,
        sympy_max_segments=8,
        screen_coarse_samples=257,
        screen_refine_steps=2,
        screen_refine_samples=257,
    )

    print(f"[condition length] linear path(K={K}, M={loss_cfg.samples_per_segment}): {cl_linear.item():.16e}")
    print(f"[condition length] optimized piecewise-linear path (K={K}, M={loss_cfg.samples_per_segment}): {cl_opt.item():.16e}")

    # ------------------------------------------------------------
    # Batched multi-path optimization example (R paths in parallel)
    # ------------------------------------------------------------
    start_time = time.time()
    P_bests, info_b = optimize_piecewise_linear_paths_batched(
        p_start=p_start,
        p_target=p_target,
        num_segments=3,
        num_paths=20,
        init="linear",
        init_imag_noise_scale=1.0,
        loss_cfg=loss_cfg,
        optim_cfg=optim_cfg,
        device=torch.device("cpu"),
        dtype=torch.float64,
        return_cpu=True,
    )
    end_time = time.time()
    print(f"[piecewise batched] Time taken: {end_time - start_time} seconds")
    print("[piecewise batched] P_bests shape:", tuple(P_bests.shape))  # (R, K+1, degree, 2)
    print("[piecewise batched] best_loss_vec:", info_b["best_loss_vec"])

    if "best_loss_cl_vec" in info_b:
        best_cl_vec = info_b["best_loss_cl_vec"]
        best_idx = int(torch.argmin(best_cl_vec).item())
        print("[piecewise batched] best_loss_cl_vec[min]:", float(best_cl_vec[best_idx]), "argmin:", best_idx)

        P_best_b = P_bests[best_idx]
        cl_best = calculate_piecewise_linear_condition_length_numeric(
            P_best_b,
            loss_cfg=loss_cfg,
            disc_zero_threshold=1e-14,
            use_sylvester_smin_screen=True,
            sylvester_smin_rel_threshold=1e-5,
            sympy_verify_suspicious_segments=True,
            sylvester_smin_coarse_samples=33,
            sylvester_smin_refine_steps=1,
            sylvester_smin_refine_samples=33,
            sympy_max_segments=8,
            screen_coarse_samples=257,
            screen_refine_steps=2,
            screen_refine_samples=257,
        )
        print(
            f"[condition length] batched best path (idx={best_idx}, K={K}, M={loss_cfg.samples_per_segment}): "
            f"{cl_best.item():.16e}"
        )


if __name__ == "__main__":
    main()


