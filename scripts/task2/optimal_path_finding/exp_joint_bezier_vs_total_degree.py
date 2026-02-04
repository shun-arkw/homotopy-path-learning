from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

"""
Compare Julia HomotopyContinuation.jl effort for:
  (A) optimized Bezier homotopy path with joint start polynomial (c,rho,theta) + mid control points
  (B) total-degree baseline solve (Julia `solve(...; start_system=:total_degree)`)

Run as a module to keep relative imports stable:
    python3 -m scripts.task2.optimal_path_finding.exp_joint_bezier_vs_total_degree --warmup-runs 1 --runs 3

Important (stability):
  Import juliacall BEFORE torch when we plan to call Julia.
"""

# ------------------------------------------------------------
# Determinism (IMPORTANT: set env vars BEFORE importing torch)
# ------------------------------------------------------------
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

from .utils.julia_hc import (  # noqa: E402  (intentionally before torch)
    JuliaTrackerOptions,
    _maybe_import_julia,
    benchmark_bezier_curve_julia,
    benchmark_piecewise_linear_julia,
    benchmark_total_degree_julia,
    summarize_julia_runs,
)


def _roots_for_joint_start(*, c: complex, s: complex, degree: int) -> list[list[complex]]:
    """
    Roots of G(x) = (x - c)^n - s^n are:
        x_k = c + s * exp(2π i k / n),  k=0..n-1
    Returns in Julia starts format: [[z1],[z2],...]
    """
    n = int(degree)
    roots: list[list[complex]] = []
    for k in range(n):
        omega = np.exp(2j * np.pi * (k / n))
        roots.append([complex(c + s * omega)])
    return roots


def _fmt_pm(mean: float | None, std: float | None, *, fmt: str) -> str:
    if mean is None:
        return "N/A"
    if std is None:
        return format(float(mean), fmt)
    return f"{format(float(mean), fmt)}±{format(float(std), fmt)}"


def _print_julia_summary(name: str, s: dict | None) -> None:
    if not s:
        print(f"[HC solver (julia)] {name}: N/A")
        return

    success = float(s.get("success_rate", 0.0))
    steps = _fmt_pm(s.get("total_steps_mean"), s.get("total_steps_std"), fmt=".2f")
    tsec = _fmt_pm(s.get("wall_time_mean"), s.get("wall_time_std"), fmt=".6f")
    newton = _fmt_pm(s.get("total_newton_iters_mean"), s.get("total_newton_iters_std"), fmt=".2f")

    print(
        f"[HC solver (julia)] {name}: "
        f"success={success*100:.1f}% "
        f"steps(mean±std)={steps} "
        f"time_s(mean±std)={tsec} "
        f"newton(mean±std)={newton}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-julia", action="store_true", help="Skip Julia HC tracking comparison.")
    parser.add_argument("--bezier-degree", type=int, default=1, help="Bezier degree d (control points d+1).")

    # Optimization parameters
    parser.add_argument("--steps", type=int, default=1500, help="Optimization steps.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Optimizer learning rate.")
    parser.add_argument("--imag-noise", type=float, default=1e-1, help="Init imag noise scale for mid control points.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for init noise.")

    # Julia measurement repetition
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warm-up runs (not measured).")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs (report avg/std).")

    parser.add_argument("--out", type=str, default="", help="Optional JSON output path to save results.")
    args = parser.parse_args()

    jl = _maybe_import_julia(enable=not args.no_julia)

    import torch  # noqa: E402  (intentionally after juliacall)

    from .config import BezierLossConfig, OptimConfig  # noqa: E402
    from .optimizer import optimize_bezier_path_joint_start  # noqa: E402
    from .utils.condition_length import calculate_bezier_condition_length_numeric  # noqa: E402

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Same target as exp_bezier_joint_start.py (edit here if needed)
    p_target = torch.tensor([1, -1, 2, 1, 5, 100, -3, -100, 100, 2, 3, -1, 30, 1, 4, 1, 2, 100, 1, 3, -3, 1, -1, 2, 1, 5, 1, -3, -1, 1, 2], dtype=torch.complex128)
    # p_target = torch.tensor([1, -1, 2, 1, 5, 1, -3, -1, 1, 2], dtype=torch.complex128)

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
    optim_cfg = OptimConfig(lr=float(args.lr), steps=int(args.steps), print_every=100, grad_clip=1.0)

    t0 = time.perf_counter()
    P_ctrl_best, info = optimize_bezier_path_joint_start(
        p_target=p_target,
        bezier_degree=int(args.bezier_degree),
        init_imag_noise_scale=float(args.imag_noise),
        init_seed=int(args.seed),
        loss_cfg=loss_cfg,
        optim_cfg=optim_cfg,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float64,
        c_re_init=0.0,
        c_im_init=0.0,
        rho_init=1.0,
        theta_init=0.0,
        rho_eps=1e-8,
    )
    # print(P_ctrl_best)
    opt_wall = float(time.perf_counter() - t0)
    device = info.get("device", "n/a")

    print("\n=== joint-start (optimized) vs baselines ===")
    print(
        f"bezier_degree={int(args.bezier_degree)}  "
        f"opt_steps={int(args.steps)}  lr={float(args.lr)}  seed={int(args.seed)}  "
        f"M={int(loss_cfg.samples_per_segment)}  device={device}"
    )
    print(f"[optimization] wall_time_sec={opt_wall:.6f}")

    def _eval_condition_length(name: str, P_ctrl: torch.Tensor) -> float:
        t_cl0 = time.perf_counter()
        cl = calculate_bezier_condition_length_numeric(
            P_ctrl,
            loss_cfg=loss_cfg,
            bezier_derivative_factor="degree",
            return_inf_if_disc_hits_zero=True,
            disc_zero_threshold=1e-14,
            use_sylvester_smin_screen=True,
            sylvester_smin_rel_threshold=1e-10,
            sympy_verify_suspicious_segments=False,
            sympy_max_segments=8,
            sympy_max_denominator=10**6,
            sympy_nroots_digits=50,
            sympy_nroots_maxsteps=200,
            sympy_imag_tol=1e-10,
            sympy_interval_tol=1e-12,
        )
        t_cl1 = time.perf_counter()
        cl_f = float(cl.detach().cpu().item())
        print(f"[condition length][{name}] {cl_f:.16e} (eval {float(t_cl1 - t_cl0):.6f}s)")
        return cl_f

    # Condition length for (A) optimized Bezier
    cl_bez = _eval_condition_length("bezier optimized", P_ctrl_best)

    # Condition length for (C) optimized-start linear path (Bezier degree 1)
    P_ctrl_linear = torch.stack([P_ctrl_best[0], P_ctrl_best[-1]], dim=0)
    cl_lin = _eval_condition_length("optimized-start linear", P_ctrl_linear)

    print(f"[condition length] bezier optimized:       {cl_bez:.16e}")
    print(f"[condition length] optimized-start linear: {cl_lin:.16e}")

    if jl is None:
        print("Julia: disabled (--no-julia)")
        return

    # Fixed tracker options for fair comparison.
    warmup_opts = JuliaTrackerOptions()
    meas_opts = JuliaTrackerOptions(
        max_steps=50_000,
        max_step_size=0.05,
        max_initial_step_size=0.05,
        min_step_size=1e-12,
        min_rel_step_size=1e-12,
        extended_precision=True,
    )

    # Prepare analytic start solutions from joint-start parameters.
    best_start = info.get("best_start") or {}
    degree = int(info.get("degree"))
    c = complex(float(best_start["c_re"]), float(best_start["c_im"]))
    rho = float(best_start["rho"])
    theta = float(best_start["theta"])
    s = complex(rho * np.cos(theta), rho * np.sin(theta))
    start_solutions = _roots_for_joint_start(c=c, s=s, degree=degree)

    # (A) Bezier optimized path
    meas_bez = benchmark_bezier_curve_julia(
        jl,
        P_ctrl_best.detach().cpu().numpy(),
        warmup_runs=int(args.warmup_runs),
        runs=int(args.runs),
        warmup_tracker_opts=warmup_opts,
        measured_tracker_opts=meas_opts,
        start_solutions=start_solutions,
    )
    sum_bez = summarize_julia_runs(meas_bez).__dict__ if meas_bez else None

    # (C) Optimized start + linear path to target (single segment)
    P0_ri_np = P_ctrl_best[0].detach().cpu().numpy()
    Pd_ri_np = P_ctrl_best[-1].detach().cpu().numpy()
    P_linear_np = np.stack([P0_ri_np, Pd_ri_np], axis=0)  # (2, degree, 2)
    meas_lin = benchmark_piecewise_linear_julia(
        jl,
        P_linear_np,
        warmup_runs=int(args.warmup_runs),
        runs=int(args.runs),
        warmup_tracker_opts=warmup_opts,
        measured_tracker_opts=meas_opts,
        start_solutions=start_solutions,
    )
    sum_lin = summarize_julia_runs(meas_lin).__dict__ if meas_lin else None

    # (B) total-degree baseline solve (target only)
    p_target_ri_np = np.stack([p_target.real.detach().cpu().numpy(), p_target.imag.detach().cpu().numpy()], axis=-1)
    meas_td = benchmark_total_degree_julia(
        jl,
        p_target_ri_np,
        warmup_runs=int(args.warmup_runs),
        runs=int(args.runs),
        warmup_tracker_opts=warmup_opts,
        measured_tracker_opts=meas_opts,
    )
    sum_td = summarize_julia_runs(meas_td).__dict__ if meas_td else None

    print(f"\n[Julia] warmup_runs={int(args.warmup_runs)} runs={int(args.runs)}")
    _print_julia_summary("bezier optimized", sum_bez)
    _print_julia_summary("optimized-start linear", sum_lin)
    _print_julia_summary("total_degree baseline", sum_td)

    payload = {
        "args": vars(args),
        "optimization": {
            "opt_wall_time_sec": opt_wall,
            "info": info,
            "condition_length": {
                "bezier_optimized": cl_bez,
                "optimized_start_linear": cl_lin,
            },
        },
        "julia": {"bezier_optimized": sum_bez, "optimized_start_linear": sum_lin, "total_degree_baseline": sum_td},
    }
    if args.out:
        with open(str(args.out), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()


