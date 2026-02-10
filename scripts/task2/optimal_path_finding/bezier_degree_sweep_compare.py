from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

"""
Sweep Bezier degree d and compare:
  - condition length (linear baseline vs optimized Bezier)
  - optimization runtime
  - Julia HomotopyContinuation.jl tracking effort (linear baseline vs optimized Bezier)

Important (stability):
  Import juliacall BEFORE torch when we plan to call Julia (see `julia_hc_compare.py`).

Run as a module to keep relative imports stable:
    python3 -m scripts.task2.optimal_path_finding.bezier_degree_sweep_compare --degrees 2,4,8,16 --warmup-runs 1 --runs 3
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
    parse_int_list,
    summarize_julia_runs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degrees", type=str, default="2,4,8,16", help="Comma-separated Bezier degrees to sweep.")
    parser.add_argument("--no-julia", action="store_true", help="Skip Julia HC tracking comparison.")

    # Optimization / condition length parameters
    parser.add_argument("--steps", type=int, default=400, help="Optimization steps.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Optimizer learning rate.")
    parser.add_argument("--imag-noise", type=float, default=1e-1, help="Init imag noise scale for mid control points.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for init noise.")

    # Julia measurement repetition
    parser.add_argument("--warmup-runs", type=int, default=0, help="Warm-up tracking runs (not measured) per path per degree.")
    parser.add_argument("--runs", type=int, default=1, help="Measured tracking runs per path per degree.")
    parser.add_argument(
        "--tracker-parameters",
        type=str,
        default="fast",
        choices=["default", "conservative", "fast"],
        help="Julia tracker parameter preset.",
    )

    parser.add_argument("--out", type=str, default="", help="Optional JSON output path to save the sweep results.")
    args = parser.parse_args()

    degrees = parse_int_list(str(args.degrees))

    # Import Julia BEFORE torch if enabled.
    jl = _maybe_import_julia(enable=not args.no_julia)

    import torch  # noqa: E402  (intentionally after juliacall)

    from .config import BezierLossConfig, OptimConfig  # noqa: E402
    from .optimizer import optimize_bezier_path  # noqa: E402
    from .utils.complex_repr import to_ri  # noqa: E402
    from .utils.condition_length import calculate_bezier_condition_length_numeric  # noqa: E402

    # Endpoints: keep consistent with exp_bezier.py unless overridden by editing this file.
    # NOTE: These are tail coefficients for monic polynomial x^d + a_{d-1}x^{d-1}+...+a_0.
    p_start = torch.tensor([-1, -1+1j, 2, 3, 1, 1, 4, 2, -2, -1], dtype=torch.complex128)
    p_target = torch.tensor([1, -1+1j, 2, 1, 5, 1, -3, -1, 1, 2], dtype=torch.complex128)
    M = 128
    loss_cfg = BezierLossConfig(
        samples_per_segment=int(M),
        lambda_smooth=0.0,
        eps_soft=1e-12,
        delta_soft=1e-12,
        disc_eps=0.0,
        lead_eps=1e-24,
        disc_backend="complex",
        bezier_eval_method="casteljau",
        alpha=0,
        beta=0,
    )
    optim_cfg = OptimConfig(lr=float(args.lr), steps=int(args.steps), print_every=0, grad_clip=1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Linear baseline for Julia tracking: a single straight-line homotopy between endpoints.
    p0_ri = to_ri(p_start).detach().cpu()
    p1_ri = to_ri(p_target).detach().cpu()
    P_linear_for_julia = torch.stack([p0_ri, p1_ri], dim=0).numpy()  # shape (2, degree, 2)

    # Condition length baseline: the same straight line represented as a degree-1 Bezier curve.
    P_ctrl_linear = torch.stack([to_ri(p_start), to_ri(p_target)], dim=0).to(device=device, dtype=torch.float64)

    def eval_bezier_condition_length(P_ctrl: torch.Tensor) -> float:
        cl = calculate_bezier_condition_length_numeric(
            P_ctrl,
            loss_cfg=loss_cfg,
            bezier_derivative_factor="degree",
            return_inf_if_disc_hits_zero=False,
            disc_zero_threshold=1e-20,
            use_sylvester_smin_screen=True,
            sylvester_smin_rel_threshold=1e-20,
            sympy_verify_suspicious_segments=False,
            sympy_max_segments=8,
            sympy_max_denominator=10**6,
            sympy_nroots_digits=50,
            sympy_nroots_maxsteps=200,
            sympy_imag_tol=1e-10,
            sympy_interval_tol=1e-12,
        )
        return float(cl.detach().cpu().item())

    # Fixed Julia tracker options for fair comparison.
    warmup_opts = JuliaTrackerOptions(parameters=str(args.tracker_parameters))
    meas_opts = JuliaTrackerOptions(
        max_steps=50000,
        max_step_size=0.05,
        max_initial_step_size=0.05,
        min_step_size=1e-12,
        min_rel_step_size=1e-12,
        extended_precision=True,
        parameters=str(args.tracker_parameters),
    )

    rows: list[dict] = []

    print("\n=== Bezier degree sweep comparison ===")
    print(f"degrees={degrees}")
    print(f"condition-length M={int(M)}, optimize steps={int(args.steps)}, device={device}")
    if jl is None:
        print("Julia: disabled (--no-julia)")
    else:
        print(
            f"Julia: enabled, warmup_runs={int(args.warmup_runs)}, runs={int(args.runs)} "
            f"(per path per degree), tracker_parameters={args.tracker_parameters}"
        )

    # Baseline evaluations (independent of Bezier degree).
    cl_linear = eval_bezier_condition_length(P_ctrl_linear)
    julia_linear = None
    if jl is not None:
        meas_linear = benchmark_piecewise_linear_julia(
            jl,
            P_linear_for_julia,
            warmup_runs=int(args.warmup_runs),
            runs=int(args.runs),
            warmup_tracker_opts=warmup_opts,
            measured_tracker_opts=meas_opts,
        )
        julia_linear = summarize_julia_runs(meas_linear).__dict__ if meas_linear else None

    for d in degrees:
        d_int = int(d)
        if d_int < 1:
            raise ValueError("All Bezier degrees must be >= 1.")

        t0 = time.perf_counter()
        P_ctrl_best, info = optimize_bezier_path(
            p_start=p_start,
            p_target=p_target,
            bezier_degree=d_int,
            init_imag_noise_scale=float(args.imag_noise),
            init_seed=int(args.seed),
            loss_cfg=loss_cfg,
            optim_cfg=optim_cfg,
            device=device,
            dtype=torch.float64,
        )
        opt_wall = float(time.perf_counter() - t0)

        t1 = time.perf_counter()
        cl_opt = eval_bezier_condition_length(P_ctrl_best)
        cl_wall = float(time.perf_counter() - t1)

        julia_bezier = None
        if jl is not None:
            meas_bez = benchmark_bezier_curve_julia(
                jl,
                P_ctrl_best.detach().cpu().numpy(),
                warmup_runs=int(args.warmup_runs),
                runs=int(args.runs),
                warmup_tracker_opts=warmup_opts,
                measured_tracker_opts=meas_opts,
            )
            julia_bezier = summarize_julia_runs(meas_bez).__dict__ if meas_bez else None

        row = {
            "bezier_degree": d_int,
            "opt_wall_time_sec": opt_wall,
            "condition_length_linear": cl_linear,
            "condition_length_opt": cl_opt,
            "condition_length_opt_eval_wall_time_sec": cl_wall,
            "julia_linear": julia_linear,
            "julia_bezier": julia_bezier,
            "info": info,
        }
        rows.append(row)

        print(f"\n--- bezier_degree={d_int} ---")
        print(f"[optimization] runtime[s]: {opt_wall:.6f}")
        print(f"[optimization] average d1_l2: {info['best_diag']['mean_speed']:.6f}")
        print(f"[optimization] average d2_l2: {info['best_diag']['mean_accel']:.6f}")
        print(f"[condition length] linear: {cl_linear:.6f}")
        print(f"[condition length] optimized: {cl_opt:.6f} (eval {cl_wall:.6f}s)")
        if julia_linear is not None:
            newton_str = (
                f"newton_iters(mean±std)={julia_linear['total_newton_iters_mean']:.1f}±{julia_linear['total_newton_iters_std']:.1f} "
                if julia_linear.get("total_newton_iters_mean") is not None
                else "newton_iters=N/A "
            )
            print(
                f"[HC solver (julia)] linear: success={julia_linear['success_rate']*100:.1f}% "
                f"steps(mean±std)={julia_linear['total_steps_mean']:.1f}±{julia_linear['total_steps_std']:.1f} "
                f"time(mean±std)={julia_linear['wall_time_mean']:.6f}±{julia_linear['wall_time_std']:.6f} "
                f"{newton_str}"
            )
        if julia_bezier is not None:
            newton_str = (
                f"newton_iters(mean±std)={julia_bezier['total_newton_iters_mean']:.1f}±{julia_bezier['total_newton_iters_std']:.1f} "
                if julia_bezier.get("total_newton_iters_mean") is not None
                else "newton_iters=N/A "
            )
            print(
                f"[HC solver (julia)] bezier: success={julia_bezier['success_rate']*100:.1f}% "
                f"steps(mean±std)={julia_bezier['total_steps_mean']:.1f}±{julia_bezier['total_steps_std']:.1f} "
                f"time(mean±std)={julia_bezier['wall_time_mean']:.6f}±{julia_bezier['wall_time_std']:.6f} "
                f"{newton_str}"
            )

    if args.out:
        payload = {"degrees": degrees, "args": vars(args), "results": rows}
        with open(str(args.out), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()


