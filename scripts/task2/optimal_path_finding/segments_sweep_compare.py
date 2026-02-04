from __future__ import annotations

"""
Sweep K (number of segments) and compare:
  - condition length (linear vs optimized)
  - optimization runtime
  - Julia HomotopyContinuation.jl tracking effort (linear vs optimized)

IMPORTANT (stability):
  Import juliacall BEFORE torch when we plan to call Julia (see `julia_hc_compare.py`).

Run as a module to keep relative imports stable:
    python3 -m scripts.task2.optimal_path_finding.segments_sweep_compare --Ks 2,4,8,16 --warmup-runs 1 --runs 3

"""

import argparse
import json
import time

import numpy as np

from .utils.julia_hc import (
    JuliaTrackerOptions,
    _maybe_import_julia,
    adaptive_julia_tracker_opts_per_segment,
    parse_int_list,
    benchmark_piecewise_linear_julia,
    summarize_julia_runs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ks", type=str, default="1,5,10,20,40", help="Comma-separated K values to sweep.")
    parser.add_argument("--no-julia", action="store_true", help="Skip Julia HC tracking comparison.")
    parser.add_argument(
        "--julia-adapt-per-segment",
        action="store_true",
        help="If set, adapt Julia TrackerOptions per segment using per-segment condition length (K>=2 only).",
    )
    parser.add_argument("--julia-adapt-alpha-steps", type=float, default=0.5, help="Exponent for max_steps scaling vs cond. length.")
    parser.add_argument(
        "--julia-adapt-beta-step-size",
        type=float,
        default=0.5,
        help="Exponent for max_step_size scaling vs cond. length (inverse relation).",
    )
    parser.add_argument("--julia-adapt-min-steps", type=int, default=5, help="Clamp: minimum max_steps per segment.")
    parser.add_argument("--julia-adapt-max-steps", type=int, default=50_000, help="Clamp: maximum max_steps per segment.")
    parser.add_argument("--julia-adapt-min-max-step-size", type=float, default=1e-4, help="Clamp: minimum max_step_size per segment.")
    parser.add_argument("--julia-adapt-max-max-step-size", type=float, default=0.2, help="Clamp: maximum max_step_size per segment.")

    # Optimization / condition length parameters (match julia_hc_compare defaults unless noted)
    parser.add_argument("--M", type=int, default=16, help="Samples per segment (condition length eval).")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps.")
    parser.add_argument("--imag-noise", type=float, default=1e-1, help="Init imag noise scale for mid control points.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for init noise.")

    # Screening / condition length safety
    parser.add_argument("--disc-threshold", type=float, default=1e-14, help="Threshold for treating Disc≈0 (torch screen).")
    parser.add_argument("--screen-coarse", type=int, default=257, help="Torch screen coarse samples on [0,1].")
    parser.add_argument("--screen-refine-steps", type=int, default=2, help="Torch screen refinement rounds.")
    parser.add_argument("--screen-refine", type=int, default=257, help="Torch screen samples per refinement.")

    # Julia measurement repetition (defaults are conservative because sweep multiplies cost)
    parser.add_argument("--warmup-runs", type=int, default=0, help="Warm-up tracking runs (not measured) per path per K.")
    parser.add_argument("--runs", type=int, default=1, help="Measured tracking runs per path per K.")

    parser.add_argument("--out", type=str, default="", help="Optional JSON output path to save the sweep results.")
    args = parser.parse_args()

    Ks = parse_int_list(str(args.Ks))

    # Import Julia BEFORE torch if enabled.
    jl = _maybe_import_julia(enable=not args.no_julia)

    import torch  # noqa: E402  (intentionally after juliacall)

    from .config import OptimConfig, PiecewiseLossConfig  # noqa: E402
    from .optimizer import optimize_piecewise_linear_path  # noqa: E402
    from .utils.condition_length import (  # noqa: E402
        build_linear_control_points,
        calculate_piecewise_linear_condition_length_numeric,
        calculate_piecewise_linear_condition_length_per_segment_numeric,
    )

    # ------------------------------------------------------------
    # Endpoints: keep identical to julia_hc_compare.py for consistency.
    # NOTE: These are tail coefficients for monic polynomial x^d + a_{d-1}x^{d-1}+...+a_0.
    # ------------------------------------------------------------
    p_start = torch.tensor([-1, -1, 2, 3, 1, 1], dtype=torch.complex128)
    p_target = torch.tensor([1, -1, 2, 1, 5, 1], dtype=torch.complex128)

    loss_cfg = PiecewiseLossConfig(
        samples_per_segment=int(args.M),
        lambda_smooth=0.0,
        eps_soft=1e-12,
        delta_soft=1e-12,
        disc_eps=0.0,
        lead_eps=1e-24,
    )
    optim_cfg = OptimConfig(lr=1e-2, steps=int(args.steps), print_every=0, grad_clip=1.0)

    def eval_condition_length(P: torch.Tensor) -> float:
        cl = calculate_piecewise_linear_condition_length_numeric(
            P,
            loss_cfg=loss_cfg,
            disc_zero_threshold=float(args.disc_threshold),
            use_sylvester_smin_screen=True,
            sylvester_smin_rel_threshold=1e-4,
            sylvester_smin_refine_steps=2,
            sylvester_smin_coarse_samples=256,
            sylvester_smin_refine_samples=256,
            sympy_verify_suspicious_segments=True,
            sympy_max_segments=8,
            screen_coarse_samples=int(args.screen_coarse),
            screen_refine_steps=int(args.screen_refine_steps),
            screen_refine_samples=int(args.screen_refine),
        )
        return float(cl.detach().cpu().item())

    rows: list[dict] = []

    print("\n=== K sweep comparison ===")
    print(f"Ks={Ks}")
    print(f"condition-length M={int(args.M)}, optimize steps={int(args.steps)}")
    if jl is None:
        print("Julia: disabled (--no-julia)")
    else:
        print(f"Julia: enabled, warmup_runs={int(args.warmup_runs)}, runs={int(args.runs)} (per path per K)")

    for K in Ks:
        K_int = int(K)
        if K_int < 1:
            raise ValueError("All K must be >= 1.")

        # ---- optimize (and time it)
        #
        # IMPORTANT:
        #   For K=1 there are no optimization variables, so reporting a runtime can be misleading:
        #   it would mainly measure Python/Torch overhead, not an actual optimization loop.
        #   Therefore we record runtime as None for K=1.
        t0 = time.perf_counter()
        P_best, info = optimize_piecewise_linear_path(
            p_start=p_start,
            p_target=p_target,
            num_segments=K_int,
            init="linear",
            init_imag_noise_scale=float(args.imag_noise),
            init_seed=int(args.seed),
            loss_cfg=loss_cfg,
            optim_cfg=optim_cfg,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float64,
        )
        opt_wall_raw = float(time.perf_counter() - t0)
        opt_wall: float | None = None if K_int == 1 else opt_wall_raw

        # ---- linear control points (same K segmentation)
        P_linear = build_linear_control_points(p_start, p_target, K_int, device=P_best.device, dtype=P_best.dtype)
        # For Julia tracking, the linear baseline should be a single straight-line homotopy
        # between endpoints (NOT K sequential solves). So we always pass only the endpoints.
        P_linear_for_julia = P_linear[[0, -1]]  # shape (2, degree, 2)

        # ---- condition length
        cl_linear = eval_condition_length(P_linear)
        cl_opt = eval_condition_length(P_best)

        # ---- Julia tracking
        warmup_opts = JuliaTrackerOptions(
            max_steps=500,
            max_step_size=0.05,
            max_initial_step_size=0.05,
            min_step_size=1e-12,
            min_rel_step_size=1e-12,
            extended_precision=True,
        )
        meas_linear_opts = JuliaTrackerOptions(
            max_steps=200,
            max_step_size=0.05,
            max_initial_step_size=0.05,
            min_step_size=1e-12,
            min_rel_step_size=1e-12,
            extended_precision=True,
        )
        meas_opts = JuliaTrackerOptions(
            max_steps=100,
            max_step_size=0.11,
            max_initial_step_size=0.1,
            min_step_size=1e-12,
            min_rel_step_size=1e-12,
            extended_precision=True,
        )

        julia_linear_meas = benchmark_piecewise_linear_julia(
            jl,
            P_linear_for_julia.detach().cpu().numpy(),
            warmup_runs=int(args.warmup_runs),
            runs=int(args.runs),
            warmup_tracker_opts=warmup_opts,
            measured_tracker_opts=meas_linear_opts,
        )

        warmup_tracker_opts_per_segment: list[JuliaTrackerOptions] | None = None
        measured_tracker_opts_per_segment: list[JuliaTrackerOptions] | None = None
        cl_opt_per_seg: list[float] | None = None
        if bool(args.julia_adapt_per_segment) and K_int >= 2:
            clps = calculate_piecewise_linear_condition_length_per_segment_numeric(
                P_best,
                loss_cfg=loss_cfg,
                disc_zero_threshold=float(args.disc_threshold),
                return_inf_if_disc_hits_zero=True,
                fill_all_inf_if_any_disc_hit=False,
                use_sylvester_smin_screen=True,
                sylvester_smin_rel_threshold=1e-4,
                sylvester_smin_refine_steps=2,
                sylvester_smin_coarse_samples=256,
                sylvester_smin_refine_samples=256,
                sympy_verify_suspicious_segments=True,
                sympy_max_segments=8,
                screen_coarse_samples=int(args.screen_coarse),
                screen_refine_steps=int(args.screen_refine_steps),
                screen_refine_samples=int(args.screen_refine),
            )
            cl_opt_per_seg = [float(x) for x in clps.detach().cpu().numpy().tolist()]
            measured_tracker_opts_per_segment = adaptive_julia_tracker_opts_per_segment(
                np.asarray(cl_opt_per_seg, dtype=np.float64),
                base_opts=meas_opts,
                alpha_steps=float(args.julia_adapt_alpha_steps),
                beta_step_size=float(args.julia_adapt_beta_step_size),
                min_steps=int(args.julia_adapt_min_steps),
                max_steps=int(args.julia_adapt_max_steps),
                min_max_step_size=float(args.julia_adapt_min_max_step_size),
                max_max_step_size=float(args.julia_adapt_max_max_step_size),
            )
            warmup_tracker_opts_per_segment = adaptive_julia_tracker_opts_per_segment(
                np.asarray(cl_opt_per_seg, dtype=np.float64),
                base_opts=warmup_opts,
                alpha_steps=float(args.julia_adapt_alpha_steps),
                beta_step_size=float(args.julia_adapt_beta_step_size),
                min_steps=int(args.julia_adapt_min_steps),
                max_steps=int(args.julia_adapt_max_steps),
                min_max_step_size=float(args.julia_adapt_min_max_step_size),
                max_max_step_size=float(args.julia_adapt_max_max_step_size),
            )

        julia_opt_meas = benchmark_piecewise_linear_julia(
            jl,
            P_best.detach().cpu().numpy(),
            warmup_runs=int(args.warmup_runs),
            runs=int(args.runs),
            warmup_tracker_opts=warmup_opts,
            measured_tracker_opts=meas_opts,
            # warmup_tracker_opts_per_segment=warmup_tracker_opts_per_segment,
            measured_tracker_opts_per_segment=measured_tracker_opts_per_segment,
        )
        julia_linear = summarize_julia_runs(julia_linear_meas) if julia_linear_meas else None
        julia_opt = summarize_julia_runs(julia_opt_meas) if julia_opt_meas else None

        row = {
            "K": K_int,
            "opt_wall_time_sec": opt_wall,
            "opt_overhead_wall_time_sec": opt_wall_raw if K_int == 1 else None,
            "condition_length_linear": cl_linear,
            "condition_length_opt": cl_opt,
            "condition_length_opt_per_segment": cl_opt_per_seg,
            "julia_adapt_per_segment": bool(args.julia_adapt_per_segment) and K_int >= 2,
            "julia_linear": None if julia_linear is None else julia_linear.__dict__,
            "julia_opt": None if julia_opt is None else julia_opt.__dict__,
            "info_note": info.get("note", ""),
        }
        rows.append(row)

        # Pretty print per K (short, high-signal)
        print(f"\n--- K={K_int} ---")
        if row["info_note"]:
            print(f"[optimization] note: {row['info_note']}")
        if opt_wall is None:
            print("[optimization] runtime[s]: None (K=1; no optimization variables)")
        else:
            print(f"[optimization] runtime[s]: {opt_wall:.6f}")
        print(f"[condition length] linear path: {cl_linear:.16e}")
        print(f"[condition length] optimized path:    {cl_opt:.16e}")
        if julia_linear is not None:
            print(
                f"[HC solver (julia)] linear path: success={julia_linear.success_rate*100:.1f}% "
                f"steps(mean±std)={julia_linear.total_steps_mean:.1f}±{julia_linear.total_steps_std:.1f} "
                f"time(mean±std)={julia_linear.wall_time_mean:.6f}±{julia_linear.wall_time_std:.6f}"
            )
        if julia_opt is not None:
            print(
                f"[HC solver (julia)] optimized path: success={julia_opt.success_rate*100:.1f}% "
                f"steps(mean±std)={julia_opt.total_steps_mean:.1f}±{julia_opt.total_steps_std:.1f} "
                f"time(mean±std)={julia_opt.wall_time_mean:.6f}±{julia_opt.wall_time_std:.6f}"
            )

    if args.out:
        out_path = str(args.out)
        payload = {
            "Ks": Ks,
            "args": vars(args),
            "results": rows,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()


