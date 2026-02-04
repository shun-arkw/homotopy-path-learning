from __future__ import annotations
"""
Compare HC solver effort (steps/runtime) for:
  - linear path in coefficient space (same K segments)
  - optimized piecewise-linear path (P_best)

We use Julia HomotopyContinuation.jl via juliacall, tracking each segment as a straight-line
homotopy H(x,t)=(1-t)Q(x)+tP(x). For a piecewise-linear coefficient path, we simply run the
tracker sequentially over segments, passing the end solutions of one segment as start
solutions of the next.

NOTE (important for stability):
  Import juliacall BEFORE torch when we plan to call Julia, to avoid potential segfaults.
  This mirrors the pattern used in scripts/task1/multi_steps_root_finding.py.
"""

import argparse
import time

import numpy as np


from .utils.julia_hc import (
    JuliaTrackerOptions,
    _maybe_import_julia,
    benchmark_piecewise_linear_julia,
    summarize_julia_runs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-julia", action="store_true", help="Skip Julia HC tracking comparison.")
    parser.add_argument("--K", type=int, default=10, help="Number of segments for both paths.")
    parser.add_argument("--M", type=int, default=16, help="Samples per segment (condition length eval).")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps.")
    parser.add_argument("--imag-noise", type=float, default=1e-1, help="Init imag noise scale for mid control points.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for init noise.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warm-up tracking runs (not measured) per path.")
    parser.add_argument("--runs", type=int, default=3, help="Measured tracking runs per path (report avg/std).")
    # parser.add_argument(
    #     "--sympy-screen-report",
    #     action="store_true",
    #     help="Also run SymPy Disc=0 diagnostics per segment (slow, but more reliable).",
    # )
    # parser.add_argument(
    #     "--torch-screen-report",
    #     action="store_true",
    #     help="Report torch-only min|Disc| estimate per segment (prints min log|Disc| and t*).",
    # )
    parser.add_argument("--disc-threshold", type=float, default=1e-14, help="Threshold for treating Disc≈0 (torch screen).")
    parser.add_argument("--screen-coarse", type=int, default=257, help="Torch screen coarse samples on [0,1].")
    parser.add_argument("--screen-refine-steps", type=int, default=2, help="Torch screen refinement rounds.")
    parser.add_argument("--screen-refine", type=int, default=257, help="Torch screen samples per refinement.")
    parser.add_argument(
        "--swap-order",
        action="store_true",
        help="Measure in order: optimized then linear (default: linear then optimized).",
    )
    args = parser.parse_args()

    # NOTE: These imports bring in torch (directly or indirectly). Keep them *after*
    # `_maybe_import_julia(...)` to honor the "import juliacall BEFORE torch" rule above.
    from .config import OptimConfig, PiecewiseLossConfig
    from .optimizer import optimize_piecewise_linear_path
    from .utils.condition_length import build_linear_control_points, calculate_piecewise_linear_condition_length_numeric
    from .utils.condition_length.piecewise_linear_sympy_screen import diagnose_disc_zero_screen
    from .utils.condition_length.piecewise_linear_torch_screen import torch_screen_summary
    from .utils.complex_repr import to_ri

    jl = _maybe_import_julia(enable=not args.no_julia)

    import torch  # noqa: E402  (intentionally after juliacall)

    # Demo endpoints (degree=2)
    # p_start = torch.tensor([-1.0, 1e-6], dtype=torch.complex128)
    # p_target = torch.tensor([1.0, 1e-6], dtype=torch.complex128)
    # p_start = torch.tensor([-1, -1, 2, 3, 1, 1, 7, 1, -10, 1], dtype=torch.complex128)
    # p_target = torch.tensor([1, -1, 2, 1, 5, 1, 8, 1, 10, 1], dtype=torch.complex128)
    p_start = torch.tensor([-1, -1, 2, 3, 1, 1], dtype=torch.complex128)
    p_target = torch.tensor([1, -1, 2, 1, 5, 1], dtype=torch.complex128)

    loss_cfg = PiecewiseLossConfig(
        samples_per_segment=int(args.M),
        # Default to pure condition-length-like objective for fair comparison.
        lambda_smooth=0.0,
        eps_soft=1e-12,
        delta_soft=1e-12,
        disc_eps=0.0,
        lead_eps=1e-24,
    )

    optim_cfg = OptimConfig(lr=1e-2, steps=int(args.steps), print_every=50, grad_clip=1.0)

    time_start = time.perf_counter()
    P_best, info = optimize_piecewise_linear_path(
        p_start=p_start,
        p_target=p_target,
        num_segments=int(args.K),
        init="linear",
        init_imag_noise_scale=float(args.imag_noise),
        init_seed=int(args.seed),
        loss_cfg=loss_cfg,
        optim_cfg=optim_cfg,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float64,
    )
    time_end = time.perf_counter()
    print(f"[optimize] optimized piecewise-linear path: {time_end - time_start} seconds")

    # Condition length evaluation (torch, discretized)
    P_linear = build_linear_control_points(p_start, p_target, int(args.K), device=P_best.device, dtype=P_best.dtype)

    # ------------------------------------------------------------
    # Torch screens + condition length (loop over both paths)
    # ------------------------------------------------------------
    print("--------------------------------")
    for name, P in [("linear", P_linear), ("opt", P_best)]:

        time_start = time.perf_counter()
        condition_length = calculate_piecewise_linear_condition_length_numeric(
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
        time_end = time.perf_counter()

        print(
            f"[condition length][{name}] condition length: {condition_length.item():.16e}, "
            f"Time taken: {time_end - time_start} seconds"
        )

        if not torch.isfinite(condition_length):
            diag = diagnose_disc_zero_screen(
                P,
                sympy_max_denominator=10**6,
                sympy_nroots_digits=50,
                sympy_nroots_maxsteps=200,
                sympy_imag_tol=1e-10,
                sympy_interval_tol=1e-12,
                conservative_on_failure=True,
                stop_at_first=True,
            )
            print(f"[condition length][{name}] {diag}")

        s = torch_screen_summary(
            P,
            loss_cfg=loss_cfg,
            disc_coarse_samples=int(args.screen_coarse),
            disc_refine_steps=int(args.screen_refine_steps),
            disc_refine_samples=int(args.screen_refine),
            sylvester_smin_coarse_samples=33,
            sylvester_smin_refine_steps=1,
            sylvester_smin_refine_samples=33,
            sylvester_smin_on_cpu=True,
        )
        print(
            f"[torch screen][{name}] "
            f"min_|Disc|={s['min_abs_disc']:.6e} "
            f"(min_log|Disc|={s['min_logabs_disc']:.6e}) "
            f"at segment={int(s['disc_segment'])}, "
            f"t*={s['disc_t_star']:.6e} "
            "| "
            f"min_rel_smin={s['min_rel_smin']:.6e} "
            f"(min_log(rel_smin)={s['min_logrel_smin']:.6e}) "
            f"at segment={int(s['smin_segment'])}, "
            f"t*={s['smin_t_star']:.6e}"
        )
        print("--------------------------------")

    # Julia HC tracking comparison
    if jl is not None:
        # Linear baseline for HC should be the single straight-line homotopy between endpoints.
        # `track_piecewise_linear_julia` expects control points with shape (K+1, degree, 2),
        # so we pass just the endpoints (K=1).
        p0_ri = to_ri(p_start).detach().cpu()
        p1_ri = to_ri(p_target).detach().cpu()
        P_linear_1_np = torch.stack([p0_ri, p1_ri], dim=0).numpy()
        P_opt_np = P_best.detach().cpu().numpy()

        order = [("linear", P_linear_1_np), ("optimized", P_opt_np)]
        if args.swap_order:
            order = list(reversed(order))

        print("\n[Julia HomotopyContinuation tracking] warm-up + measured runs ...")
        for name, P_np in order:
            print(f"\n--- {name} ---")
            meas = benchmark_piecewise_linear_julia(
                jl,
                P_np,
                warmup_runs=int(args.warmup_runs),
                runs=int(args.runs),
                warmup_tracker_opts=JuliaTrackerOptions(),
                measured_tracker_opts=JuliaTrackerOptions(),
            )
            s = summarize_julia_runs(meas, topk=12) if meas else None
            if s is None:
                print(f"{name}: no measured runs")
                continue
            print(f"{name}: success_rate={s.success_rate*100:.1f}% (n={len(meas)})")
            print(f"  total_steps:          mean={s.total_steps_mean:.2f}, std={s.total_steps_std:.2f}")
            if s.total_accepted_steps_mean is not None and s.total_accepted_steps_std is not None:
                print(f"  total_accepted_steps: mean={s.total_accepted_steps_mean:.2f}, std={s.total_accepted_steps_std:.2f}")
            print(
                f"  total_rejected_steps: mean={s.total_rejected_steps_mean:.2f}, std={s.total_rejected_steps_std:.2f}"
            )
            print(f"  runtime[s]:           mean={s.wall_time_mean:.6f}, std={s.wall_time_std:.6f}")
            if s.return_code_counts_top:
                print("  return_code counts (top):")
                for code, cnt in s.return_code_counts_top:
                    print(f"    {code}: {cnt}")
            if s.return_code_total is not None and s.return_code_non_success_total is not None:
                print(f"  return_code non-success total: {s.return_code_non_success_total} / {s.return_code_total}")
            if s.min_n_success_per_segment is not None:
                print(
                    f"  min n_success per segment observed: {s.min_n_success_per_segment} "
                    f"(at segment {s.min_n_success_segment})"
                )


if __name__ == "__main__":
    main()


