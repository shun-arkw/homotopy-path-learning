from __future__ import annotations

"""
Julia warmup (JIT/using/import) を N 個の問題で毎回払わないための検証スクリプト。

ポイント:
  - `juliacall` は同一 Python プロセス内では Julia セッションが維持される。
  - したがって (p_start, p_target) が N 個あっても、重い warmup は最初の 1 回に集約できる。

Run as a module to keep relative imports stable:
    python3 -m scripts.task2.optimal_path_finding.exp_julia_batch --n 50 --degree 10
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from .utils.julia_hc import (  # NOTE: import juliacall BEFORE torch (torch is not used here)
    JuliaTrackerOptions,
    _maybe_import_julia,
    track_piecewise_linear_julia,
)


def _to_ri(p: np.ndarray) -> np.ndarray:
    """complex (degree,) -> (degree, 2) as (Re, Im)."""
    p = np.asarray(p)
    if p.ndim != 1:
        raise ValueError(f"p must be 1D, got shape={p.shape}")
    return np.stack([p.real, p.imag], axis=-1)


def _make_linear_path_np(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """(degree,) complex endpoints -> (2, degree, 2) control points for K=1 piecewise-linear path."""
    return np.stack([_to_ri(p0), _to_ri(p1)], axis=0)


@dataclass
class Report:
    degree: int
    n_pairs: int
    seed: int
    extended_precision: bool | None
    import_julia_sec: float | None
    warmup_sec: float | None
    mean_sec: float | None
    std_sec: float | None
    ok_rate: float | None
    mean_total_steps: float | None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-julia", action="store_true", help="Skip Julia (sanity-check mode).")
    ap.add_argument("--degree", type=int, default=6, help="Length of monic tail coefficient vector.")
    ap.add_argument("--n", type=int, default=50, help="Number of (p_start, p_target) pairs to measure (warmup excluded).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    ap.add_argument(
        "--extended-precision",
        action="store_true",
        help="Enable extended precision in HomotopyContinuation tracker options (can be much slower).",
    )
    ap.add_argument(
        "--compute-newton-iters",
        action="store_true",
        help="Also compute Newton-iteration totals by running debug tracking per start solution (SLOW).",
    )
    ap.add_argument("--print-every", type=int, default=10, help="Print progress every N cases (0 disables).")
    ap.add_argument("--out", type=str, default="", help="Optional JSON output path.")
    ap.add_argument("--debug-type", action="store_true", help="Debug: print System types (first segment only).")
    args = ap.parse_args()

    degree = int(args.degree)
    n = int(args.n)
    seed = int(args.seed)
    if degree < 2:
        raise ValueError("--degree must be >= 2")
    if n < 0:
        raise ValueError("--n must be >= 0")

    # Import/initialize Julia once
    t0 = time.perf_counter()
    jl = _maybe_import_julia(enable=not args.no_julia)
    t_import = time.perf_counter() - t0

    if jl is None:
        rep = Report(
            degree=degree,
            n_pairs=n,
            seed=seed,
            extended_precision=None,
            import_julia_sec=None,
            warmup_sec=None,
            mean_sec=None,
            std_sec=None,
            ok_rate=None,
            mean_total_steps=None,
        )
        print("[exp_julia_batch] Julia disabled (--no-julia). Exiting.")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(asdict(rep), f, indent=2)
        return

    # Tracker options: keep consistent with other experiments
    opts = JuliaTrackerOptions(
        max_steps=50000,
        max_step_size=0.05,
        max_initial_step_size=0.05,
        min_step_size=1e-12,
        min_rel_step_size=1e-12,
        extended_precision=bool(args.extended_precision),
    )

    rng = np.random.default_rng(seed)

    # Warmup (exclude from measured statistics)
    p0 = rng.standard_normal(degree) + 1j * rng.standard_normal(degree)
    p1 = rng.standard_normal(degree) + 1j * rng.standard_normal(degree)
    P = _make_linear_path_np(p0, p1)

    t1 = time.perf_counter()
    _ = track_piecewise_linear_julia(
        jl,
        P,
        tracker_opts=opts,
        suppress_julia_output=True,
        compute_newton_iters=bool(args.compute_newton_iters),
    )
    t_warmup = time.perf_counter() - t1

    # Prepare batch data: collect all (p_start, p_target) pairs
    pairs_list: list[tuple[np.ndarray, np.ndarray]] = []
    starts_list: list[list[list[complex]]] = []
    for i in range(n):
        p0 = rng.standard_normal(degree) + 1j * rng.standard_normal(degree)
        p1 = rng.standard_normal(degree) + 1j * rng.standard_normal(degree)
        P = _make_linear_path_np(p0, p1)
        
        # Compute starts for this pair
        from .utils.julia_hc.tracker import _p_ri_to_tail_complex, _poly_roots_monic_tail_coeffs
        p0_tail = _p_ri_to_tail_complex(P[0])
        roots0 = _poly_roots_monic_tail_coeffs(p0_tail)
        starts = [[complex(r)] for r in roots0.tolist()]
        
        pairs_list.append((p0, p1))
        starts_list.append(starts)
    
    # Batch processing: call Julia once with all pairs
    # This avoids Python-Julia overhead and allows System construction to be optimized
    t_batch0 = time.perf_counter()
    from .utils.julia_hc.tracker import track_piecewise_linear_julia_batch
    batch_results = track_piecewise_linear_julia_batch(
        jl,
        pairs_list,
        starts_list,
        tracker_opts=opts,
        suppress_julia_output=True,
        compute_newton_iters=bool(args.compute_newton_iters),
    )
    t_batch = time.perf_counter() - t_batch0
    
    # Extract results from batch
    times_solve: list[float] = []
    times_sys_build: list[float] = []
    times_py: list[float] = []
    times_warmup: list[float] = []
    times_after_warmup: list[float] = []
    oks: list[bool] = []
    total_steps_list: list[int] = []
    
    for i, out in enumerate(batch_results):
        times_py.append(float(t_batch / n))  # Average Python call time per sample
        times_after_warmup.append(float(out.get("wall_time_sec", 0.0)))
        times_solve.append(float(out.get("wall_time_sec", 0.0)))
        times_sys_build.append(float(out.get("sys_build_time_sec", 0.0)))
        oks.append(bool(out.get("ok", False)))
        total_steps_list.append(int(out.get("total_steps", 0)))
        times_warmup.append(0.0)  # No per-sample warmup in batch mode
        
        pe = int(args.print_every)
        if pe > 0 and ((i + 1) % pe == 0):
            print(
                f"[exp_julia_batch] progress {i+1}/{n} "
                f"(solve_wall_time_sec={times_solve[-1]:.6f}, batch_avg_sec={times_py[-1]:.6f})"
            )

    arr_solve = np.asarray(times_solve, dtype=float)
    arr_sys_build = np.asarray(times_sys_build, dtype=float)
    arr_py = np.asarray(times_py, dtype=float)
    arr_warmup = np.asarray(times_warmup, dtype=float) if len(times_warmup) > 0 else np.array([], dtype=float)
    arr_after_warmup = np.asarray(times_after_warmup, dtype=float) if len(times_after_warmup) > 0 else np.array([], dtype=float)
    arr_steps = np.asarray(total_steps_list, dtype=float)
    rep = Report(
        degree=degree,
        n_pairs=n,
        seed=seed,
        extended_precision=bool(args.extended_precision),
        import_julia_sec=float(t_import),
        warmup_sec=float(t_warmup),
        mean_sec=float(arr_solve.mean()) if len(arr_solve) else None,
        std_sec=float(arr_solve.std(ddof=1)) if len(arr_solve) > 1 else (0.0 if len(arr_solve) == 1 else None),
        ok_rate=float(np.mean(oks)) if len(oks) else None,
        mean_total_steps=float(arr_steps.mean()) if len(arr_steps) else None,
    )

    print("[exp_julia_batch] single-process Julia warmup amortization check")
    print(f"  degree={rep.degree} n_pairs={rep.n_pairs} seed={rep.seed}")
    if rep.extended_precision is not None:
        print(f"  extended_precision={bool(rep.extended_precision)}")
    print(f"  import_julia_sec={rep.import_julia_sec:.6f}")
    print(f"  warmup_sec={rep.warmup_sec:.6f}")
    if rep.mean_sec is not None and rep.std_sec is not None and rep.ok_rate is not None:
        # rep.mean_sec is the mean of solve_wall_time_sec (from Julia's time_ns() around solve(...))
        print(f"  solve_wall_time_sec_mean={rep.mean_sec:.6f} ± {rep.std_sec:.6f}   ok_rate={rep.ok_rate*100:.1f}%")
        if len(arr_sys_build) > 0:
            sys_build_mean = float(arr_sys_build.mean())
            sys_build_std = float(arr_sys_build.std(ddof=1)) if len(arr_sys_build) > 1 else 0.0
            print(f"  sys_build_time_sec_mean={sys_build_mean:.6f} ± {sys_build_std:.6f}")
            # Show solve-only time (solve - sys_build) regardless of sys_build size
            solve_only_mean = rep.mean_sec - sys_build_mean
            print(f"  solve_only_time_sec_mean={solve_only_mean:.6f} (solve_wall - sys_build)")
        # Check per-sample warmup vs after-warmup timing
        if len(arr_warmup) > 0 and len(arr_after_warmup) > 0:
            warmup_mean = float(arr_warmup.mean())
            after_warmup_mean = float(arr_after_warmup.mean())
            warmup_std = float(arr_warmup.std(ddof=1)) if len(arr_warmup) > 1 else 0.0
            after_warmup_std = float(arr_after_warmup.std(ddof=1)) if len(arr_after_warmup) > 1 else 0.0
            ratio = warmup_mean / after_warmup_mean if after_warmup_mean > 0 else 1.0
            print(f"  per_sample_warmup_sec_mean={warmup_mean:.6f} ± {warmup_std:.6f}")
            print(f"  after_warmup_sec_mean={after_warmup_mean:.6f} ± {after_warmup_std:.6f} (ratio={ratio:.3f})")
            if ratio > 1.5:  # Warmup is >50% slower
                print(f"    -> JIT compilation happening per sample (warmup much slower)")
            elif ratio < 1.1:  # Warmup and after-warmup are similar
                print(f"    -> No JIT compilation per sample (warmup and after-warmup similar)")
        # Check if first few runs are slower (possible JIT compilation per solve)
        if len(arr_solve) >= 5:
            first_3_mean = float(arr_solve[:3].mean())
            last_3_mean = float(arr_solve[-3:].mean())
            ratio = first_3_mean / last_3_mean if last_3_mean > 0 else 1.0
            print(f"  first_3_solve_sec_mean={first_3_mean:.6f} vs last_3_solve_sec_mean={last_3_mean:.6f} (ratio={ratio:.3f})")
            if ratio > 1.1:  # First 3 are >10% slower
                print(f"    -> possible JIT compilation per solve (first runs slower)")
        if rep.mean_total_steps is not None:
            print(f"  mean_total_steps={rep.mean_total_steps:.1f}")
            # Per-step time estimate
            if rep.mean_sec is not None and rep.mean_total_steps > 0:
                per_step_ms = (rep.mean_sec / rep.mean_total_steps) * 1000.0
                print(f"  estimated_per_step_ms={per_step_ms:.3f}")
        if len(arr_py):
            print(
                f"  python_call_mean_sec={float(arr_py.mean()):.6f} ± "
                f"{float(arr_py.std(ddof=1)) if len(arr_py) > 1 else 0.0:.6f}"
            )
    else:
        print("  measured: N/A")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(asdict(rep), f, indent=2)


if __name__ == "__main__":
    main()


