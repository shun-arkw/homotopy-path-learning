from __future__ import annotations

"""
Parameter-homotopy speed check for univariate monic polynomials.

Idea:
  - Build ONE fixed System F(x; c) with parameters c = (a0, a1, ..., a_{d-1}) (monic tail).
  - For each pair (p_start, p_target), run:
        solve(F, starts; start_parameters=p_start, target_parameters=p_target)
    where `starts` are roots of the start polynomial.

This avoids rebuilding System(H(x,t)) per sample.

Run:
  python3 -m scripts._julia_toy_examples.exp_julia_param_hc --n 10000 --degree 10
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np


# --------------------------------------------------------------------------------------
# Helpers: roots of monic polynomial x^d + a_{d-1}x^{d-1} + ... + a0
# Here, tail is (a_{d-1}, ..., a0) length d in your codebase,
# but parameters we use in Julia are (a0, a1, ..., a_{d-1}) length d.
# We'll convert accordingly.
# --------------------------------------------------------------------------------------
def _roots_monic_tail_desc(tail_desc: np.ndarray) -> np.ndarray:
    """tail_desc: (d,) complex = [a_{d-1}, ..., a0]. returns roots (d,) complex."""
    tail_desc = np.asarray(tail_desc)
    d = int(tail_desc.shape[0])
    # numpy.roots expects coefficients highest degree first: [1, a_{d-1}, ..., a0]
    coeffs = np.concatenate([np.array([1.0 + 0.0j]), tail_desc.astype(np.complex128)], axis=0)
    roots = np.roots(coeffs)
    return roots.astype(np.complex128)


def _tail_desc_to_params_asc(tail_desc: np.ndarray) -> np.ndarray:
    """[a_{d-1},...,a0] -> [a0,a1,...,a_{d-1}]"""
    tail_desc = np.asarray(tail_desc)
    return tail_desc[::-1].copy()


# --------------------------------------------------------------------------------------
# Julia init + functions
# --------------------------------------------------------------------------------------
_JULIA_INIT = r"""
using HomotopyContinuation
using DynamicPolynomials

const __PY_PARAM_CACHE__ = Dict{Int, Any}()

# Get or build cached univariate monic system of degree d with parameters c[1:d] = (a0..a_{d-1}).
function __py_get_univar_param_system(d::Int)
    if haskey(__PY_PARAM_CACHE__, d)
        return __PY_PARAM_CACHE__[d]
    end
    DynamicPolynomials.@polyvar x
    DynamicPolynomials.@polyvar c[1:d]

    f = x^d
    for k in 1:d
        f += c[k] * x^(k-1)
    end

    sys = System([f]; variables=[x], parameters=vec(c))
    __PY_PARAM_CACHE__[d] = (sys=sys, x=x, c=c)
    return __PY_PARAM_CACHE__[d]
end

# Make TrackerOptions from numbers (kept simple for speed tests).
function __py_make_opts(max_steps::Int, max_step_size::Float64, max_initial_step_size::Float64,
                       min_step_size::Float64, min_rel_step_size::Float64, extended_precision::Bool)
    return HomotopyContinuation.TrackerOptions(
        max_steps=max_steps,
        max_step_size=max_step_size,
        max_initial_step_size=max_initial_step_size,
        min_step_size=min_step_size,
        min_rel_step_size=min_rel_step_size,
        extended_precision=extended_precision,
    )
end

# Run parameter homotopy for one pair on cached system.
function __py_solve_param_pair(d::Int, p_start::Vector{ComplexF64}, p_target::Vector{ComplexF64},
                              starts::Vector{Vector{ComplexF64}}, opts::HomotopyContinuation.TrackerOptions)
    sys = (__py_get_univar_param_system(d)).sys

    t0 = time_ns()
    res = solve(sys, starts;
        start_parameters=p_start,
        target_parameters=p_target,
        tracker_options=opts,
        show_progress=false,  # ついでにプログレスバー抑制
    )
    wall = (time_ns() - t0) * 1e-9

    prs = getproperty(res, :path_results)
    total_steps = 0
    ok = true
    for pr in prs
        # PathResult には accepted_steps / rejected_steps がある
        acc = getproperty(pr, :accepted_steps)
        rej = getproperty(pr, :rejected_steps)
        total_steps += acc + rej

        if getproperty(pr, :return_code) != :success
            ok = false
        end
    end
    return (ok=ok, total_steps=total_steps, solve_wall_time_sec=wall)
end
"""



def _import_julia():
    # Import juliacall as early as possible to avoid torch-segfault warning
    from juliacall import Main as jl
    return jl


@dataclass
class Report:
    degree: int
    n_pairs: int
    seed: int
    import_julia_sec: float
    init_julia_sec: float
    warmup_sec: float
    mean_solve_sec: float
    std_solve_sec: float
    ok_rate: float
    mean_steps: float
    est_per_step_ms: float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=10)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print-every", type=int, default=200)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--extended-precision", action="store_true")
    ap.add_argument("--max-steps", type=int, default=50000)
    ap.add_argument("--max-step-size", type=float, default=0.05)
    ap.add_argument("--max-initial-step-size", type=float, default=0.05)
    ap.add_argument("--min-step-size", type=float, default=1e-12)
    ap.add_argument("--min-rel-step-size", type=float, default=1e-12)
    args = ap.parse_args()

    d = int(args.degree)
    n = int(args.n)
    seed = int(args.seed)
    if d < 2:
        raise ValueError("--degree must be >= 2")
    if n < 0:
        raise ValueError("--n must be >= 0")

    # 1) import Julia
    t0 = time.perf_counter()
    jl = _import_julia()
    t_import = time.perf_counter() - t0

    # 2) define Julia functions once
    t1 = time.perf_counter()
    jl.seval(_JULIA_INIT)
    t_init = time.perf_counter() - t1

    # 3) make opts once
    opts = jl.__py_make_opts(
        int(args.max_steps),
        float(args.max_step_size),
        float(args.max_initial_step_size),
        float(args.min_step_size),
        float(args.min_rel_step_size),
        bool(args.extended_precision),
    )

    rng = np.random.default_rng(seed)

    def make_random_tail_desc() -> np.ndarray:
        # tail_desc = [a_{d-1},...,a0]
        return (rng.standard_normal(d) + 1j * rng.standard_normal(d)).astype(np.complex128)

    # 4) warmup once
    tail0 = make_random_tail_desc()
    tail1 = make_random_tail_desc()
    roots0 = _roots_monic_tail_desc(tail0)
    p_start = _tail_desc_to_params_asc(tail0)
    p_target = _tail_desc_to_params_asc(tail1)

    # to Julia types (ComplexF64, Vector{Vector{ComplexF64}})
    p_start_jl = jl.Vector[jl.ComplexF64]([complex(z.real, z.imag) for z in p_start.tolist()])
    p_target_jl = jl.Vector[jl.ComplexF64]([complex(z.real, z.imag) for z in p_target.tolist()])
    starts_jl = jl.Vector[jl.Vector[jl.ComplexF64]]([[complex(r.real, r.imag)] for r in roots0.tolist()])

    tw0 = time.perf_counter()
    _ = jl.__py_solve_param_pair(int(d), p_start_jl, p_target_jl, starts_jl, opts)
    t_warmup = time.perf_counter() - tw0

    # 5) measured runs
    solve_times = []
    steps_list = []
    oks = []

    for i in range(n):
        tail0 = make_random_tail_desc()
        tail1 = make_random_tail_desc()
        roots0 = _roots_monic_tail_desc(tail0)

        p_start = _tail_desc_to_params_asc(tail0)
        p_target = _tail_desc_to_params_asc(tail1)

        p_start_jl = jl.Vector[jl.ComplexF64]([complex(z.real, z.imag) for z in p_start.tolist()])
        p_target_jl = jl.Vector[jl.ComplexF64]([complex(z.real, z.imag) for z in p_target.tolist()])
        starts_jl = jl.Vector[jl.Vector[jl.ComplexF64]]([[complex(r.real, r.imag)] for r in roots0.tolist()])

        out = jl.__py_solve_param_pair(int(d), p_start_jl, p_target_jl, starts_jl, opts)

        solve_times.append(float(out.solve_wall_time_sec))
        steps_list.append(int(out.total_steps))
        oks.append(bool(out.ok))

        pe = int(args.print_every)
        if pe > 0 and ((i + 1) % pe == 0):
            print(
                f"[exp_julia_param_homotopy] progress {i+1}/{n} "
                f"(solve_wall_time_sec={solve_times[-1]:.6f}, steps={steps_list[-1]}, ok={oks[-1]})"
            )

    arr_t = np.asarray(solve_times, dtype=float)
    arr_s = np.asarray(steps_list, dtype=float)
    ok_rate = float(np.mean(oks)) if len(oks) else 0.0

    mean_t = float(arr_t.mean()) if len(arr_t) else 0.0
    std_t = float(arr_t.std(ddof=1)) if len(arr_t) > 1 else 0.0
    mean_steps = float(arr_s.mean()) if len(arr_s) else 0.0
    est_per_step_ms = (mean_t / mean_steps) * 1000.0 if mean_steps > 0 else 0.0

    rep = Report(
        degree=d,
        n_pairs=n,
        seed=seed,
        import_julia_sec=float(t_import),
        init_julia_sec=float(t_init),
        warmup_sec=float(t_warmup),
        mean_solve_sec=mean_t,
        std_solve_sec=std_t,
        ok_rate=ok_rate,
        mean_steps=mean_steps,
        est_per_step_ms=float(est_per_step_ms),
    )

    print("[exp_julia_param_homotopy] fixed-System parameter-homotopy benchmark")
    print(f"  degree={rep.degree} n_pairs={rep.n_pairs} seed={rep.seed}")
    print(f"  import_julia_sec={rep.import_julia_sec:.6f}")
    print(f"  init_julia_sec={rep.init_julia_sec:.6f}")
    print(f"  warmup_sec={rep.warmup_sec:.6f}")
    print(f"  solve_wall_time_sec_mean={rep.mean_solve_sec:.6f} ± {rep.std_solve_sec:.6f}   ok_rate={rep.ok_rate*100:.1f}%")
    print(f"  mean_total_steps={rep.mean_steps:.1f}")
    print(f"  estimated_per_step_ms={rep.est_per_step_ms:.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(asdict(rep), f, indent=2)


if __name__ == "__main__":
    main()
