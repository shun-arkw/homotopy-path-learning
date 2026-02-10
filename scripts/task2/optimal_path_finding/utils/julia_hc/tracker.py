from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np


# =============================================================================
# Types / public data contracts
# =============================================================================

@dataclass
class JuliaTrackerOptions:
    max_steps: int = 50_000
    max_step_size: float = 0.05
    max_initial_step_size: float = 0.05
    min_step_size: float = 1e-12
    min_rel_step_size: float = 1e-12
    extended_precision: bool = True
    parameters: Literal["default", "conservative", "fast"] = "fast"


class JuliaSegmentDetail(TypedDict):
    segment: int
    nresults: int
    nsolutions: int
    n_success: int
    return_codes: list[str]
    tracker_opts: dict[str, Any]
    sum_steps: int
    sum_accepted: int
    sum_rejected: int
    # Optional / mode-specific metrics.
    solve_wall_time_sec: float
    sum_newton_iters: int


class JuliaRunResult(TypedDict):
    ok: bool
    total_steps: int
    total_accepted_steps: int
    total_rejected_steps: int
    total_newton_iters: int
    wall_time_sec: float
    detail: list[JuliaSegmentDetail]


# =============================================================================
# Numeric helpers (Python-side)
# =============================================================================

def _poly_roots_monic_tail_coeffs(tail_coeffs: np.ndarray) -> np.ndarray:
    """
    For monic polynomial x^d + a_{d-1} x^{d-1} + ... + a_0,
    input tail_coeffs=[a_{d-1},...,a_0], return its d complex roots.
    """

    coeffs = np.concatenate(([1.0 + 0.0j], tail_coeffs.astype(np.complex128)))
    return np.roots(coeffs)


def _p_ri_to_tail_complex(p_ri) -> np.ndarray:
    # p_ri: (degree, 2) torch-like array; accept numpy too.
    p_ri = np.asarray(p_ri)
    return p_ri[:, 0].astype(np.float64) + 1j * p_ri[:, 1].astype(np.float64)


def _eval_monic_poly_tail_desc(tail_coeffs: np.ndarray, z: complex) -> complex:
    """
    Evaluate monic polynomial in descending powers:
        p(x) = x^n + a_{n-1} x^{n-1} + ... + a_0
    where tail_coeffs = [a_{n-1}, ..., a_0].

    Uses Horner scheme:
        y = 1
        for a in tail_coeffs: y = y*z + a
    """

    y: complex = 1.0 + 0.0j
    for a in np.asarray(tail_coeffs, dtype=np.complex128):
        y = y * complex(z) + complex(a)
    return y


def _validate_start_solutions_for_monic_tail(
    tail_coeffs: np.ndarray,
    starts: list[list[complex]],
    *,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> None:
    """Lightweight defense: check `starts` are (approximately) roots of the monic polynomial."""

    tail_coeffs = np.asarray(tail_coeffs, dtype=np.complex128)
    degree = int(tail_coeffs.shape[0])
    if len(starts) != degree:
        raise ValueError(f"start_solutions must have length={degree}, got {len(starts)}.")
    zs: list[complex] = []
    for i, s in enumerate(starts):
        if not isinstance(s, (list, tuple)) or len(s) != 1:
            raise ValueError(f"start_solutions[{i}] must be a length-1 list/tuple, got {type(s)} with len={len(s)}.")
        zs.append(complex(s[0]))

    vals = np.array([_eval_monic_poly_tail_desc(tail_coeffs, z) for z in zs], dtype=np.complex128)
    err = float(np.max(np.abs(vals))) if vals.size else 0.0
    scale = float(1.0 + np.max(np.abs(tail_coeffs))) if tail_coeffs.size else 1.0
    if not (err <= float(atol) + float(rtol) * scale):
        raise ValueError(
            "start_solutions check failed: "
            f"max|p0(z)|={err:.3e} (tol={float(atol) + float(rtol) * scale:.3e}, scale={scale:.3e})."
        )


# =============================================================================
# Julia-side implementations (injected via juliacall.seval)
# =============================================================================

# --- Piecewise-linear tracking (single segment)
_JULIA_HC_DEFINE_SNIPPET = r"""
function __py_hc_poly_from_coeffs_desc(coeffs::Vector{ComplexF64}, x)
    d = length(coeffs) - 1
    p = zero(x)
    for (i, c) in enumerate(coeffs)  # i=1..d+1
        p += c * x^(d - (i - 1))
    end
    return p
end

# Batch tracking: process multiple (p_start, p_target) pairs in a single Julia call
# This avoids Python-Julia overhead by processing all pairs in a Julia-side loop
# NOTE: We still rebuild System per sample because coefficients are embedded in polynomial structure
# However, doing this in Julia avoids Python-Julia overhead
function __py_hc_track_batch(pairs_list::Vector{Tuple{Vector{ComplexF64}, Vector{ComplexF64}}}, starts_list::Vector{Vector{Vector{ComplexF64}}}, opts::HomotopyContinuation.TrackerOptions, suppress::Bool, compute_newton_iters::Bool)
    n_batch = length(pairs_list)
    if n_batch == 0
        return Vector{Any}(undef, 0)
    end
    
    results_batch = Vector{Any}(undef, n_batch)
    
    # Process each pair in the batch (loop inside Julia to avoid Python-Julia overhead)
    for i in 1:n_batch
        (q_coeffs, p_coeffs) = pairs_list[i]
        starts = starts_list[i]
        
        # Build System for this specific problem
        # NOTE: We still need to rebuild because coefficients are embedded in the polynomial structure
        # However, by doing this in Julia, we avoid Python-Julia overhead
        Q = [__py_hc_poly_from_coeffs_desc(q_coeffs, x)]
        P = [__py_hc_poly_from_coeffs_desc(p_coeffs, x)]
        H = (1 - t) .* Q .+ t .* P
        Hsys = System(H; variables=[x], parameters=[t])
        
        # Solve
        res = suppress ? redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                solve(Hsys, starts;
                    start_parameters=[0.0],
                    target_parameters=[1.0],
                    tracker_options=opts,
                )
            end
        end : solve(Hsys, starts;
            start_parameters=[0.0],
            target_parameters=[1.0],
            tracker_options=opts,
        )
        
        rr = results(res)
        rcs = string.(getproperty.(rr, :return_code))
        n_success = count(==(Symbol(:success)), getproperty.(rr, :return_code))
        
        sum_newton_iters = 0
        if compute_newton_iters
            start_params  = [0.0]
            target_params = [1.0]
            Hh = parameter_homotopy(Hsys; start_parameters=start_params, target_parameters=target_params)
            tracker = Tracker(Hh; options=opts)
            @inbounds for s in starts
                pipe = Pipe()
                redirect_stdout(pipe) do
                    redirect_stderr(pipe) do
                        track(tracker, s, 1.0, 0.0; debug=true)
                    end
                end
                close(pipe.in)
                log = String(read(pipe))
                for m in eachmatch(r"iters\s*→\s*(\d+)", log)
                    sum_newton_iters += parse(Int, m.captures[1])
                end
            end
        end
        
        results_batch[i] = (;
            ok=(n_success == length(starts)),
            total_steps=sum(steps.(rr)),
            total_accepted=sum(accepted_steps.(rr)),
            total_rejected=sum(rejected_steps.(rr)),
            solve_wall_time_sec=0.0,  # Will be measured in Python
            sum_newton_iters=sum_newton_iters,
            end_solutions=solutions(res),
            return_codes=rcs,
        )
    end
    
    return results_batch
end

function __py_hc_track_segment(q_coeffs::Vector{ComplexF64}, p_coeffs::Vector{ComplexF64}, starts::Vector{Vector{ComplexF64}}, opts::HomotopyContinuation.TrackerOptions, suppress::Bool, compute_newton_iters::Bool, debug_type::Bool=false)
    t_sys0 = time_ns()
    # Type is already ComplexF64, no conversion needed
    Q = [__py_hc_poly_from_coeffs_desc(q_coeffs, x)]
    P = [__py_hc_poly_from_coeffs_desc(p_coeffs, x)]
    H = (1 - t) .* Q .+ t .* P
    # Force System type to be stable by using explicit type annotation
    Hsys::System = System(H; variables=[x], parameters=[t])
    sys_build_time_sec = (time_ns() - t_sys0) / 1e9

    # Debug: check System type
    if debug_type
        println("DEBUG: typeof(Hsys) = ", typeof(Hsys))
        println("DEBUG: typeof(H) = ", typeof(H))
        println("DEBUG: typeof(H[1]) = ", typeof(H[1]))
    end
    
    t0 = time_ns()
    res = suppress ? redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            solve(Hsys, starts;
                start_parameters=[0.0],
                target_parameters=[1.0],
                tracker_options=opts,
            )
        end
    end : solve(Hsys, starts;
        start_parameters=[0.0],
        target_parameters=[1.0],
        tracker_options=opts,
    )
    solve_wall_time_sec = (time_ns() - t0) / 1e9

    rr = results(res)
    rcs = string.(getproperty.(rr, :return_code))
    n_success = count(==(Symbol(:success)), getproperty.(rr, :return_code))

    # Sum Newton iteration counts by parsing debug logs.
    #
    # NOTE:
    # - HomotopyContinuation.jl does not expose Newton iteration totals via solve(...) results.
    # - Therefore we run `track(...; debug=true)` once per start solution and parse "iters → N".
    # - This is intentionally NOT included in any wall-time measurements on the Python side.
    sum_newton_iters = 0
    if compute_newton_iters
        start_params  = [0.0]
        target_params = [1.0]
        Hh = parameter_homotopy(Hsys; start_parameters=start_params, target_parameters=target_params)
        tracker = Tracker(Hh; options=opts)
        @inbounds for s in starts
            pipe = Pipe()
            redirect_stdout(pipe) do
                redirect_stderr(pipe) do
                    track(tracker, s, 1.0, 0.0; debug=true)
                end
            end
            close(pipe.in)
            log = String(read(pipe))
            for m in eachmatch(r"iters\s*→\s*(\d+)", log)
                sum_newton_iters += parse(Int, m.captures[1])
            end
        end
    end

    return (;
        nsolutions=nsolutions(res),
        nresults=length(rr),
        n_success=n_success,
        return_codes=rcs,
        sum_steps=sum(steps.(rr)),
        sum_accepted=sum(accepted_steps.(rr)),
        sum_rejected=sum(rejected_steps.(rr)),
        solve_wall_time_sec=solve_wall_time_sec,
        sys_build_time_sec=sys_build_time_sec,
        sum_newton_iters=sum_newton_iters,
        end_solutions=solutions(res),
    )
end

"""


_JULIA_HC_DEFINE_PIECEWISE_SNIPPET = r"""
function __py_hc_track_piecewise(coeffs_list, starts, opts_list, default_opts, suppress::Bool, compute_newton_iters::Bool, debug_type::Bool=false)
    K = length(coeffs_list) - 1
    total_steps = 0
    total_accepted = 0
    total_rejected = 0
    total_newton_iters = 0
    total_solve_wall_time_sec = 0.0
    total_sys_build_time_sec = 0.0
    details = Vector{Any}(undef, 0)
    ok = true

    # Degree of monic polynomial: x^d + ...
    degree = length(coeffs_list[1]) - 1

    for seg in 1:K
        q_coeffs = coeffs_list[seg]
        p_coeffs = coeffs_list[seg + 1]
        opts = isnothing(opts_list) ? default_opts : opts_list[seg]

        out = __py_hc_track_segment(q_coeffs, p_coeffs, starts, opts, suppress, compute_newton_iters, debug_type && seg == 1)
        push!(
            details,
            (;
                segment=seg - 1,
                nresults=out.nresults,
                nsolutions=out.nsolutions,
                n_success=out.n_success,
                return_codes=out.return_codes,
                sum_steps=out.sum_steps,
                sum_accepted=out.sum_accepted,
                sum_rejected=out.sum_rejected,
                solve_wall_time_sec=out.solve_wall_time_sec,
                sys_build_time_sec=out.sys_build_time_sec,
                sum_newton_iters=out.sum_newton_iters,
            ),
        )

        total_steps += out.sum_steps
        total_accepted += out.sum_accepted
        total_rejected += out.sum_rejected
        total_newton_iters += out.sum_newton_iters
        total_solve_wall_time_sec += out.solve_wall_time_sec
        total_sys_build_time_sec += out.sys_build_time_sec

        if out.n_success != degree
            ok = false
            break
        end
        starts = out.end_solutions
    end

    return (;
        ok=ok,
        total_steps=total_steps,
        total_accepted_steps=total_accepted,
        total_rejected_steps=total_rejected,
        total_newton_iters=total_newton_iters,
        solve_wall_time_sec=total_solve_wall_time_sec,
        sys_build_time_sec=total_sys_build_time_sec,
        detail=details,
    )
end
"""


_JULIA_HC_DEFINE_BEZIER_SNIPPET = r"""
# --- (A) Bezier control points -> power-basis coefficients (your original)
function __py_hc_bezier_tail_ctrls_to_power_coeffs(ctrls)
    d = length(ctrls) - 1
    a = Vector{eltype(ctrls)}(undef, d + 1)
    @inbounds for k in 0:d
        s = zero(ctrls[1])
        @inbounds for i in 0:k
            s += ctrls[i + 1] *
                 binomial(d, i) *
                 binomial(d - i, k - i) *
                 (isodd(k - i) ? (-one(s)) : one(s))
        end
        a[k + 1] = s
    end
    return a
end

# --- (B) Horner evaluation of c(t) in power basis: c(t)=Σ_{k=0..d} a[k+1] t^k
@inline function __py_hc_eval_power_poly_horner(a, t)
    d = length(a) - 1
    c = a[d + 1]
    @inbounds for k in (d-1):-1:0
        c = c * t + a[k + 1]
    end
    return c
end

# --- (C) Build p(x,t) using Horner in x, and Horner in t for each tail coefficient
function __py_hc_poly_from_power_tail_coeffs_desc(tail_power_by_coeff, x, t)
    deg = length(tail_power_by_coeff)          # monic polynomial degree
    p = one(x)
    @inbounds for j in 1:deg
        a = tail_power_by_coeff[j]
        c = __py_hc_eval_power_poly_horner(a, t)
        p = p * x + c
    end
    return p
end

# --- (D) Track function: precompute tail_power_by_coeff ONCE, then build System
function __py_hc_track_bezier(tail_ctrls_by_coeff, starts, opts, suppress::Bool)
    deg = length(tail_ctrls_by_coeff)

    tail_power_by_coeff = Vector{Vector{eltype(tail_ctrls_by_coeff[1])}}(undef, deg)
    @inbounds for j in 1:deg
        tail_power_by_coeff[j] = __py_hc_bezier_tail_ctrls_to_power_coeffs(tail_ctrls_by_coeff[j])
    end

    P = [__py_hc_poly_from_power_tail_coeffs_desc(tail_power_by_coeff, x, t)]
    Hsys = System(P; variables=[x], parameters=[t])

    start_params  = [0.0]
    target_params = [1.0]

    # 1) Allocation measurement (discard the solve result)
    GC.gc()
    bytes = suppress ? redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            @allocated solve(Hsys, starts;
                start_parameters=start_params,
                target_parameters=target_params,
                tracker_options=opts,
            )
        end
    end : @allocated solve(Hsys, starts;
        start_parameters=start_params,
        target_parameters=target_params,
        tracker_options=opts,
    )

    # 2) Wall-clock time + result (the "real" solve)
    t0 = time_ns()
    res = suppress ? redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            solve(Hsys, starts;
                start_parameters=start_params,
                target_parameters=target_params,
                tracker_options=opts,
            )
        end
    end : solve(Hsys, starts;
        start_parameters=start_params,
        target_parameters=target_params,
        tracker_options=opts,
    )
    solve_wall_time_sec = (time_ns() - t0) / 1e9

    println("bytes: ", bytes)


    rr = results(res)
    rcs = string.(getproperty.(rr, :return_code))
    n_success = count(==(Symbol(:success)), getproperty.(rr, :return_code))

    # 3) Sum Newton iteration counts by parsing debug logs.
    #
    # NOTE:
    # - HomotopyContinuation.jl does not expose "Newton iterations per correction"
    #   via the `solve(...)` result object (at least in the versions we've used).
    # - Therefore we run `track(...; debug=true)` once per start solution and parse
    #   lines like "iters → N". This is intentionally NOT included in solve_wall_time_sec.
    H = parameter_homotopy(Hsys; start_parameters=start_params, target_parameters=target_params)
    tracker = Tracker(H; options=opts)

    sum_newton_iters = 0
    @inbounds for s in starts
        pipe = Pipe()
        redirect_stdout(pipe) do
            redirect_stderr(pipe) do
                track(tracker, s, 1.0, 0.0; debug=true)
            end
        end
        close(pipe.in)
        log = String(read(pipe))
        for m in eachmatch(r"iters\s*→\s*(\d+)", log)
            sum_newton_iters += parse(Int, m.captures[1])
        end
    end

    return (;
        nsolutions=nsolutions(res),
        nresults=length(rr),
        n_success=n_success,
        return_codes=rcs,
        sum_steps=sum(steps.(rr)),
        sum_accepted=sum(accepted_steps.(rr)),
        sum_rejected=sum(rejected_steps.(rr)),
        solve_wall_time_sec=solve_wall_time_sec,
        allocated_bytes=bytes,
        sum_newton_iters=sum_newton_iters,
    )
end
"""


_JULIA_HC_DEFINE_TOTAL_DEGREE_SNIPPET = r"""
function __py_hc_solve_total_degree(p_coeffs, opts, suppress::Bool)
    Psys = System([__py_hc_poly_from_coeffs_desc(p_coeffs, x)]; variables=[x])

    start_t = time_ns()
    res = suppress ? redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            solve(Psys; start_system=:total_degree, tracker_options=opts)
        end
    end : solve(Psys; start_system=:total_degree, tracker_options=opts)
    solve_wall_time_sec = (time_ns() - start_t) / 1e9

    rr = results(res)
    rcs = string.(getproperty.(rr, :return_code))
    n_success = count(==(Symbol(:success)), getproperty.(rr, :return_code))

    # Sum Newton iteration counts by parsing `track(...; debug=true)` logs.
    #
    # NOTE:
    # - This is intentionally NOT included in solve_wall_time_sec.
    # - We build the tracker+starts using total_degree(...) to match the baseline path setup.
    tracker, start_iter = total_degree(Psys; tracker_options=opts)
    starts = collect(start_iter)
    sum_newton_iters = 0
    @inbounds for s in starts
        pipe = Pipe()
        redirect_stdout(pipe) do
            redirect_stderr(pipe) do
                track(tracker, s; debug=true)
            end
        end
        close(pipe.in)
        log = String(read(pipe))
        for m in eachmatch(r"iters\s*→\s*(\d+)", log)
            sum_newton_iters += parse(Int, m.captures[1])
        end
    end

    return (;
        nsolutions=nsolutions(res),
        nresults=length(rr),
        n_success=n_success,
        return_codes=rcs,
        sum_steps=sum(steps.(rr)),
        sum_accepted=sum(accepted_steps.(rr)),
        sum_rejected=sum(rejected_steps.(rr)),
        solve_wall_time_sec=solve_wall_time_sec,
        sum_newton_iters=sum_newton_iters,
    )
end
"""




# =============================================================================
# Julia HC initialization / option conversion
# =============================================================================

def _tracker_opts_to_julia(jl, opts: JuliaTrackerOptions):
    """Convert Python tracker options to a Julia HomotopyContinuation.TrackerOptions object."""
    params = str(opts.parameters).strip().lower()
    valid_params = {"default", "conservative", "fast"}
    if params not in valid_params:
        raise ValueError(f"Invalid tracker parameters '{opts.parameters}'. Must be one of {sorted(valid_params)}.")
    HC = jl.HomotopyContinuation
    return HC.TrackerOptions(
        max_steps=int(opts.max_steps),
        max_step_size=float(opts.max_step_size),
        max_initial_step_size=float(opts.max_initial_step_size),
        min_step_size=float(opts.min_step_size),
        min_rel_step_size=float(opts.min_rel_step_size),
        extended_precision=bool(opts.extended_precision),
        parameters=jl.Symbol(params),
    )


def _ensure_julia_hc_initialized(jl) -> None:
    """
    Idempotent Julia-side initialization.

    This avoids re-running `using HomotopyContinuation` / function definitions on every call,
    which is slow and makes stack traces noisier.
    """

    # NOTE:
    #   We intentionally split `using ...` and `@polyvar ...` across separate eval calls.
    #   Julia resolves macros at parse time, so putting `using DynamicPolynomials` and
    #   `@polyvar` in the same `eval` string can raise:
    #     UndefVarError: `@polyvar` not defined in `Main`
    #   (even though it would work interactively).
    if bool(jl.seval("isdefined(Main, :__HC_PY_INIT__)")):
        return
    jl.seval("using HomotopyContinuation, DynamicPolynomials")
    jl.seval("DynamicPolynomials.@polyvar x t")
    # Note: We'll create coefficient parameters dynamically in __py_hc_track_batch
    jl.seval(_JULIA_HC_DEFINE_SNIPPET)
    jl.seval(_JULIA_HC_DEFINE_PIECEWISE_SNIPPET)
    jl.seval(_JULIA_HC_DEFINE_BEZIER_SNIPPET)
    jl.seval(_JULIA_HC_DEFINE_TOTAL_DEGREE_SNIPPET)
    
    # Precompile the solve function with a dummy System to avoid JIT compilation on each call
    # This forces Julia to compile solve(...) for the specific System type we use
    # NOTE: This may not fully solve the problem if solve(...) internally does value-dependent type inference
    jl.seval(r"""
    if !isdefined(Main, :__HC_PY_PRECOMPILED__)
        # Create a dummy System with the same type structure we'll use
        # Use a higher degree to cover more cases
        dummy_degree = 10
        dummy_q = ComplexF64[1.0 + 0.0im; zeros(ComplexF64, dummy_degree)]
        dummy_p = ComplexF64[1.0 + 0.0im; zeros(ComplexF64, dummy_degree)]
        dummy_Q = [__py_hc_poly_from_coeffs_desc(dummy_q, x)]
        dummy_P = [__py_hc_poly_from_coeffs_desc(dummy_p, x)]
        dummy_H = (1 - t) .* dummy_Q .+ t .* dummy_P
        dummy_Hsys = System(dummy_H; variables=[x], parameters=[t])
        dummy_starts = Vector{ComplexF64}[[0.0 + 0.0im] for _ in 1:dummy_degree]
        dummy_opts = HomotopyContinuation.TrackerOptions(
            max_steps=50000,
            max_step_size=0.05,
            max_initial_step_size=0.05,
            min_step_size=1e-12,
            min_rel_step_size=1e-12,
            extended_precision=false,
        )
        # Force compilation by calling solve with the dummy System
        # Call it multiple times to ensure all code paths are compiled
        try
            for _ in 1:3
                solve(dummy_Hsys, dummy_starts;
                    start_parameters=[0.0],
                    target_parameters=[1.0],
                    tracker_options=dummy_opts,
                )
            end
        catch
            # Ignore errors in precompilation
        end
        const __HC_PY_PRECOMPILED__ = true
    end
    """)
    
    jl.seval("const __HC_PY_INIT__ = true")


# =============================================================================
# Public API: piecewise-linear coefficient path
# =============================================================================

def track_piecewise_linear_julia(
    jl,
    P_ri,
    *,
    tracker_opts: JuliaTrackerOptions = JuliaTrackerOptions(),
    tracker_opts_per_segment: list[JuliaTrackerOptions] | None = None,
    suppress_julia_output: bool = True,
    start_solutions: list[list[complex]] | None = None,
    compute_newton_iters: bool = True,
    debug_type: bool = False,
) -> JuliaRunResult:
    """
    Track all roots along a piecewise-linear coefficient path P_ri using Julia HC.

    Args:
        jl: juliacall.Main
        P_ri: control points (K+1, degree, 2) in (Re,Im)
        tracker_opts: tracker options (fixed for fair comparisons)

    Returns:
        dict with keys:
          - ok: bool  (True iff all paths succeed on all segments)
          - total_steps: int (sum of steps over all returned result objects and all segments)
          - total_accepted_steps: int
          - total_rejected_steps: int
          - wall_time_sec: float
          - detail: list[dict] per segment
    """

    P_ri = np.asarray(P_ri)
    if P_ri.ndim != 3 or P_ri.shape[-1] != 2:
        raise ValueError("P_ri must have shape (K+1, degree, 2).")

    K = int(P_ri.shape[0] - 1)
    degree = int(P_ri.shape[1])
    if degree < 2:
        raise ValueError("degree must be >= 2.")

    _ensure_julia_hc_initialized(jl)

    if tracker_opts_per_segment is not None:
        if len(tracker_opts_per_segment) != K:
            raise ValueError(f"tracker_opts_per_segment must have length K={K}, got {len(tracker_opts_per_segment)}.")

    # Start solutions:
    # - Default: compute roots of first polynomial in Python and pass to Julia.
    # - Optional override: accept externally supplied starts (and validate by substitution).
    p0_tail = _p_ri_to_tail_complex(P_ri[0])
    if start_solutions is None:
        roots0 = _poly_roots_monic_tail_coeffs(p0_tail)
        starts = [[complex(r)] for r in roots0.tolist()]  # [[z1],[z2],...]
    else:
        _validate_start_solutions_for_monic_tail(p0_tail, start_solutions)
        starts = start_solutions

    t_start = time.perf_counter()
    # Build coefficients for all control points once.
    # Explicitly convert to ComplexF64 to ensure type stability
    coeffs_list: list[list[complex]] = []
    for i in range(K + 1):
        tail = _p_ri_to_tail_complex(P_ri[i])
        # Convert to ComplexF64 explicitly for type stability
        coeffs_f64 = [complex(1.0 + 0.0j)] + [complex(complex(x).real, complex(x).imag) for x in tail.tolist()]
        coeffs_list.append(coeffs_f64)

    # Tracker options (fixed default, and optional per-segment override list).
    default_opts_jl = _tracker_opts_to_julia(jl, tracker_opts)
    opts_list_jl = None
    if tracker_opts_per_segment is not None:
        opts_list_jl = [_tracker_opts_to_julia(jl, o) for o in tracker_opts_per_segment]

    # Explicitly convert starts to Vector{Vector{ComplexF64}} for type stability
    starts_f64 = [[complex(s[0].real, s[0].imag)] for s in starts]
    
    # Convert Python lists to Julia Vector{ComplexF64} explicitly
    # This ensures type stability and avoids JIT compilation on each call
    coeffs_list_jl = jl.Vector[jl.Vector[jl.ComplexF64]](coeffs_list)
    starts_f64_jl = jl.Vector[jl.Vector[jl.ComplexF64]](starts_f64)
    
    out = jl.__py_hc_track_piecewise(
        coeffs_list_jl,
        starts_f64_jl,
        opts_list_jl,
        default_opts_jl,
        bool(suppress_julia_output),
        bool(compute_newton_iters),
        bool(debug_type),
    )

    # Use Julia-side solve wall time only (consistent with `track_bezier_curve_julia`).
    # This intentionally excludes Python↔Julia overhead and the Newton-iter debug-log pass.
    wall = float(getattr(out, "solve_wall_time_sec", time.perf_counter() - t_start))
    sys_build_time = float(getattr(out, "sys_build_time_sec", 0.0))
    total_newton_iters = int(getattr(out, "total_newton_iters", 0))
    details: list[dict] = []
    # Convert Julia NamedTuples into plain Python dicts with stable string return codes.
    for d in list(out.detail) if hasattr(out, "detail") else []:
        seg = int(getattr(d, "segment"))
        # Depending on juliacall versions, return codes may stringify as ":success" or "success".
        rcs = [str(x) for x in list(getattr(d, "return_codes"))]
        opts = tracker_opts_per_segment[seg] if tracker_opts_per_segment is not None else tracker_opts
        details.append(
            {
                "segment": seg,
                "nresults": int(getattr(d, "nresults")),
                "nsolutions": int(getattr(d, "nsolutions")),
                "n_success": int(getattr(d, "n_success")),
                "return_codes": rcs,
                "tracker_opts": {
                    "max_steps": int(opts.max_steps),
                    "max_step_size": float(opts.max_step_size),
                    "max_initial_step_size": float(opts.max_initial_step_size),
                    "min_step_size": float(opts.min_step_size),
                    "min_rel_step_size": float(opts.min_rel_step_size),
                    "extended_precision": bool(opts.extended_precision),
                    "parameters": str(opts.parameters),
                },
                "sum_steps": int(getattr(d, "sum_steps")),
                "sum_accepted": int(getattr(d, "sum_accepted")),
                "sum_rejected": int(getattr(d, "sum_rejected")),
                "solve_wall_time_sec": float(getattr(d, "solve_wall_time_sec", 0.0)),
                "sys_build_time_sec": float(getattr(d, "sys_build_time_sec", 0.0)),
                "sum_newton_iters": int(getattr(d, "sum_newton_iters", 0)),
            }
        )

    return {
        "ok": bool(getattr(out, "ok")),
        "total_steps": int(getattr(out, "total_steps")),
        "total_accepted_steps": int(getattr(out, "total_accepted_steps")),
        "total_rejected_steps": int(getattr(out, "total_rejected_steps")),
        "total_newton_iters": total_newton_iters,
        "wall_time_sec": wall,
        "sys_build_time_sec": sys_build_time,
        "detail": details,
    }


def track_piecewise_linear_julia_batch(
    jl,
    pairs_list: list[tuple[np.ndarray, np.ndarray]],
    starts_list: list[list[list[complex]]],
    *,
    tracker_opts: JuliaTrackerOptions = JuliaTrackerOptions(),
    suppress_julia_output: bool = True,
    compute_newton_iters: bool = True,
) -> list[JuliaRunResult]:
    """
    Batch version: process multiple (p_start, p_target) pairs in a single Julia call.
    
    This avoids Python-Julia overhead and allows System construction to be optimized
    by processing all pairs in a single Julia-side loop.
    
    Args:
        jl: juliacall.Main
        pairs_list: list of (p_start, p_target) tuples, each is (degree,) complex array
        starts_list: list of start solutions, each is list[list[complex]]
        tracker_opts: tracker options
    
    Returns:
        list of JuliaRunResult-compatible dicts, one per pair
    """
    _ensure_julia_hc_initialized(jl)
    
    # Convert pairs to coefficient format
    coeffs_pairs: list[tuple[list[complex], list[complex]]] = []
    for p0, p1 in pairs_list:
        # Convert to tail coefficients (monic polynomial)
        # p0, p1 are already complex arrays, convert to (Re, Im) format first
        p0_ri = np.stack([np.asarray(p0).real, np.asarray(p0).imag], axis=-1)
        p1_ri = np.stack([np.asarray(p1).real, np.asarray(p1).imag], axis=-1)
        tail0 = _p_ri_to_tail_complex(p0_ri)
        tail1 = _p_ri_to_tail_complex(p1_ri)
        coeffs0 = [complex(1.0 + 0.0j)] + [complex(x) for x in tail0.tolist()]
        coeffs1 = [complex(1.0 + 0.0j)] + [complex(x) for x in tail1.tolist()]
        coeffs_pairs.append((coeffs0, coeffs1))
    
    # Convert to Julia types
    opts_jl = _tracker_opts_to_julia(jl, tracker_opts)
    pairs_jl = jl.Vector[jl.Tuple[jl.Vector[jl.ComplexF64], jl.Vector[jl.ComplexF64]]](coeffs_pairs)
    starts_jl = jl.Vector[jl.Vector[jl.Vector[jl.ComplexF64]]](starts_list)
    
    # Call Julia batch function
    t_start = time.perf_counter()
    batch_results = jl.__py_hc_track_batch(
        pairs_jl,
        starts_jl,
        opts_jl,
        bool(suppress_julia_output),
        bool(compute_newton_iters),
    )
    t_end = time.perf_counter()
    
    # Convert results to Python format
    results: list[JuliaRunResult] = []
    for i, res in enumerate(list(batch_results)):
        # Calculate per-sample time (total time / n_samples)
        avg_time = (t_end - t_start) / len(batch_results) if len(batch_results) > 0 else 0.0
        
        results.append({
            "ok": bool(getattr(res, "ok", False)),
            "total_steps": int(getattr(res, "total_steps", 0)),
            "total_accepted_steps": int(getattr(res, "total_accepted", 0)),
            "total_rejected_steps": int(getattr(res, "total_rejected", 0)),
            "total_newton_iters": int(getattr(res, "sum_newton_iters", 0)),
            "wall_time_sec": avg_time,  # Average time per sample
            "sys_build_time_sec": 0.0,  # Not measured separately in batch mode
            "detail": [],  # Simplified for batch mode
        })
    
    return results


# =============================================================================
# Public API: Bezier coefficient curve
# =============================================================================

def track_bezier_curve_julia(
    jl,
    P_ctrl_ri,
    *,
    tracker_opts: JuliaTrackerOptions = JuliaTrackerOptions(),
    suppress_julia_output: bool = True,
    start_solutions: list[list[complex]] | None = None,
) -> JuliaRunResult:
    """
    Track all roots along a Bezier coefficient curve using Julia HC (single solve).

    Args:
        jl: juliacall.Main
        P_ctrl_ri: Bezier control points (d+1, degree, 2) in (Re,Im)
        tracker_opts: tracker options (fixed for fair comparisons)

    Returns:
        JuliaRunResult-compatible dict with a single segment entry in `detail`.
    """
    P_ctrl_ri = np.asarray(P_ctrl_ri)
    if P_ctrl_ri.ndim != 3 or P_ctrl_ri.shape[-1] != 2:
        raise ValueError("P_ctrl_ri must have shape (d+1, degree, 2).")

    d_plus_1 = int(P_ctrl_ri.shape[0])
    degree = int(P_ctrl_ri.shape[1])
    if d_plus_1 < 2:
        raise ValueError("P_ctrl_ri must have at least two control points (d+1 >= 2).")
    if degree < 2:
        raise ValueError("degree must be >= 2.")

    _ensure_julia_hc_initialized(jl)

    # Start solutions:
    # - Default: compute roots for the polynomial at t=0 (first control point).
    # - Optional override: accept externally supplied starts (and validate by substitution).
    p0_tail = _p_ri_to_tail_complex(P_ctrl_ri[0])
    if start_solutions is None:
        roots0 = _poly_roots_monic_tail_coeffs(p0_tail)
        starts = [[complex(r)] for r in roots0.tolist()]  # [[z1],[z2],...]
    else:
        _validate_start_solutions_for_monic_tail(p0_tail, start_solutions)
        starts = start_solutions

    # Build tail coefficient control points per coefficient index.
    # tail_ctrls_by_coeff[j][i] = j-th tail coefficient at i-th Bezier control point.
    tail_ctrls_by_coeff: list[list[complex]] = []
    for j in range(degree):
        ctrls_j: list[complex] = []
        for i in range(d_plus_1):
            tail_i = _p_ri_to_tail_complex(P_ctrl_ri[i])
            ctrls_j.append(complex(tail_i[j]))
        tail_ctrls_by_coeff.append(ctrls_j)

    opts_jl = _tracker_opts_to_julia(jl, tracker_opts)

    out = jl.__py_hc_track_bezier(
        tail_ctrls_by_coeff,
        starts,
        opts_jl,
        bool(suppress_julia_output),
    )
    # Record ONLY the Julia-side solve(...) wall time (exclude Python↔Julia overhead).
    wall = float(getattr(out, "solve_wall_time_sec"))
    sum_newton_iters = int(getattr(out, "sum_newton_iters"))

    rcs = [str(x) for x in list(getattr(out, "return_codes"))]
    n_success = int(getattr(out, "n_success"))
    ok = bool(n_success == degree)

    detail: list[JuliaSegmentDetail] = [
        {
            "segment": 0,
            "nresults": int(getattr(out, "nresults")),
            "nsolutions": int(getattr(out, "nsolutions")),
            "n_success": n_success,
            "return_codes": rcs,
            "solve_wall_time_sec": wall,
            "tracker_opts": {
                "max_steps": int(tracker_opts.max_steps),
                "max_step_size": float(tracker_opts.max_step_size),
                "max_initial_step_size": float(tracker_opts.max_initial_step_size),
                "min_step_size": float(tracker_opts.min_step_size),
                "min_rel_step_size": float(tracker_opts.min_rel_step_size),
                "extended_precision": bool(tracker_opts.extended_precision),
                "parameters": str(tracker_opts.parameters),
            },
            "sum_steps": int(getattr(out, "sum_steps")),
            "sum_accepted": int(getattr(out, "sum_accepted")),
            "sum_rejected": int(getattr(out, "sum_rejected")),
            "sum_newton_iters": sum_newton_iters,
        }
    ]

    return {
        "ok": ok,
        "total_steps": int(getattr(out, "sum_steps")),
        "total_accepted_steps": int(getattr(out, "sum_accepted")),
        "total_rejected_steps": int(getattr(out, "sum_rejected")),
        "total_newton_iters": sum_newton_iters,
        "wall_time_sec": wall,
        "detail": detail,
    }


def track_total_degree_julia(
    jl,
    p_target_ri,
    *,
    tracker_opts: JuliaTrackerOptions = JuliaTrackerOptions(),
    suppress_julia_output: bool = True,
) -> JuliaRunResult:
    """
    Baseline: solve target polynomial using HomotopyContinuation.jl with total-degree start system.

    Args:
        jl: juliacall.Main
        p_target_ri: target tail coefficients (degree, 2) in (Re,Im) for monic poly.
        tracker_opts: tracker options

    Returns:
        JuliaRunResult-like dict with a single segment entry in `detail`.
    """
    p_target_ri = np.asarray(p_target_ri)
    if p_target_ri.ndim != 2 or p_target_ri.shape[-1] != 2:
        raise ValueError("p_target_ri must have shape (degree, 2).")
    degree = int(p_target_ri.shape[0])
    if degree < 2:
        raise ValueError("degree must be >= 2.")

    _ensure_julia_hc_initialized(jl)

    tail = _p_ri_to_tail_complex(p_target_ri)
    coeffs = [1.0 + 0.0j] + [complex(x) for x in tail.tolist()]
    opts_jl = _tracker_opts_to_julia(jl, tracker_opts)

    out = jl.__py_hc_solve_total_degree(coeffs, opts_jl, bool(suppress_julia_output))
    wall = float(getattr(out, "solve_wall_time_sec"))
    sum_newton_iters = int(getattr(out, "sum_newton_iters"))

    rcs = [str(x) for x in list(getattr(out, "return_codes"))]
    n_success = int(getattr(out, "n_success"))
    ok = bool(n_success == degree)

    detail: list[dict] = [
        {
            "segment": 0,
            "nresults": int(getattr(out, "nresults")),
            "nsolutions": int(getattr(out, "nsolutions")),
            "n_success": n_success,
            "return_codes": rcs,
            "solve_wall_time_sec": wall,
            "tracker_opts": {
                "max_steps": int(tracker_opts.max_steps),
                "max_step_size": float(tracker_opts.max_step_size),
                "max_initial_step_size": float(tracker_opts.max_initial_step_size),
                "min_step_size": float(tracker_opts.min_step_size),
                "min_rel_step_size": float(tracker_opts.min_rel_step_size),
                "extended_precision": bool(tracker_opts.extended_precision),
                "parameters": str(tracker_opts.parameters),
            },
            "sum_steps": int(getattr(out, "sum_steps")),
            "sum_accepted": int(getattr(out, "sum_accepted")),
            "sum_rejected": int(getattr(out, "sum_rejected")),
            "sum_newton_iters": sum_newton_iters,
        }
    ]

    return {
        "ok": ok,
        "total_steps": int(getattr(out, "sum_steps")),
        "total_accepted_steps": int(getattr(out, "sum_accepted")),
        "total_rejected_steps": int(getattr(out, "sum_rejected")),
        "total_newton_iters": sum_newton_iters,
        "wall_time_sec": wall,
        "detail": detail,
    }
