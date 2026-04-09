#!/usr/bin/env julia
# Compare Linear vs Bezier: time per Newton iteration (same ctrl, n_runs each).
# Usage: julia -t 1 bench_time_per_newton.jl [degree] [bezier_degree] [seed] [n_runs]
# Example: julia -t 1 bench_time_per_newton.jl 80 4 1 20

cd(@__DIR__)
include("linear_univar.jl")
include("bezier_univar.jl")

function main()
    degree        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 80
    bezier_degree = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
    seed          = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    n_runs        = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 20

    println("Time per Newton iteration (same ctrl): degree=$degree bezier_degree=$bezier_degree seed=$seed n_runs=$n_runs  threads=$(Threads.nthreads())")
    if Threads.nthreads() != 1
        println("  (for comparison use: julia -t 1 bench_time_per_newton.jl ...)")
    end

    init_linear_univar(; degree, seed=0)
    init_bezier_univar(; degree, bezier_degree, seed=0)
    ctrl = build_ctrl_univar(degree, bezier_degree; seed=seed)
    start_coeffs = ctrl[1, :]
    target_coeffs = ctrl[end, :]

    # --- Linear ---
    LINEAR_EVAL_COUNTS[] = Dict{String,Int}()
    LINEAR_ENABLE_EVAL_COUNTS[] = true
    total_time_lin = 0.0
    for _ in 1:n_runs
        out = track_linear_paths_univar(degree, start_coeffs, target_coeffs)
        total_time_lin += out.tracking_time_sec
    end
    n_newton_lin = get(LINEAR_EVAL_COUNTS[], "evaluate_and_jacobian", 0)
    LINEAR_ENABLE_EVAL_COUNTS[] = false

    # --- Bezier ---
    EVAL_COUNTS[] = Dict{String,Int}()
    ENABLE_EVAL_COUNTS[] = true
    total_time_bez = 0.0
    for _ in 1:n_runs
        out = track_bezier_paths_univar(degree, bezier_degree, ctrl)
        total_time_bez += out.tracking_time_sec
    end
    n_newton_bez = get(EVAL_COUNTS[], "evaluate_and_jacobian", 0)
    ENABLE_EVAL_COUNTS[] = false

    # --- Report ---
    μs_per_sec = 1e6
    time_per_newton_lin = n_newton_lin > 0 ? (total_time_lin / n_newton_lin) * μs_per_sec : NaN
    time_per_newton_bez = n_newton_bez > 0 ? (total_time_bez / n_newton_bez) * μs_per_sec : NaN
    ratio = (n_newton_lin > 0 && time_per_newton_lin > 0) ? (time_per_newton_bez / time_per_newton_lin) : NaN

    println()
    println("## Linear (same endpoints as Bezier)")
    println("  total tracking time (s): ", round(total_time_lin; digits=6))
    println("  total evaluate_and_jacobian! calls: ", n_newton_lin)
    println("  time per Newton iteration (μs): ", round(time_per_newton_lin; digits=4))
    println()
    println("## Bezier")
    println("  total tracking time (s): ", round(total_time_bez; digits=6))
    println("  total evaluate_and_jacobian! calls: ", n_newton_bez)
    println("  time per Newton iteration (μs): ", round(time_per_newton_bez; digits=4))
    println()
    println("  Bezier/Linear time-per-Newton ratio: ", round(ratio; digits=4))
    return nothing
end

main()
