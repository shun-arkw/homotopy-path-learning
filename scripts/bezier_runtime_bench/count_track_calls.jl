#!/usr/bin/env julia
# Count homotopy evaluation calls during one full track (Bezier). Run in Docker (23VIF).
# Usage: julia -t 1 count_track_calls.jl [degree] [bezier_degree] [seed]
# Then: estimated time share = count * (per-call μs from profile_eval.jl) for each key.

cd(@__DIR__)
include("bezier_univar.jl")

function main()
    degree        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 80
    bezier_degree = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
    seed          = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1

    EVAL_COUNTS[] = Dict{String,Int}()
    ENABLE_EVAL_COUNTS[] = true

    init_bezier_univar(; degree, bezier_degree, seed=0)
    ctrl = build_ctrl_univar(degree, bezier_degree; seed=seed)

    out = track_bezier_paths_univar(degree, bezier_degree, ctrl)

    ENABLE_EVAL_COUNTS[] = false
    counts = EVAL_COUNTS[]

    println("degree=$degree bezier_degree=$bezier_degree seed=$seed  threads=$(Threads.nthreads())")
    println("tracking_time_sec = ", out.tracking_time_sec)
    println()
    println("Call counts during track (all paths):")
    for (k, v) in sort(collect(counts), by=x->x[2], rev=true)
        println("  ", rpad(k, 25), " ", v)
    end
    # From profile_eval (degree=160, bezier_degree=4): eval_coeffs0 ~0.3μs, poly_* ~0.25μs
    # So estimated: eval_coeffs0 * 0.3 + (poly_only + poly_deriv) * 0.25 μs per path, then * n_paths
    println()
    println("Rough share: eval_coeffs0 (Bernstein+sum) is called for every evaluate! and taylor!;")
    println("  if dominate, bottleneck is Bezier coeff evaluation (see profile_eval.jl per-call μs).")
    return nothing
end

main()
