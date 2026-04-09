#!/usr/bin/env julia
# Profile Linear vs Bezier path tracking (same ctrl) to see where extra time is spent.
# Run in Docker (23VIF): julia -t 1 profile_track.jl [degree] [bezier_degree] [seed] [n_profile]
# Example: julia -t 1 profile_track.jl 80 4 1 50

using Profile
cd(@__DIR__)
include("linear_univar.jl")
include("bezier_univar.jl")

function main()
    degree        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 80
    bezier_degree = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
    seed          = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    n_profile     = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 20

    println("Profiling Linear vs Bezier: $n_profile runs each (degree=$degree, bezier_degree=$bezier_degree, seed=$seed)  threads=$(Threads.nthreads())")
    init_linear_univar(; degree, seed=0)
    init_bezier_univar(; degree, bezier_degree, seed=0)
    ctrl = build_ctrl_univar(degree, bezier_degree; seed=seed)
    start_coeffs = ctrl[1, :]
    target_coeffs = ctrl[end, :]

    try
        println("Warmup (compile hot path)...")
        track_linear_paths_univar(degree, start_coeffs, target_coeffs)
        track_bezier_paths_univar(degree, bezier_degree, ctrl)

        # --- Linear ---
        println()
        println("=== Linear: profiling $n_profile runs ===")
        Profile.@profile for _ in 1:n_profile
            track_linear_paths_univar(degree, start_coeffs, target_coeffs)
        end
        data_lin = Profile.fetch()
        total_linear = length(data_lin)
        println("Top functions (mincount=50):")
        Profile.print(mincount=50, maxdepth=20, C=true)
        Profile.clear()

        # --- Bezier ---
        println()
        println("=== Bezier: profiling $n_profile runs ===")
        Profile.@profile for _ in 1:n_profile
            track_bezier_paths_univar(degree, bezier_degree, ctrl)
        end
        data_bez = Profile.fetch()
        total_bez = length(data_bez)
        println("Top functions (mincount=50):")
        Profile.print(mincount=50, maxdepth=20, C=true)

        # --- Comparison ---
        ratio = total_linear > 0 ? total_bez / total_linear : NaN
        println()
        println("=== Comparison (same ctrl, $n_profile runs each) ===")
        println("  Total snapshots:  Linear ", total_linear, ",  Bezier ", total_bez)
        println("  Bezier/Linear ratio: ", round(ratio; digits=4))
    catch e
        if isa(e, TaskFailedException) && occursin("index [0]", string(e))
            println("ERROR: This environment uses HomotopyContinuation I1faM (tx[1][0] fails).")
            println("Run this script inside Docker where 23VIF is used:")
            println("  julia -t 1 profile_track.jl $degree $bezier_degree $seed $n_profile")
            println("Alternatively run count_track_calls.jl in Docker to see call counts.")
        else
            rethrow(e)
        end
    end
    return nothing
end

main()
