#!/usr/bin/env julia
# Experimental: Profile.track on one extracted case (same layout as analyze_problem_instance.jl).
# Does not modify analyze_problem_instance.jl; use this to explore stack hotspots before integrating.
#
# Usage:
#   julia -t 1 profile_extracted_case.jl /path/to/case.jl [n_profile] [warmup]
#
# case.jl must define DEGREE, BEZIER_DEGREE, START_COEFFS, TARGET_COEFFS, GAMMA, BEZIER_CTRL
# (from extract_problem_instance.py).

using Profile

cd(@__DIR__)
include("linear_univar.jl")
include("bezier_univar.jl")

function main()
    if length(ARGS) < 1
        println("Usage: julia -t 1 profile_extracted_case.jl /path/to/case.jl [n_profile] [warmup]")
        return
    end
    case_file = ARGS[1]
    n_profile = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
    warmup = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2

    include(case_file)
    degree = DEGREE
    bezier_degree = BEZIER_DEGREE
    start_path = GAMMA .* START_COEFFS
    target = TARGET_COEFFS
    ctrl = BEZIER_CTRL

    println("Profile extracted case: degree=$degree bezier_degree=$bezier_degree n_profile=$n_profile warmup=$warmup threads=$(Threads.nthreads())")
    if Threads.nthreads() != 1
        println("  (recommended: julia -t 1 for clearer profiles)")
    end

    init_linear_univar(; degree, seed=0)
    init_bezier_univar(; degree, bezier_degree, seed=0)

    try
        for _ in 1:warmup
            track_linear_paths_univar(degree, start_path, target; compute_newton_iters=false)
            track_bezier_paths_univar(degree, bezier_degree, ctrl; compute_newton_iters=false)
        end

        println()
        println("=== Linear: Profile.@profile × $n_profile (compute_newton_iters=false) ===")
        Profile.clear()
        Profile.@profile for _ in 1:n_profile
            track_linear_paths_univar(degree, start_path, target; compute_newton_iters=false)
        end
        data_lin = Profile.fetch()
        println("Total snapshots: ", length(data_lin))
        Profile.print(mincount=50, maxdepth=20, C=true)
        Profile.clear()

        println()
        println("=== Bezier: Profile.@profile × $n_profile (compute_newton_iters=false) ===")
        Profile.@profile for _ in 1:n_profile
            track_bezier_paths_univar(degree, bezier_degree, ctrl; compute_newton_iters=false)
        end
        data_bez = Profile.fetch()
        println("Total snapshots: ", length(data_bez))
        Profile.print(mincount=50, maxdepth=20, C=true)

        ratio = length(data_lin) > 0 ? length(data_bez) / length(data_lin) : NaN
        println()
        println("=== Snapshot count ratio (Bezier/Linear, statistical; not wall time) ===")
        println("  ", round(ratio; digits=4))
        println()
        println("Tip: ProfileView.view() after fetch() for flame graphs (requires ProfileView.jl).")
    catch e
        if isa(e, TaskFailedException) && occursin("index [0]", string(e))
            println("ERROR: HomotopyContinuation tx indexing issue in this environment.")
            println("Run inside Docker with 23VIF, e.g.:")
            println("  julia -t 1 profile_extracted_case.jl $case_file $n_profile $warmup")
        else
            rethrow(e)
        end
    end
    return nothing
end

main()
