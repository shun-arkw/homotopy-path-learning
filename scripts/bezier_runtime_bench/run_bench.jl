#!/usr/bin/env julia
# Run from scripts/bezier_runtime_bench. No dependency on scripts/bezier_hc_ppo.
# Same sample: Linear (start->target) vs Bezier (same endpoints + middle controls).
# For comparison use 1 thread: julia -t 1 run_bench.jl [degree] [bezier_degree] [n_runs]
# Example: julia -t 1 run_bench.jl 160 4 20

cd(@__DIR__)
include("linear_univar.jl")
include("bezier_univar.jl")

function main()
    degree        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 160
    bezier_degree = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
    n_runs        = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20

    println("degree=$degree bezier_degree=$bezier_degree n_runs=$n_runs  threads=$(Threads.nthreads())")
    if Threads.nthreads() != 1
        println("  (for comparison use: julia -t 1 run_bench.jl ...)")
    end

    init_linear_univar(; degree, seed=0)
    init_bezier_univar(; degree, bezier_degree, seed=0)

    linear_ms = Float64[]
    bezier_ms = Float64[]
    for i in 1:n_runs
        ctrl = build_ctrl_univar(degree, bezier_degree; seed=i)
        start_coeffs = ctrl[1, :]
        target_coeffs = ctrl[end, :]

        out_lin = track_linear_paths_univar(degree, start_coeffs, target_coeffs)
        out_bez = track_bezier_paths_univar(degree, bezier_degree, ctrl)

        push!(linear_ms, out_lin.tracking_time_sec * 1000)
        push!(bezier_ms, out_bez.tracking_time_sec * 1000)
    end

    μ_lin = sum(linear_ms) / length(linear_ms)
    σ_lin = sqrt(sum(x -> (x - μ_lin)^2, linear_ms) / length(linear_ms))
    μ_bez = sum(bezier_ms) / length(bezier_ms)
    σ_bez = sqrt(sum(x -> (x - μ_bez)^2, bezier_ms) / length(bezier_ms))

    println()
    println("## Linear (same endpoints as Bezier)")
    println("  tracking_time mean (ms): ", round(μ_lin; digits=4))
    println("  tracking_time std  (ms): ", round(σ_lin; digits=4))
    println("  tracking_time min  (ms): ", round(minimum(linear_ms); digits=4))
    println("  tracking_time max  (ms): ", round(maximum(linear_ms); digits=4))
    println()
    println("## Bezier")
    println("  tracking_time mean (ms): ", round(μ_bez; digits=4))
    println("  tracking_time std  (ms): ", round(σ_bez; digits=4))
    println("  tracking_time min  (ms): ", round(minimum(bezier_ms); digits=4))
    println("  tracking_time max  (ms): ", round(maximum(bezier_ms); digits=4))
    println()
    println("  Bezier/Linear mean ratio: ", round(μ_bez / μ_lin; digits=4))
    return nothing
end

main()
