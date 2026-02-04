# test_track_only_bezier_univar.jl
include("bezier_univar.jl")

function test_track_only_bezier_univar()
    n = 5
    m = 3
    seed = 0

    # Build homotopy and start solutions
    H = make_homotopy_univar(n, m; seed=seed)
    starts = total_degree_start_solutions_univar(n)

    # Create TrackerOptions
    opts = HomotopyContinuation.TrackerOptions(
        max_steps=50_000,
        max_step_size=0.05,
        max_initial_step_size=0.05,
        min_step_size=1e-12,
        min_rel_step_size=1e-12,
        extended_precision=false,
    )

    # Create Tracker
    tracker = Tracker(H; options=opts)

    # Run track only for verification
    tot = 0
    tot_accepted = 0
    tot_rejected = 0
    ok = true
    sum_newton_iters = 0
    path_results = Vector{Any}()
    solutions = Vector{Vector{ComplexF64}}()
    return_codes = Vector{Symbol}()
    path_steps = Vector{Int}()
    path_accepted_steps = Vector{Int}()
    path_rejected_steps = Vector{Int}()

    for (i, s) in enumerate(starts)
        # Run track (capture log with debug=true; Pipe + async reader to avoid buffer deadlock)
        pipe = Pipe()
        buf = IOBuffer()
        reader = @async begin
            while true
                chunk = read(pipe, 8192)
                isempty(chunk) && break
                write(buf, chunk)
            end
        end
        pr = redirect_stdout(pipe) do
            redirect_stderr(pipe) do
                track(tracker, s; debug=true)
            end
        end
        close(pipe.in)
        wait(reader)

        # Get info from TrackerResult
        path_steps_i = steps(pr)
        path_accepted_i = accepted_steps(pr)
        path_rejected_i = rejected_steps(pr)

        tot += path_steps_i
        tot_accepted += path_accepted_i
        tot_rejected += path_rejected_i
        ok &= (pr.return_code == :success)

        push!(path_results, pr)
        push!(solutions, pr.solution)
        push!(return_codes, pr.return_code)
        push!(path_steps, path_steps_i)
        push!(path_accepted_steps, path_accepted_i)
        push!(path_rejected_steps, path_rejected_i)

        # Inspect PathResult once (first path) to check for Newton/corrector stats
        if i == 1
            println("=== PathResult inspection ===")
            println("typeof(pr): ", typeof(pr))
            println("propertynames(pr): ", propertynames(pr))
            T = typeof(pr)
            if isconcretetype(T)
                println("fieldnames(typeof(pr)): ", fieldnames(T))
            end
            for name in (:newton_iters, :corrector_iters, :iterations, :n_iters, :total_iters, :path_result)
                if hasproperty(pr, name)
                    println("  pr.", name, " = ", getproperty(pr, name))
                end
            end
            println("=== end inspection ===")
        end

        # Parse Newton iteration count from log
        log = String(take!(buf))
        for mt in eachmatch(r"iters\s*→\s*(\d+)", log)
            sum_newton_iters += parse(Int, mt.captures[1])
        end
    end

    println("=== Summary ===")
    println("ok: ", ok)
    println("total_steps: ", tot)
    println("total_accepted_steps: ", tot_accepted)
    println("total_rejected_steps: ", tot_rejected)
    println("sum_newton_iters: ", sum_newton_iters)
    println()
    println("=== Per-path details ===")
    for i in 1:length(starts)
        println("Path $i:")
        println("  return_code: ", return_codes[i])
        println("  steps: ", path_steps[i])
        println("  accepted_steps: ", path_accepted_steps[i])
        println("  rejected_steps: ", path_rejected_steps[i])
        println("  solution: ", solutions[i])
        # TrackerResult may not have residual/accuracy; commented out
        # println("  residual: ", residuals[i])
        # println("  accuracy: ", accuracies[i])
    end
    
    return (ok=ok, total_steps=tot, total_accepted_steps=tot_accepted, 
            total_rejected_steps=tot_rejected, sum_newton_iters=sum_newton_iters,
            solutions=solutions, return_codes=return_codes, path_steps=path_steps,
            path_accepted_steps=path_accepted_steps, path_rejected_steps=path_rejected_steps)
end

test_track_only_bezier_univar()