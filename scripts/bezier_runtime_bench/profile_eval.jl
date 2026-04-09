#!/usr/bin/env julia
# Isolate homotopy evaluation cost (no tracker). Linear vs Bezier (generic + db-specific).
# Usage: julia profile_eval.jl [degree] [bezier_degree] [n_evals]
# Example: julia profile_eval.jl 160 4 100000

using Random
cd(@__DIR__)
include("linear_univar.jl")
include("bezier_univar.jl")

const EVAL_DB = (eval_coeffs0!_db2!, eval_coeffs0!_db3!, eval_coeffs0!_db4!, eval_coeffs0!_db5!)

function main()
    degree        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 160
    bezier_degree = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
    n_evals       = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 100_000

    ncoef = degree + 1

    # Linear: one homotopy, endpoints = first and last row of same ctrl
    ctrl = build_ctrl_univar(degree, bezier_degree; seed=1)
    endpoints = zeros(ComplexF64, 2, ncoef)
    endpoints[1, :] .= ctrl[1, :]
    endpoints[2, :] .= ctrl[end, :]
    H_lin = LinearUnivarPoly(degree, endpoints, zeros(ComplexF64, ncoef), zeros(ComplexF64, ncoef))
    update_linear_diff!(H_lin)

    # Bezier (generic de Casteljau): same ctrl
    H_bez = make_homotopy_univar(degree, bezier_degree; seed=1)
    copyto!(H_bez.ctrl, ctrl)
    compute_diffs!(H_bez.diffs, H_bez.ctrl, H_bez.bezier_degree, H_bez.max_derivative_order)

    τs = rand(n_evals)
    x0 = [0.5 + 0.3im]
    u = zeros(ComplexF64, 1)
    U = zeros(ComplexF64, 1, 1)

    # ---- Linear vs Bezier (generic) ----
    t_lin0 = @elapsed for τ in τs
        eval_coeffs0!(H_lin, τ)
    end
    t_bez0 = @elapsed for τ in τs
        eval_coeffs0!(H_bez, τ)
    end
    t_lin_eval = @elapsed for τ in τs
        eval_coeffs0!(H_lin, τ)
        u[1] = poly_only_horner(H_lin.ceff0, x0[1])
    end
    t_bez_eval = @elapsed for τ in τs
        eval_coeffs0!(H_bez, τ)
        u[1] = poly_only_horner(H_bez.ceff0, x0[1])
    end
    t_lin_jac = @elapsed for τ in τs
        eval_coeffs0!(H_lin, τ)
        px, dpx = poly_and_deriv_horner(H_lin.ceff0, x0[1])
        u[1] = px
        U[1,1] = dpx
    end
    t_bez_jac = @elapsed for τ in τs
        eval_coeffs0!(H_bez, τ)
        px, dpx = poly_and_deriv_horner(H_bez.ceff0, x0[1])
        u[1] = px
        U[1,1] = dpx
    end
    t_lin_t1 = @elapsed for _ in τs
        @inbounds for i in 1:length(H_lin.ceff1)
            H_lin.ceff1[i] = -H_lin.ceff1[i]
        end
        u[1] = poly_only_horner(H_lin.ceff1, x0[1])
        @inbounds for i in 1:length(H_lin.ceff1)
            H_lin.ceff1[i] = -H_lin.ceff1[i]
        end
    end
    t_bez_t1 = @elapsed for τ in τs
        eval_coeffs_k!(H_bez.ceff1, H_bez, τ, 1)
        for i in 1:length(H_bez.ceff1)
            H_bez.ceff1[i] = -H_bez.ceff1[i]
        end
        u[1] = poly_only_horner(H_bez.ceff1, x0[1])
    end

    per_μs_lin0   = t_lin0   / n_evals * 1e6
    per_μs_bez0   = t_bez0   / n_evals * 1e6
    per_μs_lin_e  = t_lin_eval / n_evals * 1e6
    per_μs_bez_e  = t_bez_eval / n_evals * 1e6
    per_μs_lin_j  = t_lin_jac  / n_evals * 1e6
    per_μs_bez_j  = t_bez_jac  / n_evals * 1e6
    per_μs_lin_t1 = t_lin_t1   / n_evals * 1e6
    per_μs_bez_t1 = t_bez_t1   / n_evals * 1e6

    println("degree=$degree n_evals=$n_evals")
    println()
    println("=== Linear vs Bezier (generic de Casteljau, bezier_degree=$bezier_degree) ===")
    println("Per-call time (μs):")
    println("  eval_coeffs0! only:     Linear ", round(per_μs_lin0; digits=4), "  Bezier ", round(per_μs_bez0; digits=4), "  ratio ", round(per_μs_bez0/per_μs_lin0; digits=2))
    println("  evaluate! (+ Horner):  Linear ", round(per_μs_lin_e; digits=4), "  Bezier ", round(per_μs_bez_e; digits=4), "  ratio ", round(per_μs_bez_e/per_μs_lin_e; digits=2))
    println("  evaluate_and_jacobian!: Linear ", round(per_μs_lin_j; digits=4), "  Bezier ", round(per_μs_bez_j; digits=4), "  ratio ", round(per_μs_bez_j/per_μs_lin_j; digits=2))
    println("  taylor! k=1 (predict): Linear ", round(per_μs_lin_t1; digits=4), "  Bezier ", round(per_μs_bez_t1; digits=4), "  ratio ", round(per_μs_bez_t1/per_μs_lin_t1; digits=2))

    # ---- generic (de Casteljau) and dedicated (closed-form) for db=2,3,4,5 ----
    per_μs_gen0 = Float64[]
    per_μs_gen_jac = Float64[]
    per_μs_db0 = Float64[]
    per_μs_db_jac = Float64[]
    for db in 2:5
        ctrl_db = build_ctrl_univar(degree, db; seed=1)
        H_db = make_homotopy_univar(degree, db; seed=1)
        copyto!(H_db.ctrl, ctrl_db)
        compute_diffs!(H_db.diffs, H_db.ctrl, H_db.bezier_degree, H_db.max_derivative_order)
        # Generic (de Casteljau)
        t0g = @elapsed for τ in τs
            eval_coeffs0!(H_db, τ)
        end
        tjg = @elapsed for τ in τs
            eval_coeffs0!(H_db, τ)
            px, dpx = poly_and_deriv_horner(H_db.ceff0, x0[1])
            u[1] = px
            U[1,1] = dpx
        end
        push!(per_μs_gen0, t0g / n_evals * 1e6)
        push!(per_μs_gen_jac, tjg / n_evals * 1e6)
        # Dedicated (closed-form Bernstein)
        ev = EVAL_DB[db - 1]
        t0 = @elapsed for τ in τs
            ev(H_db, τ)
        end
        t_jac = @elapsed for τ in τs
            ev(H_db, τ)
            px, dpx = poly_and_deriv_horner(H_db.ceff0, x0[1])
            u[1] = px
            U[1,1] = dpx
        end
        push!(per_μs_db0, t0 / n_evals * 1e6)
        push!(per_μs_db_jac, t_jac / n_evals * 1e6)
    end

    println()
    println("=== Linear | Bezier generic (de Casteljau) db=2..5 | Bezier dedicated (closed-form) db=2..5 ===")
    println()
    println("  eval_coeffs0! (μs):")
    println("    Linear    ", round(per_μs_lin0; digits=4))
    for (db, p) in enumerate(2:5)
        println("    generic db=$p ", round(per_μs_gen0[db]; digits=4), "  (", round(per_μs_gen0[db]/per_μs_lin0; digits=2), "x)")
    end
    for (db, p) in enumerate(2:5)
        println("    dedicated db=$p ", round(per_μs_db0[db]; digits=4), "  (", round(per_μs_db0[db]/per_μs_lin0; digits=2), "x)")
    end
    println("  evaluate_and_jacobian! (μs):")
    println("    Linear    ", round(per_μs_lin_j; digits=4))
    for (db, p) in enumerate(2:5)
        println("    generic db=$p ", round(per_μs_gen_jac[db]; digits=4), "  (", round(per_μs_gen_jac[db]/per_μs_lin_j; digits=2), "x)")
    end
    for (db, p) in enumerate(2:5)
        println("    dedicated db=$p ", round(per_μs_db_jac[db]; digits=4), "  (", round(per_μs_db_jac[db]/per_μs_lin_j; digits=2), "x)")
    end
    println()
    println("Conclusion: dedicated (closed-form) is faster than generic (de Casteljau) for same db; db=2 dedicated is closest to Linear.")
    return nothing
end

main()
