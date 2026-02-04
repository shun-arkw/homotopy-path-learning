using HomotopyContinuation
using HomotopyContinuation.ModelKit
using Random

# ============================================================
# 1) Bernstein weights (small degree, fast)
# ============================================================
# Compute weights for degree n at s in [0,1]:
# w[i+1] = binom(n,i) * s^i * (1-s)^(n-i), i=0..n
function bernstein_weights!(w::Vector{Float64}, n::Int, s::Float64, binom::Vector{Float64})
    @assert length(w) == n + 1
    u = 1.0 - s
    # powers (n <= 5 so just do direct)
    sp = Vector{Float64}(undef, n + 1)
    up = Vector{Float64}(undef, n + 1)
    sp[1] = 1.0
    up[1] = 1.0
    for k in 2:(n + 1)
        sp[k] = sp[k - 1] * s
        up[k] = up[k - 1] * u
    end
    @inbounds for i in 0:n
        w[i + 1] = binom[i + 1] * sp[i + 1] * up[n - i + 1]
    end
    return w
end

# binomials for degree n: binom(n,i)
function binomials(n::Int)
    b = zeros(Float64, n + 1)
    b[1] = 1.0
    for i in 1:n
        b[i + 1] = b[i] * (n - (i - 1)) / i
    end
    return b
end

# ============================================================
# 2) Bezier coefficient evaluation via forward differences
# ============================================================
# If control points are P_0..P_m (Bezier degree m),
# then k-th derivative wrt s is:
#   d^k/ds^k C(s) = (m)(m-1)...(m-k+1) * Σ_{i=0}^{m-k} B_{i,m-k}(s) * Δ^k P_i
# We precompute Δ^k tables on ctrl update, so evaluation is cheap.
#
# Our homotopy uses τ = 1 - t so that:
#   t=1 -> τ=0 (start = G)
#   t=0 -> τ=1 (target = F)
# And d^k/dt^k = (-1)^k d^k/dτ^k.
#
# ============================================================

# ctrl: ComplexF64[neq, nterm, m+1] with control index j=1..m+1
# diffs[k]: ComplexF64[neq, nterm, (m+1-k)] for k=1..kmax
function compute_diffs!(diffs::Vector{Array{ComplexF64,3}}, ctrl::Array{ComplexF64,3}, m::Int, kmax::Int)
    # k=1
    if kmax >= 1
        D1 = diffs[1]
        @assert size(D1,3) == m
        @inbounds for j in 1:m
            D1[:,:,j] .= ctrl[:,:,j+1] .- ctrl[:,:,j]
        end
    end
    for k in 2:kmax
        Dk = diffs[k]
        Dkm1 = diffs[k-1]
        len = m + 1 - k
        @assert size(Dk,3) == len
        @inbounds for j in 1:len
            Dk[:,:,j] .= Dkm1[:,:,j+1] .- Dkm1[:,:,j]
        end
    end
    return diffs
end

# ============================================================
# 3) Custom homotopy: 2x2, sparse support, deg<=3
# ============================================================

struct BezierSparse2x2Deg3 <: AbstractHomotopy
    m::Int                      # Bezier degree
    neq::Int                    # 2
    nterm::Int                  # 3
    # exps[eq, term, var]  (var: 1->x, 2->y)
    exps::Array{Int,3}
    # control coefficients ctrl[eq, term, j], j=1..m+1
    ctrl::Array{ComplexF64,3}
    # forward differences for ctrl: diffs[k][eq, term, j]
    # k=1..kmax (we use up to 4)
    diffs::Vector{Array{ComplexF64,3}}
    kmax::Int

    # precomputed binomials for degrees: m, m-1, m-2, m-3, m-4 (as needed)
    binoms::Vector{Vector{Float64}}
    # scratch weights for degrees m, m-1, ..., m-4
    wbufs::Vector{Vector{Float64}}

    # scratch effective coeffs for 0th and derivatives
    ceff0::Matrix{ComplexF64}   # [eq,term]
    ceff1::Matrix{ComplexF64}
    ceff2::Matrix{ComplexF64}
    ceff3::Matrix{ComplexF64}
    ceff4::Matrix{ComplexF64}
end

Base.size(H::BezierSparse2x2Deg3) = (2, 2)
ModelKit.variables(::BezierSparse2x2Deg3) = [Variable(:x), Variable(:y)]
ModelKit.parameters(::BezierSparse2x2Deg3) = Variable[]

# falling factorial m*(m-1)*...*(m-k+1)
@inline function fallfac(m::Int, k::Int)
    v = 1.0
    for i in 0:(k-1)
        v *= (m - i)
    end
    return v
end

function eval_coeffs0!(H::BezierSparse2x2Deg3, τ::Float64)
    # ceff0 = Σ B_{j,m}(τ) ctrl[:,:,j]
    w = H.wbufs[1]
    bernstein_weights!(w, H.m, τ, H.binoms[1])
    fill!(H.ceff0, 0.0 + 0im)
    @inbounds for j in 1:(H.m+1)
        H.ceff0 .+= w[j] .* H.ctrl[:,:,j]
    end
    return H.ceff0
end

function eval_coeffs_k!(out::Matrix{ComplexF64}, H::BezierSparse2x2Deg3, τ::Float64, k::Int)
    # out = d^k/dτ^k coeffs (Bezier), using forward diffs
    fill!(out, 0.0 + 0im)
    if k == 0
        return eval_coeffs0!(H, τ)
    end
    if k > H.m
        return out
    end
    deg = H.m - k
    # weight buffer index: k+1 (deg=m-k)
    w = H.wbufs[k+1]
    bernstein_weights!(w, deg, τ, H.binoms[k+1])

    Dk = H.diffs[k]  # size (..., deg+1) since m+1-k = deg+1
    fac = fallfac(H.m, k)
    @inbounds for j in 1:(deg+1)
        out .+= (fac * w[j]) .* Dk[:,:,j]
    end
    return out
end

# fast monomial + partials for deg<=3 using precomputed powers
@inline function powers_deg3(z::ComplexF64)
    z0 = 1.0 + 0im
    z1 = z
    z2 = z1 * z1
    z3 = z2 * z1
    return (z0, z1, z2, z3)
end

function eval_system_and_jac!(u::Vector{ComplexF64}, U::AbstractMatrix{ComplexF64},
                             H::BezierSparse2x2Deg3, x::AbstractVector{ComplexF64},
                             ceff::Matrix{ComplexF64})
    xv = x[1]; yv = x[2]
    (x0,x1,x2,x3) = powers_deg3(xv)
    (y0,y1,y2,y3) = powers_deg3(yv)

    # lookup pow by exponent 0..3
    xp = (x0,x1,x2,x3)
    yp = (y0,y1,y2,y3)

    # u: 2
    u[1] = 0.0 + 0im
    u[2] = 0.0 + 0im
    # U: 2x2
    U[1,1] = 0.0 + 0im
    U[1,2] = 0.0 + 0im
    U[2,1] = 0.0 + 0im
    U[2,2] = 0.0 + 0im

    @inbounds for eq in 1:2
        for term in 1:3
            a = H.exps[eq, term, 1]
            b = H.exps[eq, term, 2]
            c = ceff[eq, term]

            mon = xp[a+1] * yp[b+1]
            u[eq] += c * mon

            if a > 0
                dmon_dx = a * xp[a] * yp[b+1]  # x^(a-1) is xp[a], since xp index shift
                U[eq,1] += c * dmon_dx
            end
            if b > 0
                dmon_dy = b * xp[a+1] * yp[b]  # y^(b-1) is yp[b]
                U[eq,2] += c * dmon_dy
            end
        end
    end
    return nothing
end

function eval_system_only!(u::Vector{ComplexF64}, H::BezierSparse2x2Deg3,
                           x::AbstractVector{ComplexF64}, ceff::Matrix{ComplexF64})
    xv = x[1]; yv = x[2]
    (x0,x1,x2,x3) = powers_deg3(xv)
    (y0,y1,y2,y3) = powers_deg3(yv)
    xp = (x0,x1,x2,x3)
    yp = (y0,y1,y2,y3)

    u[1] = 0.0 + 0im
    u[2] = 0.0 + 0im
    @inbounds for eq in 1:2
        for term in 1:3
            a = H.exps[eq, term, 1]
            b = H.exps[eq, term, 2]
            u[eq] += ceff[eq, term] * (xp[a+1] * yp[b+1])
        end
    end
    return nothing
end

function ModelKit.evaluate!(u, H::BezierSparse2x2Deg3, x, t, p=nothing)
    τ = 1.0 - real(t)
    eval_coeffs0!(H, τ)
    eval_system_only!(u, H, x, H.ceff0)
    return nothing
end

function ModelKit.evaluate_and_jacobian!(u, U, H::BezierSparse2x2Deg3, x, t, p=nothing)
    τ = 1.0 - real(t)
    eval_coeffs0!(H, τ)
    eval_system_and_jac!(u, U, H, x, H.ceff0)
    return nothing
end

# taylor!(Val{k}) returns ∂^k/∂t^k H(x,t)
# Version for Vector{ComplexF64}
function ModelKit.taylor!(u, ::Val{k}, H::BezierSparse2x2Deg3, x::Vector{ComplexF64}, t) where {k}
    τ = 1.0 - real(t)
    # compute d^k/dτ^k coeffs into corresponding buffer
    buf = k == 1 ? H.ceff1 :
          k == 2 ? H.ceff2 :
          k == 3 ? H.ceff3 :
          k == 4 ? H.ceff4 : nothing

    if buf === nothing || k > H.kmax
        fill!(u, 0.0 + 0im)
        return u
    end

    eval_coeffs_k!(buf, H, τ, k)
    # d^k/dt^k = (-1)^k d^k/dτ^k
    if isodd(k)
        buf .= -buf
    end
    eval_system_only!(u, H, x, buf)
    return u
end

# Version for TaylorVector - extract constant terms
function ModelKit.taylor!(u, ::Val{k}, H::BezierSparse2x2Deg3, tx, t) where {k}
    τ = 1.0 - real(t)
    # compute d^k/dτ^k coeffs into corresponding buffer
    buf = k == 1 ? H.ceff1 :
          k == 2 ? H.ceff2 :
          k == 3 ? H.ceff3 :
          k == 4 ? H.ceff4 : nothing

    if buf === nothing || k > H.kmax
        fill!(u, 0.0 + 0im)
        return u
    end

    eval_coeffs_k!(buf, H, τ, k)
    # d^k/dt^k = (-1)^k d^k/dτ^k
    if isodd(k)
        buf .= -buf
    end
    # Extract constant term (0th order) from TruncatedTaylorSeries
    # TruncatedTaylorSeries supports getindex with [0] for constant term
    x1_val = tx[1] isa ComplexF64 ? tx[1] : tx[1][0]
    x2_val = tx[2] isa ComplexF64 ? tx[2] : tx[2][0]
    x_vec = [x1_val, x2_val]
    eval_system_only!(u, H, x_vec, buf)
    return u
end

# in-place update of intermediate control points and diff tables
function set_ctrl!(H::BezierSparse2x2Deg3, ctrl_new::Array{ComplexF64,3})
    @assert size(ctrl_new) == size(H.ctrl)
    copyto!(H.ctrl, ctrl_new)
    compute_diffs!(H.diffs, H.ctrl, H.m, H.kmax)
    return H
end

# ============================================================
# 4) Problem setup: supports, G, F, starts (total-degree style)
# ============================================================

function build_default_support()
    # 2 equations, 3 terms each
    # eq1: x^3, x*y, 1
    # eq2: y^3, x*y, 1
    exps = zeros(Int, 2, 3, 2)
    exps[1,1,:] .= (3,0)
    exps[1,2,:] .= (1,1)
    exps[1,3,:] .= (0,0)
    exps[2,1,:] .= (0,3)
    exps[2,2,:] .= (1,1)
    exps[2,3,:] .= (0,0)
    return exps
end

function build_ctrl(m::Int; seed::Int=0)
    Random.seed!(seed)
    ctrl = zeros(ComplexF64, 2, 3, m+1)

    # G: x^3 - 1, y^3 - 1
    ctrl[1,1,1] = 1.0 + 0im   # x^3
    ctrl[1,2,1] = 0.0 + 0im   # x*y
    ctrl[1,3,1] = -1.0 + 0im  # 1

    ctrl[2,1,1] = 1.0 + 0im   # y^3
    ctrl[2,2,1] = 0.0 + 0im   # x*y
    ctrl[2,3,1] = -1.0 + 0im  # 1

    # F: random but small scale
    # eq1: x^3 + α x y + β
    # eq2: y^3 + γ x y + δ
    ctrl[1,1,m+1] = 1.0 + 0im
    ctrl[1,2,m+1] = 0.3 + 0.2im
    ctrl[1,3,m+1] = 0.1 - 0.4im

    ctrl[2,1,m+1] = 1.0 + 0im
    ctrl[2,2,m+1] = -0.25 + 0.15im
    ctrl[2,3,m+1] = -0.2 + 0.3im

    # intermediate control points: random
    for j in 2:m
        for eq in 1:2, term in 1:3
            ctrl[eq,term,j] = 0.2 * (randn() + randn()*im)
        end
        # optional: keep leading terms near 1.0 to avoid insane scaling
        ctrl[1,1,j] += 1.0 + 0im
        ctrl[2,1,j] += 1.0 + 0im
    end
    return ctrl
end

function total_degree_starts_deg3()
    # for x^3 - 1 = 0 and y^3 - 1 = 0
    ω = exp(2π * im / 3)
    xs = [1.0 + 0im, ω, ω^2]
    ys = [1.0 + 0im, ω, ω^2]
    starts = Vector{Vector{ComplexF64}}()
    for x in xs, y in ys
        push!(starts, [x, y])
    end
    return starts
end

function make_homotopy(m::Int; seed::Int=0)
    exps = build_default_support()
    ctrl = build_ctrl(m; seed=seed)

    kmax = min(4, m)  # implement up to 4th derivative (or m if smaller)
    diffs = Vector{Array{ComplexF64,3}}(undef, kmax)
    for k in 1:kmax
        diffs[k] = zeros(ComplexF64, 2, 3, (m+1-k))
    end
    compute_diffs!(diffs, ctrl, m, kmax)

    # binoms and weight buffers for degrees: m, m-1, ..., m-kmax
    binoms = Vector{Vector{Float64}}(undef, kmax + 1)
    wbufs  = Vector{Vector{Float64}}(undef, kmax + 1)
    for kk in 0:kmax
        deg = m - kk
        binoms[kk+1] = binomials(deg)
        wbufs[kk+1]  = zeros(Float64, deg + 1)
    end

    H = BezierSparse2x2Deg3(
        m, 2, 3, exps, ctrl, diffs, kmax,
        binoms, wbufs,
        zeros(ComplexF64, 2, 3),
        zeros(ComplexF64, 2, 3),
        zeros(ComplexF64, 2, 3),
        zeros(ComplexF64, 2, 3),
        zeros(ComplexF64, 2, 3),
    )
    return H
end

# ============================================================
# 5) Benchmark loop: update middle control points, solve, collect stats
# ============================================================

function main(; m::Int=4, n_eval::Int=200, seed::Int=0)
    H = make_homotopy(m; seed=seed)
    starts = total_degree_starts_deg3()

    opts = HomotopyContinuation.TrackerOptions(
        max_steps=50_000,
        max_step_size=0.05,
        max_initial_step_size=0.05,
        min_step_size=1e-12,
        min_rel_step_size=1e-12,
        extended_precision=false,
    )

    # Pre-allocate solver (recommended for repeated solves)
    solver, _ = solver_startsolutions(H, starts; tracker_options=opts, show_progress=false)  # :contentReference[oaicite:3]{index=3}

    times = Float64[]
    steps = Int[]
    okcnt = 0

    # helper to randomize only intermediate control points (j=2..m), keep endpoints fixed
    function randomize_middle!(ctrl::Array{ComplexF64,3})
        for j in 2:m
            for eq in 1:2, term in 1:3
                ctrl[eq,term,j] = 0.2 * (randn() + randn()*im)
            end
            ctrl[1,1,j] += 1.0 + 0im
            ctrl[2,1,j] += 1.0 + 0im
        end
    end

    for it in 1:n_eval
        # update control points in-place
        randomize_middle!(H.ctrl)
        compute_diffs!(H.diffs, H.ctrl, H.m, H.kmax)

        t0 = time()
        res = solve(solver, starts; show_progress=false)
        dt = time() - t0

        # stats
        prs = getproperty(res, :path_results)
        tot = 0
        ok = true
        for pr in prs
            acc = getproperty(pr, :accepted_steps)
            rej = getproperty(pr, :rejected_steps)
            tot += acc + rej
            ok &= (getproperty(pr, :return_code) == :success)
        end

        push!(times, dt)
        push!(steps, tot)
        okcnt += ok ? 1 : 0

        if it % 20 == 0
            println("iter=$it  dt=$(round(dt, digits=6))s  steps=$tot  ok=$ok")
        end
    end

    mean_t = sum(times) / length(times)
    mean_s = sum(steps) / length(steps)
    ok_rate = okcnt / n_eval
    per_step_ms = mean_s > 0 ? (mean_t / mean_s) * 1e3 : 0.0

    println("\n[bezier_hc_speedcheck] 2x2 deg<=3 sparse(3 terms/eq)")
    println("  bezier_degree m=$m  evals=$n_eval  seed=$seed")
    println("  mean_solve_sec=$(round(mean_t, digits=6))  ok_rate=$(round(ok_rate*100, digits=1))%")
    println("  mean_total_steps=$(round(mean_s, digits=1))  est_per_step_ms=$(round(per_step_ms, digits=4))")
end

main(m=4, n_eval=200, seed=0)