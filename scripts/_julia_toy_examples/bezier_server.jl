# bezier_server.jl
using HomotopyContinuation
using HomotopyContinuation.ModelKit
using Random


# ============================================================
# 0) Utilities
# ============================================================

@inline function fallfac(m::Int, k::Int)::Float64
    v = 1.0
    @inbounds for i in 0:(k-1)
        v *= (m - i)
    end
    return v
end

# ============================================================
# 1) Bernstein weights without allocations (de Casteljau)
# ============================================================

# Correct Bernstein weights (no allocations, in-place)
# After call: w[i+1] = B_{i,m}(s), i=0..m
function bernstein_weights_casteljau!(w::Vector{Float64}, s::Float64)
    m = length(w) - 1
    u = 1.0 - s
    fill!(w, 0.0)
    w[1] = 1.0
    @inbounds for j in 1:m
        for i in (j+1):-1:2
            w[i] = u * w[i] + s * w[i-1]
        end
        w[1] = u * w[1]
    end
    return w
end

# ============================================================
# 2) Forward differences
# ============================================================

function compute_diffs!(diffs::Vector{Array{ComplexF64,3}},
                        ctrl::Array{ComplexF64,3},
                        m::Int,
                        kmax::Int)
    @assert size(ctrl, 3) == m + 1
    if kmax >= 1
        D1 = diffs[1]
        @assert size(D1,3) == m
        @inbounds for j in 1:m, eq in 1:size(ctrl,1), term in 1:size(ctrl,2)
            D1[eq,term,j] = ctrl[eq,term,j+1] - ctrl[eq,term,j]
        end
    end
    @inbounds for k in 2:kmax
        Dk   = diffs[k]
        Dkm1 = diffs[k-1]
        len = m + 1 - k
        @assert size(Dk,3) == len
        for j in 1:len, eq in 1:size(ctrl,1), term in 1:size(ctrl,2)
            Dk[eq,term,j] = Dkm1[eq,term,j+1] - Dkm1[eq,term,j]
        end
    end
    return diffs
end

# ============================================================
# 3) Custom homotopy
# ============================================================

struct BezierSparse2x2Deg3 <: AbstractHomotopy
    m::Int
    kmax::Int
    exps::Array{Int,3}          # (2,3,2), exponents in 0..3
    ctrl::Array{ComplexF64,3}   # (2,3,m+1) control coeffs (ComplexF64 is fine for speed tests)
    diffs::Vector{Array{ComplexF64,3}}
    wbufs::Vector{Vector{Float64}}
    ceff0::Matrix{ComplexF64}
    ceff1::Matrix{ComplexF64}
    ceff2::Matrix{ComplexF64}
    ceff3::Matrix{ComplexF64}
    ceff4::Matrix{ComplexF64}
end

Base.size(::BezierSparse2x2Deg3) = (2, 2)
ModelKit.variables(::BezierSparse2x2Deg3) = [Variable(:x), Variable(:y)]
ModelKit.parameters(::BezierSparse2x2Deg3) = Variable[]

# --- generalized powers for any numeric type (ComplexF64, Complex{DoubleF64}, etc.) ---
@inline function powers_deg3(z)
    z0 = one(z)
    z1 = z
    z2 = z1 * z1
    z3 = z2 * z1
    return (z0, z1, z2, z3)
end

# Evaluate system only: u is any complex vector, x/y can be Complex{T}
@inline function eval_system_only_xy!(u::AbstractVector,
                                     H::BezierSparse2x2Deg3,
                                     x,
                                     y,
                                     ceff::AbstractMatrix)
    (x0,x1,x2,x3) = powers_deg3(x)
    (y0,y1,y2,y3) = powers_deg3(y)
    xp = (x0,x1,x2,x3)
    yp = (y0,y1,y2,y3)

    u1 = zero(x)
    u2 = zero(x)

    @inbounds for term in 1:3
        a1 = H.exps[1, term, 1]; b1 = H.exps[1, term, 2]
        a2 = H.exps[2, term, 1]; b2 = H.exps[2, term, 2]

        # promote coefficient to x's type (important when x is Complex{DoubleF64})
        c1 = convert(typeof(x), ceff[1,term])
        c2 = convert(typeof(x), ceff[2,term])

        u1 += c1 * (xp[a1+1] * yp[b1+1])
        u2 += c2 * (xp[a2+1] * yp[b2+1])
    end

    u[1] = u1
    u[2] = u2
    return nothing
end

@inline function eval_system_and_jac_xy!(u::AbstractVector,
                                        U::AbstractMatrix,
                                        H::BezierSparse2x2Deg3,
                                        x,
                                        y,
                                        ceff::AbstractMatrix)
    (x0,x1,x2,x3) = powers_deg3(x)
    (y0,y1,y2,y3) = powers_deg3(y)
    xp = (x0,x1,x2,x3)
    yp = (y0,y1,y2,y3)

    u1 = zero(x)
    u2 = zero(x)
    j11 = zero(x); j12 = zero(x); j21 = zero(x); j22 = zero(x)

    @inbounds for term in 1:3
        # eq1
        a = H.exps[1, term, 1]
        b = H.exps[1, term, 2]
        c = convert(typeof(x), ceff[1,term])
        mon = xp[a+1] * yp[b+1]
        u1 += c * mon
        if a > 0
            j11 += c * (a * xp[a] * yp[b+1])
        end
        if b > 0
            j12 += c * (b * xp[a+1] * yp[b])
        end

        # eq2
        a = H.exps[2, term, 1]
        b = H.exps[2, term, 2]
        c = convert(typeof(x), ceff[2,term])
        mon = xp[a+1] * yp[b+1]
        u2 += c * mon
        if a > 0
            j21 += c * (a * xp[a] * yp[b+1])
        end
        if b > 0
            j22 += c * (b * xp[a+1] * yp[b])
        end
    end

    u[1] = u1
    u[2] = u2
    U[1,1] = j11
    U[1,2] = j12
    U[2,1] = j21
    U[2,2] = j22
    return nothing
end

function eval_coeffs0!(H::BezierSparse2x2Deg3, τ::Float64)
    w = H.wbufs[1]
    bernstein_weights_casteljau!(w, τ)
    @inbounds for eq in 1:2, term in 1:3
        s = 0.0 + 0im
        for j in 1:(H.m+1)
            s += w[j] * H.ctrl[eq,term,j]
        end
        H.ceff0[eq,term] = s
    end
    return H.ceff0
end

function eval_coeffs_k!(out::Matrix{ComplexF64}, H::BezierSparse2x2Deg3, τ::Float64, k::Int)
    fill!(out, 0.0 + 0im)
    if k == 0
        eval_coeffs0!(H, τ)
        out .= H.ceff0
        return out
    end
    if k > H.m
        return out
    end
    deg = H.m - k
    w = H.wbufs[k+1]
    bernstein_weights_casteljau!(w, τ)
    Dk = H.diffs[k]
    fac = fallfac(H.m, k)
    @inbounds for eq in 1:2, term in 1:3
        s = 0.0 + 0im
        for j in 1:(deg+1)
            s += w[j] * Dk[eq,term,j]
        end
        out[eq,term] = fac * s
    end
    return out
end

function ModelKit.evaluate!(u, H::BezierSparse2x2Deg3, x, t, p=nothing)
    τ = 1.0 - Float64(real(t))
    eval_coeffs0!(H, τ)
    eval_system_only_xy!(u, H, x[1], x[2], H.ceff0)
    return nothing
end

function ModelKit.evaluate_and_jacobian!(u, U, H::BezierSparse2x2Deg3, x, t, p=nothing)
    τ = 1.0 - Float64(real(t))
    eval_coeffs0!(H, τ)
    eval_system_and_jac_xy!(u, U, H, x[1], x[2], H.ceff0)
    return nothing
end

function ModelKit.taylor!(u, ::Val{k}, H::BezierSparse2x2Deg3, x::Vector{ComplexF64}, t) where {k}
    if k > H.kmax
        fill!(u, 0.0 + 0im); return u
    end
    τ = 1.0 - Float64(real(t))
    buf = k == 1 ? H.ceff1 :
          k == 2 ? H.ceff2 :
          k == 3 ? H.ceff3 :
          k == 4 ? H.ceff4 : nothing
    if buf === nothing
        fill!(u, 0.0 + 0im); return u
    end
    eval_coeffs_k!(buf, H, τ, k)
    if isodd(k)
        @inbounds for eq in 1:2, term in 1:3
            buf[eq,term] = -buf[eq,term]
        end
    end
    eval_system_only_xy!(u, H, x[1], x[2], buf)
    return u
end

# TaylorVector-like (tx elements may be ComplexF64 or Taylor series)
function ModelKit.taylor!(u, ::Val{k}, H::BezierSparse2x2Deg3, tx, t) where {k}
    if k > H.kmax
        fill!(u, 0.0 + 0im); return u
    end
    τ = 1.0 - Float64(real(t))
    buf = k == 1 ? H.ceff1 :
          k == 2 ? H.ceff2 :
          k == 3 ? H.ceff3 :
          k == 4 ? H.ceff4 : nothing
    if buf === nothing
        fill!(u, 0.0 + 0im); return u
    end
    eval_coeffs_k!(buf, H, τ, k)
    if isodd(k)
        @inbounds for eq in 1:2, term in 1:3
            buf[eq,term] = -buf[eq,term]
        end
    end
    xval = (tx[1] isa ComplexF64) ? tx[1] : tx[1][0]
    yval = (tx[2] isa ComplexF64) ? tx[2] : tx[2][0]
    eval_system_only_xy!(u, H, xval, yval, buf)
    return u
end

# ============================================================
# 4) Setup
# ============================================================

function build_default_support()
    exps = zeros(Int, 2, 3, 2)
    exps[1,1,:] .= (3,0)  # x^3
    exps[1,2,:] .= (1,1)  # x*y
    exps[1,3,:] .= (0,0)  # 1
    exps[2,1,:] .= (0,3)  # y^3
    exps[2,2,:] .= (1,1)  # x*y
    exps[2,3,:] .= (0,0)  # 1
    return exps
end

function build_ctrl(m::Int; seed::Int=0)
    Random.seed!(seed)
    ctrl = zeros(ComplexF64, 2, 3, m+1)

    # G: x^3 - 1, y^3 - 1
    ctrl[1,1,1] = 1.0 + 0im
    ctrl[1,2,1] = 0.0 + 0im
    ctrl[1,3,1] = -1.0 + 0im
    ctrl[2,1,1] = 1.0 + 0im
    ctrl[2,2,1] = 0.0 + 0im
    ctrl[2,3,1] = -1.0 + 0im

    # F endpoint
    ctrl[1,1,m+1] = 1.0 + 0im
    ctrl[1,2,m+1] = 0.3 + 0.2im
    ctrl[1,3,m+1] = 0.1 - 0.4im
    ctrl[2,1,m+1] = 1.0 + 0im
    ctrl[2,2,m+1] = -0.25 + 0.15im
    ctrl[2,3,m+1] = -0.2 + 0.3im

    for j in 2:m
        for eq in 1:2, term in 1:3
            ctrl[eq,term,j] = 0.2 * (randn() + randn()*im)
        end
        ctrl[1,1,j] += 1.0 + 0im
        ctrl[2,1,j] += 1.0 + 0im
    end
    return ctrl
end

function total_degree_starts_deg3()
    ω = exp(2π * im / 3)
    xs = (1.0 + 0im, ω, ω^2)
    ys = (1.0 + 0im, ω, ω^2)
    starts = Vector{Vector{ComplexF64}}()
    for x in xs, y in ys
        push!(starts, [x, y])
    end
    return starts
end

function make_homotopy(m::Int; seed::Int=0)
    exps = build_default_support()
    ctrl = build_ctrl(m; seed=seed)
    kmax = min(4, m)

    diffs = Vector{Array{ComplexF64,3}}(undef, kmax)
    for k in 1:kmax
        diffs[k] = zeros(ComplexF64, 2, 3, (m+1-k))
    end
    compute_diffs!(diffs, ctrl, m, kmax)

    wbufs = Vector{Vector{Float64}}(undef, kmax + 1)
    for kk in 0:kmax
        deg = m - kk
        wbufs[kk+1] = zeros(Float64, deg + 1)
    end

    H = BezierSparse2x2Deg3(
        m, kmax, exps, ctrl, diffs, wbufs,
        zeros(ComplexF64, 2, 3),
        zeros(ComplexF64, 2, 3),
        zeros(ComplexF64, 2, 3),
        zeros(ComplexF64, 2, 3),
        zeros(ComplexF64, 2, 3),
    )
    return H
end

"""------------------------------------------------------------------------------------------------"""
const __STATE__ = Dict{Int, Any}()

"""
init_bezier(m; seed=0, extended_precision=false)
  - homotopy/solver/startsを作ってキャッシュ
"""
function init_bezier(m::Int; seed::Int=0, extended_precision::Bool=false)
    Random.seed!(seed)
    H = make_homotopy(m; seed=seed)
    starts = total_degree_starts_deg3()

    opts = HomotopyContinuation.TrackerOptions(
        max_steps=50_000,
        max_step_size=0.05,
        max_initial_step_size=0.05,
        min_step_size=1e-12,
        min_rel_step_size=1e-12,
        extended_precision=extended_precision,
    )

    solver, _ = solver_startsolutions(H, starts; tracker_options=opts, show_progress=false)
    __STATE__[m] = (H=H, solver=solver, starts=starts)
    return nothing
end

"""
set_ctrl_bang(m, ctrl_new)
  ctrl_new: ComplexF64[2,3,m+1]
"""
function set_ctrl_bang(m::Int, ctrl_new::AbstractArray{<:Complex,3})
    st = __STATE__[m]
    H = st.H
    @assert size(ctrl_new) == size(H.ctrl)

    # NumPy由来(PyArray)でもOK：Julia側バッファにコピー
    copyto!(H.ctrl, ctrl_new)

    compute_diffs!(H.diffs, H.ctrl, H.m, H.kmax)
    return nothing
end


"""
eval_once(m) -> (ok, total_steps)
  solverを走らせてsteps等を返す（あなたの集計と同じ）
"""
function eval_once(m::Int)
    st = __STATE__[m]
    res = solve(st.solver, st.starts; show_progress=false)

    prs = getproperty(res, :path_results)
    tot = 0
    ok = true
    @inbounds for pr in prs
        tot += getproperty(pr, :accepted_steps) + getproperty(pr, :rejected_steps)
        ok &= (getproperty(pr, :return_code) == :success)
    end
    return (ok=ok, total_steps=tot)
end