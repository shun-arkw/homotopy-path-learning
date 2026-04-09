using HomotopyContinuation
using HomotopyContinuation.ModelKit
using Random
using Base.Threads

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
# 1) Bernstein weights (de Casteljau), no allocations
# ============================================================

# After call: w[i+1] = B_{i,deg}(s), i=0..deg where deg=length(w)-1
function bernstein_weights_casteljau!(w::Vector{Float64}, s::Float64)
    deg = length(w) - 1
    u = 1.0 - s
    fill!(w, 0.0)
    w[1] = 1.0
    @inbounds for j in 1:deg
        for i in (j+1):-1:2
            w[i] = u * w[i] + s * w[i-1]
        end
        w[1] = u * w[1]
    end
    return w
end

# ============================================================
# 2) Forward differences for control points
# ctrl: (bezier_degree+1, ncoef), row j = j-th control point's coeff vector [a_n..a_0]
# diffs[k]: (bezier_degree+1-k, ncoef)
# ============================================================

function compute_diffs!(diffs::Vector{Matrix{ComplexF64}},
                        ctrl::Matrix{ComplexF64},
                        bezier_degree::Int,
                        max_derivative_order::Int)
    @assert size(ctrl, 1) == bezier_degree + 1
    ncoef = size(ctrl, 2)

    if max_derivative_order >= 1
        D1 = diffs[1]
        @assert size(D1, 1) == bezier_degree
        @assert size(D1, 2) == ncoef
        @inbounds for i in 1:ncoef, j in 1:bezier_degree
            D1[j, i] = ctrl[j+1, i] - ctrl[j, i]
        end
    end

    @inbounds for k in 2:max_derivative_order
        Dk   = diffs[k]
        Dkm1 = diffs[k-1]
        len = bezier_degree + 1 - k
        @assert size(Dk, 1) == len
        @assert size(Dk, 2) == ncoef
        for i in 1:ncoef, j in 1:len
            Dk[j, i] = Dkm1[j+1, i] - Dkm1[j, i]
        end
    end
    return diffs
end

# ============================================================
# 3) Polynomial evaluation (generic numeric type)
# coeffs: [a_n, a_{n-1}, ..., a_0] (descending)
# ============================================================

@inline function poly_and_deriv_horner(coeffs::AbstractVector, x)
    # returns (P(x), P'(x)) with one pass
    b = convert(typeof(x), coeffs[1])
    c = zero(x)
    @inbounds for i in 2:length(coeffs)
        c = c * x + b
        b = b * x + convert(typeof(x), coeffs[i])
    end
    return (b, c)
end

@inline function poly_only_horner(coeffs::AbstractVector, x)
    b = convert(typeof(x), coeffs[1])
    @inbounds for i in 2:length(coeffs)
        b = b * x + convert(typeof(x), coeffs[i])
    end
    return b
end

# ============================================================
# 4) Custom homotopy: univariate polynomial degree, coefficients Bezier in tau
# ============================================================

struct BezierUnivarPoly <: AbstractHomotopy
    degree::Int                # polynomial degree
    bezier_degree::Int         # Bezier degree (control points bezier_degree+1)
    max_derivative_order::Int                  # derivatives w.r.t t supported (<=4, <=bezier_degree)

    # control coefficients: ctrl[j, :] = j-th control point's coeff vector [a_n..a_0]
    # ctrl has size (bezier_degree+1, degree+1)
    ctrl::Matrix{ComplexF64}

    # forward differences: diffs[k] has size (bezier_degree+1-k, degree+1)
    diffs::Vector{Matrix{ComplexF64}}

    # weight buffers for degrees bezier_degree, bezier_degree-1, ..., bezier_degree-max_derivative_order
    wbufs::Vector{Vector{Float64}}

    # effective coeff buffers (descending order)
    ceff0::Vector{ComplexF64}
    ceff1::Vector{ComplexF64}
    ceff2::Vector{ComplexF64}
    ceff3::Vector{ComplexF64}
    ceff4::Vector{ComplexF64}
end

Base.size(::BezierUnivarPoly) = (1, 1)
ModelKit.variables(::BezierUnivarPoly) = [Variable(:x)]
ModelKit.parameters(::BezierUnivarPoly) = Variable[]

# ---------- db=2,3,4,5 dedicated (closed-form Bernstein, no de Casteljau) ----------
function _eval_coeffs0!_db2!(H::BezierUnivarPoly, τ::Float64)
    u = 1.0 - τ
    u2 = u * u
    s2 = τ * τ
    w0 = u2
    w1 = 2.0 * u * τ
    w2 = s2
    ncoef = H.degree + 1
    c = H.ctrl
    out = H.ceff0
    @inbounds for i in 1:ncoef
        out[i] = w0 * c[1, i] + w1 * c[2, i] + w2 * c[3, i]
    end
    return out
end

function _eval_coeffs0!_db3!(H::BezierUnivarPoly, τ::Float64)
    u = 1.0 - τ
    u2 = u * u
    u3 = u2 * u
    s2 = τ * τ
    s3 = s2 * τ
    w0 = u3
    w1 = 3.0 * u2 * τ
    w2 = 3.0 * u * s2
    w3 = s3
    ncoef = H.degree + 1
    c = H.ctrl
    out = H.ceff0
    @inbounds for i in 1:ncoef
        out[i] = w0 * c[1, i] + w1 * c[2, i] + w2 * c[3, i] + w3 * c[4, i]
    end
    return out
end

function _eval_coeffs0!_db4!(H::BezierUnivarPoly, τ::Float64)
    u = 1.0 - τ
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    s2 = τ * τ
    s3 = s2 * τ
    s4 = s3 * τ
    w0 = u4
    w1 = 4.0 * u3 * τ
    w2 = 6.0 * u2 * s2
    w3 = 4.0 * u * s3
    w4 = s4
    ncoef = H.degree + 1
    c = H.ctrl
    out = H.ceff0
    @inbounds for i in 1:ncoef
        out[i] = w0 * c[1, i] + w1 * c[2, i] + w2 * c[3, i] + w3 * c[4, i] + w4 * c[5, i]
    end
    return out
end

function _eval_coeffs0!_db5!(H::BezierUnivarPoly, τ::Float64)
    u = 1.0 - τ
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    s2 = τ * τ
    s3 = s2 * τ
    s4 = s3 * τ
    s5 = s4 * τ
    w0 = u5
    w1 = 5.0 * u4 * τ
    w2 = 10.0 * u3 * s2
    w3 = 10.0 * u2 * s3
    w4 = 5.0 * u * s4
    w5 = s5
    ncoef = H.degree + 1
    c = H.ctrl
    out = H.ceff0
    @inbounds for i in 1:ncoef
        out[i] = w0 * c[1, i] + w1 * c[2, i] + w2 * c[3, i] + w3 * c[4, i] + w4 * c[5, i] + w5 * c[6, i]
    end
    return out
end

function eval_coeffs0!(H::BezierUnivarPoly, τ::Float64)
    db = H.bezier_degree
    if db == 2
        return _eval_coeffs0!_db2!(H, τ)
    elseif db == 3
        return _eval_coeffs0!_db3!(H, τ)
    elseif db == 4
        return _eval_coeffs0!_db4!(H, τ)
    elseif db == 5
        return _eval_coeffs0!_db5!(H, τ)
    end
    # generic: de Casteljau
    w = H.wbufs[1]
    bernstein_weights_casteljau!(w, τ)
    ncoef = H.degree + 1
    @inbounds for i in 1:ncoef
        s = 0.0 + 0im
        for j in 1:(H.bezier_degree+1)
            s += w[j] * H.ctrl[j, i]
        end
        H.ceff0[i] = s
    end
    return H.ceff0
end

# Closed-form Bernstein weights for degree 1..4 (for eval_coeffs_k! when db=2..5)
@inline function _bernstein_weights_deg1!(w::Vector{Float64}, s::Float64)
    u = 1.0 - s
    w[1] = u
    w[2] = s
    return w
end
@inline function _bernstein_weights_deg2!(w::Vector{Float64}, s::Float64)
    u = 1.0 - s
    u2 = u * u
    s2 = s * s
    w[1] = u2
    w[2] = 2.0 * u * s
    w[3] = s2
    return w
end
@inline function _bernstein_weights_deg3!(w::Vector{Float64}, s::Float64)
    u = 1.0 - s
    u2 = u * u
    u3 = u2 * u
    s2 = s * s
    s3 = s2 * s
    w[1] = u3
    w[2] = 3.0 * u2 * s
    w[3] = 3.0 * u * s2
    w[4] = s3
    return w
end
@inline function _bernstein_weights_deg4!(w::Vector{Float64}, s::Float64)
    u = 1.0 - s
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    w[1] = u4
    w[2] = 4.0 * u3 * s
    w[3] = 6.0 * u2 * s2
    w[4] = 4.0 * u * s3
    w[5] = s4
    return w
end

function eval_coeffs_k!(out::Vector{ComplexF64}, H::BezierUnivarPoly, τ::Float64, k::Int)
    fill!(out, 0.0 + 0im)
    if k == 0
        eval_coeffs0!(H, τ)
        out .= H.ceff0
        return out
    end
    if k > H.bezier_degree
        return out
    end
    deg = H.bezier_degree - k
    w = H.wbufs[k+1]                  # length deg+1
    Dk = H.diffs[k]                   # (deg+1, degree+1)
    fac = fallfac(H.bezier_degree, k)
    ncoef = H.degree + 1

    # Use closed-form weights when db=2..5 and deg=1..4
    db = H.bezier_degree
    if 2 <= db <= 5 && 1 <= deg <= 4
        if deg == 1
            _bernstein_weights_deg1!(w, τ)
        elseif deg == 2
            _bernstein_weights_deg2!(w, τ)
        elseif deg == 3
            _bernstein_weights_deg3!(w, τ)
        else
            _bernstein_weights_deg4!(w, τ)
        end
    else
        bernstein_weights_casteljau!(w, τ)
    end

    @inbounds for i in 1:ncoef
        s = 0.0 + 0im
        for j in 1:(deg+1)
            s += w[j] * Dk[j, i]
        end
        out[i] = fac * s
    end
    return out
end

function ModelKit.evaluate!(u, H::BezierUnivarPoly, x, t, p=nothing)
    τ = 1.0 - Float64(real(t))
    eval_coeffs0!(H, τ)
    u[1] = poly_only_horner(H.ceff0, x[1])
    return nothing
end

function ModelKit.evaluate_and_jacobian!(u, U, H::BezierUnivarPoly, x, t, p=nothing)
    τ = 1.0 - Float64(real(t))
    eval_coeffs0!(H, τ)
    (px, dpx) = poly_and_deriv_horner(H.ceff0, x[1])
    u[1] = px
    U[1,1] = dpx
    return nothing
end

# d^k/dt^k H(x,t) (k<=max_derivative_order). We use tau=1-t => d/dt = -d/dtau
function ModelKit.taylor!(u, ::Val{k}, H::BezierUnivarPoly, x::Vector{ComplexF64}, t) where {k}
    if k > H.max_derivative_order
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
        @inbounds for i in 1:length(buf)
            buf[i] = -buf[i]
        end
    end
    u[1] = poly_only_horner(buf, x[1])
    return u
end

function ModelKit.taylor!(u, ::Val{k}, H::BezierUnivarPoly, tx, t) where {k}
    if k > H.max_derivative_order
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
        @inbounds for i in 1:length(buf)
            buf[i] = -buf[i]
        end
    end
    xval = (tx[1] isa ComplexF64) ? tx[1] : tx[1][0]
    u[1] = poly_only_horner(buf, xval)
    return u
end

# ============================================================
# 5) Starts for G(x)=x^n-1 (total-degree style, univariate)
# ============================================================

function total_degree_start_solutions_univar(degree::Int)
    roots = Vector{Vector{ComplexF64}}(undef, degree)
    @inbounds for k in 0:(degree-1)
        roots[k+1] = [exp(2π * im * k / degree)]
    end
    return roots
end

# ============================================================
# 6) Default control points builder (for benchmark)
# G fixed: x^n - 1
# F random: leading coefficient not necessarily 1
# ============================================================

function build_ctrl_univar(degree::Int, bezier_degree::Int; seed::Int=0, sigma_mid::Float64=0.2, sigma_F::Float64=0.3)
    Random.seed!(seed)
    ncoef = degree + 1
    ctrl = zeros(ComplexF64, bezier_degree+1, ncoef)

    # --- G endpoint: row 1 = control point 1 (x^degree - 1) ---
    ctrl[1, 1] = 1.0 + 0im        # a_n
    for i in 2:ncoef-1
        ctrl[1, i] = 0.0 + 0im
    end
    ctrl[1, ncoef] = -1.0 + 0im   # a_0

    # --- F endpoint: row bezier_degree+1 = control point bezier_degree+1 (random coeffs) ---
    for i in 1:ncoef
        ctrl[bezier_degree+1, i] = sigma_F * (randn() + randn()*im)
    end
    if abs(ctrl[bezier_degree+1, 1]) < 1e-3
        ctrl[bezier_degree+1, 1] += 1.0 + 0im
    end

    # --- intermediate control points: rows 2..bezier_degree ---
    for j in 2:bezier_degree
        for i in 1:ncoef
            ctrl[j, i] = sigma_mid * (randn() + randn()*im)
        end
        ctrl[j, 1] += 1.0 + 0im
    end

    return ctrl
end

function make_homotopy_univar(degree::Int, bezier_degree::Int; seed::Int=0)
    @assert degree >= 1
    @assert bezier_degree >= 1
    ncoef = degree + 1
    max_derivative_order = min(4, bezier_degree)

    ctrl = build_ctrl_univar(degree, bezier_degree; seed=seed)

    diffs = Vector{Matrix{ComplexF64}}(undef, max_derivative_order)
    for k in 1:max_derivative_order
        diffs[k] = zeros(ComplexF64, (bezier_degree+1-k), ncoef)
    end
    compute_diffs!(diffs, ctrl, bezier_degree, max_derivative_order)

    wbufs = Vector{Vector{Float64}}(undef, max_derivative_order + 1)
    for kk in 0:max_derivative_order
        deg = bezier_degree - kk
        wbufs[kk+1] = zeros(Float64, deg + 1)
    end

    H = BezierUnivarPoly(
        degree, bezier_degree, max_derivative_order,
        ctrl, diffs, wbufs,
        zeros(ComplexF64, ncoef),
        zeros(ComplexF64, ncoef),
        zeros(ComplexF64, ncoef),
        zeros(ComplexF64, ncoef),
        zeros(ComplexF64, ncoef),
    )
    return H
end

# ============================================================
# 7) State with per-thread cache (for parallel path tracking)
# ============================================================

mutable struct BezierUnivarState
    degree::Int
    bezier_degree::Int
    H0::BezierUnivarPoly
    tracker0::Any
    opts::HomotopyContinuation.TrackerOptions
    starts::Vector{Vector{ComplexF64}}

    nthreads_cached::Int
    Hs::Vector{BezierUnivarPoly}
    trackers::Vector{Any}
end

const __STATE__ = Dict{Tuple{Int,Int}, BezierUnivarState}()

function _build_thread_cache!(st::BezierUnivarState)
    nt = Threads.nthreads()
    st.nthreads_cached = nt
    st.Hs = Vector{BezierUnivarPoly}(undef, nt)
    st.trackers = Vector{Any}(undef, nt)
    @inbounds for tid in 1:nt
        Hloc = deepcopy(st.H0)
        st.Hs[tid] = Hloc
        st.trackers[tid] = HomotopyContinuation.Tracker(Hloc; options=st.opts)
    end
    return nothing
end

function _ensure_thread_cache!(st::BezierUnivarState)
    if st.nthreads_cached != Threads.nthreads() || length(st.trackers) != Threads.nthreads()
        _build_thread_cache!(st)
    end
    return nothing
end

# ============================================================
# 8) API for Python (juliacall)
# ============================================================

function init_bezier_univar(;
    degree::Int,
    bezier_degree::Int,
    seed::Int = 0,
    compute_newton_iters::Bool = false,
    extended_precision::Bool = false,
    max_steps::Int = 50_000,
    max_step_size::Float64 = 0.05,
    max_initial_step_size::Float64 = 0.05,
    min_step_size::Float64 = 1e-12,
    # HC TrackerParameters (shared with linear_univar.jl)
    hc_a::Float64 = 0.125,
    hc_beta_a::Float64 = 1.0,
    hc_beta_omega_p::Float64 = 0.8,
    hc_beta_tau::Float64 = 0.85,
    hc_strict_beta_tau::Float64 = 0.8,
    hc_min_newton_iters::Int = 1,
)
    Random.seed!(seed)

    H0 = make_homotopy_univar(degree, bezier_degree; seed=seed)
    starts = total_degree_start_solutions_univar(degree)

    p_custom = HomotopyContinuation.TrackerParameters(
        hc_a,
        hc_beta_a,
        hc_beta_omega_p,
        hc_beta_tau,
        hc_strict_beta_tau,
        hc_min_newton_iters,
    )

    opts = HomotopyContinuation.TrackerOptions(
        automatic_differentiation = 1,
        max_steps = max_steps,
        max_step_size = max_step_size,
        max_initial_step_size = max_initial_step_size,
        min_step_size = min_step_size,
        extended_precision = extended_precision,
        parameters = p_custom,
    )

    tracker0 = HomotopyContinuation.Tracker(H0; options=opts)

    st = BezierUnivarState(
        degree,
        bezier_degree,
        H0,
        tracker0,
        opts,
        starts,
        0,
        BezierUnivarPoly[],
        Any[],
    )
    _build_thread_cache!(st)

    __STATE__[(degree, bezier_degree)] = st
    return nothing
end

# Set control points, then run path tracking for all start solutions.
# control_points: (bezier_degree+1, degree+1) complex, row j = j-th control point's coeff vector [a_n..a_0]
# When compute_newton_iters=true: sequential (stdout/stderr capture); else: parallel with per-thread trackers.
function track_bezier_paths_univar(degree::Int, bezier_degree::Int, control_points::AbstractArray{<:Complex,2}; compute_newton_iters::Bool=false)
    t0 = time()
    st = __STATE__[(degree, bezier_degree)]
    starts = st.starts
    n = length(starts)

    if compute_newton_iters
        # Sequential: safe with stdout/stderr capture for Newton iteration logging
        H = st.H0
        @assert size(control_points) == size(H.ctrl)
        copyto!(H.ctrl, control_points)
        compute_diffs!(H.diffs, H.ctrl, H.bezier_degree, H.max_derivative_order)

        success_flag = true
        total_step_attempts = 0
        total_newton_iterations = 0
        total_accepted_steps = 0
        total_rejected_steps = 0
        tracking_time_sec = 0.0

        for s in starts
            pipe = Pipe()
            buf = IOBuffer()
            reader = @async begin
                while true
                    chunk = read(pipe, 8192)
                    isempty(chunk) && break
                    write(buf, chunk)
                end
            end
            track_start = time()
            pr = redirect_stdout(pipe) do
                redirect_stderr(pipe) do
                    track(st.tracker0, s; debug=true)
                end
            end
            tracking_time_sec += time() - track_start
            close(pipe.in)
            wait(reader)
            log = String(take!(buf))
            for mt in eachmatch(r"iters\s*→\s*(\d+)", log)
                total_newton_iterations += parse(Int, mt.captures[1])
            end

            _accepted_steps = accepted_steps(pr)
            _rejected_steps = rejected_steps(pr)
            total_step_attempts += _accepted_steps + _rejected_steps
            total_accepted_steps += _accepted_steps
            total_rejected_steps += _rejected_steps
            success_flag &= (pr.return_code == :success)
        end

        runtime_sec = time() - t0
        return (
            success_flag=success_flag,
            total_step_attempts=total_step_attempts,
            total_newton_iterations=total_newton_iterations,
            total_accepted_steps=total_accepted_steps,
            total_rejected_steps=total_rejected_steps,
            runtime_sec=runtime_sec,
            tracking_time_sec=tracking_time_sec,
        )
    end

    # Parallel path tracking with cached per-thread trackers
    _ensure_thread_cache!(st)
    nt = st.nthreads_cached

    # Copy control_points into each thread's homotopy (sequential; cheap)
    @inbounds for tid in 1:nt
        Hloc = st.Hs[tid]
        @assert size(control_points) == size(Hloc.ctrl)
        copyto!(Hloc.ctrl, control_points)
        compute_diffs!(Hloc.diffs, Hloc.ctrl, Hloc.bezier_degree, Hloc.max_derivative_order)
    end

    acc = zeros(Int, n)
    rej = zeros(Int, n)
    ok  = trues(n)
    tsec = zeros(Float64, n)

    Threads.@threads for i in 1:n
        tid = Threads.threadid()
        tr = st.trackers[tid]
        s = starts[i]

        tstart = time()
        pr = track(tr, s; debug=false)
        tsec[i] = time() - tstart

        acc[i] = accepted_steps(pr)
        rej[i] = rejected_steps(pr)
        ok[i]  = (pr.return_code == :success)
    end

    total_accepted_steps = sum(acc)
    total_rejected_steps = sum(rej)
    total_step_attempts  = total_accepted_steps + total_rejected_steps
    success_flag         = all(ok)
    tracking_time_sec    = sum(tsec)
    runtime_sec          = time() - t0

    return (
        success_flag=success_flag,
        total_step_attempts=total_step_attempts,
        total_newton_iterations=0,
        total_accepted_steps=total_accepted_steps,
        total_rejected_steps=total_rejected_steps,
        runtime_sec=runtime_sec,
        tracking_time_sec=tracking_time_sec,
    )
end
