# Standalone copy for runtime bench. No dependency on scripts/bezier_hc_ppo.
# TrackerOptions include automatic_differentiation=1 to match Bezier (fair comparison).
using HomotopyContinuation
using HomotopyContinuation.ModelKit
using Random
using Base.Threads

# Optional: set to true and clear LINEAR_EVAL_COUNTS before track to count calls (time-per-Newton bench)
const LINEAR_ENABLE_EVAL_COUNTS = Ref(false)
const LINEAR_EVAL_COUNTS = Ref(Dict{String,Int}())
@inline _inc_linear(name::String) = LINEAR_ENABLE_EVAL_COUNTS[] && (LINEAR_EVAL_COUNTS[][name] = get(LINEAR_EVAL_COUNTS[], name, 0) + 1)

# ============================================================
# 1) Polynomial evaluation
# ============================================================

@inline function poly_and_deriv_horner(coeffs::AbstractVector, x)
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
# 2) Custom homotopy: linear in tau
# ============================================================

struct LinearUnivarPoly <: AbstractHomotopy
    degree::Int
    endpoints::Matrix{ComplexF64}
    ceff0::Vector{ComplexF64}
    ceff1::Vector{ComplexF64}
end

Base.size(::LinearUnivarPoly) = (1, 1)
ModelKit.variables(::LinearUnivarPoly) = [Variable(:x)]
ModelKit.parameters(::LinearUnivarPoly) = Variable[]

function eval_coeffs0!(H::LinearUnivarPoly, τ::Float64)
    ncoef = H.degree + 1
    @inbounds for i in 1:ncoef
        H.ceff0[i] = (1.0 - τ) * H.endpoints[1, i] + τ * H.endpoints[2, i]
    end
    return H.ceff0
end

function update_linear_diff!(H::LinearUnivarPoly)
    ncoef = H.degree + 1
    @inbounds for i in 1:ncoef
        H.ceff1[i] = H.endpoints[2, i] - H.endpoints[1, i]
    end
    return H.ceff1
end

function ModelKit.evaluate!(u, H::LinearUnivarPoly, x, t, p=nothing)
    τ = 1.0 - Float64(real(t))
    eval_coeffs0!(H, τ)
    u[1] = poly_only_horner(H.ceff0, x[1])
    return nothing
end

function ModelKit.evaluate_and_jacobian!(u, U, H::LinearUnivarPoly, x, t, p=nothing)
    _inc_linear("evaluate_and_jacobian")
    τ = 1.0 - Float64(real(t))
    eval_coeffs0!(H, τ)
    (px, dpx) = poly_and_deriv_horner(H.ceff0, x[1])
    u[1] = px
    U[1,1] = dpx
    return nothing
end

function ModelKit.taylor!(u, ::Val{k}, H::LinearUnivarPoly, x::Vector{ComplexF64}, t) where {k}
    if k == 0
        τ = 1.0 - Float64(real(t))
        eval_coeffs0!(H, τ)
        u[1] = poly_only_horner(H.ceff0, x[1])
        return u
    elseif k == 1
        @inbounds for i in 1:length(H.ceff1)
            H.ceff1[i] = -H.ceff1[i]
        end
        u[1] = poly_only_horner(H.ceff1, x[1])
        @inbounds for i in 1:length(H.ceff1)
            H.ceff1[i] = -H.ceff1[i]
        end
        return u
    else
        fill!(u, 0.0 + 0im); return u
    end
end

function ModelKit.taylor!(u, ::Val{k}, H::LinearUnivarPoly, tx, t) where {k}
    if k == 0
        τ = 1.0 - Float64(real(t))
        eval_coeffs0!(H, τ)
        xval = (tx[1] isa ComplexF64) ? tx[1] : tx[1][0]
        u[1] = poly_only_horner(H.ceff0, xval)
        return u
    elseif k == 1
        @inbounds for i in 1:length(H.ceff1)
            H.ceff1[i] = -H.ceff1[i]
        end
        xval = (tx[1] isa ComplexF64) ? tx[1] : tx[1][0]
        u[1] = poly_only_horner(H.ceff1, xval)
        @inbounds for i in 1:length(H.ceff1)
            H.ceff1[i] = -H.ceff1[i]
        end
        return u
    else
        fill!(u, 0.0 + 0im); return u
    end
end

# ============================================================
# 3) Starts
# ============================================================

function total_degree_start_solutions_univar(degree::Int)
    roots = Vector{Vector{ComplexF64}}(undef, degree)
    @inbounds for k in 0:(degree-1)
        roots[k+1] = [exp(2π * im * k / degree)]
    end
    return roots
end

# ============================================================
# 4) State and thread cache
# ============================================================

mutable struct LinearUnivarState
    degree::Int
    H0::LinearUnivarPoly
    tracker0::Any
    opts::HomotopyContinuation.TrackerOptions
    starts::Vector{Vector{ComplexF64}}
    nthreads_cached::Int
    Hs::Vector{LinearUnivarPoly}
    trackers::Vector{Any}
end

const __STATE_LINEAR__ = Dict{Int, LinearUnivarState}()

function _build_thread_cache!(st::LinearUnivarState)
    nt = Threads.nthreads()
    st.nthreads_cached = nt
    st.Hs = Vector{LinearUnivarPoly}(undef, nt)
    st.trackers = Vector{Any}(undef, nt)
    @inbounds for tid in 1:nt
        st.Hs[tid] = deepcopy(st.H0)
        st.trackers[tid] = HomotopyContinuation.Tracker(st.Hs[tid]; options=st.opts)
    end
    return nothing
end

function _ensure_thread_cache!(st::LinearUnivarState)
    if st.nthreads_cached != Threads.nthreads() || length(st.trackers) != Threads.nthreads()
        _build_thread_cache!(st)
    end
    return nothing
end

# ============================================================
# 5) Init and track API
# ============================================================

function init_linear_univar(;
    degree::Int,
    seed::Int = 0,
    max_steps::Int = 50_000,
    max_step_size::Float64 = 0.05,
    max_initial_step_size::Float64 = 0.05,
    min_step_size::Float64 = 1e-12,
    hc_a::Float64 = 0.125,
    hc_beta_a::Float64 = 1.0,
    hc_beta_omega_p::Float64 = 0.8,
    hc_beta_tau::Float64 = 0.85,
    hc_strict_beta_tau::Float64 = 0.8,
    hc_min_newton_iters::Int = 1,
)
    Random.seed!(seed)
    ncoef = degree + 1
    endpoints = zeros(ComplexF64, 2, ncoef)
    endpoints[1, 1] = 1.0 + 0im
    for i in 2:ncoef-1
        endpoints[1, i] = 0.0 + 0im
    end
    endpoints[1, ncoef] = -1.0 + 0im
    for i in 1:ncoef
        endpoints[2, i] = 1e-3 * (randn() + randn()*im)
    end

    H0 = LinearUnivarPoly(
        degree,
        endpoints,
        zeros(ComplexF64, ncoef),
        zeros(ComplexF64, ncoef),
    )
    update_linear_diff!(H0)
    starts = total_degree_start_solutions_univar(degree)

    p_custom = HomotopyContinuation.TrackerParameters(
        hc_a, hc_beta_a, hc_beta_omega_p, hc_beta_tau, hc_strict_beta_tau, hc_min_newton_iters,
    )

    opts = HomotopyContinuation.TrackerOptions(
        automatic_differentiation = 1,
        max_steps = max_steps,
        max_step_size = max_step_size,
        max_initial_step_size = max_initial_step_size,
        min_step_size = min_step_size,
        extended_precision = false,
        parameters = p_custom,
    )

    tracker0 = HomotopyContinuation.Tracker(H0; options=opts)
    st = LinearUnivarState(
        degree, H0, tracker0, opts, starts,
        0, LinearUnivarPoly[], Any[],
    )
    _build_thread_cache!(st)
    __STATE_LINEAR__[degree] = st
    return nothing
end

function track_linear_paths_univar(degree::Int, start_coeffs::AbstractVector{<:Complex}, target_coeffs::AbstractVector{<:Complex})
    t0 = time()
    st = __STATE_LINEAR__[degree]
    @assert length(start_coeffs) == degree + 1 && length(target_coeffs) == degree + 1
    _ensure_thread_cache!(st)
    nt = st.nthreads_cached
    @inbounds for tid in 1:nt
        Hloc = st.Hs[tid]
        @inbounds for i in 1:(degree+1)
            Hloc.endpoints[1, i] = start_coeffs[i]
            Hloc.endpoints[2, i] = target_coeffs[i]
        end
        update_linear_diff!(Hloc)
    end
    n = length(st.starts)
    acc = zeros(Int, n)
    rej = zeros(Int, n)
    ok  = trues(n)
    tsec = zeros(Float64, n)
    Threads.@threads for i in 1:n
        tid = Threads.threadid()
        tr = st.trackers[tid]
        s = st.starts[i]
        tstart = time()
        pr = track(tr, s; debug=false)
        tsec[i] = time() - tstart
        acc[i] = accepted_steps(pr)
        rej[i] = rejected_steps(pr)
        ok[i]  = (pr.return_code == :success)
    end
    tracking_time_sec = sum(tsec)
    runtime_sec = time() - t0
    return (
        success_flag = all(ok),
        total_step_attempts = sum(acc) + sum(rej),
        total_accepted_steps = sum(acc),
        total_rejected_steps = sum(rej),
        runtime_sec = runtime_sec,
        tracking_time_sec = tracking_time_sec,
    )
end
