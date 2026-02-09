using HomotopyContinuation
using HomotopyContinuation.ModelKit
using Random

# ============================================================
# 0) Utilities
# ============================================================

# ============================================================
# 1) Polynomial evaluation (generic numeric type)
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
# 2) Custom homotopy: univariate polynomial degree, coefficients linear in tau
# ============================================================

struct LinearUnivarPoly <: AbstractHomotopy
    degree::Int                # polynomial degree
    # endpoints: row 1 = start coeffs, row 2 = target coeffs
    endpoints::Matrix{ComplexF64}  # size (2, degree+1)
    # effective coeff buffers (descending order)
    ceff0::Vector{ComplexF64}
    ceff1::Vector{ComplexF64}  # d/dtau coefficients (constant)
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
    τ = 1.0 - Float64(real(t))
    eval_coeffs0!(H, τ)
    (px, dpx) = poly_and_deriv_horner(H.ceff0, x[1])
    u[1] = px
    U[1,1] = dpx
    return nothing
end

# d^k/dt^k H(x,t). We use tau=1-t => d/dt = -d/dtau
function ModelKit.taylor!(u, ::Val{k}, H::LinearUnivarPoly, x::Vector{ComplexF64}, t) where {k}
    if k == 0
        τ = 1.0 - Float64(real(t))
        eval_coeffs0!(H, τ)
        u[1] = poly_only_horner(H.ceff0, x[1])
        return u
    elseif k == 1
        # d/dt = -d/dtau
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
# 3) Starts for G(x)=x^n-1 (total-degree style, univariate)
# ============================================================

function total_degree_start_solutions_univar(degree::Int)
    roots = Vector{Vector{ComplexF64}}(undef, degree)
    @inbounds for k in 0:(degree-1)
        roots[k+1] = [exp(2π * im * k / degree)]
    end
    return roots
end

# ============================================================
# 4) API for Python (juliacall)
# ============================================================

const __STATE_LINEAR__ = Dict{Int, Any}()

function init_linear_univar(;
    degree::Int,
    seed::Int = 0,
    compute_newton_iters::Bool = false,
    extended_precision::Bool = false,
    max_steps::Int = 50_000,
    max_step_size::Float64 = 0.05,
    max_initial_step_size::Float64 = 0.05,
    min_step_size::Float64 = 1e-12,
    min_rel_step_size::Float64 = 1e-12,
)
    Random.seed!(seed)
    ncoef = degree + 1
    endpoints = zeros(ComplexF64, 2, ncoef)
    # Dummy warmup endpoints: start = x^degree - 1, target = small random
    endpoints[1, 1] = 1.0 + 0im
    for i in 2:ncoef-1
        endpoints[1, i] = 0.0 + 0im
    end
    endpoints[1, ncoef] = -1.0 + 0im
    for i in 1:ncoef
        endpoints[2, i] = 1e-3 * (randn() + randn()*im)
    end

    H = LinearUnivarPoly(
        degree,
        endpoints,
        zeros(ComplexF64, ncoef),
        zeros(ComplexF64, ncoef),
    )
    update_linear_diff!(H)
    starts = total_degree_start_solutions_univar(degree)

    opts = HomotopyContinuation.TrackerOptions(
        max_steps = max_steps,
        max_step_size = max_step_size,
        max_initial_step_size = max_initial_step_size,
        min_step_size = min_step_size,
        min_rel_step_size = min_rel_step_size,
        extended_precision = extended_precision,
    )

    tracker = HomotopyContinuation.Tracker(H; options=opts)
    __STATE_LINEAR__[degree] = (H=H, tracker=tracker, starts=starts)
    return nothing
end

function track_linear_paths_univar(degree::Int, start_coeffs::AbstractVector{<:Complex}, target_coeffs::AbstractVector{<:Complex}; compute_newton_iters::Bool=false)
    t0 = time()
    st = __STATE_LINEAR__[degree]
    H = st.H
    @assert length(start_coeffs) == H.degree + 1
    @assert length(target_coeffs) == H.degree + 1
    @inbounds for i in 1:(H.degree+1)
        H.endpoints[1, i] = start_coeffs[i]
        H.endpoints[2, i] = target_coeffs[i]
    end
    update_linear_diff!(H)
    success_flag = true
    total_step_attempts = 0
    total_newton_iterations = 0
    total_accepted_steps = 0
    total_rejected_steps = 0
    tracking_time_sec = 0.0

    for s in st.starts
        if compute_newton_iters
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
                    track(st.tracker, s; debug=true)
                end
            end
            tracking_time_sec += time() - track_start
            close(pipe.in)
            wait(reader)
            log = String(take!(buf))
            for mt in eachmatch(r"iters\s*→\s*(\d+)", log)
                total_newton_iterations += parse(Int, mt.captures[1])
            end
        else
            track_start = time()
            pr = track(st.tracker, s; debug=false)
            tracking_time_sec += time() - track_start
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
