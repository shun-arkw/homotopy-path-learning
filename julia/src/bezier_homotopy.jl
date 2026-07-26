"""Multivariate Bezier homotopy for fixed Pham-type supports."""

mutable struct BezierPhamHomotopy <: AbstractHomotopy
    spec::PolynomialSystemSpec
    bezier_degree::Int
    control_points::Matrix{ComplexF64}
    max_derivative_order::Int
    coefficient_differences::Vector{Matrix{ComplexF64}}
    weight_buffers::Vector{Vector{Float64}}
    coefficient_buffers::Vector{Vector{ComplexF64}}
    variables_cache::Vector{Variable}
end

function _finite_complex_matrix(name::AbstractString, values::AbstractMatrix{<:Complex})
    out = ComplexF64.(collect(values))
    all(z -> isfinite(real(z)) && isfinite(imag(z)), out) ||
        throw(ArgumentError("$name must not contain NaN or Inf."))
    return out
end

function validate_control_points(
    spec::PolynomialSystemSpec,
    control_points::AbstractMatrix{<:Complex},
    bezier_degree::Int,
)
    bezier_degree >= 1 || throw(ArgumentError("bezier_degree must be positive."))
    size(control_points) == (bezier_degree + 1, ncoeffs(spec)) ||
        throw(ArgumentError("control_points must have shape (bezier_degree + 1, M)."))
    all(z -> isfinite(real(z)) && isfinite(imag(z)), control_points) ||
        throw(ArgumentError("control_points must not contain NaN or Inf."))

    for row in 1:size(control_points, 1), q in spec.leading_indices
        control_points[row, q] == 1.0 + 0.0im ||
            throw(ArgumentError("all leading coefficients in control_points must be exactly 1 + 0im."))
    end
    return control_points
end

function BezierPhamHomotopy(
    spec::PolynomialSystemSpec,
    bezier_degree::Int,
    control_points::AbstractMatrix{<:Complex};
    max_derivative_order::Int = min(4, bezier_degree),
)
    bezier_degree >= 1 || throw(ArgumentError("bezier_degree must be positive."))
    0 <= max_derivative_order <= bezier_degree ||
        throw(ArgumentError("max_derivative_order must satisfy 0 <= k <= bezier_degree."))

    ctrl = _finite_complex_matrix("control_points", control_points)
    validate_control_points(spec, ctrl, bezier_degree)
    diffs = compute_forward_differences(ctrl, max_derivative_order)
    weight_buffers = [zeros(Float64, bezier_degree - order + 1) for order in 0:max_derivative_order]
    coefficient_buffers = [zeros(ComplexF64, ncoeffs(spec)) for _ in 0:max_derivative_order]
    variables_cache = [Variable(Symbol("x", i)) for i in 1:spec.nvars]

    return BezierPhamHomotopy(
        spec,
        bezier_degree,
        ctrl,
        max_derivative_order,
        diffs,
        weight_buffers,
        coefficient_buffers,
        variables_cache,
    )
end

function set_control_points!(H::BezierPhamHomotopy, control_points::AbstractMatrix{<:Complex})
    ctrl = _finite_complex_matrix("control_points", control_points)
    validate_control_points(H.spec, ctrl, H.bezier_degree)
    copyto!(H.control_points, ctrl)
    H.coefficient_differences = compute_forward_differences(H.control_points, H.max_derivative_order)
    return H
end

function coefficient_vector!(
    out::AbstractVector{ComplexF64},
    H::BezierPhamHomotopy,
    tau::Real,
    derivative_order::Int = 0,
)
    length(out) == ncoeffs(H.spec) || throw(ArgumentError("out must have length M."))
    derivative_order >= 0 || throw(ArgumentError("derivative_order must be nonnegative."))
    τ = Float64(tau)
    isfinite(τ) || throw(ArgumentError("tau must be finite."))

    if derivative_order > H.bezier_degree
        fill!(out, 0.0 + 0.0im)
        return out
    end
    derivative_order <= H.max_derivative_order ||
        throw(ArgumentError("requested derivative_order was not precomputed for this homotopy."))

    source = derivative_order == 0 ? H.control_points : H.coefficient_differences[derivative_order]
    degree = H.bezier_degree - derivative_order
    weights = H.weight_buffers[derivative_order + 1]
    bernstein_weights!(weights, τ)
    factor = falling_factorial(H.bezier_degree, derivative_order)

    for q in 1:ncoeffs(H.spec)
        value = 0.0 + 0.0im
        for row in 1:(degree + 1)
            value += weights[row] * source[row, q]
        end
        out[q] = factor * value
    end
    return out
end

function coefficients_at(H::BezierPhamHomotopy, tau::Real; derivative_order::Int = 0)
    out = zeros(ComplexF64, ncoeffs(H.spec))
    coefficient_vector!(out, H, tau, derivative_order)
    return out
end

@inline function _monomial_value(x, exponents::AbstractMatrix{Int}, row::Int)
    value = one(x[1])
    for variable_index in 1:length(x)
        exponent = exponents[row, variable_index]
        if exponent != 0
            value *= x[variable_index] ^ exponent
        end
    end
    return value
end

@inline function _partial_monomial_value(x, exponents::AbstractMatrix{Int}, row::Int, variable_index::Int)
    value = one(x[1])
    for j in 1:length(x)
        exponent = exponents[row, j]
        if j == variable_index
            exponent -= 1
        end
        if exponent != 0
            value *= x[j] ^ exponent
        end
    end
    return value
end

function _coefficient_like_x(coeff, x0, scale)
    return convert(typeof(x0), scale * coeff)
end

function evaluate_system!(
    out,
    spec::PolynomialSystemSpec,
    coefficients::AbstractVector{<:Complex},
    x;
    coefficient_scale = 1,
)
    length(out) == spec.nvars || throw(ArgumentError("out must have length nvars."))
    length(x) == spec.nvars || throw(ArgumentError("x must have length nvars."))
    length(coefficients) == ncoeffs(spec) || throw(ArgumentError("coefficients must have length M."))

    for equation_index in 1:spec.nvars
        value = zero(x[1])
        block_start = spec.offsets[equation_index]
        block_stop = spec.offsets[equation_index + 1] - 1
        for q in block_start:block_stop
            coeff = _coefficient_like_x(coefficients[q], x[1], coefficient_scale)
            value += coeff * _monomial_value(x, spec.exponents, q)
        end
        out[equation_index] = value
    end
    return out
end

function evaluate_jacobian!(
    jacobian,
    spec::PolynomialSystemSpec,
    coefficients::AbstractVector{<:Complex},
    x;
    coefficient_scale = 1,
)
    size(jacobian) == (spec.nvars, spec.nvars) ||
        throw(ArgumentError("jacobian must have shape (nvars, nvars)."))
    length(x) == spec.nvars || throw(ArgumentError("x must have length nvars."))
    length(coefficients) == ncoeffs(spec) || throw(ArgumentError("coefficients must have length M."))

    for equation_index in 1:spec.nvars, variable_index in 1:spec.nvars
        value = zero(x[1])
        block_start = spec.offsets[equation_index]
        block_stop = spec.offsets[equation_index + 1] - 1
        for q in block_start:block_stop
            exponent = spec.exponents[q, variable_index]
            exponent == 0 && continue
            coeff = _coefficient_like_x(coefficients[q], x[1], coefficient_scale)
            value += coeff * exponent * _partial_monomial_value(x, spec.exponents, q, variable_index)
        end
        jacobian[equation_index, variable_index] = value
    end
    return jacobian
end

function evaluate_system_and_jacobian!(
    out,
    jacobian,
    spec::PolynomialSystemSpec,
    coefficients::AbstractVector{<:Complex},
    x;
    coefficient_scale = 1,
)
    evaluate_system!(out, spec, coefficients, x; coefficient_scale = coefficient_scale)
    evaluate_jacobian!(jacobian, spec, coefficients, x; coefficient_scale = coefficient_scale)
    return out, jacobian
end

Base.size(H::BezierPhamHomotopy) = (H.spec.nvars, H.spec.nvars)
ModelKit.variables(H::BezierPhamHomotopy) = H.variables_cache
ModelKit.parameters(::BezierPhamHomotopy) = Variable[]

function ModelKit.evaluate!(out, H::BezierPhamHomotopy, x, t, p = nothing)
    tau = 1.0 - Float64(real(t))
    coeffs = coefficient_vector!(H.coefficient_buffers[1], H, tau, 0)
    evaluate_system!(out, H.spec, coeffs, x)
    return nothing
end

function ModelKit.evaluate_and_jacobian!(out, jacobian, H::BezierPhamHomotopy, x, t, p = nothing)
    tau = 1.0 - Float64(real(t))
    coeffs = coefficient_vector!(H.coefficient_buffers[1], H, tau, 0)
    evaluate_system_and_jacobian!(out, jacobian, H.spec, coeffs, x)
    return nothing
end

function _constant_term(value)
    value isa Number && return value
    try
        return value[0]
    catch
        return value[1]
    end
end

function ModelKit.taylor!(out, ::Val{k}, H::BezierPhamHomotopy, tx, t) where {k}
    if k > H.max_derivative_order
        fill!(out, 0.0 + 0.0im)
        return out
    end
    tau = 1.0 - Float64(real(t))
    coeffs = coefficient_vector!(H.coefficient_buffers[k + 1], H, tau, k)
    x = [_constant_term(tx[i]) for i in 1:H.spec.nvars]
    scale = isodd(k) ? -1 : 1
    evaluate_system!(out, H.spec, coeffs, x; coefficient_scale = scale)
    return out
end
