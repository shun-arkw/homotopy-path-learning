"""Bernstein basis and Bezier finite-difference helpers."""

function falling_factorial(n::Int, k::Int)
    k < 0 && throw(ArgumentError("k must be nonnegative."))
    k > n && return 0.0
    value = 1.0
    for i in 0:(k - 1)
        value *= n - i
    end
    return value
end

function bernstein_weights!(weights::AbstractVector{Float64}, tau::Real)
    degree = length(weights) - 1
    degree >= 0 || throw(ArgumentError("weights must not be empty."))
    τ = Float64(tau)
    isfinite(τ) || throw(ArgumentError("tau must be finite."))

    u = 1.0 - τ
    fill!(weights, 0.0)
    weights[1] = 1.0
    for j in 1:degree
        for i in (j + 1):-1:2
            weights[i] = u * weights[i] + τ * weights[i - 1]
        end
        weights[1] = u * weights[1]
    end
    return weights
end

function bernstein_weights(degree::Int, tau::Real)
    degree >= 0 || throw(ArgumentError("degree must be nonnegative."))
    weights = zeros(Float64, degree + 1)
    return bernstein_weights!(weights, tau)
end

function compute_forward_differences(control_points::AbstractMatrix{<:Complex}, max_order::Int)
    max_order >= 0 || throw(ArgumentError("max_order must be nonnegative."))
    bezier_degree = size(control_points, 1) - 1
    max_order <= bezier_degree || throw(ArgumentError("max_order cannot exceed bezier_degree."))
    M = size(control_points, 2)
    diffs = Vector{Matrix{ComplexF64}}(undef, max_order)

    if max_order >= 1
        diffs[1] = zeros(ComplexF64, bezier_degree, M)
        for q in 1:M, row in 1:bezier_degree
            diffs[1][row, q] = control_points[row + 1, q] - control_points[row, q]
        end
    end

    for order in 2:max_order
        rows = bezier_degree + 1 - order
        diffs[order] = zeros(ComplexF64, rows, M)
        previous = diffs[order - 1]
        for q in 1:M, row in 1:rows
            diffs[order][row, q] = previous[row + 1, q] - previous[row, q]
        end
    end

    return diffs
end
