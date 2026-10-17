const SMOKE_DEGREES = [2, 2]
const SMOKE_EXPONENTS = [
    2 0
    1 0
    0 1
    0 0
    0 2
    1 0
    0 1
    0 0
]
const SMOKE_OFFSETS = [0, 4, 8]
const SMOKE_LEADING = [0, 4]
const SMOKE_CONSTANT = [3, 7]

function smoke_spec()
    return from_python_spec(
        SMOKE_DEGREES,
        SMOKE_EXPONENTS,
        SMOKE_OFFSETS,
        SMOKE_LEADING,
        SMOKE_CONSTANT,
    )
end

function smoke_target_coefficients()
    return ComplexF64[
        1.0 + 0.0im,
        0.25 + 0.125im,
        -0.5 + 0.75im,
        0.1 - 0.2im,
        1.0 + 0.0im,
        -0.3 + 0.4im,
        0.6 - 0.1im,
        -0.7 + 0.2im,
    ]
end

function diagonal_target_coefficients()
    coeffs = zeros(ComplexF64, 8)
    coeffs[[1, 5]] .= 1.0 + 0.0im
    coeffs[[4, 8]] .= -2.0 + 0.0im
    return coeffs
end

function linear_control_points(start_coeffs, target_coeffs, bezier_degree::Int)
    control_points = Matrix{ComplexF64}(undef, bezier_degree + 1, length(start_coeffs))
    for row in 1:(bezier_degree + 1)
        tau = (row - 1) / bezier_degree
        control_points[row, :] .= (1.0 - tau) .* start_coeffs .+ tau .* target_coeffs
    end
    return control_points
end

function curved_control_points(start_coeffs, target_coeffs, bezier_degree::Int)
    control_points = linear_control_points(start_coeffs, target_coeffs, bezier_degree)
    control_points[2, 2] += 0.2 - 0.1im
    control_points[2, 3] += -0.15 + 0.05im
    control_points[3, 6] += 0.1 + 0.2im
    control_points[3, 7] += -0.05 - 0.1im
    control_points[:, [1, 5]] .= 1.0 + 0.0im
    return control_points
end

function manual_system(spec, coeffs, x)
    out = zeros(ComplexF64, spec.nvars)
    for eq in 1:spec.nvars
        for q in spec.offsets[eq]:(spec.offsets[eq + 1] - 1)
            mon = 1.0 + 0.0im
            for j in 1:spec.nvars
                mon *= x[j] ^ spec.exponents[q, j]
            end
            out[eq] += coeffs[q] * mon
        end
    end
    return out
end

function manual_jacobian(spec, coeffs, x)
    J = zeros(ComplexF64, spec.nvars, spec.nvars)
    for eq in 1:spec.nvars, var in 1:spec.nvars
        for q in spec.offsets[eq]:(spec.offsets[eq + 1] - 1)
            exponent = spec.exponents[q, var]
            exponent == 0 && continue
            mon = 1.0 + 0.0im
            for j in 1:spec.nvars
                power = spec.exponents[q, j] - (j == var ? 1 : 0)
                mon *= x[j] ^ power
            end
            J[eq, var] += coeffs[q] * exponent * mon
        end
    end
    return J
end

# Pham spec with linear lower-order terms: per equation the leading monomial,
# x_1, ..., x_n and the constant term (the same layout as the Python side).
function pham_test_spec(degrees::Vector{Int})
    n = length(degrees)
    rows = Vector{Vector{Int}}()
    offsets = Int[0]
    leading = Int[]
    constant = Int[]
    for i in 1:n
        lead = zeros(Int, n)
        lead[i] = degrees[i]
        push!(leading, length(rows))
        push!(rows, lead)
        for j in 1:n
            e = zeros(Int, n)
            e[j] = 1
            push!(rows, e)
        end
        push!(constant, length(rows))
        push!(rows, zeros(Int, n))
        push!(offsets, length(rows))
    end
    exponents = permutedims(reduce(hcat, rows))
    return from_python_spec(degrees, exponents, offsets, leading, constant)
end

function random_pham_coefficients(spec, seed::Int)
    coeffs = zeros(ComplexF64, size(spec.exponents, 1))
    for q in eachindex(coeffs)
        a = sin(1.3 * q + seed)
        b = cos(0.7 * q - seed)
        coeffs[q] = complex(a, b)
    end
    coeffs[spec.leading_indices] .= 1.0 + 0.0im
    return coeffs
end

function random_control_points(spec, start, target, bezier_degree::Int, seed::Int)
    control_points = linear_control_points(start, target, bezier_degree)
    for row in 2:bezier_degree, q in axes(control_points, 2)
        control_points[row, q] += 0.2 * complex(sin(seed + 2.1 * row + q), cos(seed - row + 0.3 * q))
    end
    control_points[:, spec.leading_indices] .= 1.0 + 0.0im
    return control_points
end

# The same homotopy built symbolically with ModelKit (t = 1 at the start system).
function reference_bezier_homotopy(spec, control_points, bezier_degree::Int)
    n = spec.nvars
    @var x[1:n] t
    τ = 1 - t
    equations = Expression[]
    for i in 1:n
        f = Expression(0)
        for q in spec.offsets[i]:(spec.offsets[i + 1] - 1)
            c = sum(
                binomial(bezier_degree, r) * τ^r * (1 - τ)^(bezier_degree - r) * control_points[r + 1, q]
                for r in 0:bezier_degree
            )
            f += c * prod(x[j]^spec.exponents[q, j] for j in 1:n)
        end
        push!(equations, f)
    end
    return InterpretedHomotopy(Homotopy(equations, x, t))
end
