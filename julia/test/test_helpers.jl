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
