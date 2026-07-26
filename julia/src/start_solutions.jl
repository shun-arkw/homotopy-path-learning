"""Start solutions for the Pham start system G_i(x)=x_i^d_i-1."""

function _validate_degrees_for_starts(degrees::AbstractVector{<:Integer})
    degree_vec = Int.(collect(degrees))
    isempty(degree_vec) && throw(ArgumentError("degrees must not be empty."))
    all(d -> d > 0, degree_vec) || throw(ArgumentError("degrees must be positive."))
    return degree_vec
end

function pham_start_solutions(degrees::AbstractVector{<:Integer})
    degree_vec = _validate_degrees_for_starts(degrees)
    nvars = length(degree_vec)
    npaths = prod(degree_vec)
    roots_by_variable = Vector{Vector{ComplexF64}}(undef, nvars)

    for variable_index in 1:nvars
        degree = degree_vec[variable_index]
        roots = Vector{ComplexF64}(undef, degree)
        for k in 0:(degree - 1)
            roots[k + 1] = exp((2.0 * pi * im * k) / degree)
        end
        roots_by_variable[variable_index] = roots
    end

    solutions = Matrix{ComplexF64}(undef, npaths, nvars)
    row = 1
    for root_tuple in Iterators.product(roots_by_variable...)
        for variable_index in 1:nvars
            solutions[row, variable_index] = root_tuple[variable_index]
        end
        row += 1
    end
    return solutions
end

pham_start_solutions(spec::PolynomialSystemSpec) = pham_start_solutions(spec.degrees)
