"""Polynomial-system support representation for fixed Pham-type systems."""

struct PolynomialSystemSpec
    nvars::Int
    degrees::Vector{Int}
    exponents::Matrix{Int}
    offsets::Vector{Int}
    leading_indices::Vector{Int}
    constant_indices::Vector{Int}
end

ncoeffs(spec::PolynomialSystemSpec) = size(spec.exponents, 1)

function _as_int_vector(name::AbstractString, values::AbstractVector{<:Integer})
    out = Int.(collect(values))
    isempty(out) && throw(ArgumentError("$name must not be empty."))
    return out
end

function _as_int_matrix(name::AbstractString, values::AbstractMatrix{<:Integer})
    out = Int.(collect(values))
    ndims(out) == 2 || throw(ArgumentError("$name must be a matrix."))
    return out
end

function _row_tuple(A::AbstractMatrix{Int}, row::Int)
    return Tuple(A[row, col] for col in 1:size(A, 2))
end

function _normalize_index_base(
    offsets::Vector{Int},
    leading_indices::Vector{Int},
    constant_indices::Vector{Int},
    index_base::Int,
)
    if index_base == 0
        return offsets .+ 1, leading_indices .+ 1, constant_indices .+ 1
    elseif index_base == 1
        return offsets, leading_indices, constant_indices
    end
    throw(ArgumentError("index_base must be 0 for Python-style indices or 1 for Julia-style indices."))
end

function PolynomialSystemSpec(
    degrees::AbstractVector{<:Integer},
    exponents::AbstractMatrix{<:Integer},
    offsets::AbstractVector{<:Integer},
    leading_indices::AbstractVector{<:Integer},
    constant_indices::AbstractVector{<:Integer};
    index_base::Int = 1,
)
    degree_vec = _as_int_vector("degrees", degrees)
    nvars = length(degree_vec)
    all(d -> d > 0, degree_vec) || throw(ArgumentError("degrees must contain positive integers."))

    exponent_mat = _as_int_matrix("exponents", exponents)
    size(exponent_mat, 2) == nvars || throw(ArgumentError("exponents must have shape (M, nvars)."))
    all(e -> e >= 0, exponent_mat) || throw(ArgumentError("exponents must be nonnegative."))

    offset_vec0 = _as_int_vector("offsets", offsets)
    leading_vec0 = _as_int_vector("leading_indices", leading_indices)
    constant_vec0 = _as_int_vector("constant_indices", constant_indices)
    offset_vec, leading_vec, constant_vec = _normalize_index_base(
        offset_vec0,
        leading_vec0,
        constant_vec0,
        index_base,
    )

    length(offset_vec) == nvars + 1 || throw(ArgumentError("offsets must have length nvars + 1."))
    length(leading_vec) == nvars || throw(ArgumentError("leading_indices must have length nvars."))
    length(constant_vec) == nvars || throw(ArgumentError("constant_indices must have length nvars."))

    M = size(exponent_mat, 1)
    offset_vec[1] == 1 || throw(ArgumentError("Julia-side offsets must start at 1."))
    offset_vec[end] == M + 1 || throw(ArgumentError("offsets[end] must equal M + 1."))
    all(diff(offset_vec) .> 0) || throw(ArgumentError("offsets must be strictly increasing."))
    all(i -> 1 <= i <= M, leading_vec) || throw(ArgumentError("leading_indices are out of range."))
    all(i -> 1 <= i <= M, constant_vec) || throw(ArgumentError("constant_indices are out of range."))

    spec = PolynomialSystemSpec(nvars, degree_vec, exponent_mat, offset_vec, leading_vec, constant_vec)
    validate_spec(spec)
    return spec
end

function from_python_spec(
    degrees::AbstractVector{<:Integer},
    exponents::AbstractMatrix{<:Integer},
    offsets::AbstractVector{<:Integer},
    leading_indices::AbstractVector{<:Integer},
    constant_indices::AbstractVector{<:Integer},
)
    return PolynomialSystemSpec(
        degrees,
        exponents,
        offsets,
        leading_indices,
        constant_indices;
        index_base = 0,
    )
end

function validate_spec(spec::PolynomialSystemSpec)
    M = ncoeffs(spec)
    zero_exponent = zeros(Int, spec.nvars)

    for equation_index in 1:spec.nvars
        block_start = spec.offsets[equation_index]
        block_stop = spec.offsets[equation_index + 1] - 1
        block_start <= block_stop || throw(ArgumentError("each equation must contain at least one exponent."))

        leading_exponent = zeros(Int, spec.nvars)
        leading_exponent[equation_index] = spec.degrees[equation_index]
        leading_matches = Int[]
        constant_matches = Int[]
        seen = Set{Tuple{Vararg{Int}}}()

        for q in block_start:block_stop
            exponent_tuple = _row_tuple(spec.exponents, q)
            exponent_tuple in seen && throw(ArgumentError("equation $equation_index contains duplicate exponent $exponent_tuple."))
            push!(seen, exponent_tuple)

            if all(spec.exponents[q, col] == leading_exponent[col] for col in 1:spec.nvars)
                push!(leading_matches, q)
            end
            if all(spec.exponents[q, col] == zero_exponent[col] for col in 1:spec.nvars)
                push!(constant_matches, q)
            end
        end

        length(leading_matches) == 1 || throw(ArgumentError("equation $equation_index must contain its leading monomial exactly once."))
        spec.leading_indices[equation_index] == leading_matches[1] ||
            throw(ArgumentError("leading_indices[$equation_index] does not point to the leading monomial."))

        length(constant_matches) == 1 || throw(ArgumentError("equation $equation_index must contain the constant term exactly once."))
        spec.constant_indices[equation_index] == constant_matches[1] ||
            throw(ArgumentError("constant_indices[$equation_index] does not point to the constant term."))

        for q in block_start:block_stop
            q == spec.leading_indices[equation_index] && continue
            total_degree = sum(spec.exponents[q, :])
            total_degree < spec.degrees[equation_index] ||
                throw(ArgumentError("non-leading exponent in equation $equation_index must have total degree < d_i."))
        end
    end
    return spec
end

function start_coefficients(spec::PolynomialSystemSpec)
    coeffs = zeros(ComplexF64, ncoeffs(spec))
    coeffs[spec.leading_indices] .= 1.0 + 0.0im
    coeffs[spec.constant_indices] .= -1.0 + 0.0im
    return coeffs
end
