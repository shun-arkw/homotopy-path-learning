"""Public Julia API boundary for Python callers."""

mutable struct BezierPhamState
    spec::PolynomialSystemSpec
    bezier_degree::Int
    homotopy::BezierPhamHomotopy
    starts::Matrix{ComplexF64}
    tracker_options::Any
end

const _BEZIER_PHAM_STATE = Ref{Union{Nothing, BezierPhamState}}(nothing)

function _linear_control_points(start_coeffs::AbstractVector{<:Complex}, target_coeffs::AbstractVector{<:Complex}, bezier_degree::Int)
    bezier_degree >= 1 || throw(ArgumentError("bezier_degree must be positive."))
    control_points = Matrix{ComplexF64}(undef, bezier_degree + 1, length(start_coeffs))
    for row in 1:(bezier_degree + 1)
        tau = (row - 1) / bezier_degree
        control_points[row, :] .= (1.0 - tau) .* start_coeffs .+ tau .* target_coeffs
    end
    return control_points
end

function init_bezier_pham!(
    degrees,
    exponents,
    offsets,
    leading_indices,
    constant_indices,
    bezier_degree;
    tracker_options = nothing,
)
    spec = from_python_spec(degrees, exponents, offsets, leading_indices, constant_indices)
    db = Int(bezier_degree)
    db >= 1 || throw(ArgumentError("bezier_degree must be positive."))
    start = start_coefficients(spec)
    control_points = _linear_control_points(start, start, db)
    homotopy = BezierPhamHomotopy(spec, db, control_points)
    starts = pham_start_solutions(spec)
    opts = tracker_options === nothing ? _default_tracker_options() : tracker_options
    _BEZIER_PHAM_STATE[] = BezierPhamState(spec, db, homotopy, starts, opts)
    return nothing
end

function _require_state()
    state = _BEZIER_PHAM_STATE[]
    state === nothing && throw(ArgumentError("Bezier Pham state is not initialized. Call init_bezier_pham! first."))
    return state
end

function track_bezier_paths!(control_points; tracker_options = nothing)
    state = _require_state()
    set_control_points!(state.homotopy, control_points)
    opts = tracker_options === nothing ? state.tracker_options : tracker_options
    return track_all_paths(state.homotopy, state.starts; tracker_options = opts)
end

function warmup!()
    state = _require_state()
    return track_all_paths(state.homotopy, state.starts; tracker_options = state.tracker_options)
end

function clear_state!()
    _BEZIER_PHAM_STATE[] = nothing
    return nothing
end
