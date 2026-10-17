"""Public Julia API boundary for Python callers."""

mutable struct BezierPhamState
    spec::PolynomialSystemSpec
    bezier_degree::Int
    homotopy::BezierPhamHomotopy
    starts::Matrix{ComplexF64}
    tracker_options::Any
    predictor_order::Int
end

const _BEZIER_PHAM_STATE = Ref{Union{Nothing, BezierPhamState}}(nothing)

"""
    make_tracker_options(; max_steps, max_step_size, max_initial_step_size, min_step_size,
                         extended_precision, a, beta_a, beta_omega_p, beta_tau,
                         strict_beta_tau, min_newton_iters)

Build `HomotopyContinuation.TrackerOptions`. The last six keywords are the fields of
`HomotopyContinuation.TrackerParameters` (`a`, `β_a`, `β_ω_p`, `β_τ`, `strict_β_τ`,
`min_newton_iters`); their defaults are those of `DEFAULT_TRACKER_PARAMETERS`, and
`strict_beta_tau = nothing` means HomotopyContinuation.jl's rule `min(0.75 β_τ, 0.4)`.
"""
function make_tracker_options(;
    max_steps::Integer = 50_000,
    max_step_size::Real = 0.05,
    max_initial_step_size::Real = 0.05,
    min_step_size::Real = 1e-12,
    extended_precision::Bool = false,
    a::Real = 0.125,
    beta_a::Real = 1.0,
    beta_omega_p::Real = 3.0,
    beta_tau::Real = 0.4,
    strict_beta_tau::Union{Nothing,Real} = nothing,
    min_newton_iters::Integer = 2,
)
    max_steps > 0 || throw(ArgumentError("max_steps must be positive."))
    all(isfinite, (max_step_size, max_initial_step_size, min_step_size)) ||
        throw(ArgumentError("step sizes must be finite."))
    max_step_size > 0 || throw(ArgumentError("max_step_size must be positive."))
    max_initial_step_size > 0 || throw(ArgumentError("max_initial_step_size must be positive."))
    min_step_size > 0 || throw(ArgumentError("min_step_size must be positive."))
    min_step_size <= max_step_size || throw(ArgumentError("min_step_size must be <= max_step_size."))
    min_step_size <= max_initial_step_size ||
        throw(ArgumentError("min_step_size must be <= max_initial_step_size."))
    all(isfinite, (a, beta_a, beta_omega_p, beta_tau)) ||
        throw(ArgumentError("tracker parameters must be finite."))
    0 < a < 1 || throw(ArgumentError("a must satisfy 0 < a < 1."))
    beta_a > 0 || throw(ArgumentError("beta_a must be positive."))
    beta_omega_p > 0 || throw(ArgumentError("beta_omega_p must be positive."))
    0 < beta_tau <= 1 || throw(ArgumentError("beta_tau must satisfy 0 < beta_tau <= 1."))
    min_newton_iters > 0 || throw(ArgumentError("min_newton_iters must be positive."))

    parameters = if strict_beta_tau === nothing
        HomotopyContinuation.TrackerParameters(
            a = Float64(a),
            β_a = Float64(beta_a),
            β_ω_p = Float64(beta_omega_p),
            β_τ = Float64(beta_tau),
            min_newton_iters = Int(min_newton_iters),
        )
    else
        (isfinite(strict_beta_tau) && 0 < strict_beta_tau <= 1) ||
            throw(ArgumentError("strict_beta_tau must satisfy 0 < strict_beta_tau <= 1."))
        HomotopyContinuation.TrackerParameters(
            a = Float64(a),
            β_a = Float64(beta_a),
            β_ω_p = Float64(beta_omega_p),
            β_τ = Float64(beta_tau),
            strict_β_τ = Float64(strict_beta_tau),
            min_newton_iters = Int(min_newton_iters),
        )
    end

    return HomotopyContinuation.TrackerOptions(
        max_steps = Int(max_steps),
        max_step_size = Float64(max_step_size),
        max_initial_step_size = Float64(max_initial_step_size),
        min_step_size = Float64(min_step_size),
        extended_precision = extended_precision,
        parameters = parameters,
    )
end

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
    predictor_order::Integer = 1,
)
    spec = from_python_spec(degrees, exponents, offsets, leading_indices, constant_indices)
    db = Int(bezier_degree)
    db >= 1 || throw(ArgumentError("bezier_degree must be positive."))
    start = start_coefficients(spec)
    control_points = _linear_control_points(start, start, db)
    homotopy = BezierPhamHomotopy(spec, db, control_points)
    starts = pham_start_solutions(spec)
    opts = tracker_options === nothing ? _default_tracker_options() : tracker_options
    order = _check_predictor_order(predictor_order)
    _BEZIER_PHAM_STATE[] = BezierPhamState(spec, db, homotopy, starts, opts, order)
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
    return track_all_paths(
        state.homotopy,
        state.starts;
        tracker_options = opts,
        predictor_order = state.predictor_order,
    )
end

function warmup!()
    state = _require_state()
    return track_all_paths(
        state.homotopy,
        state.starts;
        tracker_options = state.tracker_options,
        predictor_order = state.predictor_order,
    )
end

function clear_state!()
    _BEZIER_PHAM_STATE[] = nothing
    return nothing
end

bezier_pham_state_initialized() = _BEZIER_PHAM_STATE[] !== nothing

function bezier_pham_state_snapshot()
    state = _require_state()
    return (
        initialized = true,
        nvars = state.spec.nvars,
        degrees = copy(state.spec.degrees),
        exponents = copy(state.spec.exponents),
        offsets = copy(state.spec.offsets),
        leading_indices = copy(state.spec.leading_indices),
        constant_indices = copy(state.spec.constant_indices),
        bezier_degree = state.bezier_degree,
        control_points = copy(state.homotopy.control_points),
        starts = copy(state.starts),
        predictor_order = state.predictor_order,
    )
end

function tracking_result_payload(result::TrackingResult)
    return (
        success = result.success,
        n_paths = result.n_paths,
        n_success = result.n_success,
        n_failed = result.n_failed,
        accepted_steps = result.accepted_steps,
        rejected_steps = result.rejected_steps,
        per_path_accepted_steps = copy(result.per_path_accepted_steps),
        per_path_rejected_steps = copy(result.per_path_rejected_steps),
        path_success = copy(result.path_success),
        endpoints = copy(result.endpoints),
        residual_norms = copy(result.residual_norms),
        failure_codes = Tuple(result.failure_codes),
    )
end
