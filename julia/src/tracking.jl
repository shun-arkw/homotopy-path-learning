"""Sequential path tracking helpers for Bezier Pham homotopies."""

struct TrackingResult
    success::Bool
    n_paths::Int
    n_success::Int
    n_failed::Int
    accepted_steps::Int
    rejected_steps::Int
    per_path_accepted_steps::Vector{Int}
    per_path_rejected_steps::Vector{Int}
    path_success::Vector{Bool}
    endpoints::Matrix{ComplexF64}
    residual_norms::Vector{Float64}
    failure_codes::Vector{String}
end

function _default_tracker_options()
    return HomotopyContinuation.TrackerOptions(
        max_steps = 50_000,
        max_step_size = 0.05,
        max_initial_step_size = 0.05,
        min_step_size = 1e-12,
        extended_precision = false,
    )
end

function _accepted_steps(path_result)
    try
        return Int(HomotopyContinuation.accepted_steps(path_result))
    catch
    end
    try
        return Int(getproperty(path_result, :accepted_steps))
    catch
    end
    return 0
end

function _rejected_steps(path_result)
    try
        return Int(HomotopyContinuation.rejected_steps(path_result))
    catch
    end
    try
        return Int(getproperty(path_result, :rejected_steps))
    catch
    end
    return 0
end

function _path_success(path_result)
    try
        return Bool(HomotopyContinuation.is_success(path_result))
    catch
    end
    try
        return getproperty(path_result, :return_code) == :success
    catch
    end
    return false
end

function _failure_code(path_result)
    try
        return string(getproperty(path_result, :return_code))
    catch
    end
    return "unknown"
end

function _endpoint(path_result, nvars::Int)
    candidates = (
        () -> HomotopyContinuation.solution(path_result),
        () -> HomotopyContinuation.solution_candidate(path_result),
        () -> HomotopyContinuation.solution_approximation(path_result),
        () -> getproperty(path_result, :solution),
    )
    for getter in candidates
        try
            value = getter()
            return ComplexF64.(collect(value))
        catch
        end
    end
    return fill(ComplexF64(NaN, NaN), nvars)
end

function _target_residual_norm(H::BezierPhamHomotopy, endpoint::AbstractVector{<:Complex})
    any(z -> !isfinite(real(z)) || !isfinite(imag(z)), endpoint) && return Inf
    coeffs = coefficient_vector!(H.coefficient_buffers[1], H, 1.0, 0)
    residual = zeros(ComplexF64, H.spec.nvars)
    evaluate_system!(residual, H.spec, coeffs, endpoint)
    return norm(residual)
end

function _homotopy_for_tracking(H::BezierPhamHomotopy)
    # HC 2.6.4's Pade predictor can become fragile on exactly linear
    # coefficient paths when higher-order forward differences are roundoff
    # sized rather than mathematically zero. First-order Taylor data is enough
    # for a correct predictor-corrector smoke implementation.
    H.max_derivative_order == 1 && return H
    return BezierPhamHomotopy(
        H.spec,
        H.bezier_degree,
        H.control_points;
        max_derivative_order = 1,
    )
end

function track_all_paths(
    H::BezierPhamHomotopy,
    starts::AbstractMatrix{<:Complex};
    tracker_options = nothing,
)
    size(starts, 2) == H.spec.nvars || throw(ArgumentError("starts must have shape (n_paths, nvars)."))

    tracking_H = _homotopy_for_tracking(H)
    npaths = size(starts, 1)
    accepted = zeros(Int, npaths)
    rejected = zeros(Int, npaths)
    ok = falses(npaths)
    endpoints = Matrix{ComplexF64}(undef, npaths, tracking_H.spec.nvars)
    residuals = zeros(Float64, npaths)
    failure_codes = Vector{String}(undef, npaths)

    opts = tracker_options === nothing ? _default_tracker_options() : tracker_options
    tracker = HomotopyContinuation.Tracker(tracking_H; options = opts)

    for path_index in 1:npaths
        start = ComplexF64.(vec(starts[path_index, :]))
        path_result = HomotopyContinuation.track(tracker, start; debug = false)
        accepted[path_index] = _accepted_steps(path_result)
        rejected[path_index] = _rejected_steps(path_result)
        ok[path_index] = _path_success(path_result)
        endpoint = _endpoint(path_result, tracking_H.spec.nvars)
        endpoints[path_index, :] .= endpoint
        residuals[path_index] = _target_residual_norm(tracking_H, endpoint)
        failure_codes[path_index] = ok[path_index] ? "" : _failure_code(path_result)
    end

    nsuccess = count(ok)
    nfailed = npaths - nsuccess
    return TrackingResult(
        all(ok),
        npaths,
        nsuccess,
        nfailed,
        sum(accepted),
        sum(rejected),
        accepted,
        rejected,
        ok,
        endpoints,
        residuals,
        failure_codes,
    )
end
