module HomotopyPathLearning

using LinearAlgebra
using HomotopyContinuation
using HomotopyContinuation.ModelKit

include("system_spec.jl")
include("bernstein.jl")
include("start_solutions.jl")
include("bezier_homotopy.jl")
include("tracking.jl")
include("api.jl")

export PolynomialSystemSpec,
    from_python_spec,
    start_coefficients,
    bernstein_weights!,
    bernstein_weights,
    compute_forward_differences,
    pham_start_solutions,
    BezierPhamHomotopy,
    set_control_points!,
    coefficients_at,
    coefficient_vector!,
    evaluate_system!,
    evaluate_jacobian!,
    evaluate_system_and_jacobian!,
    validate_control_points,
    TrackingResult,
    track_all_paths,
    init_bezier_pham!,
    track_bezier_paths!,
    warmup!,
    clear_state!

end
