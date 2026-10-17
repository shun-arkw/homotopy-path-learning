@testset "sequential tracking smoke" begin
    spec = smoke_spec()
    start = start_coefficients(spec)
    target = diagonal_target_coefficients()
    control_points = linear_control_points(start, target, 3)
    H = BezierPhamHomotopy(spec, 3, control_points)
    starts = pham_start_solutions(spec)

    result = track_all_paths(H, starts)
    @test result.n_paths == 4
    @test result.n_success == 4
    @test result.n_failed == 0
    @test result.success
    @test all(result.path_success)
    @test maximum(result.residual_norms) < 1e-8
end

@testset "public API tracking smoke" begin
    clear_state!()
    @test !bezier_pham_state_initialized()
    options = make_tracker_options()
    init_bezier_pham!(
        SMOKE_DEGREES,
        SMOKE_EXPONENTS,
        SMOKE_OFFSETS,
        SMOKE_LEADING,
        SMOKE_CONSTANT,
        3,
        tracker_options = options,
    )
    @test bezier_pham_state_initialized()
    snapshot = bezier_pham_state_snapshot()
    @test snapshot.offsets == [1, 5, 9]
    @test snapshot.leading_indices == [1, 5]
    @test snapshot.constant_indices == [4, 8]
    @test size(snapshot.control_points) == (4, 8)

    spec = smoke_spec()
    control_points = linear_control_points(start_coefficients(spec), diagonal_target_coefficients(), 3)
    result = track_bezier_paths!(control_points)
    payload = tracking_result_payload(result)

    @test payload.n_paths == 4
    @test payload.success
    @test size(payload.endpoints) == (4, 2)
    @test maximum(payload.residual_norms) < 1e-8
    clear_state!()
    @test !bezier_pham_state_initialized()
end

function _sorted_endpoints(endpoints)
    rows = [endpoints[i, :] for i in 1:size(endpoints, 1)]
    return sort(rows; by = r -> (round(real(r[1]); digits = 6), round(imag(r[1]); digits = 6),
                                 round(real(r[end]); digits = 6), round(imag(r[end]); digits = 6)))
end

@testset "predictor order 3 finds the same solutions" begin
    for degrees in ([2, 2], [3, 3], [2, 2, 2])
        spec = pham_test_spec(degrees)
        start = start_coefficients(spec)
        target = random_pham_coefficients(spec, 5)
        control_points = random_control_points(spec, start, target, 3, 3)
        H = BezierPhamHomotopy(spec, 3, control_points)
        starts = pham_start_solutions(spec)

        first_order = track_all_paths(H, starts)
        third_order = track_all_paths(H, starts; predictor_order = 3)
        @test first_order.success
        @test third_order.success
        @test maximum(third_order.residual_norms) < 1e-8
        a = _sorted_endpoints(first_order.endpoints)
        b = _sorted_endpoints(third_order.endpoints)
        @test all(isapprox(a[i], b[i]; atol = 1e-8) for i in eachindex(a))
        # all solutions are distinct
        @test minimum(norm(b[i] - b[j]) for i in eachindex(b) for j in eachindex(b) if i < j) > 1e-6
    end
    spec = smoke_spec()
    H = BezierPhamHomotopy(spec, 2, linear_control_points(start_coefficients(spec), diagonal_target_coefficients(), 2))
    @test_throws ArgumentError track_all_paths(H, pham_start_solutions(spec); predictor_order = 2)
    @test track_all_paths(H, pham_start_solutions(spec)).accepted_steps ==
          track_all_paths(H, pham_start_solutions(spec); predictor_order = 1).accepted_steps
end

@testset "tracker parameters" begin
    defaults = make_tracker_options().parameters
    reference = HomotopyContinuation.DEFAULT_TRACKER_PARAMETERS
    for field in fieldnames(HomotopyContinuation.TrackerParameters)
        @test getfield(defaults, field) == getfield(reference, field)
    end

    options = make_tracker_options(
        a = 0.2,
        beta_a = 1.5,
        beta_omega_p = 0.8,
        beta_tau = 0.85,
        strict_beta_tau = 0.8,
        min_newton_iters = 1,
    )
    p = options.parameters
    @test (p.a, p.β_a, p.β_ω_p, p.β_τ, p.strict_β_τ, p.min_newton_iters) == (0.2, 1.5, 0.8, 0.85, 0.8, 1)
    @test make_tracker_options(beta_tau = 0.75).parameters.strict_β_τ == min(0.75 * 0.75, 0.4)

    @test_throws ArgumentError make_tracker_options(a = 0.0)
    @test_throws ArgumentError make_tracker_options(a = 1.0)
    @test_throws ArgumentError make_tracker_options(beta_omega_p = -1.0)
    @test_throws ArgumentError make_tracker_options(beta_tau = 1.5)
    @test_throws ArgumentError make_tracker_options(strict_beta_tau = 0.0)
    @test_throws ArgumentError make_tracker_options(min_newton_iters = 0)
    @test_throws ArgumentError make_tracker_options(beta_a = Inf)
end

@testset "public API predictor order" begin
    clear_state!()
    spec = smoke_spec()
    control_points = linear_control_points(start_coefficients(spec), diagonal_target_coefficients(), 3)
    init_bezier_pham!(SMOKE_DEGREES, SMOKE_EXPONENTS, SMOKE_OFFSETS, SMOKE_LEADING, SMOKE_CONSTANT, 3)
    @test bezier_pham_state_snapshot().predictor_order == 1
    first_order = track_bezier_paths!(control_points)
    init_bezier_pham!(
        SMOKE_DEGREES,
        SMOKE_EXPONENTS,
        SMOKE_OFFSETS,
        SMOKE_LEADING,
        SMOKE_CONSTANT,
        3;
        predictor_order = 3,
    )
    @test bezier_pham_state_snapshot().predictor_order == 3
    third_order = track_bezier_paths!(control_points)
    @test third_order.success
    @test isapprox(_sorted_endpoints(third_order.endpoints), _sorted_endpoints(first_order.endpoints); atol = 1e-8)
    @test_throws ArgumentError init_bezier_pham!(
        SMOKE_DEGREES,
        SMOKE_EXPONENTS,
        SMOKE_OFFSETS,
        SMOKE_LEADING,
        SMOKE_CONSTANT,
        3;
        predictor_order = 4,
    )
    clear_state!()
end
