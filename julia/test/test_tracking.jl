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
