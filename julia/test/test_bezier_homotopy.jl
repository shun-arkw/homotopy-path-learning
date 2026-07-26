@testset "bezier homotopy endpoints and coefficients" begin
    spec = smoke_spec()
    start = start_coefficients(spec)
    target = smoke_target_coefficients()
    control_points = linear_control_points(start, target, 3)
    H = BezierPhamHomotopy(spec, 3, control_points)

    @test size(H) == (2, 2)
    @test length(ModelKit.variables(H)) == 2
    @test isempty(ModelKit.parameters(H))

    @test coefficients_at(H, 0.0) == start
    @test coefficients_at(H, 1.0) == target
    @test coefficients_at(H, 0.5) ≈ 0.5 .* start .+ 0.5 .* target

    x = ComplexF64[0.3 + 0.2im, -0.4 + 0.1im]
    out = zeros(ComplexF64, 2)

    ModelKit.evaluate!(out, H, x, 1.0 + 0.0im)
    @test out ≈ manual_system(spec, start, x)

    ModelKit.evaluate!(out, H, x, 0.0 + 0.0im)
    @test out ≈ manual_system(spec, target, x)
end

@testset "control point validation" begin
    spec = smoke_spec()
    start = start_coefficients(spec)
    target = smoke_target_coefficients()
    control_points = linear_control_points(start, target, 3)
    bad = copy(control_points)
    bad[2, spec.leading_indices[1]] = 1.0 + 0.1im

    @test_throws ArgumentError BezierPhamHomotopy(spec, 3, bad)
    @test_throws ArgumentError BezierPhamHomotopy(spec, 3, control_points[1:3, :])
end

@testset "bernstein weights and forward differences" begin
    weights = bernstein_weights(3, 0.25)
    @test sum(weights) ≈ 1.0
    @test weights ≈ [0.421875, 0.421875, 0.140625, 0.015625]

    spec = smoke_spec()
    start = start_coefficients(spec)
    target = smoke_target_coefficients()
    control_points = linear_control_points(start, target, 3)
    diffs = compute_forward_differences(control_points, 2)
    @test size(diffs[1]) == (3, 8)
    @test size(diffs[2]) == (2, 8)
    @test norm(diffs[2]) <= 1e-14
end
