@testset "taylor coefficients in HC t direction" begin
    spec = smoke_spec()
    start = start_coefficients(spec)
    target = smoke_target_coefficients()
    H = BezierPhamHomotopy(spec, 3, curved_control_points(start, target, 3))

    x = ComplexF64[0.3 + 0.2im, -0.4 + 0.1im]
    t = 0.37 + 0.0im
    h1 = 1e-6
    h2 = 1e-4

    u1 = zeros(ComplexF64, 2)
    u2 = zeros(ComplexF64, 2)
    ModelKit.taylor!(u1, Val(1), H, x, t)
    ModelKit.taylor!(u2, Val(2), H, x, t)

    fp = zeros(ComplexF64, 2)
    fm = zeros(ComplexF64, 2)
    f0 = zeros(ComplexF64, 2)
    ModelKit.evaluate!(fp, H, x, t + h1)
    ModelKit.evaluate!(fm, H, x, t - h1)
    @test (fp .- fm) ./ (2h1) ≈ u1 atol=1e-5 rtol=1e-5

    # With x fixed, the order-2 Taylor coefficient is half the second t-derivative.
    ModelKit.evaluate!(fp, H, x, t + h2)
    ModelKit.evaluate!(f0, H, x, t)
    ModelKit.evaluate!(fm, H, x, t - h2)
    @test (fp .- 2 .* f0 .+ fm) ./ (2 * h2^2) ≈ u2 atol=1e-4 rtol=1e-4
end

@testset "taylor coefficients match a ModelKit reference homotopy" begin
    for degrees in ([2, 2], [3, 3], [2, 2, 2], [3, 3, 3]), bezier_degree in (2, 3)
        spec = pham_test_spec(degrees)
        n = spec.nvars
        start = start_coefficients(spec)
        target = random_pham_coefficients(spec, 11 + bezier_degree + sum(degrees))
        control_points = random_control_points(spec, start, target, bezier_degree, 7 + n)
        H = BezierPhamHomotopy(spec, bezier_degree, control_points; max_derivative_order = 3)
        reference = reference_bezier_homotopy(spec, control_points, bezier_degree)

        x = ComplexF64[0.3 - 0.2im, -0.5 + 0.4im, 0.7 + 0.1im][1:n]
        t = 0.41 + 0.0im
        for k in 1:3
            tx = if k == 1
                x
            else
                data = zeros(ComplexF64, k, n)
                data[1, :] .= x
                for order in 2:k, i in 1:n
                    data[order, i] = complex(0.1 * order + 0.05 * i, -0.07 * i + 0.02 * order)
                end
                ModelKit.TaylorVector{k}(data)
            end
            ours = zeros(ComplexF64, n)
            expected = zeros(ComplexF64, n)
            ModelKit.taylor!(ours, Val(k), H, tx, t)
            ModelKit.taylor!(expected, Val(k), reference, tx, t)
            @test isapprox(ours, expected; rtol = 1e-10, atol = 1e-12)
        end
    end
end

@testset "taylor orders above max_derivative_order are zero" begin
    spec = smoke_spec()
    start = start_coefficients(spec)
    target = smoke_target_coefficients()
    H = BezierPhamHomotopy(spec, 2, curved_control_points(start, target, 2); max_derivative_order = 1)
    data = ComplexF64[0.3 + 0.2im -0.4 + 0.1im; 0.1 0.2im]
    u = ones(ComplexF64, 2)
    ModelKit.taylor!(u, Val(2), H, ModelKit.TaylorVector{2}(data), 0.5 + 0.0im)
    @test all(iszero, u)
    @test_throws ArgumentError BezierPhamHomotopy(
        spec,
        2,
        curved_control_points(start, target, 2);
        max_derivative_order = 5,
    )
end
