@testset "jacobian" begin
    spec = smoke_spec()
    start = start_coefficients(spec)
    target = smoke_target_coefficients()
    H = BezierPhamHomotopy(spec, 3, curved_control_points(start, target, 3))

    t = 0.37 + 0.0im
    tau = 1.0 - real(t)
    coeffs = coefficients_at(H, tau)
    x = ComplexF64[0.3 + 0.2im, -0.4 + 0.1im]
    out = zeros(ComplexF64, 2)
    J = zeros(ComplexF64, 2, 2)

    ModelKit.evaluate_and_jacobian!(out, J, H, x, t)
    @test out ≈ manual_system(spec, coeffs, x)
    @test J ≈ manual_jacobian(spec, coeffs, x)

    h = 1e-6
    for var in 1:2
        xp = copy(x)
        xm = copy(x)
        xp[var] += h
        xm[var] -= h
        fp = zeros(ComplexF64, 2)
        fm = zeros(ComplexF64, 2)
        ModelKit.evaluate!(fp, H, xp, t)
        ModelKit.evaluate!(fm, H, xm, t)
        @test (fp .- fm) ./ (2h) ≈ J[:, var] atol=1e-5 rtol=1e-5
    end
end
