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

    ModelKit.evaluate!(fp, H, x, t + h2)
    ModelKit.evaluate!(f0, H, x, t)
    ModelKit.evaluate!(fm, H, x, t - h2)
    @test (fp .- 2 .* f0 .+ fm) ./ (h2^2) ≈ u2 atol=1e-4 rtol=1e-4
end
