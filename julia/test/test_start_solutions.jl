@testset "start solutions" begin
    spec = smoke_spec()
    starts = pham_start_solutions(spec)
    coeffs = start_coefficients(spec)

    @test size(starts) == (4, 2)
    for row in 1:size(starts, 1)
        residual = manual_system(spec, coeffs, vec(starts[row, :]))
        @test norm(residual) <= 1e-12
    end
end
