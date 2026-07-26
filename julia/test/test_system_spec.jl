@testset "system spec" begin
    spec = smoke_spec()

    @test spec.nvars == 2
    @test spec.degrees == [2, 2]
    @test size(spec.exponents) == (8, 2)
    @test spec.offsets == [1, 5, 9]
    @test spec.leading_indices == [1, 5]
    @test spec.constant_indices == [4, 8]

    @test start_coefficients(spec) == ComplexF64[
        1, 0, 0, -1, 1, 0, 0, -1
    ]

    @test_throws ArgumentError from_python_spec([2, 0], SMOKE_EXPONENTS, SMOKE_OFFSETS, SMOKE_LEADING, SMOKE_CONSTANT)
    @test_throws ArgumentError from_python_spec(SMOKE_DEGREES, [-1 0; 0 0], [0, 1, 2], [0, 1], [1, 1])
    @test_throws ArgumentError from_python_spec(SMOKE_DEGREES, SMOKE_EXPONENTS, SMOKE_OFFSETS, [1, 4], SMOKE_CONSTANT)
end
