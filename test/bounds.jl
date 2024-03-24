using Test
using PearsonCorrelationMatch
using Distributions


@testset verbose=true "Correlation Bounds" begin
    @testset "Uniform-Uniform" begin
        U = Uniform(0, 1)
        Ginv(p) = asin(p / 2) * 6 / π
        pl, pu = pearson_bounds(U, U)
        @test pl ≈ Ginv(-1.0) atol=0.005
        @test pu ≈ Ginv( 1.0) atol=0.005
    end

    @testset "Uniform-Binomial" begin
        U = Uniform(0, 1)
        B = Binomial(1, 0.5)
        Ginv(p) = asin(p / sqrt(2)) * 2 * sqrt(3) / π
        pl, pu = pearson_bounds(U, B)
        @test pl ≈ Ginv(-1.0) atol=0.005
        @test pu ≈ Ginv( 1.0) atol=0.005
    end

    @testset "Uniform-Normal" begin
        U = Uniform(0, 1)
        N = Normal(0, 1)
        Ginv(p) = p / sqrt(π/3)
        pl, pu = pearson_bounds(U, N)
        @test pl ≈ Ginv(-1.0) atol=0.005
        @test pu ≈ Ginv( 1.0) atol=0.005
    end

    @testset "Binomial-Normal" begin
        B = Binomial(1, 0.5)
        N = Normal(0, 1)
        Ginv(p) = p / sqrt(π/2)
        pl, pu = pearson_bounds(B, N)
        @test pl ≈ Ginv(-1.0) atol=0.005
        @test pu ≈ Ginv( 1.0) atol=0.005
    end
end
