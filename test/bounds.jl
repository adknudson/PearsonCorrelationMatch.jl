using Test
using Distributions
using PearsonCorrelationMatch

@testset verbose = true "Correlation Bounds" begin
    U = Uniform(0, 1)
    B = Binomial(1, 0.5)
    N = Normal(0, 1)
    C = Cauchy(0, 1)

    @testset "Pairwise Exact Analytical Bounds" begin
        @testset "Uniform-Uniform" begin
            Ginv = p -> asin(p / 2) * 6 / π
            pl, pu = pearson_bounds(U, U)
            @test isapprox(pl, Ginv(-1.0), atol = 1.0e-4)
            @test isapprox(pu, Ginv(1.0), atol = 1.0e-4)
        end

        @testset "Uniform-Binomial" begin
            Ginv = p -> asin(p / sqrt(2)) * 2 * sqrt(3) / π
            pl, pu = pearson_bounds(U, B)
            @test isapprox(pl, Ginv(-1.0), atol = 1.0e-4)
            @test isapprox(pu, Ginv(1.0), atol = 1.0e-4)
        end

        @testset "Uniform-Normal" begin
            Ginv = p -> p / sqrt(π / 3)
            pl, pu = pearson_bounds(U, N)
            @test isapprox(pl, Ginv(-1.0), atol = 1.0e-4)
            @test isapprox(pu, Ginv(1.0), atol = 1.0e-4)
        end

        @testset "Binomial-Normal" begin
            Ginv = p -> p / sqrt(π / 2)
            pl, pu = pearson_bounds(B, N)
            @test isapprox(pl, Ginv(-1.0), atol = 1.0e-4)
            @test isapprox(pu, Ginv(1.0), atol = 1.0e-4)
        end
    end

    @testset "Matrix Pearson Bounds" begin
        dists = [U, B, N]
        bounds = pearson_bounds(dists)

        # 1. Structure & Dimensions
        @test bounds isa NamedTuple
        @test haskey(bounds, :lower) && haskey(bounds, :upper)
        @test size(bounds.lower) == (3, 3)
        @test size(bounds.upper) == (3, 3)

        # 2. Symmetry
        @test isapprox(bounds.lower, bounds.lower')
        @test isapprox(bounds.upper, bounds.upper')

        # 3. Admissible Bound Ranges [-1, 1]
        @test all(-1.0 .<= bounds.lower .<= 1.0)
        @test all(-1.0 .<= bounds.upper .<= 1.0)

        # 4. Matrix Entries Match Pairwise Scalar Functions
        for i in 1:3, j in 1:3
            pl, pu = pearson_bounds(dists[i], dists[j])
            @test isapprox(bounds.lower[i, j], pl, atol = 1.0e-6)
            @test isapprox(bounds.upper[i, j], pu, atol = 1.0e-6)
        end
    end

    @testset "Variance Checking & Non-finite Variances" begin
        # Scalar tests
        @test_throws ArgumentError pearson_bounds(C, N; check_variance = true)

        pl_nan, pu_nan = pearson_bounds(C, N; check_variance = false)
        @test isnan(pl_nan) && isnan(pu_nan)

        # Matrix tests
        dists_with_cauchy = [U, C, N]

        @test_throws ArgumentError pearson_bounds(dists_with_cauchy; check_variance = true)

        bounds_nan = pearson_bounds(dists_with_cauchy; check_variance = false)

        # Row 2 & Col 2 (Cauchy entries) should propagate NaNs
        @test all(isnan, bounds_nan.lower[2, :])
        @test all(isnan, bounds_nan.lower[:, 2])
        @test all(isnan, bounds_nan.upper[2, :])
        @test all(isnan, bounds_nan.upper[:, 2])

        # Valid entries (1, 3) should remain correct
        pl13, pu13 = pearson_bounds(U, N)
        @test isapprox(bounds_nan.lower[1, 3], pl13, atol = 1.0e-6)
        @test isapprox(bounds_nan.upper[1, 3], pu13, atol = 1.0e-6)
    end
end
