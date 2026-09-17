using Test
using Distributions
using PearsonCorrelationMatch

@testset verbose = true "Correlation Matching" begin
    dA = Beta(2, 3)
    dB = Binomial(2, 0.2)
    dC = Binomial(20, 0.2)

    U = Uniform(0, 1)
    B = Binomial(1, 0.5)
    N = Normal(0, 1)
    LN = LogNormal(0, 0.5)
    C = Cauchy(0, 1)

    @testset "Polynomial Approximation Pairs" begin
        @testset "Continuous-Continuous" begin
            @test pearson_match(-0.9, dA, dA) ≈ -0.914 atol = 0.005
            @test pearson_match(-0.6, dA, dA) ≈ -0.611 atol = 0.005
            @test pearson_match(-0.3, dA, dA) ≈ -0.306 atol = 0.005
            @test pearson_match(0.3, dA, dA) ≈ 0.304 atol = 0.005
            @test pearson_match(0.6, dA, dA) ≈ 0.606 atol = 0.005
            @test pearson_match(0.9, dA, dA) ≈ 0.904 atol = 0.005
        end

        @testset "Discrete-Discrete" begin
            @test pearson_match(-0.5, dB, dB) ≈ -0.937 atol = 0.005
            @test pearson_match(-0.3, dB, dB) ≈ -0.501 atol = 0.005
            @test pearson_match(-0.2, dB, dB) ≈ -0.322 atol = 0.005
            @test pearson_match(0.3, dB, dB) ≈ 0.418 atol = 0.005
            @test pearson_match(0.6, dB, dB) ≈ 0.769 atol = 0.005
            @test pearson_match(0.8, dB, dB) ≈ 0.944 atol = 0.005

            @test pearson_match(-0.9, dC, dC) ≈ -0.939 atol = 0.005
            @test pearson_match(-0.6, dC, dC) ≈ -0.624 atol = 0.005
            @test pearson_match(-0.3, dC, dC) ≈ -0.311 atol = 0.005
            @test pearson_match(0.3, dC, dC) ≈ 0.31 atol = 0.005
            @test pearson_match(0.6, dC, dC) ≈ 0.618 atol = 0.005
            @test pearson_match(0.9, dC, dC) ≈ 0.925 atol = 0.005
        end

        @testset "Mixed Continuous-Discrete" begin
            @test pearson_match(-0.7, dB, dA) ≈ -0.89 atol = 0.005
            @test pearson_match(-0.5, dB, dA) ≈ -0.632 atol = 0.005
            @test pearson_match(-0.3, dB, dA) ≈ -0.377 atol = 0.005
            @test pearson_match(0.3, dB, dA) ≈ 0.366 atol = 0.005
            @test pearson_match(0.5, dB, dA) ≈ 0.603 atol = 0.005
            @test pearson_match(0.8, dB, dA) ≈ 0.945 atol = 0.005

            @test pearson_match(-0.9, dC, dA) ≈ -0.928 atol = 0.005
            @test pearson_match(-0.6, dC, dA) ≈ -0.618 atol = 0.005
            @test pearson_match(-0.3, dC, dA) ≈ -0.309 atol = 0.005
            @test pearson_match(0.3, dC, dA) ≈ 0.308 atol = 0.005
            @test pearson_match(0.6, dC, dA) ≈ 0.613 atol = 0.005
            @test pearson_match(0.9, dC, dA) ≈ 0.916 atol = 0.005
        end
    end

    @testset "Exact Analytical Overloads" begin
        @testset "Normal-Normal" begin
            @test pearson_match(0.5, N, N) == 0.5
            @test pearson_match(-0.8, N, N) == -0.8
        end

        @testset "Uniform-Uniform" begin
            G = p -> 2 * sinpi(p / 6)
            Ginv = p -> asin(p / 2) * 6 / π
            pl, pu = clamp(Ginv(-1), -1, 1), clamp(Ginv(1), -1, 1)

            for p in range(pl, pu; length = 10)
                @test pearson_match(p, U, U) ≈ G(p) atol = 0.005
            end
        end

        @testset "Uniform-Binomial" begin
            G = p -> sqrt(2) * sinpi(p / (2 * sqrt(3)))
            Ginv = p -> asin(p / sqrt(2)) * 2 * sqrt(3) / π
            pl, pu = Ginv(-1), Ginv(1)

            for p in range(pl, pu; length = 10)
                @test pearson_match(p, U, B) ≈ G(p) atol = 0.005
            end
        end

        @testset "Uniform-Normal" begin
            G = p -> sqrt(π / 3) * p
            Ginv = p -> p / sqrt(π / 3)
            pl, pu = Ginv(-1), Ginv(1)

            for p in range(pl, pu; length = 10)
                @test pearson_match(p, U, N) ≈ G(p) atol = 0.005
            end
        end

        @testset "Binomial-Normal" begin
            G = p -> sqrt(π / 2) * p
            Ginv = p -> p / sqrt(π / 2)
            pl, pu = Ginv(-1), Ginv(1)

            for p in range(pl, pu; length = 10)
                @test pearson_match(p, B, N) ≈ G(p) atol = 0.005
            end
        end

        @testset "LogNormal-LogNormal" begin
            s1, s2 = LN.σ, LN.σ
            denom = sqrt(expm1(s1^2) * expm1(s2^2))
            target_rho = 0.3
            expected_rho_z = log(1.0 + target_rho * denom) / (s1 * s2)

            @test pearson_match(target_rho, LN, LN) ≈ expected_rho_z atol = 1.0e-12
        end
    end

    @testset "Boundary Clamping" begin
        # Target correlation exceeding theoretical bounds should return boundary values
        pl, pu = pearson_bounds(dB, dB)
        @test pearson_match(pu + 0.1, dB, dB) == 1.0
        @test pearson_match(pl - 0.1, dB, dB) == -1.0
    end

    @testset "Correlation Matrix Matching" begin
        margins = [dA, dB, dC]
        R_x = [
            1.0  -0.59  0.68
            -0.59  1.0   0.19
            0.68  0.19  1.0
        ]

        # 1. Dimension validation
        @test_throws AssertionError pearson_match(R_x[1:2, 1:2], margins)

        # 2. Matrix execution & properties
        R_z = pearson_match(R_x, margins)

        @test size(R_z) == (3, 3)
        @test isapprox(R_z, R_z') # Matrix symmetry

        # Diagonal entries must be 1.0
        for i in 1:3
            @test R_z[i, i] ≈ 1.0 atol = 1.0e-12
        end

        # Off-diagonal entries must match scalar calculations
        @test R_z[1, 2] ≈ pearson_match(R_x[1, 2], dA, dB)
        @test R_z[1, 3] ≈ pearson_match(R_x[1, 3], dA, dC)
        @test R_z[2, 3] ≈ pearson_match(R_x[2, 3], dB, dC)
    end

    @testset "Non-Finite Variance Checking" begin
        # Scalar tests
        @test_throws ArgumentError pearson_match(0.5, N, C; check_variance = true)
        @test isnan(pearson_match(0.5, N, C; check_variance = false))

        # Matrix tests
        margins = [N, C, N, N]
        R_x = [
            1.0 0.5 0.4 0.3
            0.5 1.0 0.6 0.2
            0.4 0.6 1.0 0.1
            0.3 0.2 0.1 1.0
        ]

        @test_throws ArgumentError pearson_match(R_x, margins; check_variance = true)

        R_y = pearson_match(R_x, margins; check_variance = false)

        # Diagonal values must remain 1.0
        for i in 1:4
            @test R_y[i, i] == 1.0
        end

        # Off-diagonal Cauchy entries (Row 2 & Col 2) should be NaN
        off_diag = [1, 3, 4]
        @test all(isnan, R_y[2, off_diag])
        @test all(isnan, R_y[off_diag, 2])

        # Valid pairs (non-Cauchy entries) should compute correctly
        @test R_y[1, 3] ≈ pearson_match(R_x[1, 3], N, N)
        @test R_y[1, 4] ≈ pearson_match(R_x[1, 4], N, N)
        @test R_y[3, 4] ≈ pearson_match(R_x[3, 4], N, N)
    end
end
