using Test
using PearsonCorrelationMatch
using Distributions
using LinearAlgebra: isposdef


@testset "Pearson Correlation Matching" begin
    dA = Beta(2, 3)
    dB = Binomial(2, 0.2)
    dC = Binomial(20, 0.2)

    @testset "Continuous-Continuous" begin
        @test -0.914 ≈ pearson_match(-0.9, dA, dA, 21) atol=0.005
        @test -0.611 ≈ pearson_match(-0.6, dA, dA, 21) atol=0.005
        @test -0.306 ≈ pearson_match(-0.3, dA, dA, 21) atol=0.005
        @test  0.304 ≈ pearson_match( 0.3, dA, dA, 21) atol=0.005
        @test  0.606 ≈ pearson_match( 0.6, dA, dA, 21) atol=0.005
        @test  0.904 ≈ pearson_match( 0.9, dA, dA, 21) atol=0.005
    end

    @testset "Discrete-Discrete" begin
        @test -0.937 ≈ pearson_match(-0.5, dB, dB, 21) atol=0.005
        @test -0.501 ≈ pearson_match(-0.3, dB, dB, 21) atol=0.005
        @test -0.322 ≈ pearson_match(-0.2, dB, dB, 21) atol=0.005
        @test  0.418 ≈ pearson_match( 0.3, dB, dB, 21) atol=0.005
        @test  0.769 ≈ pearson_match( 0.6, dB, dB, 21) atol=0.005
        @test  0.944 ≈ pearson_match( 0.8, dB, dB, 21) atol=0.005

        @test -0.939 ≈ pearson_match(-0.9, dC, dC, 21) atol=0.005
        @test -0.624 ≈ pearson_match(-0.6, dC, dC, 21) atol=0.005
        @test -0.311 ≈ pearson_match(-0.3, dC, dC, 21) atol=0.005
        @test  0.310 ≈ pearson_match( 0.3, dC, dC, 21) atol=0.005
        @test  0.618 ≈ pearson_match( 0.6, dC, dC, 21) atol=0.005
        @test  0.925 ≈ pearson_match( 0.9, dC, dC, 21) atol=0.005
    end

    @testset "Mixed" begin
        @test -0.890 ≈ pearson_match(-0.7, dB, dA, 21) atol=0.005
        @test -0.632 ≈ pearson_match(-0.5, dB, dA, 21) atol=0.005
        @test -0.377 ≈ pearson_match(-0.3, dB, dA, 21) atol=0.005
        @test  0.366 ≈ pearson_match( 0.3, dB, dA, 21) atol=0.005
        @test  0.603 ≈ pearson_match( 0.5, dB, dA, 21) atol=0.005
        @test  0.945 ≈ pearson_match( 0.8, dB, dA, 21) atol=0.005

        @test -0.928 ≈ pearson_match(-0.9, dC, dA, 21) atol=0.005
        @test -0.618 ≈ pearson_match(-0.6, dC, dA, 21) atol=0.005
        @test -0.309 ≈ pearson_match(-0.3, dC, dA, 21) atol=0.005
        @test  0.308 ≈ pearson_match( 0.3, dC, dA, 21) atol=0.005
        @test  0.613 ≈ pearson_match( 0.6, dC, dA, 21) atol=0.005
        @test  0.916 ≈ pearson_match( 0.9, dC, dA, 21) atol=0.005
    end

    @testset "Uniform-Uniform" begin
        U = Uniform(0, 1)

        G(p) = 2 * sinpi(p / 6)
        Ginv(p) = asin(p / 2) * 6 / π

        pl = clamp(Ginv(-1), -1, 1)
        pu = clamp(Ginv(1), -1, 1)

        for p in range(pl, pu; length=10)
            @test pearson_match(p, U, U, 21) ≈ G(p) atol=0.005
        end
    end

    @testset "Uniform-Binomial" begin
        U = Uniform(0, 1)
        B = Binomial(1, 0.5)

        G(p) = sqrt(2) * sinpi(p / (2 * sqrt(3)))
        Ginv(p) = asin(p / sqrt(2)) * 2 * sqrt(3) / π

        pl, pu = Ginv(-1), Ginv(1)

        for p in range(pl, pu; length=10)
            @test pearson_match(p, U, B, 21) ≈ G(p) atol=0.005
        end
    end

    @testset "Uniform-Normal" begin
        U = Uniform(0, 1)
        N = Normal(0, 1)

        G(p) = sqrt(π/3) * p
        Ginv(p) = p / sqrt(π/3)

        pl, pu = Ginv(-1), Ginv(1)

        for p in range(pl, pu; length=10)
            @test pearson_match(p, U, N, 21) ≈ G(p) atol=0.005
        end
    end

    @testset "Binomial-Normal" begin
        B = Binomial(1, 0.5)
        N = Normal(0, 1)

        G(p) = sqrt(π/2) * p
        Ginv(p) = p / sqrt(π/2)

        pl, pu = Ginv(-1), Ginv(1)

        for p in range(pl, pu; length=10)
            @test pearson_match(p, B, N, 21) ≈ G(p) atol=0.005
        end
    end

    @testset "Match Correlation Matrix" begin
        margins = (dA, dB, dC)

        r0 = [
             1.09 -0.59 0.68
            -0.59  1.00 0.19
             0.68  0.19 1.00
        ]

        r = pearson_match(r0, margins)

        @test isposdef(r)
    end
end
