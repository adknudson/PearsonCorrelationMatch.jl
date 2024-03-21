using Test
using PearsonCorrelationMatch
using PearsonCorrelationMatch: _generate_coefs, _Gn0d, _Gn0m
using PearsonCorrelationMatch: _hermite, _hermite_normpdf
using PearsonCorrelationMatch: _is_real, _real_roots, _feasible_roots, _nearest_root, _best_root
using Polynomials: coeffs, fromroots
using Distributions


@testset "Internals" begin
    @testset "Generate Coefficients" begin
        dA = Binomial(20, 0.2)
        dB = NegativeBinomial(20, 0.002)
        dC = LogitNormal(3, 1)
        dD = Beta(5, 3)

        @test_nowarn _generate_coefs(dA, 7)
        @test_nowarn _generate_coefs(dB, 7)
        @test_nowarn _generate_coefs(dC, 7)
        @test_nowarn _generate_coefs(dD, 7)
    end

    @testset "Gn0d" begin
    end

    @testset "Gn0m" begin
    end

    @testset "Hermite Evaluation" begin
        @test typeof(_hermite(3.0,   5)) === Float64
        @test typeof(_hermite(3,     5)) === Float64
        @test typeof(_hermite(3//1,  5)) === Float64
        @test typeof(_hermite(3.0f0, 5)) === Float64

        @test typeof(_hermite_normpdf(3.0,   5)) === Float64
        @test typeof(_hermite_normpdf(3,     5)) === Float64
        @test typeof(_hermite_normpdf(3//1,  5)) === Float64
        @test typeof(_hermite_normpdf(3.0f0, 5)) === Float64
        @test typeof(_hermite_normpdf(Inf,  10)) === Float64
        @test typeof(_hermite_normpdf(-Inf, 10)) === Float64

        @test_throws InexactError _hermite(3 + 4im, 5)
        @test_throws InexactError _hermite(3.00, 5.5)

        @test _hermite_normpdf( Inf, 10) ≈ 0 atol=sqrt(eps())
        @test _hermite_normpdf(-Inf, 10) ≈ 0 atol=sqrt(eps())
        @test _hermite_normpdf(1.0,   5) ≈ 1.45182435
    end

    @testset "Root Finding" begin
        coeffs_from_roots(roots) = coeffs(fromroots(roots))

        r1 = nextfloat(-1.0)
        r2 = prevfloat(1.0)
        r3 = eps()
        r4 = 127 / 256

        P1 = coeffs_from_roots([r1, 7, 7, 8])
        P2 = coeffs_from_roots([r2, -1.14, -1.14, -1.14, -1.14, 1119])
        P3 = coeffs_from_roots([r3, 1.01, -1.01])
        P4 = coeffs_from_roots([-5, 5, r4])
        P5 = coeffs_from_roots([1.01, -1.01])
        P6 = coeffs_from_roots([-0.5, 0.5])

        # One root at -1.0
        xs = _feasible_roots(P1)
        @test only(xs) ≈ r1

        # One root at 1.0
        xs = _feasible_roots(P2)
        @test only(xs) ≈ r2

        # Roots that are just outside [-1, 1]
        xs = _feasible_roots(P3)
        @test only(xs) ≈ r3

        # Random root in (-1, 1)
        xs = _feasible_roots(P4)
        @test only(xs) ≈ r4

        # Case of two roots just outside feasible region
        xs = _feasible_roots(P5)
        @test length(xs) == 0

        # Case of multiple roots
        xs = _feasible_roots(P6)
        @test length(xs) == 2
        @test _nearest_root(-0.6, xs) ≈ -0.5
    end
end
