using Test
using PearsonCorrelationMatch
using PearsonCorrelationMatch: _generate_coefs, _Gn0_discrete, _Gn0_mixed
using PearsonCorrelationMatch: _hermite, _hermite_normpdf
using PearsonCorrelationMatch: _is_real, _real_roots, _feasible_roots, _nearest_root, _best_root
using Polynomials: coeffs, fromroots
using Distributions


function nothrow(f)
    try
        f()
    catch e
        println(e)
        return false
    end

    return true
end


@testset verbose=true "Internals" begin
    @testset "Generate Coefficients" begin
        dA = Binomial(20, 0.2)
        dB = NegativeBinomial(20, 0.002)
        dC = LogitNormal(3, 1)
        dD = Beta(5, 3)

        @test nothrow(() -> _generate_coefs(dA, 21, 28))
        @test nothrow(() -> _generate_coefs(dB, 21, 28))
        @test nothrow(() -> _generate_coefs(dC, 21, 28))
        @test nothrow(() -> _generate_coefs(dD, 21, 28))
    end

    @testset "Gn0 discrete" begin
    end

    @testset "Gn0 mixed" begin
    end

    @testset "Hermite Evaluation" begin
        # Want to always return a Float64
        for T in (Float16, Float32, Float64, Int, Rational{Int})
            @test _hermite(T(3), 5) isa Float64
            @test _hermite_normpdf(T(3), 5) isa Float64
        end
        @test _hermite_normpdf(Inf,  10) isa Float64
        @test _hermite_normpdf(-Inf, 10) isa Float64

        # Must only work for real numbers
        @test_throws InexactError _hermite(3 + 4im, 5)
        # `k` must be a non-negative integer
        @test_throws ArgumentError _hermite(1.0, -1)
        @test_throws InexactError _hermite(3.00, 5.5)

        # Must always return a real number even when evaluated at ±Inf
        @test _hermite_normpdf( Inf, 10) ≈ 0 atol=sqrt(eps())
        @test _hermite_normpdf(-Inf, 10) ≈ 0 atol=sqrt(eps())

        # Test for exactness against known polynomials
        He0(x) = 1.0
        He1(x) = x
        He2(x) = evalpoly(x, (-1, 0, 1))
        He3(x) = evalpoly(x, (0, -3, 0, 1))
        He4(x) = evalpoly(x, (3, 0, -6, 0, 1))
        He5(x) = evalpoly(x, (0, 15, 0, -10, 0, 1))
        He6(x) = evalpoly(x, (-15, 0, 45, 0, -15, 0, 1))
        He7(x) = evalpoly(x, (0, -105, 0, 105, 0, -21, 0, 1))
        He8(x) = evalpoly(x, (105, 0, -420, 0, 210, 0, -28, 0, 1))
        He9(x) = evalpoly(x, (0, 945, 0, -1260, 0, 378, 0, -36, 0, 1))
        He10(x)= evalpoly(x, (-945, 0, 4725, 0, -3150, 0, 630, 0, -45, 0, 1))

        @testset "Hermite Polynomial Function" for _ in 1:1000
            width = 10_000
            x = rand() * width - 0.5*width
            @test _hermite(x, 0) ≈ He0(x)
            @test _hermite(x, 1) ≈ He1(x)
            @test _hermite(x, 2) ≈ He2(x)
            @test _hermite(x, 3) ≈ He3(x)
            @test _hermite(x, 4) ≈ He4(x)
            @test _hermite(x, 5) ≈ He5(x)
            @test _hermite(x, 6) ≈ He6(x)
            @test _hermite(x, 7) ≈ He7(x)
            @test _hermite(x, 8) ≈ He8(x)
            @test _hermite(x, 9) ≈ He9(x)
            @test _hermite(x, 10) ≈ He10(x)
        end
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
