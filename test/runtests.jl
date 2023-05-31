using Test
using PearsonCorrelationMatch
using PearsonCorrelationMatch: _generate_coefs, _hermite, _hermite_normpdf, _feasible_roots, _nearest_root
using Polynomials


@testset "Utilities" begin
    @testset "Generate Coefficients" begin
        dA = Binomial(20, 0.2)
        dB = NegativeBinomial(20, 0.002)
        dC = LogitNormal(3, 1)
        dD = Beta(5, 3)

        @test_nowarn _generate_coefs(dA, 7)
        @test_nowarn _generate_coefs(dB, 7)
        @test_nowarn _generate_coefs(dC, 7)
        @test_nowarn _generate_coefs(dD, 7)

        @test_nowarn _generate_coefs(dA, 7.0)
        @test_nowarn _generate_coefs(dB, 7.0)
        @test_nowarn _generate_coefs(dC, 7.0)
        @test_nowarn _generate_coefs(dD, 7.0)

        @test_throws InexactError _generate_coefs(dA, 7.5)
        @test_throws InexactError _generate_coefs(dB, 7.5)
        @test_throws InexactError _generate_coefs(dC, 7.5)
        @test_throws InexactError _generate_coefs(dD, 7.5)
    end

    @testset "Hermite Evaluation" begin
        
    end

    @testset "Solve Polynomial on [-1, 1]" begin
        r1 = nextfloat(-1.0)
        r2 = prevfloat(1.0)
        r3 = eps()
        r4 = prevfloat(2.0) * rand() - prevfloat(1.0)

        P1 = coeffs(3 * fromroots([r1, 7, 7, 8]))
        P2 = coeffs(-5 * fromroots([r2, -1.14, -1.14, -1.14, -1.14, 1119]))
        P3 = coeffs(1.2 * fromroots([r3, nextfloat(1.0), prevfloat(-1.0)]))
        P4 = coeffs(fromroots([-5, 5, r4]))
        P5 = coeffs(fromroots([nextfloat(1.0), prevfloat(-1.0)]))
        P6 = coeffs(fromroots([-0.5, 0.5]))

        # One root at -1.0
        xs = _feasible_roots(P1)
        @test only(xs) ≈ r1 atol=0.001

        # One root at 1.0
        xs = _feasible_roots(P2)
        @test only(xs) ≈ r2 atol=0.001

        # Roots that are just outside [-1, 1]
        xs = _feasible_roots(P3)
        @test only(xs) ≈ r3 atol=0.001

        # Random root in (-1, 1)
        xs = _feasible_roots(P4)
        @test only(xs) ≈ r4 atol=0.001

        # Case of no roots
        xs = _feasible_roots(P5)
        @test length(xs) == 0
        
        # Case of multiple roots
        xs = _feasible_roots(P6)
        @test length(xs) == 2
        @test _nearest_root(-0.6, xs) ≈ -0.5 atol=0.001
    end
end


@testset "Pearson Correlation Matching" begin
    dA = Beta(2, 3)
    dB = Binomial(2, 0.2)
    dC = Binomial(20, 0.2)
    
    @testset "Continuous-Continuous" begin
        @test -0.914 ≈ pearson_match(-0.9, dA, dA) atol=0.01
        @test -0.611 ≈ pearson_match(-0.6, dA, dA) atol=0.01
        @test -0.306 ≈ pearson_match(-0.3, dA, dA) atol=0.01
        @test  0.304 ≈ pearson_match( 0.3, dA, dA) atol=0.01
        @test  0.606 ≈ pearson_match( 0.6, dA, dA) atol=0.01
        @test  0.904 ≈ pearson_match( 0.9, dA, dA) atol=0.01
    end

    @testset "Discrete-Discrete" begin
        @test -0.937 ≈ pearson_match(-0.5, dB, dB, 20) atol=0.02
        @test -0.501 ≈ pearson_match(-0.3, dB, dB) atol=0.01
        @test -0.322 ≈ pearson_match(-0.2, dB, dB) atol=0.01
        @test  0.418 ≈ pearson_match( 0.3, dB, dB) atol=0.01
        @test  0.769 ≈ pearson_match( 0.6, dB, dB) atol=0.01
        @test  0.944 ≈ pearson_match( 0.8, dB, dB, 20) atol=0.02

        @test -0.939 ≈ pearson_match(-0.9, dC, dC) atol=0.01
        @test -0.624 ≈ pearson_match(-0.6, dC, dC) atol=0.01
        @test -0.311 ≈ pearson_match(-0.3, dC, dC) atol=0.01
        @test  0.310 ≈ pearson_match( 0.3, dC, dC) atol=0.01
        @test  0.618 ≈ pearson_match( 0.6, dC, dC) atol=0.01
        @test  0.925 ≈ pearson_match( 0.9, dC, dC) atol=0.01
    end

    @testset "Mixed" begin
        @test -0.890 ≈ pearson_match(-0.7, dB, dA) atol=0.01
        @test -0.632 ≈ pearson_match(-0.5, dB, dA) atol=0.01
        @test -0.377 ≈ pearson_match(-0.3, dB, dA) atol=0.01
        @test  0.366 ≈ pearson_match( 0.3, dB, dA) atol=0.01
        @test  0.603 ≈ pearson_match( 0.5, dB, dA) atol=0.01
        @test  0.945 ≈ pearson_match( 0.8, dB, dA) atol=0.01

        @test -0.928 ≈ pearson_match(-0.9, dC, dA) atol=0.01
        @test -0.618 ≈ pearson_match(-0.6, dC, dA) atol=0.01
        @test -0.309 ≈ pearson_match(-0.3, dC, dA) atol=0.01
        @test  0.308 ≈ pearson_match( 0.3, dC, dA) atol=0.01
        @test  0.613 ≈ pearson_match( 0.6, dC, dA) atol=0.01
        @test  0.916 ≈ pearson_match( 0.9, dC, dA) atol=0.01
    end

end