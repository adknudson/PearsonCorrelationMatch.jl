module PearsonCorrelationMatch

using Distributions: UnivariateDistribution as UD
using Distributions: ContinuousUnivariateDistribution as CUD
using Distributions: DiscreteUnivariateDistribution as DUD
using Distributions

using FastGaussQuadrature: gausshermite
using IrrationalConstants: sqrt2, invsqrtπ
using LinearAlgebra: eigen, Symmetric, Diagonal, lmul!, rmul!, diagind
using PolynomialRoots: roots
using SharedArrays: SharedMatrix, sdata
using StatsFuns: normcdf, normpdf, norminvcdf


export pearson_bounds, pearson_match


include("common.jl")
include("bounds.jl")
include("match.jl")
include("rules.jl")


using PrecompileTools
using Distributions: Distributions

@setup_workload begin
    # put in let-block to avoid exporting temporary variables
    # TODO: check if wrapping in let-block is an anti-pattern
    let
        p = 0.5
        D = Distributions.Gamma()
        E = Distributions.Beta(5, 3)
        F = Distributions.Binomial(100, 0.3)
        G = Distributions.NegativeBinomial(20)
        margins = [D, E, F, G]

        @compile_workload begin
            pearson_bounds(D, E)
            pearson_bounds(D, F)
            pearson_bounds(G, D)
            pearson_bounds(F, G)
            pearson_bounds(margins)

            pearson_match(p, D, E)
            pearson_match(p, D, F)
            pearson_match(p, G, D)
            pearson_match(p, F, G)
            pearson_match([1.0 0.5; 0.5 1.0], [E, G])
        end
    end
end

end
