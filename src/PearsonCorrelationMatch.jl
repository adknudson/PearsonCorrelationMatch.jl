module PearsonCorrelationMatch

using Distributions
using FastGaussQuadrature: gausshermite
using IrrationalConstants: sqrt2, invsqrtπ
using PolynomialRoots: roots
using SharedArrays
using StatsFuns: normcdf, normpdf, norminvcdf
using NearestCorrelationMatrix


export pearson_bounds, pearson_match


using Reexport
@reexport using Distributions


include("common.jl")
include("pearson_bounds.jl")
include("pearson_match.jl")


using PrecompileTools
@setup_workload begin
    p = 0.5
    D = Gamma()
    E = Beta(5, 3)
    F = Binomial(100, 0.3)
    G = NegativeBinomial(20)
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
