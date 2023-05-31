module PearsonCorrelationMatch

using Distributions
using FastGaussQuadrature: gausshermite
using PolynomialRoots: roots
using StatsFuns: normcdf, normpdf, norminvcdf
using StatsFuns: sqrt2, invπ

const invsqrtπ = sqrt(invπ)

using Reexport
@reexport using Distributions

using PrecompileTools


export pearson_bounds, pearson_match


include("common.jl")
include("pearson_bounds.jl")
include("pearson_match.jl")


@setup_workload begin
    p = 0.5
    D = Gamma()
    E = Beta(5, 3)
    F = Binomial(100, 0.3)
    G = NegativeBinomial(20)

    @compile_workload begin
        pearson_bounds(D, E)

        pearson_match(p, D, E)
        pearson_match(p, D, F)
        pearson_match(p, F, G)
        pearson_match(p, G, D)
    end
end

end
