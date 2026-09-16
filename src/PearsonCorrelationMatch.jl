module PearsonCorrelationMatch

using Distributions
using FastGaussQuadrature: gausshermite
using PolynomialRoots: roots
using StatsFuns: normcdf, normpdf, norminvcdf

export pearson_bounds, pearson_match

include("common.jl")
include("bounds.jl")
include("match.jl")
include("rules.jl")

end
