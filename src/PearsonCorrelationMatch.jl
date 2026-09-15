module PearsonCorrelationMatch

using Distributions
using LinearAlgebra

export pearson_match, pearson_bounds

include("common.jl")
include("defaults.jl")
include("match.jl")
include("exact_match.jl")
include("bounds.jl")

function __init__()
    # Pre-warm common quadrature node counts so first calls don't hit the lock
    for m in (22, 25, 30, 40)
        get_gauss_hermite(m)
    end
    return nothing
end

end
