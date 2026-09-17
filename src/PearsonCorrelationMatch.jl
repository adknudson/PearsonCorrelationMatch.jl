module PearsonCorrelationMatch

using Distributions
using LinearAlgebra

export pearson_match, pearson_bounds

# Precompute 1/k! up to a reasonable default
const GLOBAL_INV_FACTORIALS = Float64[1.0 / Float64(factorial(big(k))) for k in 1:60]

# Cache for Gauss-Hermite quadrature rules: m => (nodes, weights)
const GLOBAL_GH_CACHE = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

# Lock to ensure thread safety when expanding caches at runtime
const CACHE_LOCK = ReentrantLock()

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
