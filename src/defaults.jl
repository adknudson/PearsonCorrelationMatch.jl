"""
    default_degree(d::UnivariateDistribution)

Returns the recommended polynomial truncation degree n for a given distribution type.
"""
function default_degree end

# Fallback base method
default_degree(::UnivariateDistribution) = 15

# Continuous general fallback
default_degree(::ContinuousUnivariateDistribution) = 15

# Discrete general fallback (step discontinuities require higher order expansions)
default_degree(::DiscreteUnivariateDistribution) = 20

# --- Specialized Continuous Distributions (Heavy Tails / High Skew) ---

# LogNormal: Degree scales dynamically with the shape parameter σ
function default_degree(d::LogNormal)
    sigma = d.σ
    if sigma > 1.2
        return 30
    elseif sigma > 0.8
        return 25
    else
        return 18
    end
end

# Gamma & Weibull: High skew when shape parameter α < 1
function default_degree(d::Gamma)
    return d.α < 1.0 ? 25 : 18
end

function default_degree(d::Weibull)
    return d.α < 1.0 ? 25 : 18
end

# Extremely heavy-tailed distributions
default_degree(::Cauchy) = 30
default_degree(::Pareto) = 30

# --- Specialized Discrete Distributions ---

# Unbounded / Large Support Discrete
function default_degree(d::Poisson)
    return d.λ > 20.0 ? 22 : 18
end

default_degree(::NegativeBinomial) = 22

# Bounded / Small Support Discrete
default_degree(::Bernoulli) = 15
default_degree(::Categorical) = 15
function default_degree(d::Binomial)
    return d.n <= 10 ? 15 : 18
end

# --- Pair and Collection Dispatches ---

# Two distributions: Take the maximum requirement
default_degree(d1::UnivariateDistribution, d2::UnivariateDistribution) = max(default_degree(d1), default_degree(d2))

# Vector of distributions: Take the maximum across all marginals
default_degree(dists::Vector{<:UnivariateDistribution}) = maximum(default_degree, dists)


# ---------------------------------------------------------------------------
# 2. default_m: Quadrature Point Scaling
# ---------------------------------------------------------------------------

"""
    default_m(d::UnivariateDistribution)
    default_m(dists::Vector{<:UnivariateDistribution})

Returns the recommended number of Gauss-Hermite points m.
"""
function default_m(d::UnivariateDistribution)
    n = default_degree(d)
    return max(25, round(Int, 1.5 * n))
end

function default_m(d1::UnivariateDistribution, d2::UnivariateDistribution)
    # If both are discrete, no quadrature points needed
    if d1 isa DiscreteUnivariateDistribution && d2 isa DiscreteUnivariateDistribution
        return 0
    end
    n = default_degree(d1, d2)
    return max(25, round(Int, 1.5 * n))
end

function default_m(dists::Vector{<:UnivariateDistribution})
    # If all marginals are discrete, skip quadrature entirely
    if all(d -> d isa DiscreteUnivariateDistribution, dists)
        return 0
    end
    n = default_degree(dists)
    return max(25, round(Int, 1.5 * n))
end
