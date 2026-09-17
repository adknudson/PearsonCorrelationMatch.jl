"""
    default_degree(d::UnivariateDistribution)
    default_degree(d1::UnivariateDistribution, d2::UnivariateDistribution)
    default_degree(dists::Vector{<:UnivariateDistribution})

Computes the recommended polynomial expansion degree `n` based on marginal distribution types.
"""
function default_degree end

default_degree(::UnivariateDistribution) = 15
default_degree(::ContinuousUnivariateDistribution) = 15
default_degree(::DiscreteUnivariateDistribution) = 20
default_degree(d1::UnivariateDistribution, d2::UnivariateDistribution) = max(default_degree(d1), default_degree(d2))
default_degree(dists::Vector{<:UnivariateDistribution}) = maximum(default_degree, dists)

"""
    default_m(d::UnivariateDistribution)
    default_m(d1::UnivariateDistribution, d2::UnivariateDistribution)
    default_m(dists::Vector{<:UnivariateDistribution})

Computes the recommended number of Gauss-Hermite quadrature points \$m\$ based on marginal distribution types.
"""
function default_m end

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
