"""
    pearson_match(p::Real, d1::UnivariateDistribution, d2::UnivariateDistribution, n=7)

Compute the Pearson correlation coefficient to be used in a bivariate Gaussian copula.
"""
function pearson_match(p::Real, d1::UnivariateDistribution, d2::UnivariateDistribution, n=21)
    -1 <= p <= 1 || throw(ArgumentError("`p` must be in [-1, 1]"))
    n > 0 || throw(ArgumentError("`n` must be a positive number"))
    return _pearson_match(Float64(p), d1, d2, Int(n))
end



function _pearson_match(
    p::Float64,
    D1::ContinuousUnivariateDistribution,
    D2::ContinuousUnivariateDistribution,
    n::Int
)
    m1 = mean(D1)
    m2 = mean(D2)
    s1 = std(D1)
    s2 = std(D2)

    a = _generate_coefs(D1, n)
    b = _generate_coefs(D2, n)

    c1 = -m1 * m2
    c2 = inv(s1 * s2)

    coef = zeros(Float64, n + 1)
    for k in 1:n
        coef[k+1] = c2 * a[k+1] * b[k+1] * factorial(big(k))
    end
    coef[1] = c1 * c2 + c2 * a[1] * b[1] - p

    xs = _feasible_roots(coef)
    return _best_root(p, xs)
end


function _pearson_match(
    p::Float64,
    D1::DiscreteUnivariateDistribution,
    D2::DiscreteUnivariateDistribution,
    n::Int
)
    max1 = maximum(D1)
    max2 = maximum(D2)
    max1 = isinf(max1) ? quantile(D1, prevfloat(1.0)) : max1
    max2 = isinf(max2) ? quantile(D2, prevfloat(1.0)) : max2

    min1 = minimum(D1)
    min2 = minimum(D2)

    s1 = std(D1)
    s2 = std(D2)

    A = Int(min1):Int(max1)
    B = Int(min2):Int(max2)

    # z = Φ⁻¹[F(A)], α[0] = -Inf, β[0] = -Inf
    a = [-Inf; norminvcdf.(cdf.(D1, A))]
    b = [-Inf; norminvcdf.(cdf.(D2, B))]

    c2 = inv(s1 * s2)

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k + 1] = _Gn0d(k, A, B, a, b, c2) / factorial(big(k))
    end
    coef[1] = -p

    xs = _feasible_roots(coef)
    return _best_root(p, xs)
end


function _pearson_match(
    p::Float64,
    D1::DiscreteUnivariateDistribution,
    D2::ContinuousUnivariateDistribution,
    n::Int
)
    s1 = std(D1)
    s2 = std(D2)
    min1 = minimum(D1)
    max1 = maximum(D1)
    max1 = isinf(max1) ? quantile(D1, prevfloat(1.0)) : max1

    A = Int(min1):Int(max1)
    a = [-Inf; norminvcdf.(cdf.(D1, A))]

    c2 = inv(s1 * s2)

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k+1] = _Gn0m(k, A, a, D2, c2) / factorial(big(k))
    end
    coef[1] = -p

    xs = _feasible_roots(coef)
    return _best_root(p, xs)
end


function _pearson_match(
    p::Float64,
    d1::ContinuousUnivariateDistribution,
    d2::DiscreteUnivariateDistribution,
    n::Int
)
    return _pearson_match(p, d2, d1, n)
end



"""
    pearson_match(rho, margins, n=21)

Pairwise compute the Pearson correlation coefficient to be used in a bivariate Gaussian
copula. Ensures that the resulting matrix is a valid correlation matrix.
"""
function pearson_match(rho::AbstractMatrix{T}, margins::AbstractVector{<:UnivariateDistribution}, n=21) where {T<:Real}
    d = length(margins)
    r, s = size(rho)
    (r == s == d) || throw(DimensionMismatch("The number of margins must be the same size as the correlation matrix."))

    R = SharedMatrix{Float64}(d, d)

    # Calculate the pearson matching pairs
    Base.Threads.@threads for (i, j) in _idx_subsets2(d)
        @inbounds R[i, j] = pearson_match(rho[i,j], margins[i], margins[j], n)
    end

    _symmetric!(R)
    _set_diag1!(R)

    return nearest_cor!(sdata(R), DirectProjection())
end
