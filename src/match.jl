"""
    pearson_match(p::Real, d1, d2; n::Real=12, m::Real=128, kwargs...)

Compute the Pearson correlation coefficient to be used in a bivariate Gaussian copula.

# Fields

- `p`: The target correltation between the marginal distributions.
- `d1`: The first marginal distribution.
- `d2`: The second marginal distribution.
- `n`: The degree of the polynomial used to estimate the matching correlation.
- `m`: The number of points used in the hermite polynomial interpolation.
- `kwargs`: Additional keyword arguments. Currently unused.

# Examples

```julia-repl
julia> using Distributions

julia> d1 = Beta(2, 3); d2 = Binomial(20, 0.2);

julia> pearson_match(0.6, d1, d2)
0.6127531346934495
```
"""
function pearson_match(p::Real, d1::UD, d2::UD; n::Real=20, m::Real=128, kwargs...)
    -1 <= p <= 1 || throw(ArgumentError("`p` must be in [-1, 1]"))
    n > 0 || throw(ArgumentError("`n` must be a positive number"))
    return _invrule(Float64(p), d1, d2; n=Int(n), m=Int(m))
end

_invrule(p::Float64, d1::UD, d2::UD; kwargs...) = _invrule_fallback(p, d1, d2; kwargs...)

"""
    _invrule_fallback(p::Float64, d1, d2; n::Int, m::Int, kwargs...)

Countinuous case.
"""
function _invrule_fallback(p::Float64, d1::CUD, d2::CUD; n::Int, m::Int, kwargs...)
    m1 = mean(d1)
    m2 = mean(d2)
    s1 = std(d1)
    s2 = std(d2)

    a = _generate_coefs(d1, n, m)
    b = _generate_coefs(d2, n, m)

    c1 = -m1 * m2
    c2 = inv(s1 * s2)

    coef = zeros(Float64, n + 1)
    for k in 1:n
        @inbounds coef[k+1] = c2 * a[k+1] * b[k+1] * factorial(big(k))
    end
    coef[1] = c1 * c2 + c2 * a[1] * b[1] - p

    xs = _feasible_roots(coef)
    return _best_root(p, xs)
end

"""
    _invrule_fallback(p::Float64, d1, d2; n::Int, kwargs...)

Discrete case.
"""
function _invrule_fallback(p::Float64, d1::DUD, d2::DUD; n::Int, kwargs...)
    max1 = maximum(d1)
    max2 = maximum(d2)
    max1 = isinf(max1) ? quantile(d1, prevfloat(1.0)) : max1
    max2 = isinf(max2) ? quantile(d2, prevfloat(1.0)) : max2

    min1 = minimum(d1)
    min2 = minimum(d2)

    s1 = std(d1)
    s2 = std(d2)

    A = Int(min1):Int(max1)
    B = Int(min2):Int(max2)

    # z = Φ⁻¹[F(A)], α[0] = -Inf, β[0] = -Inf
    a = [-Inf; norminvcdf.(cdf.(Ref(d1), A))]
    b = [-Inf; norminvcdf.(cdf.(Ref(d2), B))]

    c2 = inv(s1 * s2)

    coef = zeros(Float64, n + 1)
    for k in 1:n
        @inbounds coef[k+1] = _Gn0_discrete(A, B, a, b, c2, k) / factorial(big(k))
    end
    coef[1] = -p

    xs = _feasible_roots(coef)
    return _best_root(p, xs)
end

"""
    _invrule_fallback(p::Float64, d1, d2; n::Int, m::Int, kwargs...)

Mixed case.
"""
function _invrule_fallback(p::Float64, d1::DUD, d2::CUD; n::Int, m::Int, kwargs...)
    s1 = std(d1)
    s2 = std(d2)
    min1 = minimum(d1)
    max1 = maximum(d1)
    max1 = isinf(max1) ? quantile(d1, prevfloat(1.0)) : max1

    A = Int(min1):Int(max1)
    a = [-Inf; norminvcdf.(cdf.(d1, A))]

    c2 = inv(s1 * s2)

    coef = zeros(Float64, n + 1)
    for k in 1:n
        @inbounds coef[k+1] = _Gn0_mixed(A, a, d2, c2, k, m) / factorial(big(k))
    end
    coef[1] = -p

    xs = _feasible_roots(coef)
    return _best_root(p, xs)
end

"""
    _invrule_fallback(p::Float64, d1, d2; n::Int, m::Int, kwargs...)

Mixed case.
"""
function _invrule_fallback(p::Float64, d1::CUD, d2::DUD; n::Int, m::Int, kwargs...)
    return _invrule_fallback(p, d2, d1; n=n, m=m, kwargs...)
end

"""
    pearson_match(R::AbstractMatrix{<:Real}, margins; n::Real=12, m::Real=128, kwargs...)

Pairwise compute the Pearson correlation coefficient to be used in a bivariate Gaussian
copula. Ensures that the resulting matrix is a valid correlation matrix.

# Fields

- `R`: The target correltation matrix of the marginal distributions.
- `margins`: A list of marginal distributions.
- `n`: The degree of the polynomial used to estimate the matching correlation.
- `m`: The number of points used in the hermite polynomial interpolation.
- `kwargs`: Additional keyword arguments. Currently unused.

# Examples

```julia-repl
julia> using Distributions

julia> margins = [Beta(2, 3), Uniform(0, 1), Binomial(20, 0.2)];

julia> rho = [
    1.0 0.3 0.6
    0.3 1.0 0.4
    0.6 0.4 1.0
];

julia> pearson_match(rho, margins)
3×3 Matrix{Float64}:
 1.0       0.309111  0.612753
 0.309111  1.0       0.418761
 0.612753  0.418761  1.0
```
"""
function pearson_match(
    R::AbstractMatrix{<:Real}, margins; n::Real=20, m::Real=128, kwargs...
)
    d = length(margins)
    r, s = size(R)
    (r == s == d) ||
        throw(DimensionMismatch("The number of margins must be the same size as the correlation matrix."))

    n = Int(n)
    m = Int(m)

    R = SharedMatrix{Float64}(d, d)

    Base.Threads.@threads for (i, j) in _idx_subsets2(d)
        @inbounds R[i, j] = pearson_match(R[i, j], margins[i], margins[j]; n=n, m=m)
    end

    S = sdata(R)

    _project_psd!(S, sqrt(eps()))
    _cov2cor!(S)

    return Symmetric(S)
end
