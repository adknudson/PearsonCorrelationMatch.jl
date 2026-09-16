"""
    pearson_bounds(d1, d2; n::Real=20, m::Real=128, kwargs...)


Determine the range of admissible Pearson correlations between two distributions.

# Fields

- `d1`: The first marginal distribution.
- `d2`: The second marginal distribution.
- `n`: The degree of the polynomial used to estimate the bounds.
- `m`: The number of points used in the hermite polynomial interpolation.
- `kwargs`: Additional keyword arguments. Currently unused.

# Details

The accuracy of the bounds depends on the degree of the polynomial and the number of hermite
points. Be careful not to set the polynomial degree too high as Runge's theorem states that
a polynomial of too high degree would cause oscillation at the edges of the interval and
reduce accuracy.

In general raising the number of hermite points will result in better accuracy, but comes
with a small performance hit. Furthermore the number of hermite points should be higher
than the degree of the polynomial.

# Examples

```julia-repl
julia> using Distributions

julia> d1 = Exponential(3.14); d2 = NegativeBinomial(20, 0.2);

julia> pearson_bounds(d1, d2)
(lower = -0.8507790644977721, upper = 0.9315140946532414)
```
"""
function pearson_bounds(d1::UnivariateDistribution, d2::UnivariateDistribution; n::Real = 20, m::Real = 128, kwargs...)
    n = Int(n)
    m = Int(m)

    m1 = mean(d1)
    m2 = mean(d2)
    s1 = std(d1)
    s2 = std(d2)

    a = _generate_coefs(d1, n, m)
    b = _generate_coefs(d2, n, m)
    k = big.(0:n)
    kab = a .* factorial.(k) .* b

    c1 = -m1 * m2
    c2 = inv(s1 * s2)

    pl = c1 * c2 + c2 * sum((-1) .^ k .* kab)
    pu = c1 * c2 + c2 * sum(kab)

    pl = clamp(pl, -1, 1)
    pu = clamp(pu, -1, 1)

    return (lower = Float64(pl), upper = Float64(pu))
end

"""
    pearson_bounds(margins; n::Real=20, m::Real=128, kwargs...)

Determine the range of admissible Pearson correlations pairwise between a list of distributions.

# Fields

- `margins`: A list of marginal distributions.
- `n`: The degree of the polynomial used to estimate the bounds.
- `m`: The number of points used in the hermite polynomial interpolation.
- `kwargs`: Additional keyword arguments. Currently unused.

# Examples

```julia-repl
julia> using Distributions

julia> margins = [Exponential(3.14), NegativeBinomial(20, 0.2), LogNormal(2.718)];

julia> lower, upper = pearson_bounds(margins);

julia> lower
3×3 Matrix{Float64}:
 -0.644934  -0.850779  -0.488737
 -0.850779  -0.983832  -0.70092
 -0.488737  -0.70092   -0.367879

julia> upper
3×3 Matrix{Float64}:
 1.0       0.931514  0.939671
 0.931514  0.960926  0.80747
 0.939671  0.80747   1.0
```
"""
function pearson_bounds(margins; n::Real = 20, m::Real = 128, kwargs...)
    d = length(margins)
    n = Int(n)
    m = Int(m)

    lower = Matrix{Float64}(undef, d, d)
    upper = Matrix{Float64}(undef, d, d)

    for i in 1:d, j in i:d
        d1 = margins[i]
        d2 = margins[j]
        l, u = pearson_bounds(d1, d2; n = n, m = m, kwargs...)
        lower[i, j] = lower[j, i] = l
        upper[i, j] = upper[j, i] = u
    end

    return (lower = lower, upper = upper)
end
