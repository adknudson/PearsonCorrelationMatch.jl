"""
    pearson_bounds(d1::UnivariateDistribution, d2::UnivariateDistribution, n)

Determine the range of admissible Pearson correlations between two distributions.

# Examples

```julia-repl
julia> using Distributions

julia> d1 = Exponential(3.14); d2 = NegativeBinomial(20, 0.2);

julia> pearson_bounds(d1, d2)
(lower = -0.8553947509241561, upper = 0.9413665073003636)
```
"""
function pearson_bounds(d1::UD, d2::UD, n=32)
    m1 = mean(d1)
    m2 = mean(d2)
    s1 = std(d1)
    s2 = std(d2)

    n = convert(Int, n)

    a = _generate_coefs(d1, n)
    b = _generate_coefs(d2, n)

    k = big.(0:1:n)

    c1 = -m1 * m2
    c2 = inv(s1 * s2)
    kab = a .* factorial.(k) .* b

    pl = c1 * c2 + c2 * sum((-1).^k .* kab)
    pu = c1 * c2 + c2 * sum(kab)

    pl = clamp(pl, -1, 1)
    pu = clamp(pu, -1, 1)

    return (lower = Float64(pl), upper = Float64(pu))
end


"""
    pearson_bounds(margins::AbstractVector{<:UnivariateDistribution}, n::Int)

Determine the range of admissible Pearson correlations pairwise between a list of distributions.

# Examples

```julia-repl
julia> using Distributions

julia> margins = [Exponential(3.14), NegativeBinomial(20, 0.2), LogNormal(2.718)];

julia> lower, upper = pearson_bounds(margins);

julia> lower
3×3 Matrix{Float64}:
  1.0       -0.855395  -0.488737
 -0.855395   1.0       -0.704403
 -0.488737  -0.704403   1.0

julia> upper
3×3 Matrix{Float64}:
 1.0       0.941367  0.939671
 0.941367  1.0       0.815171
 0.939671  0.815171  1.0
```
"""
function pearson_bounds(margins::AbstractVector{<:UD}, n=32)
    d = length(margins)
    n = convert(Int, n)

    lower = SharedMatrix{Float64}(d, d)
    upper = SharedMatrix{Float64}(d, d)

    Base.Threads.@threads for (i,j) in _idx_subsets2(d)
        l, u = pearson_bounds(margins[i], margins[j], n)
        @inbounds lower[i,j] = l
        @inbounds upper[i,j] = u
    end

    L, U = sdata(lower), sdata(upper)

    _symmetric!(L)
    _set_diag1!(L)

    _symmetric!(U)
    _set_diag1!(U)

    (lower = L, upper = U)
end
