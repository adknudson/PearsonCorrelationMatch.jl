"""
    pearson_bounds(d1::UnivariateDistribution, d2::UnivariateDistribution, n::Int=10)

Determine the range of admissible Pearson correlations between two distributions.
"""
function pearson_bounds(d1::UnivariateDistribution, d2::UnivariateDistribution, n::Int=32)
    m1 = mean(d1)
    m2 = mean(d2)
    s1 = std(d1)
    s2 = std(d2)

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


function pearson_bounds(margins::AbstractVector{<:UnivariateDistribution}, n::Int=32)
    d = length(margins)

    lower = SharedMatrix{Float64}(d, d)
    upper = SharedMatrix{Float64}(d, d)

    Base.Threads.@threads for (i,j) in _idx_subsets2(d)
        l, u = pearson_bounds(margins[i], margins[j], n)
        lower[i,j] = l
        upper[i,j] = u
    end

    _symmetric!(lower)
    _set_diag1!(lower)

    _symmetric!(upper)
    _set_diag1!(upper)

    (lower = sdata(lower), upper = sdata(upper))
end
