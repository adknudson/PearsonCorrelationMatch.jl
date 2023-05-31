"""

"""
function pearson_bounds(D1::UnivariateDistribution, D2::UnivariateDistribution, n::Int=10)
    m1 = mean(D1)
    m2 = mean(D2)
    s1 = std(D1)
    s2 = std(D2)

    a = _generate_coefs(D1, n)
    b = _generate_coefs(D2, n)

    k = 0:1:n

    c1 = -m1 * m2
    c2 = inv(s1 * s2)
    kab = a .* factorial.(k) .* b

    pl = c1 * c2 + c2 * sum((-1).^k .* kab)
    pu = c1 * c2 + c2 * sum(kab)

    pl = min(max(pl, -1), 1)
    pu = min(max(pu, -1), 1)

    return (lower = pl, upper = pu)
end
