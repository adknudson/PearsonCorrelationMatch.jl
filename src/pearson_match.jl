function pearson_match(p::Float64, 
    D1::UnivariateDistribution,
    D2::UnivariateDistribution,
    n::Int=7)

    return _match(p, D1, D2, n)
end

pearson_match(p, D1, D2, n) = pearson_match(float(p), D1, D2, Int(n))


function _match(p::Float64, 
    D1::ContinuousUnivariateDistribution, 
    D2::ContinuousUnivariateDistribution, 
    n::Int)

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
        coef[k+1] = c2 * a[k+1] * b[k+1] * factorial(k)
    end
    coef[1] = c1 * c2 + c2 * a[1] * b[1] - p
    
    # find all real roots in the interval [-1, 1]
    xs = _real_roots(coef)
    filter!(x -> abs(x) ≤ 1, xs)

    return _best_root(xs)
end

function _match(p::Float64, 
    D1::DiscreteUnivariateDistribution, 
    D2::DiscreteUnivariateDistribution, 
    n::Int)

    max1 = maximum(D1)
    max2 = maximum(D2)
    max1 = isinf(max1) ? quantile(D1, prevfloat(1.0)) : max1
    max2 = isinf(max2) ? quantile(D2, prevfloat(1.0)) : max2

    min1 = minimum(D1)
    min2 = minimum(D2)

    s1 = std(D1)
    s2 = std(D2)

    A = min1:max1
    B = min2:max2

    # z = Φ⁻¹[F(A)], α[0] = -Inf, β[0] = -Inf
    a = [-Inf; norminvcdf.(cdf.(D1, A))]
    b = [-Inf; norminvcdf.(cdf.(D2, B))]

    c2 = inv(s1 * s2)

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k + 1] = Gn0d(k, A, B, a, b, c2) / factorial(k)
    end
    coef[1] = -p

    # find all real roots in the interval [-1, 1]
    xs = _real_roots(coef)
    filter!(x -> abs(x) ≤ 1, xs)

    return _best_root(xs)
end

function _match(p::Float64, 
    D1::DiscreteUnivariateDistribution, 
    D2::ContinuousUnivariateDistribution, 
    n::Int)

    s1 = std(D1)
    s2 = std(D2)
    min1 = minimum(D1)
    max1 = maximum(D1)
    max1 = isinf(max1) ? quantile(D1, prevfloat(1.0)) : max1

    A = min1:max1
    a = [-Inf; norminvcdf.(cdf.(D1, A))]

    c2 = inv(s1 * s2)

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k+1] = Gn0m(k, A, a, D2, c2) / factorial(k)
    end
    coef[1] = -p

    # find all real roots in the interval [-1, 1]
    xs = _real_roots(coef)
    filter!(x -> abs(x) ≤ 1, xs)

    return _best_root(xs)
end

function _match(p::Float64, 
    D1::ContinuousUnivariateDistribution, 
    D2::DiscreteUnivariateDistribution, 
    n::Int)
    return _match(p, D2, D1, n)
end



function _real_roots(coefs)
    xs = roots(coefs)
    filter!(isreal, xs)
    xs = real.(xs)
    unique!(xs)
    return xs
end

_nearest_root(p, xs) = xs[argmin(abs.(xs .- p))]

function _best_root(xs)
    # if there is only one root, then we can return it
    length(xs) == 1 && return first(xs)
    
    # if there are multiple roots, then return the one closest to the target
    length(xs) >  1 && return _nearest_root(p, xs)

    # if the target correlation is not feasible, return ±1
    return p < 0 ? nextfloat(-1.0) : prevfloat(1.0)
end