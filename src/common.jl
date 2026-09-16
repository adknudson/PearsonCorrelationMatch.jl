const _sqrt2::Float64 = sqrt(2.0)
const _invsqrtπ::Float64 = inv(sqrt(π))

"""
    _generate_coefs(dist, n, m)

Equation (25) of the reference paper.
"""
function _generate_coefs(F::UnivariateDistribution, n::Int, m::Int)
    t, w = gausshermite(m)
    t *= _sqrt2

    u = normcdf.(t)

    # If u[i] is 0 or 1, then quantile(F, u[i]) has the potential to be ±∞
    # Apply a small correction here to ensure that the values are finite
    clamp!(u, 1.0e-15, 1.0 - 1.0e-15)

    X = quantile.(F, u)

    a = zeros(Float64, n + 1)
    for i in eachindex(a)
        k = i - 1
        S = sum(w .* _hermite.(t, k) .* X)
        @inbounds a[i] = _invsqrtπ * S / factorial(big(k))
    end

    return a
end

"""
    _Gn0_discrete(A, B, a, b, invs1s2, n::Int)


Equation (41) of the reference paper.
"""
function _Gn0_discrete(A, B, a, b, invs1s2, n::Int)
    n == 0 && return zero(Float64)

    M = length(A)
    N = length(B)

    accu = zero(Float64)

    for r in 1:M, s in 1:N
        r00 = _hermite_normpdf(a[r], n - 1) * _hermite_normpdf(b[s], n - 1)
        r10 = _hermite_normpdf(a[r + 1], n - 1) * _hermite_normpdf(b[s], n - 1)
        r01 = _hermite_normpdf(a[r], n - 1) * _hermite_normpdf(b[s + 1], n - 1)
        r11 = _hermite_normpdf(a[r + 1], n - 1) * _hermite_normpdf(b[s + 1], n - 1)

        accu += A[r] * B[s] * (r11 + r00 - r01 - r10)
    end

    return accu * invs1s2
end

"""
    _Gn0_mixed(A, a, F, invs1s2, n::Int, m::Int)

Equation (49) of the reference paper.
"""
function _Gn0_mixed(A, a, F, invs1s2, n::Int, m::Int)
    n == 0 && return zero(Float64)

    M = length(A)

    accu = zero(Float64)

    for r in 1:M
        accu += A[r] * (_hermite_normpdf(a[r + 1], n - 1) - _hermite_normpdf(a[r], n - 1))
    end

    t, w = gausshermite(m)
    t *= _sqrt2
    u = normcdf.(t)

    # If u[i] is 0 or 1, then quantile(F, u[i]) has the potential to be ±∞
    # Apply a small correction here to ensure that the values are finite
    clamp!(u, 1.0e-15, 1.0 - 1.0e-15)

    X = quantile.(F, u)
    any(isinf, X) && error("Values must be real and finite")

    S = zero(Float64)

    for k in eachindex(t)
        S += w[k] * _hermite(t[k], n) * X[k]
    end

    S *= _invsqrtπ

    return -invs1s2 * accu * S
end

"""
    _hermite(x, k)

The "probabilist's" Hermite polynomial of degree ``k``.
"""
function _hermite(x::Float64, k::Int)
    k < 0 && throw(ArgumentError("'k' must be a non-negative integer"))
    k == 0 && return 1.0
    k == 1 && return x

    Hk = x
    Hkp1 = zero(x)
    Hkm1 = one(x)

    for j in 2:k
        Hkp1 = x * Hk - (j - 1) * Hkm1
        Hkm1, Hk = Hk, Hkp1
    end

    return Hkp1
end

_hermite(x, n) = _hermite(convert(Float64, x), convert(Int, n))

"""
    He(x, k) * normpdf(x)
"""
_hermite_normpdf(x::Float64, k::Int) = isinf(x) ? zero(x) : _hermite(x, k) * normpdf(x)
_hermite_normpdf(x, k) = _hermite_normpdf(convert(Float64, x), convert(Int, k))

"""
    _is_real(x::Complex)

Check if a number is real within a given tolerance.
"""
_is_real(x::Complex{T}) where {T} = abs(imag(x)) < sqrt(eps(T))

"""
    _real_roots(coeffs)

Find the real and unique roots of the polynomial coefficients.

- `coeffs`: A vector of coefficients in ascending order of degree.
"""
function _real_roots(coeffs)
    complex_roots = roots(coeffs)
    filter!(_is_real, complex_roots)
    xs = real.(complex_roots)
    return unique!(xs)
end

"""
    _feasible_roots(coeffs)

Find all real roots of the polynomial that are in the interval ``[-1, 1]``.

- `coeffs`: a vector of coefficients in ascending order of degree.
"""
function _feasible_roots(coeffs)
    xs = _real_roots(coeffs)
    return filter!(x -> abs(x) ≤ 1.0 + sqrt(eps()), xs)
end

"""
    _nearest_root(target, roots)

Find the root closest to the target value.
"""
function _nearest_root(target, roots)
    y = 0
    m = Inf

    for x0 in roots
        f = abs(x0 - target)
        if f < m
            m = f
            y = x0
        end
    end

    return y
end

"""
    _best_root(p, xs)

Consider the feasible roots and return a value.

- `p`: the target correlation
- `roots`: a vector of feasible roots
"""
function _best_root(p, roots)
    length(roots) == 1 && return clamp(first(roots), -1, 1)
    length(roots) > 1 && return _nearest_root(p, roots)
    return p < 0 ? nextfloat(-one(Float64)) : prevfloat(one(Float64))
end
