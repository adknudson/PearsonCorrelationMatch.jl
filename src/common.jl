function _generate_coefs(F::UnivariateDistribution, n::Int)
    t, w = gausshermite(2n)
    t *= sqrt2

    u = normcdf.(t)
    # If u[i] is 0 or 1, then quantile(F, u[i]) has the potential to be ±∞
    # I apply a small correction here to ensure that the values are real (not infinite)
    u = max.(u, eps())
    u = min.(u, prevfloat(1.0))
    
    X = quantile.(F, u)
    
    a = zeros(Float64, n + 1)
    for i in eachindex(a)
        k = i - 1
        S = sum(w .* _hermite.(t, k) .* X)

        @inbounds a[i] = invsqrtπ * S / factorial(k)
    end

    return a
end
_generate_coefs(F::UnivariateDistribution, n) = _generate_coefs(F, Int(n))


function _Gn0d(n::Int, 
    A::AbstractVector{Int}, 
    B::AbstractVector{Int}, 
    a::AbstractVector{Float64},
    b::AbstractVector{Float64},
    invs1s2::Float64)

    n == 0 && return zero(Float64)

    M = length(A)
    N = length(B)

    accu = zero(Float64)

    for r in 1:M, s in 1:N
        r11 = _hermite_normpdf(a[r+1], n-1) * _hermite_normpdf(b[s+1], n-1)
        r00 = _hermite_normpdf(a[r],   n-1) * _hermite_normpdf(b[s],   n-1)
        r01 = _hermite_normpdf(a[r],   n-1) * _hermite_normpdf(b[s+1], n-1)
        r10 = _hermite_normpdf(a[r+1], n-1) * _hermite_normpdf(b[s],   n-1)

        accu += A[r] * B[s] * (r11 + r00 - r01 - r10)
    end

    return accu * invs1s2
end


function _Gn0m(n::Int,
    A::AbstractVector{Int},
    a::AbstractVector{Float64},
    F::UnivariateDistribution,
    invs1s2::Float64)

    n == 0 && return zero(Float64)

    M = length(A)

    accu = zero(Float64)

    for r in 1:M
        accu += A[r] * (_hermite_normpdf(a[r+1], n-1) - _hermite_normpdf(a[r], n-1))
    end

    t, w = gausshermite(n + 4)
    t *= sqrt2
    u = normcdf.(t)
    u = max.(u, eps())
    u = min.(u, prevfloat(1.0))
    X = quantile.(F, u)

    S = zero(Float64)
    for k in eachindex(t)
        S += w[k] * _hermite(t[k], n) * X[k]
    end
    S *= invsqrtπ

    return -invs1s2 * accu * S
end


function _hermite(x::Float64, k::Int)
    k == 0 && return one(x)
    k == 1 && return x

    Hkp1, Hk, Hkm1 = zero(x), x, one(x)

    for k in 2:k
        Hkp1 = x * Hk - (k - 1) * Hkm1
        Hkm1, Hk = Hk, Hkp1
    end
    
    return Hkp1
end
_hermite(x, n) = _hermite(float(x), Int(n))

_hermite_normpdf(x::Float64, n::Int) = isinf(x) ? zero(x) : _hermite(x, n) * normpdf(x)
_hermite_normpdf(x, n) = _hermite_normpdf(float(x), Int(n))