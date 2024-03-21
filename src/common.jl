function _generate_coefs(F, n::Int)
    t, w = gausshermite(2n)
    t *= sqrt2

    u = normcdf.(t)
    # If u[i] is 0 or 1, then quantile(F, u[i]) has the potential to be ±∞
    # Apply a small correction here to ensure that the values are finite
    u = max.(u, eps(Float64))
    u = min.(u, prevfloat(one(Float64)))

    X = quantile.(F, u)

    a = zeros(Float64, n + 1)
    for i in eachindex(a)
        k = i - 1
        S = sum(w .* _hermite.(t, k) .* X)

        @inbounds a[i] = invsqrtπ * S / factorial(big(k))
    end

    return a
end


function _Gn0d(n::Int, A, B, a, b, invs1s2)
    n == 0 && return zero(Float64)

    M = length(A)
    N = length(B)

    accu = zero(Float64)

    for r in 1:M, s in 1:N
        r00 = _hermite_normpdf(a[r],   n-1) * _hermite_normpdf(b[s],   n-1)
        r10 = _hermite_normpdf(a[r+1], n-1) * _hermite_normpdf(b[s],   n-1)
        r01 = _hermite_normpdf(a[r],   n-1) * _hermite_normpdf(b[s+1], n-1)
        r11 = _hermite_normpdf(a[r+1], n-1) * _hermite_normpdf(b[s+1], n-1)

        accu += A[r] * B[s] * (r11 + r00 - r01 - r10)
    end

    return accu * invs1s2
end


function _Gn0m(n::Int, A, a, F, invs1s2)
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
    u = min.(u, prevfloat(one(Float64)))
    X = quantile.(F, u)

    S = zero(Float64)

    for k in eachindex(t)
        S += w[k] * _hermite(t[k], n) * X[k]
    end

    S *= invsqrtπ

    return -invs1s2 * accu * S
end


function _hermite(x::Float64, k::Int)
    x = convert(Float64, x)
    k == 0 && return one(x)
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

_hermite_normpdf(x::Float64, n::Int) = isinf(x) ? zero(x) : _hermite(x, n) * normpdf(x)
_hermite_normpdf(x, n) = _hermite_normpdf(convert(Float64, x), convert(Int, n))


"""
    _is_real(x::Complex)

Check if a number is real within a given tolerance.
"""
_is_real(x::Complex{T}) where T = abs(imag(x)) < _sqrteps(T)


"""
    _real_roots(coeffs)

Find the real and unique roots of ``polynomial``.

- ``polynomial`` is a vector of coefficients in ascending order of degree.
"""
function _real_roots(coeffs)
    complex_roots = roots(coeffs)
    filter!(_is_real, complex_roots)
    xs = real.(complex_roots)
    return unique!(xs)
end


"""
    _sqrteps(T)

Return the square root of machine precision for a given floating point type.
"""
_sqrteps(::Type{T}) where T = sqrt(eps(T))
_sqrteps() = sqrt(eps())


"""
    _feasible_roots(coeffs)

Find all real roots of ``polynomial`` that are in the interval `[-1, 1]`.

- ``polynomial`` is a vector of coefficients in ascending order of degree.
"""
function _feasible_roots(coeffs)
    xs = _real_roots(coeffs)
    return filter!(x -> abs(x) ≤ 1.0 + _sqrteps(), xs)
end


"""
    _nearest_root(target, xs)

Find the root closest to the target value, ``x``.
"""
function _nearest_root(target, xs)
    y = 0
    m = Inf

    for x in xs
        f = abs(x - target)
        if f < m
            m = f
            y = x
        end
    end

    return y
end


"""
    _best_root(p, xs)

Consider the feasible roots and return a value.

- ``p`` is the target correlation
- ``xs`` is a vector of feasible roots
"""
function _best_root(p, xs)
    length(xs) == 1 && return clamp(first(xs), -1, 1)
    length(xs)  > 1 && return _nearest_root(p, xs)
    return p < 0 ? nextfloat(-one(Float64)) : prevfloat(one(Float64))
end


"""
    _idx_subsets2(d)

equivalent to IterTools.subsets(1:d, Val(2)), but allocates all pairs for use in parallel
threads.
"""
function _idx_subsets2(d::Int)
    n = d * (d - 1) ÷ 2
    xs = Vector{Tuple}(undef, n)

    k = 1
    for i = 1:d-1
        for j = i+1:d
            xs[k] = (i,j)
            k += 1
        end
    end

    return xs
end


"""
    _symmetric!(X)

Copy the upper part of a matrix to its lower half.
"""
function _symmetric!(X::AbstractMatrix{T}) where T
    m, n = size(X)
    m == n || throw(DimensionMismatch("Input matrix must be square"))

    for i = 1:n-1
        for j = i+1:n
            @inbounds X[j,i] = X[i,j]
        end
    end

    return X
end


"""
    _set_diag1!(X)

Set the diagonal elements of a square matrix to `1`.
"""
function _set_diag1!(X::AbstractMatrix{T}) where T
    m, n = size(X)
    m == n || throw(DimensionMismatch("Input matrix must be square"))

    @inbounds for i in diagind(X)
        X[i] = one(T)
    end

    return X
end


"""
    _project_psd(X, ϵ)

Project `X` onto the set of PSD matrixes.
"""
function _project_psd!(X, ϵ)
    λ, P = eigen(Symmetric(X), sortby=x->-x)
    replace!(x -> max(x, ϵ), λ)
    X .= P * Diagonal(λ) * P'
    return X
end


"""
    _cov2cor!(X)

Project `X` onto the set of correlation matrices.
"""
function _cov2cor!(X::AbstractMatrix)
    D = sqrt(inv(Diagonal(X)))
    lmul!(D, X)
    rmul!(X, D)
    _set_diag1!(X)
    _symmetric!(X)
    return X
end

function _cov2cor!(X::Symmetric)
    _symmetric!(X.data)
    _cov2cor!(X.data)
    return X
end
