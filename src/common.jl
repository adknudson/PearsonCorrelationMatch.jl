"""
    get_inv_factorials(max_k)

Retrieves precomputed 1/k! values up to degree `max_k`.
Dynamically extends the global vector in a thread-safe manner if max_k > current length.
"""
function get_inv_factorials(max_k::Int)
    if max_k > length(GLOBAL_INV_FACTORIALS)
        lock(CACHE_LOCK) do
            current_len = length(GLOBAL_INV_FACTORIALS)
            if max_k > current_len
                f = factorial(Float64(current_len))
                for k in (current_len + 1):max_k
                    f *= k
                    push!(GLOBAL_INV_FACTORIALS, 1.0 / f)
                end
            end
        end
    end
    return @view GLOBAL_INV_FACTORIALS[1:max_k]
end


"""
    probabilist_hermite(k, x)

Evaluates the `k`-th order probabilist's Hermite polynomial at `x`.
"""
function probabilist_hermite(k::Int, x::Float64)
    k < 0 && return 0.0
    k == 0 && return 1.0
    k == 1 && return x

    h_prev2 = 1.0
    h_prev1 = x
    h_curr = x
    for i in 2:k
        h_curr = x * h_prev1 - (i - 1) * h_prev2
        h_prev2 = h_prev1
        h_prev1 = h_curr
    end
    return h_curr
end

"""
    gauss_hermite_prob(m)

Generates m-point Gauss-Hermite nodes and weights for integrating against
the standard normal PDF: ∫ f(x)ϕ(x)dx ≈ Σ w_i f(t_i).
Uses the Golub-Welsch algorithm.
"""
function gauss_hermite_prob(m::Int)
    # Jacobi matrix for normalized probabilist's Hermite polynomials
    J = SymTridiagonal(zeros(m), sqrt.(1:(m - 1)))
    vals, vecs = eigen(J)
    nodes = vals
    weights = vecs[1, :] .^ 2 # First row elements squared represent weights
    return nodes, weights
end

"""
    get_gauss_hermite(m)

Retrieves or computes Gauss-Hermite nodes and weights for m points.
Dynamically updates the global dictionary in a thread-safe manner.
"""
function get_gauss_hermite(m::Int)
    # Fast path: Read without lock if already present
    if haskey(GLOBAL_GH_CACHE, m)
        return GLOBAL_GH_CACHE[m]
    end

    # Slow path: Compute and cache under lock
    return lock(CACHE_LOCK) do
        return get!(GLOBAL_GH_CACHE, m) do
            gauss_hermite_prob(m)
        end
    end
end


"""
    extract_coef(d::ContinuousUnivariateDistribution, k::Int, nodes::Vector{Float64}, weights::Vector{Float64})
    extract_coef(d::DiscreteUnivariateDistribution, k::Int, ::Vector{Float64}, ::Vector{Float64})

Computes the `k`-th Hermite polynomial expansion coefficient `C_i(k)` for marginal distribution `d`.

## Arguments
- `d::UnivariateDistribution`: Marginal distribution object.
- `k::Int`: Polynomial expansion order (`k ≥ 1`).
- `nodes::Vector{Float64}`: Gauss-Hermite integration nodes (used only for continuous distributions).
- `weights::Vector{Float64}`: Gauss-Hermite integration weights (used only for continuous distributions).

## Details
- For continuous distributions, uses Gauss-Hermite quadrature.
- For discrete distributions, evaluates exact piecewise boundary differences across cumulative probabilities.
"""
function extract_coef end

function extract_coef(d::ContinuousUnivariateDistribution, k::Int, nodes::Vector{Float64}, weights::Vector{Float64})
    val = 0.0
    for i in eachindex(nodes)
        t = nodes[i]
        w = weights[i]
        # F^{-1}(Φ(t))
        p = cdf(Normal(), t)
        # Clamp to avoid domain errors at extreme float precision
        p = clamp(p, 1.0e-15, 1.0 - 1.0e-15)
        x = quantile(d, p)
        val += w * probabilist_hermite(k, t) * x
    end
    return val
end

function extract_coef(d::DiscreteUnivariateDistribution, k::Int, ::Vector{Float64}, ::Vector{Float64})
    lb = isinf(minimum(d)) ? floor(Int, quantile(d, 1.0e-10)) : minimum(d)
    ub = isinf(maximum(d)) ? ceil(Int, quantile(d, 1.0 - 1.0e-10)) : maximum(d)
    sup = lb:ub
    M = length(sup)
    val = 0.0

    # Evaluates H_{k-1}(α_r) * ϕ(α_r)
    function term(r::Int)
        if r == 0
            return 0.0
        elseif r == M && maximum(d) == sup[M]
            return 0.0 # Φ^{-1}(1.0) = ∞, ϕ(∞) = 0
        else
            A_r = sup[r]
            p = cdf(d, A_r)
            p >= 1.0 && return 0.0
            alpha_r = quantile(Normal(), p)
            return probabilist_hermite(k - 1, alpha_r) * pdf(Normal(), alpha_r)
        end
    end

    t_prev = term(0)
    for r in 1:M
        t_curr = term(r)
        delta = t_curr - t_prev
        val -= sup[r] * delta
        t_prev = t_curr
    end

    return val
end

"""
    eval_poly(coefs, z)

Evaluates the polynomial at `z` given its coefficients.
"""
function eval_poly(coefs::Vector{Float64}, z::Real)
    val = 0.0
    z_pow = 1.0
    for c_i in coefs
        z_pow *= z
        val += c_i * z_pow
    end
    return val
end
