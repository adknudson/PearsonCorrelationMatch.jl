"""
    pearson_match(rho_x, d1, d2; kwargs...)

Determines the Gaussian copula correlation parameter `rho_z` required to achieve a target
Pearson correlation `rho_x` between two marginal distributions `d1` and `d2`.

## Arguments

- `rho_x`: Target Pearson correlation coefficient.
- `d1`: First marginal distribution.
- `d2`: Second marginal distribution.

## Keyword Arguments

- `degree`: Truncation degree for polynomial approximation (default: `default_degree(d1, d2)`).
- `m`: Number of Gauss-Hermite integration points (default: `default_m(d1, d2)`).
- `maxiters`: The maximum number of iterations in the polynomial root search (default: `100`).
- `atol`: The absolute tolerance used as a stopping condition in the polynomial root search (default: `1e-12`).
- `check_variance`: Checks the requirement that marginal distributions have finite variance (default: `true`).
  If `true`, an error is thrown if `d1` or `d2` has non-finite or undefined variance.
  Otherwise, if `false`, non-finite variances result in `NaN` values being propagated.
"""
function pearson_match(
        rho_x::Real,
        d1::UnivariateDistribution,
        d2::UnivariateDistribution;
        degree::Real = default_degree(d1, d2),
        m::Real = default_m(d1, d2),
        maxiters::Real = 100,
        atol::Real = 1.0e-12,
        check_variance::Bool = true
    )
    std1, std2 = std(d1), std(d2)
    if !isfinite(std1) || !isfinite(std2)
        check_variance || return NaN
        throw(
            ArgumentError(
                "Both distributions are required to have a finite variance:\n" *
                    "  Var[$(d1)] = $(std1^2)\n" *
                    "  Var[$(d2)] = $(std2^2)"
            )
        )
    end

    degree = Int(degree)
    m = Int(m)
    G = BivariateModel(d1, d2; degree, m)

    maxiters = Int(maxiters)
    atol = float(atol)
    return pearson_match(float(rho_x), G; maxiters, atol)
end

"""
    pearson_match(R_x, dists; kwargs...)

Computes the pairwise Gaussian copula correlation matrix `R_z` corresponding to a target Pearson
correlation matrix `R_x` for a list of marginal distributions `dists`.

!!! warning
    The resulting correlation matrix may not be valid (i.e., not positive definite), and
    therefore may not work in subsequent use. In that case, you may need to compute the
    nearest valid correlation matrix. See the [NearestCorrelationMatrix.jl](https://github.com/adknudson/NearestCorrelationMatrix.jl)
    package for more details.

## Arguments

- `R_x`: Target `N × N` Pearson correlation matrix.
- `dists`: List of marginal distributions.

## Keyword Arguments

- `degree`: Truncation degree for polynomial approximation (default: `default_degree(dists)`).
- `m`: Number of Gauss-Hermite integration points (default: `default_m(dists)`).
- `maxiters`: The maximum number of iterations in the polynomial root search (default: `100`).
- `atol`: The absolute tolerance used as a stopping condition in the polynomial root search (default: `1e-12`).
- `check_variance`: Checks the requirement that marginal distributions have finite variance (default: `true`).
  If `true`, an `ArgumentError` is thrown if any distribution has a non-finite or undefined variance.
  Otherwise, if `false`, non-finite variances result in `NaN` values being propagated for those entries.
"""
function pearson_match(
        R_x::AbstractMatrix{<:Real},
        dists;
        degree::Real = default_degree(dists),
        m::Real = default_m(dists),
        maxiters::Real = 100,
        atol::Real = 1.0e-12,
        check_variance::Bool = true
    )
    R_x = float.(R_x)
    degree = Int(degree)
    m = Int(m)
    maxiters = Int(maxiters)
    atol = float(atol)

    n_dists = length(dists)
    @assert size(R_x) == (n_dists, n_dists) "R_x must be an $(n_dists)x$(n_dists) matrix"

    # 1. Precompute standard deviations upfront
    stds = zeros(Float64, n_dists)
    Threads.@threads for i in 1:n_dists
        stds[i] = std(dists[i])
    end

    # 2. Early error check for non-finite variances
    if check_variance
        invalid_indices = findall(!isfinite, stds)
        if !isempty(invalid_indices)
            msg = "All distributions are required to have a finite variance (check_variance=true)."
            for idx in invalid_indices
                msg *= "\n  Var[$(dists[idx])] = $(stds[idx]^2)"
            end
            throw(ArgumentError(msg))
        end
    end

    # 3. Precompute polynomial expansion coefficients in parallel
    inv_fact = get_inv_factorials(degree)
    has_continuous = any(d -> d isa ContinuousUnivariateDistribution, dists)
    nodes, weights = has_continuous ? get_gauss_hermite(m) : (Float64[], Float64[])

    C = Matrix{Float64}(undef, n_dists, degree)
    Threads.@threads for i in 1:n_dists
        # Only calculate coefficients if variance is finite
        if isfinite(stds[i])
            for k in 1:degree
                C[i, k] = extract_coef(dists[i], k, nodes, weights)
            end
        end
    end

    # 4. Construct upper-triangle pair indices
    has_exact_match(d1::Type, d2::Type) = hasmethod(pearson_match, Tuple{Real, d1, d2})
    pairs = [(i, j) for i in 1:n_dists for j in (i + 1):n_dists]

    R_z = similar(R_x)
    for i in 1:n_dists
        R_z[i, i] = 1.0
    end

    # 5. Parallel Pairwise Matching
    Threads.@threads for idx in eachindex(pairs)
        i, j = pairs[idx]
        d1, d2 = dists[i], dists[j]
        target_rho = R_x[i, j]

        # Handle non-finite variances when check_variance = false
        if !isfinite(stds[i]) || !isfinite(stds[j])
            R_z[i, j] = R_z[j, i] = NaN
            continue
        end

        # Fast Path: Dispatch to exact closed-form method if overload exists
        if has_exact_match(typeof(d1), typeof(d2))
            rho_z_ij = pearson_match(target_rho, d1, d2; check_variance = check_variance)
            R_z[i, j] = R_z[j, i] = rho_z_ij
            continue
        end

        # Slow Path: Inexact polynomial approximation & root-finding
        scale = 1.0 / (stds[i] * stds[j])
        poly_pos1, poly_neg1 = 0.0, 0.0
        for k in 1:degree
            c_k = C[i, k] * C[j, k] * inv_fact[k] * scale
            poly_pos1 += c_k
            poly_neg1 += c_k * (isodd(k) ? -1.0 : 1.0)
        end

        if target_rho <= poly_neg1
            R_z[i, j] = R_z[j, i] = -1.0
            continue
        elseif target_rho >= poly_pos1
            R_z[i, j] = R_z[j, i] = 1.0
            continue
        end

        low, high = -1.0, 1.0
        for _ in 1:maxiters
            mid = (low + high) * 0.5
            val = 0.0
            mid_pow = 1.0
            for k in 1:degree
                mid_pow *= mid
                val += (C[i, k] * C[j, k] * inv_fact[k] * scale) * mid_pow
            end

            if val < target_rho
                low = mid
            else
                high = mid
            end

            if high - low < atol
                break
            end
        end

        rho_z_ij = (low + high) * 0.5
        R_z[i, j] = R_z[j, i] = rho_z_ij
    end

    return R_z
end
