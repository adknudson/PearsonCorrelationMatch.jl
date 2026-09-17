"""
    pearson_bounds(d1, d2; kwargs...)

Computes the theoretical lower and upper admissible bounds for the Pearson correlation
coefficient between two marginal distributions.

## Arguments

- `d1`: First marginal distribution.
- `d2`: Second marginal distribution.

## Keyword Arguments

- `degree`: Polynomial truncation degree (default: `default_degree(d1, d2)`).
- `m`: Gauss-Hermite integration points (default: `default_m(d1, d2)`).
- `check_variance`: Checks the requirement that marginal distributions have finite variance (default: `true`).
  If `true`, an error is thrown if `d1` or `d2` has non-finite or undefined variance.
  Otherwise, if `false`, non-finite variances result in `NaN` values being propagated.
"""
function pearson_bounds(
        d1::UnivariateDistribution,
        d2::UnivariateDistribution;
        degree::Real = 20,
        m::Real = 40,
        check_variance::Bool = true
    )
    degree = Int(degree)
    m = Int(m)

    # Generate quadrature rules if at least one variable is continuous
    nodes, weights = Float64[], Float64[]
    if d1 isa ContinuousUnivariateDistribution || d2 isa ContinuousUnivariateDistribution
        nodes, weights = get_gauss_hermite(m)
    end

    std1, std2 = std(d1), std(d2)
    if !isfinite(std1) || !isfinite(std2)
        check_variance || return (NaN, NaN)
        throw(
            ArgumentError(
                "Both distributions are required to have a finite variance:\n" *
                    "  Var[$(d1)] = $(std1^2)\n" *
                    "  Var[$(d2)] = $(std2^2)"
            )
        )
    end

    max_rho = 0.0
    min_rho = 0.0

    fact_k = 1.0
    # Summing from k=1 handles the cancellation of the k=0 term with the mean subtraction
    for k in 1:degree
        fact_k *= k
        coef1 = extract_coef(d1, k, nodes, weights)
        coef2 = extract_coef(d2, k, nodes, weights)

        # c_k corresponds to the coefficient of ρ_z^k in the polynomial expansion
        c_k = (coef1 * coef2) / (fact_k * std1 * std2)

        # Evaluate polynomial at ρ_z = 1.0 for the upper bound
        max_rho += c_k * (1.0)^k

        # Evaluate polynomial at ρ_z = -1.0 for the lower bound
        min_rho += c_k * (-1.0)^k
    end

    # Due to numerical precision, clamp results strictly within standard [-1.0, 1.0] bounds
    min_rho = clamp(min_rho, -1.0, 1.0)
    max_rho = clamp(max_rho, -1.0, 1.0)

    return (min_rho, max_rho)
end

"""
    pearson_bounds(dists; kwargs...)

Computes the pairwise theoretical Pearson correlation bounds `(lower, upper)` for a list of
marginal distributions `dists`.

## Arguments

- `dists`: List of marginal distributions.

## Keyword Arguments

- `degree`: Truncation degree for polynomial approximation (default: `default_degree(dists)`).
- `m`: Number of Gauss-Hermite integration points (default: `default_m(dists)`).
- `check_variance`: Checks the requirement that marginal distributions have finite variance (default: `true`).
  If `true`, an `ArgumentError` is thrown if any distribution has a non-finite or undefined variance.
  Otherwise, if `false`, non-finite variances result in `NaN` matrix entries.
"""
function pearson_bounds(
        dists;
        degree::Real = default_degree(dists),
        m::Real = default_m(dists),
        check_variance::Bool = true
    )
    degree = Int(degree)
    m = Int(m)
    d = length(dists)

    stds = zeros(Float64, d)
    Threads.@threads for i in 1:d
        stds[i] = std(dists[i])
    end

    if check_variance
        invalid_indices = findall(!isfinite, stds)
        if !isempty(invalid_indices)
            msg = "All distributions are required to have a finite variance."
            for idx in invalid_indices
                msg *= "\n  Var[$(dists[idx])] = $(stds[idx]^2)"
            end
            throw(ArgumentError(msg))
        end
    end

    has_continuous = any(dist -> dist isa ContinuousUnivariateDistribution, dists)
    nodes, weights = has_continuous ? get_gauss_hermite(m) : (Float64[], Float64[])
    sqrt_inv_fact = sqrt.(get_inv_factorials(degree))

    A = Matrix{Float64}(undef, d, degree)
    Threads.@threads for i in 1:d
        if isfinite(stds[i])
            inv_s = 1.0 / stds[i]
            for k in 1:degree
                c_ik = extract_coef(dists[i], k, nodes, weights)
                A[i, k] = c_ik * inv_s * sqrt_inv_fact[k]
            end
        else
            A[i, :] .= NaN
        end
    end

    # Upper bound matrix: Upper = A * A'
    upper = A * A'

    # Lower bound matrix: Lower = A * B' where B[i, k] = A[i, k] * (-1)^k
    sign_vec = [isodd(k) ? -1.0 : 1.0 for k in 1:degree]
    B = A .* sign_vec'
    lower = A * B'

    clamp!(upper, -1.0, 1.0)
    clamp!(lower, -1.0, 1.0)

    return (lower = lower, upper = upper)
end
