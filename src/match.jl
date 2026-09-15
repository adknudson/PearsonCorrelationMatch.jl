"""
Determines the Gaussian copula parameter ρ_z to match the target Pearson correlation ρ_x
between two arbitrary marginal distributions (Continuous or Discrete).
"""
function pearson_match(
        rho_x::Float64,
        d1::UnivariateDistribution,
        d2::UnivariateDistribution;
        degree::Int = default_degree(d1, d2),
        m::Int = default_m(d1, d2),
        maxiters::Int = 100,
        atol::Float64 = 1.0e-12
    )
    # Generate quadrature rules if at least one variable is continuous
    nodes, weights = Float64[], Float64[]
    if d1 isa ContinuousUnivariateDistribution || d2 isa ContinuousUnivariateDistribution
        nodes, weights = get_gauss_hermite(m)
    end

    std1, std2 = std(d1), std(d2)
    scale = 1.0 / (std1 * std2)
    inv_fact = get_inv_factorials(degree)

    # Precompute coefficients C_k
    c = zeros(Float64, degree)
    for k in 1:degree
        coef1 = extract_coef(d1, k, nodes, weights)
        coef2 = extract_coef(d2, k, nodes, weights)
        c[k] = coef1 * coef2 * inv_fact[k] * scale
    end

    # Check physical admissibility bounds
    G_neg1 = eval_poly(c, -1.0)
    G_pos1 = eval_poly(c, 1.0)

    if rho_x < G_neg1
        @warn "Target ρ_x ($rho_x) is below the admissible bound ($G_neg1). Returning -1.0."
        return -1.0
    elseif rho_x > G_pos1
        @warn "Target ρ_x ($rho_x) is above the admissible bound ($G_pos1). Returning 1.0."
        return 1.0
    end

    # Bisection search to find the root on [-1, 1]
    low, high = -1.0, 1.0
    for _ in 1:maxiters
        mid = (low + high) * 0.5
        if eval_poly(c, mid) < rho_x
            low = mid
        else
            high = mid
        end
        if high - low < atol
            break
        end
    end

    return (low + high) * 0.5
end

"""
    pearson_match(R_x::AbstractMatrix{Float64}, dists::Vector{<:UnivariateDistribution}; degree::Int=15, m::Int=25)

Computes the pairwise Gaussian copula correlation matrix `R_z` corresponding to a target
Pearson correlation matrix `R_x` for a list of marginal distributions `dists`.
"""
function pearson_match(
        R_x::AbstractMatrix{Float64},
        dists::Vector{<:UnivariateDistribution};
        degree::Int = default_degree(dists),
        m::Int = default_m(dists),
        maxiters::Int = 100,
        atol::Float64 = 1.0e-12
    )
    n_dists = length(dists)
    @assert size(R_x) == (n_dists, n_dists) "R_x must be an $(n_dists)x$(n_dists) matrix"

    # Helper function to check if a pair has an exact dispatch overload
    has_exact_match(d1::Type, d2::Type) = hasmethod(pearson_match, Tuple{Float64, d1, d2})

    # Precompute standard deviations & polynomial expansion coefficients in parallel
    inv_fact = get_inv_factorials(degree)
    has_continuous = any(d -> d isa ContinuousUnivariateDistribution, dists)
    nodes, weights = has_continuous ? get_gauss_hermite(m) : (Float64[], Float64[])

    stds = zeros(Float64, n_dists)
    C = Matrix{Float64}(undef, n_dists, degree)

    Threads.@threads for i in 1:n_dists
        stds[i] = std(dists[i])
        # Skip coefficient extraction if distribution is purely part of exact pairs
        for k in 1:degree
            C[i, k] = extract_coef(dists[i], k, nodes, weights)
        end
    end

    pairs = [(i, j) for i in 1:n_dists for j in (i + 1):n_dists]
    R_z = Matrix{Float64}(undef, n_dists, n_dists)
    for i in 1:n_dists
        R_z[i, i] = 1.0
    end

    Threads.@threads for idx in eachindex(pairs)
        i, j = pairs[idx]
        d1, d2 = dists[i], dists[j]
        target_rho = R_x[i, j]

        # 1. Fast Path: Dispatch to exact closed-form method if overload exists
        if has_exact_match(typeof(d1), typeof(d2))
            rho_z_ij = pearson_match(target_rho, d1, d2)
            R_z[i, j] = R_z[j, i] = rho_z_ij
            continue
        end

        # 2. Slow Path: Inexact polynomial approximation & root-finding
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
