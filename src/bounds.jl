"""
Computes the theoretical lower and upper bounds of the Pearson correlation coefficient
between two arbitrary marginal distributions using the polynomial approximation.
"""
function pearson_bounds(
        d1::UnivariateDistribution,
        d2::UnivariateDistribution;
        degree::Int = 20,
        m::Int = 40,
        propagate_nan::Bool = false
    )
    # Generate quadrature rules if at least one variable is continuous
    nodes, weights = Float64[], Float64[]
    if d1 isa ContinuousUnivariateDistribution || d2 isa ContinuousUnivariateDistribution
        nodes, weights = get_gauss_hermite(m)
    end

    std1, std2 = std(d1), std(d2)
    if isnan(std1) || isnan(std2)
        propagate_nan && return (NaN, NaN)
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


function pearson_bounds(margins; degree::Real = 20, m::Real = 40, kwargs...)
    d = length(margins)

    lower = Matrix{Float64}(undef, d, d)
    upper = Matrix{Float64}(undef, d, d)

    for i in 1:d, j in i:d
        d1 = margins[i]
        d2 = margins[j]
        l, u = pearson_bounds(d1, d2; degree = degree, m = m, kwargs...)
        lower[i, j] = lower[j, i] = l
        upper[i, j] = upper[j, i] = u
    end

    return (lower = lower, upper = upper)
end
