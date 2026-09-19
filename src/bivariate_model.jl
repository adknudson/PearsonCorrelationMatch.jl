"""
    BivariateModel(d1, d2)


"""
mutable struct BivariateModel{D1 <: UnivariateDistribution, D2 <: UnivariateDistribution}
    const d1::D1
    const d2::D2
    degree::Int
    m::Int
    const coef::Vector{Float64}

    function BivariateModel(
            d1::UnivariateDistribution,
            d2::UnivariateDistribution;
            degree::Int = default_degree(d1, d2),
            m::Int = default_m(d1, d2)
        )
        std1, std2 = std(d1), std(d2)
        if !isfinite(std1) || !isfinite(std2)
            throw(
                ArgumentError(
                    "Both distributions are required to have a finite variance:\n" *
                        "  Var[$(d1)] = $(std1^2)\n" *
                        "  Var[$(d2)] = $(std2^2)"
                )
            )
        end

        nodes, weights = Float64[], Float64[]
        if d1 isa ContinuousUnivariateDistribution || d2 isa ContinuousUnivariateDistribution
            nodes, weights = get_gauss_hermite(m)
        end

        scale = 1.0 / (std1 * std2)
        inv_fact = get_inv_factorials(degree)

        c = zeros(Float64, degree)
        for k in 1:degree
            coef1 = extract_coef(d1, k, nodes, weights)
            coef2 = extract_coef(d2, k, nodes, weights)
            c[k] = coef1 * coef2 * inv_fact[k] * scale
        end

        return new{typeof(d1), typeof(d2)}(d1, d2, degree, m, c)
    end
end

function Base.show(io::IO, G::BivariateModel)
    return println(io, typeof(G), "(degree=$(G.degree), m=$(G.m))")
end

function update_coefficients!(G::BivariateModel, degree::Int, m::Int)
    degree > 0 || throw(ArgumentError("Degree must be positive. Got $degree"))
    m >= 2 || throw(ArgumentError("Number of quadrature points must be ≥2. Got $m"))

    empty!(G.coef)
    nodes, weights = Float64[], Float64[]
    if G.d1 isa ContinuousUnivariateDistribution || G.d2 isa ContinuousUnivariateDistribution
        nodes, weights = get_gauss_hermite(m)
    end

    std1 = std(G.d1)
    std2 = std(G.d2)
    scale = 1.0 / (std1 * std2)
    inv_fact = get_inv_factorials(degree)

    c = zeros(Float64, degree)
    for k in 1:degree
        coef1 = extract_coef(G.d1, k, nodes, weights)
        coef2 = extract_coef(G.d2, k, nodes, weights)
        c[k] = coef1 * coef2 * inv_fact[k] * scale
    end

    append!(G.coef, c)
    G.degree = degree
    G.m = m

    return G
end

"""
    model(ρ_z)

Evaluates ρ_x = G(ρ_z) for the bivariate distribution.
"""
(model::BivariateModel)(rho_z::Real) = clamp(eval_poly(model.coef, rho_z), -1, 1)

pearson_bounds(model::BivariateModel) = (model(-1), model(1))

function pearson_match(rho_x::Real, G::BivariateModel; maxiters::Int = 100, atol::Real = 1.0e-12)
    lb, ub = pearson_bounds(G)
    rho_x < lb && return -1.0
    rho_x > ub && return 1.0

    low, high = -1.0, 1.0

    for _ in 1:maxiters
        mid = (low + high) * 0.5
        if G(mid) < rho_x
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
