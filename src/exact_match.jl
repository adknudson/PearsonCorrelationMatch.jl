"""
    pearson_match(rho_x::Float64, ::Normal, ::Normal)

Analytical exact solution for Normal-Normal pairs: `rho_z = rho_x`.
"""
function pearson_match(rho_x::Float64, ::Normal, ::Normal; kwargs...)
    return clamp(rho_x, -1.0, 1.0)
end

"""
    pearson_match(rho_x::Float64, d1::LogNormal, d2::LogNormal)

Analytical exact solution for LogNormal-LogNormal pairs.
"""
function pearson_match(rho_x::Float64, d1::LogNormal, d2::LogNormal; kwargs...)
    s1, s2 = d1.σ, d2.σ
    denom = sqrt(expm1(s1^2) * expm1(s2^2))
    min_rho = expm1(-s1 * s2) / denom
    max_rho = expm1(s1 * s2) / denom
    rho_x <= min_rho && return -1.0
    rho_x >= max_rho && return 1.0
    return log(1.0 + rho_x * denom) / (s1 * s2)
end

"""
    pearson_match(rho_x::Float64, d1::Normal, d2::LogNormal)

Analytical exact solution for Normal-LogNormal pairs.
"""
function pearson_match(rho_x::Float64, ::Normal, d2::LogNormal; kwargs...)
    s = d2.σ
    factor = s / sqrt(expm1(s^2))
    rho_x <= -factor && return -1.0
    rho_x >= factor && return 1.0
    return rho_x / factor
end

pearson_match(rho_x::Float64, d1::LogNormal, d2::Normal; kwargs...) = pearson_match(rho_x, d2, d1; kwargs...)

"""
    pearson_match(rho_x::Float64, ::Uniform, ::Uniform)

Analytical exact solution for Uniform-Uniform pairs: `rho_z = 2 sin(π * rho_x / 6)`.
"""
function pearson_match(rho_x::Float64, ::Uniform, ::Uniform; kwargs...)
    rho_x_clamped = clamp(rho_x, -1.0, 1.0)
    return 2.0 * sin(pi * rho_x_clamped / 6.0)
end

"""
    pearson_match(rho_x::Float64, ::Normal, ::Uniform)

Analytical exact solution for Normal-Uniform pairs: `rho_z = rho_x * sqrt(π / 3)`.
"""
function pearson_match(rho_x::Float64, ::Normal, ::Uniform; kwargs...)
    max_rho = sqrt(3.0 / pi)
    rho_x <= -max_rho && return -1.0
    rho_x >= max_rho && return 1.0
    return rho_x * sqrt(pi / 3.0)
end

pearson_match(rho_x::Float64, d1::Uniform, d2::Normal; kwargs...) = pearson_match(rho_x, d2, d1; kwargs...)
