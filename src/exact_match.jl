# --- Normal - Normal ---
# Exact relationship: ρ_z = ρ_x
function pearson_match(rho_x::Float64, ::Normal, ::Normal; kwargs...)
    return clamp(rho_x, -1.0, 1.0)
end

# --- LogNormal - LogNormal ---
# Exact relationship: ρ_x = (exp(ρ_z * σ1 * σ2) - 1) / sqrt((exp(σ1^2) - 1) * (exp(σ2^2) - 1))
function pearson_match(rho_x::Float64, d1::LogNormal, d2::LogNormal; kwargs...)
    s1, s2 = d1.σ, d2.σ
    denom = sqrt(expm1(s1^2) * expm1(s2^2))

    # Calculate admissibility bounds
    min_rho = expm1(-s1 * s2) / denom
    max_rho = expm1(s1 * s2) / denom

    if rho_x <= min_rho
        return -1.0
    elseif rho_x >= max_rho
        return 1.0
    end

    arg = 1.0 + rho_x * denom
    return log(arg) / (s1 * s2)
end

# --- Normal - LogNormal ---
# Exact relationship: ρ_x = ρ_z * σ / sqrt(exp(σ^2) - 1)
function pearson_match(rho_x::Float64, d1::Normal, d2::LogNormal; kwargs...)
    s = d2.σ
    factor = s / sqrt(expm1(s^2))

    max_rho = factor
    if rho_x <= -max_rho
        return -1.0
    elseif rho_x >= max_rho
        return 1.0
    end

    return rho_x / factor
end

pearson_match(rho_x::Float64, d1::LogNormal, d2::Normal; kwargs...) = pearson_match(rho_x, d2, d1; kwargs...)

# --- Uniform - Uniform ---
# Exact relationship: ρ_x = (6 / π) * asin(ρ_z / 2) => ρ_z = 2 * sin(π * ρ_x / 6)
function pearson_match(rho_x::Float64, ::Uniform, ::Uniform; kwargs...)
    rho_x_clamped = clamp(rho_x, -1.0, 1.0)
    return 2.0 * sin(pi * rho_x_clamped / 6.0)
end

# --- Normal - Uniform ---
# Exact relationship: ρ_x = ρ_z * sqrt(3 / π) => ρ_z = ρ_x * sqrt(π / 3)
function pearson_match(rho_x::Float64, ::Normal, ::Uniform; kwargs...)
    max_rho = sqrt(3.0 / pi)
    if rho_x <= -max_rho
        return -1.0
    elseif rho_x >= max_rho
        return 1.0
    end

    return rho_x * sqrt(pi / 3.0)
end

pearson_match(rho_x::Float64, d1::Uniform, d2::Normal; kwargs...) = pearson_match(rho_x, d2, d1; kwargs...)
