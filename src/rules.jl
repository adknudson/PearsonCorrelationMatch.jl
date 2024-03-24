"""
    _rule(p, d1, d2, n)

Finds ``G(p) = pₓ`` where ``p`` is the correlation used in a bivariate Gaussian copula, and
``pₓ`` is the resulting correlation between the two marginals after the NORTA step.
"""
function _rule end


"""
_invrule(p, d1, d2, n)

Finds ``Ginv(pₓ) = p`` where ``pₓ`` is the target correlation between two marginals, and
``p`` is the correlation to be used in a bivariate Gaussian copula.
"""
function _invrule end


# Uniform-Uniform
_invrule(p::Float64, ::Uniform, ::Uniform, ::Int) = 2 * sinpi(p / 6)
_rule(p::Float64, ::Uniform, ::Uniform, ::Int) = asin(p / 2) * 6 / π
