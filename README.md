# PearsonCorrelationMatch

[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Determine the correlation coefficient for a bivariate Gaussian copula so that the resulting samples following a Normal-to-anything (NORTA) step have the desired correlation.

This package is based on the paper by Xiao and Zhou, [Matching a correlation coefficient by a Gaussian copula](https://doi.org/10.1080/03610926.2018.1439962).


## API

Only two methods are exported:

- `pearson_bounds`
  - Determines the range of admissible Pearson correlations between two distributions
- `pearson_match`
  - Computes the Pearson correlation coefficient to be used in a bivariate Gaussian copula


## Usage

```julia
using PearsonCorrelationMatch # also re-exports Distributions.jl
using StatsFuns

p = 0.4 # target correlation

# Distributions can be continuous, discrete, or a mix
F = Gamma()
G = NegativeBinomial(20)

# estimate the pearson correlation bounds
pearson_bounds(F, G)
# (lower = -0.8385297744531974, upper = 0.9712817585733178)

# calculate the required input correlation
p_star = pearson_match(p, F, G)
# 0.4361868405991995

# apply the NORTA step
D = MultivariateNormal([0, 0], [1 p_star; p_star 1])
Z = rand(D, 1_000_000)
U = normcdf.(Z)
X1 = quantile.(F, U[1,:])
X2 = quantile.(G, U[2,:])

cor(X1, X2)
# 0.40007047985609534
```


## Details

It is highly recommended that any user of this package reads the reference paper first. The algorithm uses Gauss-Hermite polynomial interpolation to approximate the solution of the double integral. Target correlations near the Frechet bounds may be highly sensitive to the degree of the Hermite polynomial.


## Related Packages

While this package focuses on the bivariate case, it can be used to compute the input correlations between all pairs of marginal distributions. However, the resulting adjusted correlation matrix may not be positive definite. In that case, you can use the [NearestCorrelationMatrix.jl](https://github.com/adknudson/NearestCorrelationMatrix.jl) library to fix the adjusted correlation matrix before applying the NORTA step.


## References

* Xiao, Q., & Zhou, S. (2019). Matching a correlation coefficient by a Gaussian copula. Communications in Statistics-Theory and Methods, 48(7), 1728-1747.