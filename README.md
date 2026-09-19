# PearsonCorrelationMatch

Determine the correlation coefficient for a bivariate Gaussian copula so that the resulting samples following a Normal-to-anything (NORTA) step have the desired correlation.

This package is based on the paper by Xiao and Zhou, [Matching a correlation coefficient by a Gaussian copula](https://doi.org/10.1080/03610926.2018.1439962).

## Problem

The NORTA (or Nataf) transformation is a standard technique for simulating random vectors with prescribed marginal distributions and a target dependence structure. It proceeds in two steps:

1. Draw $Z \sim \mathcal{N}(0, R_z)$ from a multivariate normal distribution with correlation matrix $R_z$.
2. Transform each component through the inverse CDF of its target marginal: $X_i = F_i^{-1}(\Phi(Z_i))$, where $F_i$ is the marginal CDF and $\Phi$ is the standard normal CDF.

Each $X_i$ then follows $F_i$ exactly, but the Pearson correlation between components of $X$ is not given directly by $R_z$. For a pair $(X_i, X_j)$ with standard deviations $\sigma_i$ and $\sigma_j$, the copula parameter $\rho_z$ (the $(i, j)$ entry of $R_z$) and the resulting Pearson correlation $\rho_x$ are linked by the integral equation (Cario & Nelson, 1997):

$$
\rho_x = G(\rho_z) = \frac{1}{\sigma_i \sigma_j} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} F_i^{-1}(\Phi(z_i)) \, F_j^{-1}(\Phi(z_j)) \, \varphi(z_i, z_j; \rho_z) \, dz_i \, dz_j
$$

where $\varphi(\cdot, \cdot; \rho_z)$ is the bivariate standard normal density with correlation $\rho_z$. The mapping $G$ is continuous and strictly increasing on $[-1, 1]$, so it is invertible: for any target $\rho_x$ inside the admissible range there exists a unique $\rho_z$ that achieves it. In practice, however:

- Except for a handful of special marginal pairs (Normal–Normal, Uniform–Uniform, ...), $G$ has no closed form.
- Evaluating the double integral repeatedly inside a root-finding loop is expensive, and when one or both marginals are discrete the integrand is discontinuous, which makes naive numerical integration inefficient or inaccurate.

This package solves that inversion problem. Given two marginal distributions and a target Pearson correlation $\rho_x$, it computes the copula correlation $\rho_z$ to place in $R_z$. It also reports the admissible range of $\rho_x$ for the pair — the Fréchet–Hoeffding bounds mapped through the Gaussian copula — outside of which no valid $\rho_z$ exists.

## API

Only two methods are exported:

- `pearson_bounds`
  - Determines the range of admissible Pearson correlations between two distributions
- `pearson_match`
  - Computes the Pearson correlation coefficient to be used in a bivariate Gaussian copula

## Usage

```julia
using PearsonCorrelationMatch, Distributions

rho_x = 0.4 # target correlation

# Distributions can be continuous, discrete, or a mix
d1 = Gamma()
d2 = NegativeBinomial(20)

# estimate the pearson correlation bounds
pearson_bounds(d1, d2)
# (lower = -0.8385297744531974, upper = 0.9712817585733178)

# calculate the required input correlation
rho_z = pearson_match(p, d1, d2)
# 0.4361868405991995

# apply the NORTA step
D = MultivariateNormal([0, 0], [1 rho_z; rho_z 1])
Z = rand(D, 1_000_000)
U = cdf.(Normal(), Z)
X1 = quantile.(d1, U[1,:])
X2 = quantile.(d2, U[2,:])

cor(X1, X2)
# 0.40007047985609534
```

## Details

It is recommended that any user of this package also read the reference paper.

Following Xiao and Zhou, the mapping $G(\rho_z)$ is approximated by a polynomial in $\rho_z$:

$$
\rho_x = G(\rho_z) \approx \sum_{k=1}^{n} c_k \rho_z^k, \qquad c_k = \frac{a_k(d_i) \, a_k(d_j)}{k! \, \sigma_i \sigma_j}
$$

whose coefficients come from Hermite polynomial expansions of the marginal transformations $F^{-1}(\Phi(z))$. For each marginal $d$, the package precomputes

$$
a_k(d) = \mathbb{E}\left[ H_k(Z) \, F^{-1}(\Phi(Z)) \right], \qquad Z \sim \mathcal{N}(0, 1)
$$

where $H_k$ is the probabilists' Hermite polynomial:

- **Continuous marginals** — $a_k$ is evaluated with an $m$-point Gauss–Hermite quadrature.
- **Discrete marginals** — $F^{-1}(\Phi(z))$ is a step function, so $a_k$ reduces to a finite sum over the support using the CDF breakpoints $\alpha_r = \Phi^{-1}(F(a_r))$. This follows from Taylor-expanding the bivariate normal CDF at $\rho_z = 0$, and avoids numerically integrating a discontinuous integrand entirely.
- **Mixed pairs** combine the two cases; the resulting polynomial has the same form.

Because $G$ is strictly increasing, inverting the polynomial only requires a bracketing (bisection) search on $[-1, 1]$, which is far cheaper than repeatedly evaluating the double integral. The admissible bounds for $\rho_x$ are obtained by evaluating the polynomial at $\rho_z = \pm 1$. For marginal pairs with known closed-form solutions (Normal–Normal, Normal–Uniform, Uniform–Uniform, Normal–LogNormal, LogNormal–LogNormal), the package bypasses the approximation and uses the exact formula.

### Choosing the approximation degree

The truncation degree $n$ and quadrature order $m$ trade accuracy against stability:

- Defaults are $n = 15$ for continuous marginals and $n = 20$ when a discrete marginal is involved (a pair uses the larger of the two), with $m = \max(25, \lfloor 1.5\, n \rfloor)$ Gauss–Hermite points ($m = 0$ if both marginals are discrete). See `default_degree` and `default_m`.
- By the Weierstrass approximation theorem a polynomial of sufficiently high degree can approximate $G$ as closely as desired, but polynomials of too high a degree oscillate near the edges of $[-1, 1]$ (Runge phenomenon), so increasing $n$ does not always improve accuracy.
- Target correlations near the Fréchet bounds are the most sensitive case: if results appear unstable there, increase $n$ and $m$ together and verify that the answer has converged before trusting it.

## Related Packages

While this package focuses on the bivariate case, it can be used to compute the input correlations between all pairs of marginal distributions. However, the resulting adjusted correlation matrix may not be positive definite. In that case, you can use the [NearestCorrelationMatrix.jl](https://github.com/adknudson/NearestCorrelationMatrix.jl) library to fix the adjusted correlation matrix before applying the NORTA step.

## References

- Cario, M. C., & Nelson, B. L. (1997). Modeling and generating random vectors with arbitrary marginal distributions and correlation matrix (pp. 1-19). Technical Report, Department of Industrial Engineering and Management Sciences, Northwestern University, Evanston, Illinois.
- Xiao, Q., & Zhou, S. (2019). Matching a correlation coefficient by a Gaussian copula. Communications in Statistics-Theory and Methods, 48(7), 1728-1747.
