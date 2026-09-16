using Logging
using BenchmarkTools
using Distributions
using PearsonCorrelationMatch

Logging.disable_logging(Logging.Warn)

const SUITE = BenchmarkGroup()

const d_all = [
    Normal(10.0, 3.14),
    Binomial(25, 0.42),
    Poisson(6.8),
    Exponential(4.2),
    Gamma(3.1, 2.7),
    Chisq(5),
    TDist(8),
    Beta(2.4, 5.6),
    Uniform(),
    FDist(7, 3),
    Logistic(1.7, 2.3),
    Weibull(2.8, 5.4),
    Geometric(0.36),
    NegativeBinomial(7, 0.61),
    Cauchy(-1.5, 0.8),
    Pareto(2.2, 0.4),
    LogNormal(-0.7, 1.1),
]

const d_c = filter(d -> d isa ContinuousUnivariateDistribution, d_all)
const d_d = filter(d -> d isa DiscreteUnivariateDistribution, d_all)

function create_benchmark_group(distributions; seconds = 1)
    grp = BenchmarkGroup()

    U = Uniform(-1, 1)

    n = length(distributions)
    for i in 1:n
        d1 = distributions[i]
        d1_n = nameof(typeof(d1))
        for j in i:n
            d2 = distributions[j]
            d2_n = nameof(typeof(d2))
            grp["$(d1_n)-$(d2_n)"] = @benchmarkable pearson_match(x, $d1, $d2) setup = (x = rand($U)) seconds = seconds
        end
    end

    return grp
end

SUITE["Pairs"] = create_benchmark_group(d_all)

function run_all_pairs!(R, target_rho, distributions)
    n = length(distributions)
    for i in 1:n
        d1 = distributions[i]
        for j in i:n
            d2 = distributions[j]
            rho_z = pearson_match(target_rho, d1, d2)
            R[i, j] = R[j, i] = rho_z
        end
    end
    return R
end

SUITE["AllPairs"] = @benchmarkable run_all_pairs!(R, 0.32, $d_all) setup = (R = zeros(Float64, length($d_all), length($d_all)))
SUITE["ContinuousPairs"] = @benchmarkable run_all_pairs!(R, 0.32, $d_c) setup = (R = zeros(Float64, length($d_c), length($d_c)))
SUITE["DiscretePairs"] = @benchmarkable run_all_pairs!(R, 0.32, $d_d) setup = (R = zeros(Float64, length($d_d), length($d_d)))

results = run(SUITE, verbose = true)
