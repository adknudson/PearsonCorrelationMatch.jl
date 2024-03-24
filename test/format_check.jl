using Test
using PearsonCorrelationMatch
using JuliaFormatter

if VERSION >= v"1.6"
    @test JuliaFormatter.format(PearsonCorrelationMatch; verbose=false, overwrite=false)
end
