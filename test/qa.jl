using Test
using Aqua, PearsonCorrelationMatch

@testset "Quality Assurance" begin
    Aqua.test_all(PearsonCorrelationMatch; ambiguities=false)
    Aqua.test_ambiguities(PearsonCorrelationMatch)
end
