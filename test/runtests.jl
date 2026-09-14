using SafeTestsets

@safetestset "Quality Assurance" include("qa.jl")
@safetestset "Pearson Matching" include("matching.jl")
@safetestset "Pearson Bounds" include("bounds.jl")
