using SafeTestsets

@safetestset "Quality Assurance" include("qa.jl")
@safetestset "Code Formatting" include("format_check.jl")
@safetestset "Utilities" include("internals.jl")
@safetestset "Pearson Matching" include("matching.jl")
@safetestset "Pearson Bounds" include("bounds.jl")
