#using Revise
using UnitfulLinearAlgebra
using Unitful
using LinearAlgebra
using SparseArrays
using DimensionalData
using DimensionalData: @dim
using Test

ENV["UNITFUL_FANCY_EXPONENTS"] = true
m = u"m"
s = u"s"
K = u"K"
m² = u"m^2"
permil = Unitful.FixedUnits(u"permille")

include("test_functions.jl")

@testset "UnitfulLinearAlgebra.jl" begin

    @testset "Unitful" begin
        include("test_Unitful.jl")
    end

    @testset "UnitfulMatrix" begin
        include("test_UnitfulMatrix.jl")
    end

    @testset "UnitfulDimMatrix" begin
        include("test_UnitfulDimMatrix.jl")
    end
    
end


        
