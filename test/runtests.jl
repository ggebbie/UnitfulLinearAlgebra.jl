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

"""
    Are two matrices within a certain tolerance?
    Use to simplify tests.
    """
#within(A::Union{Matrix{Quantity},Vector{Quantity}},B::Union{Matrix{Quantity},Vector{Quantity}},tol) =  maximum(abs.(ustrip.(A - B))) < tol
within(A,B,tol) =  maximum(abs.(ustrip.(A - B))) < tol
# next one checks for unit consistency as well. Is more stringent.
within(A::AbstractUnitfulVecOrMat,B::AbstractUnitfulVecOrMat,tol) =  maximum(abs.(parent(A - B))) < tol

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


        
