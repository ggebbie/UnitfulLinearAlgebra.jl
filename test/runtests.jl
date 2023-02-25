using Revise
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
within(A,B,tol) =  maximum(abs.(ustrip.(A - B))) < tol


@testset "UnitfulLinearAlgebra.jl" begin

    @testset "Unitful" begin
        include("test_Unitful.jl")
    end

    @testset "UnitfulMatrix" begin
        include("test_UnitfulMatrix.jl")
    end

    #@testset "UnitfulDimMatrix" begin
    #    include("test_UnitfulDimMatrix.jl")
    #end
    
end


        # NOT TESTED
        # @testset "dimarrays" begin

        #     using DimensionalData: @dim
        #     @dim Units "units"
        #     p = unit.([1.0m, 9.0s])
        #     q̃ = unit.([-1.0K, 2.0])
        #     U = zeros(Units(p),Units(q̃))
        #     Unum = [1.0 2.0; 3.0 4.0]
        #     V = DimMatrix(Unum,(Units(p),Units(q̃)),exact=true)

        #     vctr = DimMatrix(rand(2),(Units(q̃)),exact=true)

        #     Units(p.^-1)
        #     inv(Matrix(V.data))
        #     Vi = DimMatrix(inv(Unum),(Units(q̃),Units(p)));
        #     Vi*V;
            
            # years = (1990.0:2000.0)
            # ny = length(years)
            
            # # estimate defined for locations/regions
            # regions = [:NATL,:ANT]
            # nr = length(regions)

            # units1 = [u"m",u"s"]
            # units2 = [u"m*s",u"s^2"]
            # nu = length(units)

            # V = rand(Region(regions),YearCE(years))
            # #U = rand(UnitRange(units2),UnitDomain(units1))
            # U = rand(Units(units1),Units(units2))
            # #u = rand(UnitDomain(units1))
            # u = rand(Unit(units2))

            # test = dims(U)

            # E = rand(YearCE(years))
            # transpose(V)
            # V*transpose(V)
            # transpose(V)*V
            # # flatten V for vector multiplication?
            # V⃗ = vec(V)
            # V⃗c::Vector{Float64} = reshape(V,nr*ny)
            # V[:] # same thing
            # V2 = DimArray(reshape(vec(V),size(V)),(Region(regions),YearCE(years)))
            # @test V2 == V
            # @test vec(V) == V[:]
            
            # # 4D for covariance? sure, it works.
            # C = rand(Region(regions),YearCE(years),Region(regions),YearCE(years))

            # Cmat::Matrix{Float64} = reshape(C,nr*ny,nr*ny)

            # reshape(Cmat,size(C))

            # # reconstruct?
            # C2 = DimArray(reshape(Cmat,size(V)[1],size(V)[2],size(V)[1],size(V)[2]),(Region(regions),YearCE(years),Region(regions),YearCE(years)))

            # C == C2

            # V[Region(1),YearCE(5)]
            # V[Region(1)]
            # sum(V,dims=Region)

            # # NATL values
            # V[Region(At(:NATL))]
            # V[:NATL]
            
            # # get the year 1990
            # V[YearCE(Near(years[5]))]
            # V[YearCE(Near(1992.40u"yr"))]
            # V[YearCE(Interval(1992.5u"yr",1996u"yr"))]
            # size(V[YearCE()])
            
            # # order doesn't need to be known
            # test[Region(1),YearCE(5)]
            # test[YearCE(5),Region(1)]

            # x = BLUEs.State(V,C)

        #end
        
