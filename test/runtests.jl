using Revise
using UnitfulLinearAlgebra
using Unitful
using Test

@testset "UnitfulLinearAlgebra.jl" begin
    # Write your tests here.

    @testset "inverse 2x2" begin

        # matrix with columns of like type

        # any two combinations of units will work
        # can't easily get a list of units to draw from
        u1 = u"m"
        u2 = u"m/s"
        u3 = u1/u2
        
        # i.e., trend analysis
        E = hcat(randn(2),randn(2)u3)
        y = randn(2)u1
        x = [randn()u1; randn()u2] 

        E⁻¹ = inv_unitful(E)
        x̃ = E⁻¹ * (E * x)

        [@test isapprox(x̃[i],x[i]) for i in 1:length(x)]
    end

    @testset "inverse 3x3" begin
        # can't easily get a list of units to draw from
        u1 = u"m"
        u2 = u"m/s"
        u3 = u"m/s^2"
        
        # i.e., trend analysis
        K = 3
        E = hcat(randn(K),randn(K)u1/u2,randn(K)u1/u3)
        y = randn(K)u1
        x = [randn()u1; randn()u2; randn()u3] 

        E⁻¹ = inv_unitful(E)
        x̃ = E⁻¹ * (E * x)

        [@test isapprox(x̃[i],x[i]) for i in 1:length(x)]
    end    

    @testset "underdetermined problems" begin

        # matrix with columns of like type

        # any two combinations of units will work
        # can't easily get a list of units to draw from
        u1 = u"m"
        u2 = u"m/s"
        u3 = u"m/s^2"

        K = 2 # rank
        # rectangular (wide) matrices
        E = hcat(randn(K),randn(K)u1/u2,randn(K)u1/u3)

        s = vcat(randn()u1^2,randn()u2^2,randn()u3^2)
        S = diagonal_matrix(s)
        SET = S*E'
        ESET = E*SET        

        μ = inv_unitful(ESET)*y

        𝓛 = μ'*y
        @test 𝓛 == ustrip(𝓛) # nondimensional?

        # check if it runs
        SET*(iESET*y)

    end

end
