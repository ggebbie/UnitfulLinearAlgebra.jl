using Revise
using UnitfulLinearAlgebra
using Unitful
using LinearAlgebra
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

        E⁻¹ = inv(E)
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

        E⁻¹ = inv(E)
        x̃ = E⁻¹ * (E * x)

        [@test isapprox(x̃[i],x[i]) for i in 1:length(x)]
    end    

    @testset "underdetermined problems" begin

        # matrix with columns of like type

        # any two combinations of units will work
        # can't easily get a list of units to draw from
        u1 = u"m/s"
        u2 = u"m/s^2"
        u3 = u"m/s^3"

        K = 2 # rank
        # rectangular (wide) matrices
        E = hcat(randn(K)*100u"percent",randn(K)u1/u2,randn(K)u1/u3)

        s = vcat(randn()u1,randn()u2,randn()u3) # sqrt of diagonal elements
        S = diagonal_matrix(s)
        SET = S*E'
        ESET = E*SET        
        y = randn(K)u1

        #inv_unitful(ESET) # use unitful for Matrix{Any}
        μ = inv_unitful(ESET)*y

        𝓛 = μ'*y
        @test 𝓛 == ustrip(𝓛) # nondimensional?
        @test dimension(𝓛) == NoDims
        # check if it runs
        SET*μ
    end

    @testset "svd" begin
	E = [1/2 1/2; 1/4 3/4; 3/4 1/4]u"dbar*s"
	U,λ,V = svd_unitful(E)
	Λ = Diagonal(λ)
        K = length(λ) # rank
	y = 5randn(3)u"s"
	σₙ = randn(3)u"s"
	Cₙₙ = diagonal_matrix(σₙ)
	W⁻¹ = diagonal_matrix([1,1,1]u"1/s^2")
	x̃ = inv(E'*W⁻¹*E)*(E'*W⁻¹*y)
        [@test isequal(x̃[i]/ustrip(x̃[i]),1.0u"dbar^-1") for i in 1:length(x̃)]
    end
    
end
