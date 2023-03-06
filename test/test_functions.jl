"""
    Are two matrices within a certain tolerance?
    Use to simplify tests.
    """
#within(A::Union{Matrix{Quantity},Vector{Quantity}},B::Union{Matrix{Quantity},Vector{Quantity}},tol) =  maximum(abs.(ustrip.(A - B))) < tol
within(A,B,tol) =  maximum(abs.(ustrip.(A - B))) < tol
# next one checks for unit consistency as well. Is more stringent.
within(A::T,B::T,tol) where T <: UnitfulLinearAlgebra.AbstractUnitfulType =  maximum(abs.(parent(A - B))) < tol

function test_Unitful_matrices()
    for i = 1:3
        A,B,r,q = random_UnitfulMatrix_vector_pairs(i)
        test_matrices(A,B,r,q)
    end
end
function test_UnitfulDim_matrices()
    for i = 1:3
        A,B,r,q = random_UnitfulDimMatrix_vector_pairs(i)
        test_matrices(A,B,r,q)
    end
end

function test_matrices(A,B,r,q)
    test_interface(B)
    test_interface(r)
    @test A==Matrix(B)
    @test within(A*q,Matrix(B*r),1.0e-10)
    @test isequal(uniform(A),uniform(B))
    @test isequal(left_uniform(A),left_uniform(B))
    @test isequal(right_uniform(A),right_uniform(B))
    @test ~dimensionless(B)
end

function random_UnitfulMatrix_vector_pairs(i)
    # A - B matrix pairs
    # r - q vector pairs
    if i == 1
        p = [1.0m, 9.0s]
        #q̃ = [-1.0K, 2.0]
        q̃ = [-1.0K, 2.0m]
    elseif i == 2
        p = [1.0m, 3.0s, 5.0u"m/s"]
        q̃ = [-1.0K]
    elseif i == 3
        p = [1.0m, 3.0s]
        q̃ = [-1.0, 2.0]
    end
    q = ustrip.(q̃).*unit.(1 ./q̃)
    A = p*q̃'

    B = UnitfulMatrix(ustrip.(A),unit.(p),unit.(q))
    r = UnitfulMatrix(ustrip.(q),unit.(q),exact=false)
    return A,B,r,q
end

function random_UnitfulDimMatrix_vector_pairs(i)
    # A - B matrix pairs
    # r - q vector pairs
    if i == 1
        p = [1.0m, 9.0s]
        #q̃ = [-1.0K, 2.0]
        q̃ = [-1.0K, 2.0m]
    elseif i == 2
        p = [1.0m, 3.0s, 5.0u"m/s"]
        q̃ = [-1.0K]
    elseif i == 3
        p = [1.0m, 3.0s]
        q̃ = [-1.0, 2.0]
    end
    q = ustrip.(q̃).*unit.(1 ./q̃)
    A = p*q̃'

    B = UnitfulDimMatrix(ustrip.(A),unit.(p),unit.(q),dims=(:sealevel,:coefficients))
    r = UnitfulDimMatrix(ustrip.(q),unit.(q),dims=(:coefficients),exact=false)
    return A,B,r,q
end

# function test_cat()
# vcat and hcat fail, both sides of equation fail
# y1 = B*q2
# Bvcat = vcat(B,B)
# @test Bvcat*q2 == vcat(y1,y1)

# Bhcat = hcat(B,B)
# @test Bhcat*vcat(q,q) == 2y1
#
#end

function test_interface(x::AbstractUnitfulVecOrMat)
    @testset "types" begin
        @test parent(x) isa AbstractArray # Is this absolutely necessary?
        @test dims(x) isa DimensionalData.DimTuple
        #@test refdims(x) isa Tuple
    end

    @testset "size" begin
        @test length(dims(x)) == ndims(x)
        @test map(length, dims(x)) === size(x) == size(parent(x))
    end

    @testset "rebuild" begin
        # argument version
        x1 = rebuild(x, parent(x), dims(x), exact(x))
        # keyword version, will work magically using ConstructionBase.jl if you use the same fieldnames.
        # If not, define it and remap these names to your fields.
        x2 = rebuild(x; data=parent(x), dims=dims(x))
        # all should be identical. If any fields are not used, they will always be `nothing` or `()` for `refdims`
        @test parent(x) === parent(x1) === parent(x2)
        @test dims(x) === dims(x1) === dims(x2)
        #@test refdims(x) === refdims(x1) === refdims(x2)
        #@test metadata(x) === metadata(x1) === metadata(x2)
    end
end

function test_interface(x::AbstractUnitfulDimVecOrMat)
    @testset "types" begin
        @test parent(x) isa AbstractArray # Is this absolutely necessary?
        @test unitdims(x) isa DimensionalData.DimTuple
        @test dims(x) isa DimensionalData.DimTuple
        @test refdims(x) isa Tuple
    end

    @testset "size" begin
        @test length(unitdims(x)) == ndims(x)
        @test length(dims(x)) == ndims(x)
        @test map(length, dims(x)) === size(x) == size(parent(x))
        @test map(length, unitdims(x)) === size(x) == size(parent(x))
    end

    @testset "rebuild" begin
        # argument version
        x1 = rebuild(x, parent(x), unitdims(x)) #, exact=exact(x))
        # keyword version, will work magically using ConstructionBase.jl if you use the same fieldnames.
        # If not, define it and remap these names to your fields.
        x2 = rebuild(x; data=parent(x), unitdims= unitdims(x), dims=dims(x))
        # all should be identical. If any fields are not used, they will always be `nothing` or `()` for `refdims`
        @test parent(x) === parent(x1) === parent(x2)
        @test dims(x) === dims(x1) === dims(x2)
        @test unitdims(x) === unitdims(x1) === unitdims(x2)
        @test refdims(x) === refdims(x1) === refdims(x2)
        @test metadata(x) === metadata(x1) === metadata(x2)
    end
end

function Unitful_dimensionless_pair(i)
    # Not all dimensionless matrices have
    # dimensionless domain and range
    if i == 1
        p = [1.0m², 3.0m²]
    elseif i ==2
        p = [1.0m², 3.0u"m^3"]
    end
    q̃ = [-1.0u"m^-2", 2.0u"m^-2"]
    q = ustrip.(q̃).*unit.(1 ./q̃)
    # outer product to make a multipliable matrix
    A = p*q̃'
    B = UnitfulMatrix(ustrip.(A),unit.(p),unit.(q))
    return A,B
end
function UnitfulDim_dimensionless_pair(i)
    # Not all dimensionless matrices have
    # dimensionless domain and range
    if i == 1
        p = [1.0m², 3.0m²]
    elseif i ==2
        p = [1.0m², 3.0u"m^3"]
    end
    q̃ = [-1.0u"m^-2", 2.0u"m^-2"]
    q = ustrip.(q̃).*unit.(1 ./q̃)
    # outer product to make a multipliable matrix
    A = p*q̃'
    B = UnitfulDimMatrix(ustrip.(A),unit.(p),unit.(q),dims=(:one,:two))
    return A,B
end

function test_dimensionless(i,A,B)
    if i == 1
        @test dimensionless(B)
        @test dimensionless(A)
    elseif i ==2
        @test ~dimensionless(B)
        @test ~dimensionless(A)
    end
end
function test_Unitful_dimensionless()
    for i = 1:2
        A,B = Unitful_dimensionless_pair(i)
        test_dimensionless(i,A,B)
    end
end
function test_UnitfulDim_dimensionless()
    for i = 1:2
        A,B = UnitfulDim_dimensionless_pair(i)
        test_dimensionless(i,A,B)
    end
end

function vandermonde_UnitfulMatrix_vector_pair(uy,ux)
    u1 = uy
    u2 = uy/ux
    u3 = uy/ux/ux
    k = 3
    Eparent = hcat(randn(k),randn(k),randn(k))
    E = UnitfulMatrix(Eparent,fill(m,k),[u1,u2,u3])
    x = UnitfulMatrix(randn(k,1),[u1,u2,u3],[unit(1.0)])
    return E,x
end
function vandermonde_UnitfulDimMatrix_vector_pair(uy,ux)
    u1 = uy
    u2 = uy/ux
    u3 = uy/ux/ux
    k = 3
    Eparent = hcat(randn(k),randn(k),randn(k))
    E = UnitfulDimMatrix(Eparent,fill(m,k),[u1,u2,u3],dims=(:sealevel,:coefficients))
    x = UnitfulDimMatrix(randn(k,1),[u1,u2,u3],[unit(1.0)],dims=(:coefficients,:nothing))
    return E,x
end

function solve_polynomial(y,E,x)
    # E should be exact
    @test exact(E)
    @test ~singular(E)
    det(E)
    x̂ = E\y
    @test within(x̂,x, 1e-10)

    Eshift = convert_unitdomain(E,unitrange(x))
    E⁻¹ = inv(Eshift)
    Eᵀ = transpose(Eshift)
    @test Eshift[2,1] == Eᵀ[1,2]
    @test E[2,1] == Eᵀ[1,2]

    x′ = Eshift\y
    @test within(x′,x,1e-10)

    # next try to solve by LU
    #Elu = lu(Eshift)
    #@test within(E[Elu.p,:],Matrix(Elu.L*Elu.U),1e-10)
    #x̆ = Elu \ y 
    #@test within(x̆,x, 1e-10)
end

function test_polynomial_UnitfulDimMatrix()
    uy = u"m"
    ux = u"s"
    E,x=vandermonde_UnitfulDimMatrix_vector_pair(uy,ux)
    y = E*x
    solve_polynomial(y,E,x)
end    
function test_polynomial_UnitfulMatrix()
    uy = u"m"
    ux = u"s"
    E,x=vandermonde_UnitfulMatrix_vector_pair(uy,ux)
    y = E*x
    solve_polynomial(y,E,x)
end    
