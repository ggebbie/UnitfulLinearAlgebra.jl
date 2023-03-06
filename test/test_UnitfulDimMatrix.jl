@testset "interface" begin
    m = u"m"
    s = u"s"
    test_UnitfulDim_matrices()
end
@testset "dimensionless" begin
    test_UnitfulDim_dimensionless()
end
@testset "polynomial fitting" begin
    test_polynomial_UnitfulDimMatrix()
end    

@testset "time-average" begin
    m = u"m"
    # create matrix for mean value problem
    n = 10 # 10 observations

    # axis dimensions
    @dim YearCE "years Common Era"
    @dim Region "location"
    regions = [:NATL]
    yr = u"yr"
    years = range(1990.0yr,stop=1999.0yr,length=n)

    # observations have units of meters
    udomain = [m]
    urange = fill(m,n)
    Eparent = ones(n,1)

    E = UnitfulDimMatrix(Eparent,urange,udomain,dims=(YearCE(years),Region(regions)))

    # add UnitfulDimVector constructor
    x = UnitfulDimMatrix(randn(1,1),[m],[unit(1.0)],dims=(Region(regions),:mean))

    # matrix multiplication with UnitfulDimMatrix
    #y = _rebuildmul(E,x)
    y = E*x
    
end


@testset "PEMDAS" begin
    m = u"m"
    s = u"s"
    K = u"K"
    stup = (1,5)

    #can we add two matrices with same dimensions and units? 
    a1 = UnitfulDimMatrix(randn(stup), fill(K, stup[1]), fill(unit(1.0), stup[2]), dims = (X = [5m], Ti = (1:stup[2])s))
    a2_add = UnitfulDimMatrix(randn(stup), fill(K, stup[1]), fill(unit(1.0), stup[2]), dims = (X = [5m], Ti = (1:stup[2])s))
    @test parent(a1 + a2_add) == parent(a1) .+ parent(a2_add)
    #subtraction 
    @test parent(a1 - a2_add) == parent(a1) .- parent(a2_add)
    
    #do we throw an error when the two matrices use the same dimensions, but are
    #sampled at different points 
    a2_add_wrongdims = UnitfulDimMatrix(randn(stup), fill(K, stup[1]), fill(unit(1.0), stup[2]), dims = (X = [5m], Ti = (6:stup[2]+5)s))
    @test_throws DimensionMismatch a1 + a2_add_wrongdims

    #multiply by scalar
    @test parent(5K * a1) == 5 * parent(a1)
    @test parent(5 * a1) == 5 * parent(a1)        

    #inner product
    a2_multiply_inner = UnitfulDimMatrix(randn(stup[2], stup[1]), fill(unit(1.0), stup[2]), fill(K, stup[1]), dims = (Ti = (1:stup[2])s, X = [5m]))
    @test parent(a1 * a2_multiply_inner) == parent(a1) * parent(a2_multiply_inner)
    #a2_multiply_outer = copy(a1)
    #a1 * a2_multiply_outer
end

@testset "functions!" begin
    m = u"m"
    s = u"s"
    u1 = m
    u2 = m/s
    u3 = m/s/s

    k = 3
    Eparent = hcat(randn(k),randn(k),randn(k))
    #y = randn(k)u1
    E = UnitfulDimMatrix(Eparent,fill(m,k),[u1,u2,u3],dims=(:sealevel,:coefficients))
    @test parent(inv(E)) == inv(parent(E))
    det(E)
    @test ~singular(E) # fixed now (gg)
end

#  useful config for DimArrays
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
    
#     years = (1990.0:2000.0)
#     ny = length(years)
    
#     # estimate defined for locations/regions
#     regions = [:NATL,:ANT]#
#     nr = length(regions)

#     units1 = [u"m",u"s"]
#     units2 = [u"m*s",u"s^2"]
#     nu = length(units)

#     V = rand(Region(regions),YearCE(years))
#     #U = rand(UnitRange(units2),UnitDomain(units1))
#     U = rand(Units(units1),Units(units2))
#     #u = rand(UnitDomain(units1))
#     u = rand(Unit(units2))

#     test = dims(U)

#     E = rand(YearCE(years))
#     transpose(V)
#     V*transpose(V)
#     transpose(V)*V
#     # flatten V for vector multiplication?
#     V⃗ = vec(V)
#     V⃗c::Vector{Float64} = reshape(V,nr*ny)
#     V[:] # same thing
#     V2 = DimArray(reshape(vec(V),size(V)),(Region(regions),YearCE(years)))
#     @test V2 == V
#     @test vec(V) == V[:]
    
#     # 4D for covariance? sure, it works.
#     C = rand(Region(regions),YearCE(years),Region(regions),YearCE(years))

#     Cmat::Matrix{Float64} = reshape(C,nr*ny,nr*ny)

#     reshape(Cmat,size(C))

#     # reconstruct?
#     C2 = DimArray(reshape(Cmat,size(V)[1],size(V)[2],size(V)[1],size(V)[2]),(Region(regions),YearCE(years),Region(regions),YearCE(years)))

#     C == C2

#     V[Region(1),YearCE(5)]
#     V[Region(1)]
#     sum(V,dims=Region)

#     # NATL values
#     V[Region(At(:NATL))]
#     V[:NATL]
    
#     # get the year 1990
#     V[YearCE(Near(years[5]))]
#     V[YearCE(Near(1992.40u"yr"))]
#     V[YearCE(Interval(1992.5u"yr",1996u"yr"))]
#     size(V[YearCE()])
    
#     # order doesn't need to be known
#     test[Region(1),YearCE(5)]
#     test[YearCE(5),Region(1)]

#     x = BLUEs.State(V,C)

# end
