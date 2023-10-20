@testset "interface" begin
    m = u"m"
    s = u"s"
    test_Unitful_matrices()
end

@testset "dimensionless" begin
    test_Unitful_dimensionless()
end

@testset "fixed units" begin
    mat = UnitfulMatrix(ones(2,2)*permil)
    unit(diag(mat)[1]) != NoUnits

    #using Unitful, UnitfulLinearAlgebra, LinearAlgebra
    permil = Unitful.FixedUnits(u"permille")
    #urange = fill(permil, 2)
    #udomain = fill(permil, 3)
    urange = fill(u"K", 2)
    udomain = fill(u"K", 3)
    A = UnitfulMatrix(randn(length(urange),length(udomain)),urange,udomain)

    Pr = UnitfulMatrix(I(length(urange)),urange.^-1,urange)
    Pd = UnitfulMatrix(I(length(udomain)),udomain.^-1,udomain)
    U,σ,V = dsvd(A,Pr,Pd)

    #This seems to trace back to the fact that when the following is called
    #https://github.com/ggebbie/UnitfulLinearAlgebra.jl/blob/de6592342d2aaf7c2eaae4d59125cc2b04a8f690/src/dsvd.jl#L61
    #`F.Qy` and `F.U′ ` have different unitranges/domains, so the ldiv can't be called. E.g.

    F = dsvd(A,Pr,Pd)
    F.Qy.dims
    F.U'.dims

    DimensionalData.comparedims(F.Qy,F.U')
    #It seems to me like the Categorical Units versus the Sampled Units is the problem, and I'm not sure how to fix this. Would there be a way to expand on the test of equivalent dimensions that ldiv calls to make it broader? Like if the vector version of the units is equivalent, it's okay to ldiv? 

end

@testset "exact" begin
    p = [1.0m, 3.0s]
    q̃ = [-1.0K, 2.0]

    qold = ustrip.(q̃).*unit.(1 ./q̃)
    q = UnitfulMatrix(qold)
    q = UnitfulMatrix(ustrip.(q̃),unit.(1 ./q̃))

    # Does vec return unit information?
    # Does this break anything else?
    
    # outer product to make a multipliable matrix
    A = p*q̃' #Vector{Quantity} * UnitfulMatrix
    #B = MMatrix(ustrip.(A),unit.(p),unit.(q),exact=true)
    B = UnitfulMatrix(ustrip.(A),unit.(p),unit.(qold))
    Bq = B*q #UnitfulMatrix * UnitfulMatrix
    @test A==Matrix(B)
    @test isequal(A*qold,Matrix(Bq))

    #test UnitfulMatrix * Matrix
    C = UnitfulMatrix([1 2;3 4],fill(NoUnits,2), fill(m, 2))
    @test isequal(C * Matrix(I(2)), C)
    @test isequal(Matrix(I(2)) * C, C)
    #make sure the show() works
    C * Matrix(I(2))
    
    # new domain
    qnew = UnitfulMatrix(ustrip.(qold),unit.(qold).*s)
    D = convert_unitdomain(B,unitrange(qnew))
    @test Bq ∥ D*qnew

    p2 = UnitfulMatrix(ustrip.(p),unit.(p))
    pnew = p2 *s #multiply by a unit 
    qnew2 = UnitfulMatrix(ustrip.(qold),unit.(qold).*s)
    E = convert_unitrange(B,unitrange(pnew))
    @test Bq ∥ E*qnew2
end

@testset "backslash inexact" begin
    A = rand(3, 3) + I
    Au = A * 1u"1/s" # u= with units
    Amm = UnitfulMatrix(Au) # mm= multipliable matrix
    
    x = rand(3)
    xu = (x)u"mol/m^3"
    xmm = UnitfulMatrix(xu)

    # Test *
    A * x # no units
    Au * xu # should have correct units
    Au * x # missing units for x
    A * xu # missing units for A
    Amm*xu # no longer returns invalid object
    
    # Test \
    A \ x # works without units
    #Au \ x # won't work
    Amm \ xmm # gets units right

    @test !exact(Amm)
    @test !exact(xmm)
    @test !exact(Amm\xmm)
    
end

@testset "array" begin
    p = [1.0m, 3.0s]
    q̃ = [-1.0K, 2.0]
    q = ustrip.(q̃).*unit.(1 ./q̃)
    
    # outer product to make a multipliable matrix
    A = p*q̃'
    #B = MMatrix(ustrip.(A),unit.(p),unit.(q),exact=true)
    B = UnitfulMatrix(ustrip.(A),unit.(p),unit.(q),exact=true)

    # turn array into Unitful matrix
    C = UnitfulMatrix(A)
    @test A==Matrix(C)
    @test multipliable(A)
    @test ~left_uniform(A)
    @test ~endomorphic(C)            
end

@testset "endomorphic" begin

    @test endomorphic(1.0)
    @test ~endomorphic(1.0K)
    
    p = [1.0m, 1.0s]
    q̃ = 1 ./ [1.0m, 1.0s]
    q = ustrip.(q̃).*unit.(1 ./q̃)
    
    # outer product to make a multipliable matrix
    A = p*q̃'
    B = UnitfulMatrix(A)
    B2 = UnitfulMatrix(ustrip.(A),unit.(p),unit.(q))

    Bᵀ = transpose(B)
    @test Bᵀ[2,1] == B[1,2]

    Ip = identitymatrix([m,s])
    B2 + Ip
    
    @test Matrix(B)==Matrix(B2)
    @test multipliable(B)
    @test endomorphic(B2)
    @test endomorphic(B)
    @test endomorphic(A)

    # endomorphic should have dimensionless eigenvalues
    F = UnitfulLinearAlgebra.eigen(B)
    for j in F.values
        @test dimensionless(j)
    end
    
    #change domain of B3
    # convert_unitrange!(B3,[m²,s*m])
    # @test unitrange(B3) == [m²,s*m]

    # convert_unitdomain!(B3,[m,s])
    # @test unitdomain(B3) == [m,s]
end

@testset "squarable" begin
    p = [1.0m, 2.0s]
    q̃ = 1 ./ [2.0m², 3.0m*s]

    q = ustrip.(q̃).*unit.(1 ./q̃)
    
    # outer product to make a multipliable matrix
    A = p*q̃'
    B = UnitfulMatrix(ustrip.(A),unit.(p),unit.(q),exact=true)
    B2 = UnitfulMatrix(A)
    @test square(B)
    @test squarable(B)
    B2*B2 == B2^2

    # get eigenstructure
    F = eigen(B)

    # Hart, 1995, pp. 97
    @test abs(ustrip(trace(B) - sum(F.values))) < 1e-10
    @test abs(ustrip(det(B) - prod(F.values))) < 1e-10

    for k = 1:2
        #@test within(B*Matrix(F.vectors)[:,k],F.values[k]*Matrix(F.vectors)[:,k],1e-10) 
        @test within(B*F.vectors[:,k],F.values[k]*F.vectors[:,k],1e-10) 
    end
end

@testset "eigenvalues" begin
    # requires uniform, squarable matrix
    p = [1.0, 2.0]m
    q̃ = 1 ./ [ 3.0, 4.0]

    q = ustrip.(q̃).*unit.(1 ./q̃)
    
    # outer product to make a multipliable matrix
    A = p*q̃'
    B = UnitfulMatrix(ustrip.(A),unit.(p),unit.(q),exact=false)
    B[2,2] += 1 # make it non-singular
    @test square(B)
    @test squarable(B)
    B*B == B^2

    C = UnitfulLinearAlgebra.eigen(B)
    vals, vecs = C; # destructuring via iteration
    @test vals == C.values && vecs == C.vectors
    @test within(inv(B),inv(C),1e-10)

    # reconstruct using factorization
    ud = unitrange(C.vectors)
    ur = UnitfulLinearAlgebra.Units(unit.(C.values))
    Λ = Diagonal(C.values,ur,ud)
    # use matrix right divide would be best
    #transpose(transpose(C.vectors)\ (Λ*transpose(C.vectors)))
    B̃ = C.vectors * Λ* inv(C.vectors)
    @test within(B̃,B,1e-10)

    # check eigenvalue condition
    for k = 1:2
        @test within(B*C.vectors[:,k],C.values[k]*C.vectors[:,k],1e-10)
    end

    # compute det using Eigen factorization
    @test within(det(C),det(B),1e-10)
    @test isposdef(C)

end

@testset "unit symmetric" begin
    p = [2.0m, 1.0s]
    q̃ = p

    p = [m,s]
    q= p.^-1
    
    # outer product to make a multipliable matrix
    A = [1.0 0.1; 0.1 1.0]
    B = UnitfulMatrix(A,p,q ,exact=true)
    @test square(B)
    @test ~squarable(B)

    # make equivalent Diagonal matrix.
    C = Diagonal([1.0m, 4.0s],p,q)

    Anodims = ustrip.(A)
    # try cholesky decomposition
    Qnodims = cholesky(Anodims)

    Q = UnitfulLinearAlgebra.cholesky(B)
    test1 = transpose(Q.U)*Q.U
    @test within(B,test1,1e-6)
    @test within(B,Q.L*transpose(Q.L),1e-10)

    # do operations directly with Q?
    ynd = [0.5, 0.8]
    y = UnitfulMatrix(ynd)
    Qnodims.U\ynd
    Q.U\ y  # includes units
end

@testset "matrix * operations" begin
    p = [1.0m, 4.0s]
    q̃ = [-1.0K, 2.0]
    q = ustrip.(q̃).*unit.(1 ./q̃)
    
    # outer product to make a multipliable matrix
    A = p*q̃'
    B = UnitfulMatrix(ustrip.(A),unit.(p),unit.(q),exact=true)

    scalar = 2.0K 
    C = B * scalar
    @test (Matrix(C)./Matrix(B))[1,1] == scalar
    C2 = scalar *B
    @test (Matrix(C2)./Matrix(B))[1,1] == scalar

    scalar2 = 5.3
    @test exact(scalar2*B)

    # outer product to make a multipliable matrix
    B2 = UnitfulMatrix(ustrip.(A),unit.(q),unit.(p),exact=true)
    A2 = Matrix(B2)
    
    @test A*A2==Matrix(B*B2)
end

@testset "polynomial fitting" begin
    test_polynomial_UnitfulMatrix()
end    

@testset "uniform svd" begin
    
    E = [1/2 1/2; 1/4 3/4; 3/4 1/4]m
    
    E2 = UnitfulMatrix(E)
    @test size(E2)==size(E)
    Eᵀ = transpose(E2)
    @test E2[2,1] == Eᵀ[1,2]

    F = svd(ustrip.(E))
    F2 = svd(E2,full=true)
    F3 = svd(E2)

    Krank = length(F3.S)
    G = 0 .*E
    for k = 1:Krank
        # outer product
        G += F2.S[k] * F2.U[:,k] * transpose(F2.Vt[k,:])
    end
    @test within(G,E, 1e-10)

    # recover using Diagonal dimensional matrix
    # use Full SVD (thin may not work)
    Λ = diagm(F2.S,unitrange(E2),unitdomain(E2),exact=true)
    Ẽ = UnitfulMatrix(F2.U)*(Λ*UnitfulMatrix(F2.Vt))
    @test within(Matrix(Ẽ),E,1e-10)

    # solve a linear system with SVD
    # could also be solved with ldiv! but not yet implemented.
    x = [1.0, 2.0]
    y = E*x
    y2 = E2*UnitfulMatrix(x)
    x̃ = E2\y2 
    x̃2 = inv(F2)*y # find particular solution
    @test within(x̃2,x,1e-10)

    #             K = length(λ) # rank
    # 	    y = 5randn(3)u"s"
    # 	    σₙ = randn(3)u"s"
    # 	    Cₙₙ = diagonal_matrix(σₙ)
    # 	    W⁻¹ = diagonal_matrix([1,1,1]u"1/s^2")
    # 	    x̃ = inv(E'*W⁻¹*E)*(E'*W⁻¹*y)
    # #            [@test isequal(x̃[i]/ustrip(x̃[i]),1.0u"dbar^-1") for i in 1:length(x̃)]

end

@testset "dimensional svd (DSVD)" begin
    s = u"s"
    m = u"m"
    u1 = m
    u2 = m/s
    u3 = m/s/s
    
    # example: polynomial fitting
    k = 3
    #E2 = BestMultipliableMatrix(hcat(randn(k),randn(k)u1/u2,randn(k)u1/u3))
    E = UnitfulMatrix(randn(k,3),fill(m,k),[u1,u2,u3],exact=true)
    y = UnitfulMatrix(randn(k)u1)
    x = UnitfulMatrix([randn()u1; randn()u2; randn()u3])

    # Define norms for this space.
    p1 = [m,m/s,m/s/s]
    q1= p1.^-1

    # covariance for domain.
    Cd = Diagonal([1,0.1,0.01],p1,q1,exact=true)
    Pd = inv(Cd)
    #Pd = Diagonal([1m,0.1m/s,0.01m/s/s],p1,q1)

    p2 = [m,m,m]
    q2 = p2.^-1
    Cr = Diagonal([1.0,1.0,1.0],p2,q2,exact=true)
    Pr = inv(Cr)

    ##
    G = dsvd(E,Pr,Pd)

    # provides inverse of singular vectors in an efficient way.
    # are they correct?
    @test within(G.V,inv(G.V⁻¹),1e-10)
    @test within(G.U,inv(G.U⁻¹), 1e-10)
    
    # Diagonal makes dimensionless S matrix
    Ẽ = G.U * UnitfulMatrix(Diagonal(G.S)) * G.V⁻¹

    # even longer method to make S
    #F̃ = G.U * MMatrix(Matrix(Diagonal(G.S))) * G.V⁻¹
    @test within(Ẽ,E, 1e-10)

    u, s, v = G; # destructuring via iteration
    @test u == G.U && s == G.S && v == G.V

    # another way to decompose matrix.
    # recover using Diagonal dimensional matrix
    # Λ = diagm(G.S,unitrange(F),unitdomain(G),exact=true)
    Λ = diagm(size(G)[1],size(G)[2],G.S) 
    Ẽ = G.U*(Λ*G.V⁻¹)

    @test within(Ẽ,E, 1e-10)

    # test other DSVD properties
    @test within(transpose(G.Qx)*G.Qx,Pd,1e-10)
    @test within(transpose(G.Qy)*G.Qy,Pr,1e-10)

    @test dimensionless(G.U′)
    @test dimensionless(G.V′⁻¹)
    @test dimensionless(G.S[:,:]) # turn vector into matrix

    # Test orthogonality within normed space
    for n1 = 1:size(G,1)
        for n2 = n1:size(G,1)
            v1 = G.U[:,n1]
            v2 = G.U[:,n2]
            #WARNING: problem with inner product of vectors
            if n1 == n2
                #@test transpose(v1) * (Pr*v2) ≈ 1.0
                @test v1 ⋅ (Pr*v2) ≈ 1.0
            else
                #@test abs(transpose(v1)*(Pr*v2)) < 1e-10
                @test abs( v1  ⋅ (Pr*v2)) < 1e-10
            end
        end
    end

    for n1 = 1:size(G,2)
        for n2 = n1:size(G,2)
            v1 = G.V[:,n1]
            v2 = G.V[:,n2]
            # WARNING: doesn't check units, problem with rebuildmul of row vector times vector
            if n1 == n2
                #@test transpose(v1)*(Pd*v2) ≈ 1.0
                @test v1 ⋅ (Pd*v2) ≈ 1.0
            else
                #@test abs(transpose(v1)*(Pd*v2)) < 1e-10
                @test abs( v1 ⋅(Pd*v2)) < 1e-10
            end
        end
    end

    # Test domain to range connections
    # i.e., A*v1 = S1*u1, pp. 126, Hart 1995 
    k = searchsortedlast(G.S, eps(real(Float64))*G.S[1], rev=true)

    for kk = 1:k
        @test within(E*G.V[:,kk],G.S[kk]*G.U[:,kk], 1e-10)
    end

    # solve for particular solution.
    x = UnitfulMatrix(randn(size(E,2)),unitdomain(E))
    y = UnitfulMatrix(Matrix(E*x)) # need y to be inexact
    xₚ1 = E\y # find particular solution
    xₚ2 = inv(G)*y # find particular solution
    @test within(xₚ1,xₚ2,1e-10)

    # inverse of DSVD object
    @test within(Matrix(inv(E)),Matrix(inv(G)),1e-10)
    
end    

@testset "briochemc" begin
    
    A = rand(3, 3) + I
    Au = A * 1u"1/s"

    # A with multipliable matrix
    Amm = UnitfulMatrix(Au)
    
    x = rand(3)
    xu = x * 1u"mol/m^3"
    xmm = UnitfulMatrix(x)
    # Test *
    A * x
    Au * xu
    Au * x
    A * xu
    # Test \
    A \ x # works with a UniformMatrix or LeftUnitformMatrix
    #Au \ x # won't work
    Amm \ xmm # gets units right
    #A \ xu # won't work
    #Au \ xu # no existing method: limitation of Unitful.jl

    # ---------- Sparse tests ----------
    A = sprand(3, 3, 0.5) + I
    Au = A * 1u"1/s"
    Ammfull = UnitfulMatrix(Matrix(Au))# not working with SparseArray now
    Amm = UnitfulMatrix(A,fill(u"mol/m^3",3),fill(u"s*mol/m^3",3))  # use constructor, internally stores a sparse matrix
    x = rand(3)
    xu = x * 1u"mol/m^3"
    xmm = UnitfulMatrix(xu)
    
    # Test *
    A * x
    Au * x
    A * xu
    Au * xu
    convert_unitdomain(Amm,unitrange(xmm))* xmm
    # Test \

    # Problem: units not right for x to be conformable with Au.
    # change x to y
    y = rand(3);
    yu = y.*unitrange(Amm)
    ymm = UnitfulMatrix(yu)
    A \ y 
    #Au \ x # stack overflow, doesn't work at lu, no method
    #A \ yu # doesn't work, no method
    #Au \ yu, doens't work, no lu method
    Amm \ ymm # works, should be same units as x but they aren't?
end
