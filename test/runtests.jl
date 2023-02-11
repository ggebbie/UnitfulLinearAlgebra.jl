using Revise
using UnitfulLinearAlgebra
using Unitful
using LinearAlgebra
using SparseArrays
using DimensionalData
using DimensionalData: @dim
using Test

@testset "UnitfulLinearAlgebra.jl" begin

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

    @testset "scalars" begin
        c = 1m
        d = 2m
        @test c~d
        @test similarity(c,d)
        @test rand() ~ rand()
        @test parallel(rand(),rand())
        @test rand() ∥ rand()
        @test uniform(rand())
        @test uniform((rand())K)
        @test isequal(invdimension(1.0),NoDims)
        #@test isequal(invdimension(1.0K),Symbol(𝚯^-1))
        invdimension(1.0K)

        f = 1m
        g = 1 ./ f
        @test dottable(f,g)
        f ⋅ g

        h = 12.0
        j = 1 ./ h
        @test dottable(h,j)
        h ⋅ j
    end
    
    @testset "vectors" begin

        # already implemented in Unitful?
        a = [1m, 1s, 10K]
        b = [10m, -1s, 4K]
        a + b
        @test similarity(a,b)
        @test a~b
        @test parallel(a,b)
        @test a ∥ b
        #a ⋅ b
        @test ~uniform(b)
        
        c = [1m, 1s, 10K]
        d = [10m², -1s, 4K]
        @test ~similarity(c,d)
        @test ~(c~d)
        @test ~(c∥d)
        #c ⋅ d

        # inverse dimension
        invdimension.(a)

        k = 1 ./ a
        a ⋅ k
        @test dottable(a,k)
        @test ~dottable(a,b)
    end

    @testset "matrices" begin
        
        for i = 1:3
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
            q2 = UnitfulMatrix(ustrip.(q̃),unit.(q̃.^-1))
            # outer product to make a multipliable matrix
            A = p*q̃'

            #B = UnitfulMatrix(ustrip.(A),(unit.(p),unit.(q)),exact=true)
            B = UnitfulMatrix(ustrip.(A),unit.(p),unit.(q),exact=true) # MMatrix compatible
            r = UnitfulMatrix(ustrip.(q),unit.(q),exact=false) 
            #r = DimMatrix(reshape(ustrip.(q),2,1),unit.(q),[unit(1.0)],exact=true) 

            @test A==Matrix(B)
            
            # test multiplication
            @test within(A*q,Matrix(B*r),1.0e-10)
            @test isequal(uniform(A),uniform(B))
            @test isequal(left_uniform(A),left_uniform(B))
            @test isequal(right_uniform(A),right_uniform(B))
            @test ~dimensionless(B)

            # vcat and hcat fail, both sides of equation fail
            # y1 = B*q2
            # Bvcat = vcat(B,B)
            # @test Bvcat*q2 == vcat(y1,y1)

            # Bhcat = hcat(B,B)
            # @test Bhcat*vcat(q,q) == 2y1
            #
            
        end

        @testset "dimensionless" begin

            # scalar test
            @test dimensionless(1.0)
            @test ~dimensionless(1.0K)
            
            # Not all dimensionless matrices have
            # dimensionless domain and range
            for i = 1:2
                if i == 1
                    p = [1.0m²	, 3.0m²]
                elseif i ==2
                    p = [1.0m², 3.0u"m^3"]
                end
                
                q̃ = [-1.0u"m^-2", 2.0u"m^-2"]
                q = ustrip.(q̃).*unit.(1 ./q̃)
            
                # outer product to make a multipliable matrix
                A = p*q̃'
                #B = MMatrix(ustrip.(A),unit.(p),unit.(q))
                B = UnitfulMatrix(ustrip.(A),unit.(p),unit.(q))
                if i == 1
                    @test dimensionless(B)
                    @test dimensionless(A)
                elseif i ==2
                    @test ~dimensionless(B)
                    @test ~dimensionless(A)
                end
            end
        end
        
        @testset "exact" begin
            p = [1.0m, 3.0s]
            q̃ = [-1.0K, 2.0]

            qold = ustrip.(q̃).*unit.(1 ./q̃)
            q = UnitfulMatrix(ustrip.(q̃),unit.(1 ./q̃),exact=false)
            
            # outer product to make a multipliable matrix
            A = p*q̃'
            #B = MMatrix(ustrip.(A),unit.(p),unit.(q),exact=true)
            B = UnitfulMatrix(ustrip.(A),unit.(p),unit.(qold),exact=true)
            Bq = B*q
            @test A==Matrix(B)
            @test isequal(A*qold,Matrix(Bq))
            
            # new domain
            qnew = UnitfulMatrix(ustrip.(qold),unit.(qold).*s)
            D = convert_unitdomain(B,unitrange(qnew))
            #convert_unitdomain!(B,unit.(qnew)) # removed
            #@test unitrange(D) == unitrange(B)
            #@test unitdomain(D) == unitdomain(B)
            @test Bq ∥ D*qnew

            p2 = UnitfulMatrix(ustrip.(p),unit.(p))
            pnew = p2 *s
            qnew2 = UnitfulMatrix(ustrip.(qold),unit.(qold).*s)
            E = convert_unitrange(B,unitrange(pnew))
            @test Bq ∥ E*qnew2
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
            B = UnitfulMatrix(ustrip.(A),unit.(p),unit.(q),exact=false)
            B2 = UnitfulMatrix(A)
            @test square(B)
            @test squarable(B)
            B*B == B^2

            # in-place CONVERSION NOT WORKING
            # convert_unitrange!(B,K*[m,s])
            # @test unitrange(B) == K*[m,s]

            # convert_unitdomain!(B,K*[m,s])
            # @test unitdomain(B) == K*[m,s]

            # try to get eigenstructure
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
            p = [1.0, 2.0, 3.0]m
            q̃ = 1 ./ [2.0, 3.0, 4.0]

            q = ustrip.(q̃).*unit.(1 ./q̃)
            
            # outer product to make a multipliable matrix
            A = p*q̃'
            B = UnitfulMatrix(ustrip.(A),unit.(p),unit.(q),exact=false)
            B .+= 1 # make it non-singular
            @test square(B)
            @test squarable(B)
            B*B == B^2

            C = UnitfulLinearAlgebra.eigen(B)
            vals, vecs = C; # destructuring via iteration
            @test vals == C.values && vecs == C.vectors
            @test within(inv(B),inv(C),1e-10)

            # reconstruct using factorization
            ur = unitrange(C.vectors)
            ud = UnitfulLinearAlgebra.Units(unit.(C.values))
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
            @test ~isposdef(C)

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
           
            u1 = m
            u2 = m/s
            u3 = m/s/s
        
            # example: polynomial fitting
            k = 3
            E = hcat(randn(k),randn(k)u1/u2,randn(k)u1/u3)
            y = randn(k)u1
            x = UnitfulMatrix([randn()u1; randn()u2; randn()u3] )

            Z = lu(ustrip.(E))
            
            F = UnitfulMatrix(E)
            G = convert_unitdomain(F,unitrange(x))
            Z2 = lu(G)

            @test within(E[Z2.p,:],Matrix(Z2.L*Z2.U),1e-10)

            @test ~singular(F)
            det(F)

            E⁻¹ = inv(G)
            Eᵀ = transpose(G)
            @test G[2,1] == Eᵀ[1,2]
            #x̃ = E⁻¹ * (E * x) # doesn't work because Vector{Any} in parentheses, dimension() not valid, dimension deprecated?
            y = G*x

            # matrix left divide.
            # just numbers.
            x̃num = ustrip.(E) \ parent(y)

            x̂ = G \ y

            #y2 = convert(Vector{Quantity},y)
            #UnitfulLinearAlgebra.ldiv!(G,y2)
            @test within(x̂,x, 1e-10)

            # an inexact matrix
            x′ = F \ y
            @test within(x′,x,1e-10)

            #easy = [1. 0.2; 0.2 1.0]
            #tester = cholesky(easy)
            #@which ldiv!(tester,[2.1,3.1])
            
            x̃ = E⁻¹ * y
            @test within(x̃,x,1e-10)

            # Does LU solve the same problem?
            # MISSING UNITS HERE
            x̆ = Z2 \ y 
            @test within(x̆,x, 1e-10)

            # fails due to mixed matrix types
            #𝐱 = Z2.U\(Z2.L\(UnitfulMatrix(Z2.P'*y)))
            #@test within(𝐱,x,1e-10)
            #@test abs.(maximum(ustrip.(𝐱-x))) < 1e-10

        end    

        # NOT TESTED
        @testset "uniform svd" begin
            
	    E = [1/2 1/2; 1/4 3/4; 3/4 1/4]m
            
            E2 = MMatrix(E)
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
            Ẽ = F2.U*(Λ*F2.Vt)
            @test within(Matrix(Ẽ),E,1e-10)

            # solve a linear system with SVD
            # could also be solved with ldiv! but not yet implemented.
            x = [1.0, 2.0]
            y = E*x
            y2 = E2*x
            x̃ = E2\y 
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
           
            u1 = m
            u2 = m/s
            u3 = m/s/s
        
            # example: polynomial fitting
            k = 3
            E = hcat(randn(k),randn(k)u1/u2,randn(k)u1/u3)
            y = randn(k)u1
            x = [randn()u1; randn()u2; randn()u3] 

            F = MMatrix(E)
            convert_unitdomain!(F,unit.(x))

            # Define norms for this space.
            p1 = [m,m/s,m/s/s]
            q1= p1.^-1

            # covariance for domain.
            Cd = Diagonal([1,0.1,0.01],p1,q1)
            Pd = inv(Cd)
            #Pd = Diagonal([1m,0.1m/s,0.01m/s/s],p1,q1)

            p2 = [m,m,m]
            q2 = p2.^-1
            Cr = Diagonal([1.0,1.0,1.0],p2,q2)
            Pr = inv(Cr)

            ##
            G = dsvd(F,Pr,Pd)

            # provides inverse of singular vectors in an efficient way.
            # are they correct?
            @test within(G.V,inv(G.V⁻¹),1e-10)
            @test within(G.U,inv(G.U⁻¹), 1e-10)
            
            # Diagonal makes dimensionless S matrix
            # (but could usage be simplified? if uniform diagonal, make whole matrix uniform?)
            F̃ = G.U * Diagonal(G.S,fill(unit(1.0),size(F,1)),fill(unit(1.0),size(F,2))) * G.V⁻¹

            # even longer method to make S
            #F̃ = G.U * MMatrix(Matrix(Diagonal(G.S))) * G.V⁻¹
            @test within(F̃,F, 1e-10)

            u, s, v = G; # destructuring via iteration
            @test u == G.U && s == G.S && v == G.V

            # another way to decompose matrix.
            # recover using Diagonal dimensional matrix
 	    # Λ = diagm(G.S,unitrange(F),unitdomain(G),exact=true)
 	    Λ = diagm(size(F)[1],size(F)[2],G.S) 
            Ẽ = G.U*(Λ*G.V⁻¹)

            @test size(G) == size(F)
            @test within(Matrix(Ẽ),E, 1e-10)

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
                    if n1 == n2
                        @test transpose(v1)*(Pr*v2) ≈ 1.0
                    else
                        @test abs(transpose(v1)*(Pr*v2)) < 1e-10
                    end
                end
            end

            for n1 = 1:size(G,2)
                for n2 = n1:size(G,2)
                    v1 = G.V[:,n1]
                    v2 = G.V[:,n2]
                    if n1 == n2
                        @test transpose(v1)*(Pd*v2) ≈ 1.0
                    else
                        @test abs(transpose(v1)*(Pd*v2)) < 1e-10
                    end
                end
            end

            # Test domain to range connections
            # i.e., A*v1 = S1*u1, pp. 126, Hart 1995 
            k = searchsortedlast(G.S, eps(real(Float64))*G.S[1], rev=true)

            for kk = 1:k
               @test within(F*G.V[:,kk],G.S[kk]*G.U[:,kk], 1e-10)
            end

            # solve for particular solution.
            x = randn(size(F,2)).*unitdomain(F)
            y = F*x
            xₚ1 = F\y # find particular solution
            xₚ2 = inv(G)*y # find particular solution
            @test within(xₚ1,xₚ2,1e-10)

            # inverse of DSVD object
            @test within(inv(F),inv(G),1e-10)
            
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
            #Au \ xu # no existing method


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
            Amm* xmm
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

        # NOT TESTED
        @testset "dimarrays" begin

            using DimensionalData: @dim
            @dim Units "units"
            p = unit.([1.0m, 9.0s])
            q̃ = unit.([-1.0K, 2.0])
            U = zeros(Units(p),Units(q̃))
            Unum = [1.0 2.0; 3.0 4.0]
            V = DimMatrix(Unum,(Units(p),Units(q̃)),exact=true)

            vctr = DimMatrix(rand(2),(Units(q̃)),exact=true)

            Units(p.^-1)
            inv(Matrix(V.data))
            Vi = DimMatrix(inv(Unum),(Units(q̃),Units(p)));
            Vi*V;
            
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

        end
        
    end
end
