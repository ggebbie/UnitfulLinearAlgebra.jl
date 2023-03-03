"""
     function inv

     Inverse of Multipliable Matrix.
     Only defined for nonsingular matrices.
     Inverse reverses mapping from unitdomain to range.
     Is `exact` if input is exact.

    Hart, pp. 205. 
"""
LinearAlgebra.inv(A::AbstractUnitfulMatrix) = ~singular(A) ? rebuild(A,inv(parent(A)),(unitdomain(A),unitrange(A))) : error("matrix is singular")

LinearAlgebra.inv(A::AbstractUnitfulDimMatrix) = rebuild(A,inv(parent(A)), (unitdomain(A),unitrange(A)), (last(dims(A)),first(dims(A)) ))

"""
    function det

    Unitful matrix determinant.
"""
function LinearAlgebra.det(A::AbstractUnitfulType) 
    if square(A)
        detunit = prod([unitrange(A)[i]/unitdomain(A)[i] for i = 1:size(A)[1]])
        return Quantity(det(parent(A)),detunit)
    else
        error("Determinant requires square matrix")
    end
end

"""
    function eigen(A::T;permute::Bool=true, scale::Bool=true, sortby::Union{Function,Nothing}=eigsortby) where T <: AbstractMultipliableMatrix

    Thin wrapper for `UnitfulLinearAlgebra.eigen` with same keyword arguments as `LinearAlgebra.eigen`.
    There are multiple ways to distribute the units amongst the values and vectors.
    Here, physical intuition and the equation 𝐀𝐱 = λ𝐱
    dictate that the units of the eigenvectors are equal to the unit domain of 𝐀 (pp. 206, Hart, 1995).
    Only squarable matrices have eigenstructure (pp. 96, Hart, 1995).
    Ideally the AbstractArray interface would automatically handle `eigen`,
    but there is an unsolved issue with Unitful conversions.
    The following functions are available for `Eigen` objects:  [`det`](@ref), [`inv`](@ref) and [`isposdef`](@ref). Some are restricted to uniform matrices.
    `eigvals` of Eigen struct also available.
"""
function LinearAlgebra.eigen(A::AbstractUnitfulMatrix;permute::Bool=true, scale::Bool=true, sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) 
    if squarable(A) 
        F = LinearAlgebra.eigen(parent(A), permute=permute, scale=scale, sortby=sortby)
        return LinearAlgebra.Eigen(F.values.*(unitrange(A)[1]/unitdomain(A)[1]), rebuild(A,F.vectors,(unitdomain(A),Units(fill(unit(1.0),size(A,2))))))
    else
        error("UnitfulLinearAlgebra: Eigenvalue decomposition doesn't exist for for non-squarable matrices")
    end
end

"""
   Extend `isposdef` for Eigen factorizations of `MultipliableMatrix`s.
    Should the units be stripped out of the function?
    Only defined for matrices with uniform units (pp. 101, Hart, 1995). 
"""
LinearAlgebra.isposdef(A::Eigen{T,V,S,U}) where {U<: AbstractVector, S<:AbstractUnitfulMatrix, V, T <: Number} = (uniform(A.vectors) && isreal(A.values)) && all(x -> x > 0, ustrip.(A.values))

"""
   Extend `inv` for Eigen factorizations of `MultipliableMatrix`s.
    Only defined for matrices with uniform units (pp. 101, Hart, 1995). 
"""
function LinearAlgebra.inv(A::Eigen{T,V,S,U}) where {U <: AbstractVector, S <: AbstractUnitfulMatrix, V, T <: Number}

    if (uniform(A.vectors) && isreal(A.values))
        ur = unitrange(A.vectors)
        ud = Units(unit.(A.values))
        Λ⁻¹ = Diagonal(A.values.^-1,ud,ur)
        return A.vectors* transpose(transpose(A.vectors) \ Λ⁻¹)

        # LinearAlgebra.eigen uses matrix right divide, i.e., 
        #return A.vectors * Λ⁻¹ / A.vectors
        # but this is not available yet for `Multipliable Matrix`s.

    else
        error("UnitfulLinearAlgebra: Eigen factorization can only be inverted for uniform matrices")
    end
end

"""
    svd(A; full::Bool = false, alg::Algorithm = default_svd_alg(A)) -> SVD

    Singular value decomposition (SVD) of `AbstractMultipliableMatrix`.
    Only exists for uniform matrices (pp. 124, Hart, 1995).
    Functions for `SVD{AbstractMultipliableMatrix}` object: `inv`, `size`, `adjoint`, `svdvals`.
    Not implemented: `ldiv!`.
"""
function LinearAlgebra.svd(A::AbstractUnitfulMatrix;full=false,alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(parent(A))) 
    if uniform(A) 
        F = svd(parent(A), full=full, alg=alg)
        return SVD(F.U,F.S * unitrange(A)[1]./unitdomain(A)[1],F.Vt)
    else
        error("UnitfulLinearAlgebra: SVD doesn't exist for non-uniform matrices")
    end
end

"""
    function cholesky(A::AbstractMultipliableMatrix)

    Cholesky decomposition extended for matrices with units.
    Requires unit (or dimensionally) symmetric matrix.
    Functions available for LinearAlgebra.Cholesky objects: `size`, `\`, `inv`, `det`, `logdet` and `isposdef`.
    Functions available for UnitfulLinearAlgebra.Cholesky objects: `size`, `det`, and `isposdef`.
"""
function LinearAlgebra.cholesky(A::AbstractUnitfulMatrix)
    if unit_symmetric(A)
        C = LinearAlgebra.cholesky(parent(A))
        factors = rebuild(A,C.factors,(Units(unitdomain(A)./unitdomain(A)),unitdomain(A)))
        return Cholesky(factors,C.uplo,C.info)
    else
        error("requires unit symmetric matrix")
    end
end
function LinearAlgebra.cholesky(A::AbstractUnitfulDimMatrix)
    if unit_symmetric(A)
        C = LinearAlgebra.cholesky(parent(A))

        # What should the axis units for a Cholesky decomposition be? Just a guess here.
        # What if internal parts of Cholesky decomposition are simply UnitfulMatrix's. 
        factors = rebuild(A,C.factors,(Units(unitdomain(A)./unitdomain(A)),unitdomain(A)),(:Normalspace,last(dims(A))))
        return Cholesky(factors,C.uplo,C.info)
    else
        error("requires unit symmetric matrix")
    end
end

# seems like this might be from Base? Move to base.jl? 
function Base.getproperty(C::Cholesky{T,<:AbstractUnitfulMatrix}, d::Symbol) where T 
    Cfactors = getfield(C, :factors)
    Cuplo    = getfield(C, :uplo)
    if d === :U
        numbers = UpperTriangular(Cuplo === LinearAlgebra.char_uplo(d) ? parent(Cfactors) : copy(transpose(parent(Cfactors))))
        return rebuild(Cfactors,numbers,(unitrange(Cfactors),unitdomain(Cfactors)))
    elseif d === :L
        numbers = LowerTriangular(Cuplo === LinearAlgebra.char_uplo(d) ? parent(Cfactors) : copy(transpose(parent(Cfactors))))
        # use transpose to get units right
        return rebuild(Cfactors,numbers,(Units(unitdomain(Cfactors).^-1),Units(unitrange(Cfactors).^-1)))
    elseif d === :UL
        (Cuplo === 'U') ? (return rebuild(Cfactors,UpperTriangular(parent(Cfactors)))) : (return rebuild(Cfactors,LowerTriangular(parent(Cfactors)),(unitdomain(Cfactors).^-1,unitrange(Cfactors).^-1)))
    else
        #println("caution: fallback not tested")
        return getfield(C, d)
    end
end

"""
    function lu(A::AbstractUnitfulVecOrMat{T})

    Extend `lu` factorization to AbstractMultipliableMatrix.
    Related to Gaussian elimination.
    Store dimensional domain and range in "factors" attribute
    even though this is not truly a MultipliableMatrix.
    Returns `LU` type in analogy with `lu` for unitless matrices.
    Based on LDU factorization, Hart, pp. 204.
"""
function LinearAlgebra.lu(A::AbstractUnitfulVecOrMat)
    F̂ = lu(parent(A))
    factors = rebuild(A,parent(F̂.factors),(unitrange(A),unitdomain(A)))
    #factors = MMatrix(F̂.factors, unitrange(A), unitdomain(A), exact=exact(A))
    F = LU(factors,F̂.ipiv,F̂.info)
    return F
end

"""
    function getproperty(F::LU{T,<:AbstractMultipliableMatrix,Vector{Int64}}, d::Symbol) where T

    Extend LinearAlgebra.getproperty for AbstractUnitfulVecOrMat.

    LU factorization stores L and U together.
    Extract L and U while keeping consistent
    with dimensional domain and range.
"""
function Base.getproperty(F::LU{T,<:AbstractUnitfulVecOrMat,Vector{Int64}}, d::Symbol) where T
    m, n = size(F)
    if d === :L
        mmatrix = getfield(F, :factors)
        numbers = parent(mmatrix)
        #numbers = getfield(mmatrix,:numbers)
        # add ustrip to get numerical values
        Lnum = tril!(numbers[1:m, 1:min(m,n)])
        for i = 1:min(m,n); Lnum[i,i] = one(T); end
        return rebuild(mmatrix,Lnum,(unitrange(mmatrix),unitrange(mmatrix)))
    elseif d === :U
        mmatrix = getfield(F, :factors)
        numbers = parent(mmatrix)
        Unum = triu!(numbers[1:min(m,n), 1:n])
        return rebuild(mmatrix,Unum,(unitrange(mmatrix),unitdomain(mmatrix)))
    elseif d === :p
        return LinearAlgebra.ipiv2perm(getfield(F, :ipiv), m)
    elseif d === :P
        return Matrix{T}(I, m, m)[:,LinearAlgebra.invperm(F.p)]
    else
        getfield(F, d)
    end
end


"""
    function diagm(v::AbstractVector,r::Unitful.Unitlike,d::Unitful.Unitlike; exact = false)

    Construct diagonal matrix with units where the diagonal has elements `v`.
    If `v` has units, check that they conform with dimensional unit range `r`
     and dimensional unit domain `d`. Works for square or non-square matrices.
"""
LinearAlgebra.diagm(v::AbstractVector,r::Units,d::Units; exact = false) = UnitfulMatrix(spdiagm(length(r),length(d),ustrip.(v)),(r,d); exact=exact)    

# """
#     function diag(A::AbstractMultipliableMatrix)

#     Diagonal elements of matrix with units.

#     Usual `LinearAlgebra.diag` function is not working due to different type elements on diagonal
#  """
# function diag(A::AbstractMultipliableMatrix{T}) where T <: Number

#     m,n = size(A)
#     ndiag = max(m,n)
#     dimensionless(A) ? vdiag = Vector{T}(undef,ndiag) : vdiag = Vector{Quantity}(undef,ndiag)
#     for nd in 1:ndiag
#         vdiag[nd] = getindex(A,nd,nd)
#     end
#     return vdiag
# end
"""
    function Diagonal(v::AbstractVector,r::Unitful.Unitlike,d::Unitful.Unitlike; exact = false)

    Construct diagonal matrix with units where the diagonal has elements `v`.
    If `v` has units, check that they conform with dimensional unit range `r`
     and dimensional unit domain `d`.
    Like `LinearAlgebra.Diagonal`, this extension is restricted to square matrices.
"""
LinearAlgebra.Diagonal(v::AbstractVector,r::AbstractVector,d::AbstractVector; exact = false) = ((length(r) == length(d)) && (length(v) == length(d))) ? UnitfulMatrix(LinearAlgebra.Diagonal(ustrip.(v)),(r,d); exact=exact) : error("unit range and domain do not define a square matrix")   
LinearAlgebra.Diagonal(v::AbstractVector,r::Units,d::Units; exact = false) = ((length(r) == length(d)) && (length(v) == length(d))) ? UnitfulMatrix(LinearAlgebra.Diagonal(ustrip.(v)),(r,d); exact=exact) : error("unit range and domain do not define a square matrix")   

# Some lost code
# function ldiv!(A::LU{<:Any,<:StridedMatrix}, B::StridedVecOrMat)
#     if unitrange(A.L) == unit.(B)
#         LinearAlgebra._apply_ipiv_rows!(A, B)
#         nums = ldiv!(UpperTriangular(A.factors.numbers), ldiv!(UnitLowerTriangular(A.factors.numbers), ustrip.(B)))
#         return nums.*unitdomain(A.U)
#     else
#         error("units not compatible for ldiv!")
#     end
# end


"""
    function det

    Unitful matrix determinant.
    same as ULA.det
"""
function LinearAlgebra.det(A::AbstractUnitfulDimMatrix)
    if square(A)
        detunit = prod([unitrange(A)[i]/unitdomain(A)[i] for i = 1:size(A)[1]])
        return Quantity(det(parent(A)),detunit)
    else
        error("Determinant requires square matrix")
    end
end
