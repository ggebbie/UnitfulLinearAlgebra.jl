module UnitfulLinearAlgebra

using Unitful
using LinearAlgebra
using SparseArrays
using DimensionalData

export UnitfulMatrix, AbstractUnitfulVecOrMat
export DSVD
export similarity, ∥, parallel
export uniform, left_uniform, right_uniform
export square, squarable, singular, unit_symmetric
export invdimension, dottable
export getindexqty
#export getindex, setindex!,
export size, similar
export convert_unitrange, convert_unitdomain
#export convert_unitrange!, convert_unitdomain!
export exact, multipliable, dimensionless, endomorphic
export svd, dsvd
export eigen, isposdef, inv, transpose
export unitrange, unitdomain
export lu, det, trace, diag, diagm
export Diagonal, (\), cholesky
export identitymatrix, show, vcat, hcat #, rebuild #, rebuildsliced
export describe

import LinearAlgebra: inv, det, lu,
    svd, getproperty, eigen, isposdef,
    diag, diagm, Diagonal, cholesky
import Base:(~), (*), (+), (-), (\), getindex, setindex!,
    size, range, transpose, similar, show, vcat, hcat

import DimensionalData: @dim, dims, DimArray, AbstractDimArray, NoName, NoMetadata, format, print_name

@dim Units "units"


include("UnitfulMatrix.jl")
#include("UnitfulDimMatrix.jl")
include("multipliablematrices.jl")
include("base.jl")
include("cholesky.jl")
"""
    function lu(A::AbstractUnitfulVecOrMat{T})

    Extend `lu` factorization to AbstractMultipliableMatrix.
    Related to Gaussian elimination.
    Store dimensional domain and range in "factors" attribute
    even though this is not truly a MultipliableMatrix.
    Returns `LU` type in analogy with `lu` for unitless matrices.
    Based on LDU factorization, Hart, pp. 204.
"""
function lu(A::AbstractUnitfulVecOrMat)
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
function getproperty(F::LU{T,<:AbstractUnitfulVecOrMat,Vector{Int64}}, d::Symbol) where T
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
    function similarity(a,b)::Bool

    Dimensional similarity of vectors, a binary relation
    Read "a has the same dimensional form as b"
    `a` and `b` may still have different units.
    A stronger condition than being parallel.
    pp. 184, Hart
"""
 similarity(a,b)::Bool = isequal(dimension.(a),dimension.(b))

"""
    function parallel

    Vector a is dimensionally parallel to vector b if
    they have the same length and a consistent dimensional
    change relates corresponding components.
    Guaranteed if two vectors are dimensionally similar.
    True for scalars in all cases. 

    pp. 188, Hart
    Note: Hart uses ≈, but this conflicts with an existing Julia function.
"""
function parallel(a,b)::Bool

    if isequal(length(a),length(b))
        if length(a) == 1
            return true
        else
            Δdim = dimension.(a)./dimension.(b)
            for i = 2:length(a)
                if Δdim[i] ≠ Δdim[1]
                    return false
                end
            end
            return true
        end
    else
        return false
    end
end
function parallel(a::AbstractUnitfulVector,b::AbstractUnitfulVector)::Bool
    if isequal(length(a),length(b))
        if length(a) == 1
            return true
        else
            Δdim = dimension(a)./dimension(b) # inconsistent function call
            for i = 2:length(a)
                if Δdim[i] ≠ Δdim[1]
                    return false
                end
            end
            return true
        end
    else
        return false
    end
end
∥(a,b)  = parallel(a,b)

# not consistent in that it should be an element-wise function
Unitful.dimension(a::AbstractUnitfulVector) = dimension.(unitrange(a)) 

"""
    function uniform(a)

    Is the dimension of this quantity uniform?

    There must be a way to inspect the Unitful type to answer this.
    Uniform matrix: All entries have the same units
"""
uniform(a::T) where T <: Number = true # all scalars by default
function uniform(a::Union{Vector,<:DimensionalData.Dimension}) 
    dima = dimension.(a)
    for dd = 2:length(dima)
        if dima[dd] ≠ dima[1]
            return false
        end
    end
    return true
end
function uniform(A::Matrix)
    B = UnitfulMatrix(A)
    isnothing(B) ? false : uniform(B)
end
uniform(A::UnitfulMatrix) = left_uniform(A) && right_uniform(A)

"""
    function left_uniform(A)

    Definition: uniform unitrange of A
    Left uniform matrix: output of matrix has uniform units
"""
left_uniform(A::UnitfulMatrix) = uniform(unitrange(A)) ? true : false
function left_uniform(A::Matrix)
    B = UnitfulMatrix(A)
    isnothing(B) ? false : left_uniform(B)
end

"""
    function right_uniform(A)

    Does the unitdomain of A have uniform dimensions?
    Right uniform matrix: input of matrix must have uniform units
"""
right_uniform(A::UnitfulMatrix) = uniform(unitdomain(A)) ? true : false
function right_uniform(A::Matrix)
    B = UnitfulMatrix(A)
    isnothing(B) ? false : right_uniform(B)
end

"""
     function dimensionless(A)

     Not all dimensionless matrices have
     dimensionless domain and range.
"""
dimensionless(A::Union{Matrix,UnitfulMatrix}) = uniform(A) && dimension(A[1,1]) == NoDims
dimensionless(A::T) where T <: Number = (dimension(A) == NoDims)
function dimensionless(A::AbstractMatrix)
    B = UniformMatrix(A)
    isnothing(B) ? false : dimensionless(B) # fallback
end

"""
    function square(A)

    size(A)[1] == size(A)[2]
"""
square(A::T) where T <: AbstractMatrix = isequal(size(A)[1],size(A)[2]) 

"""
    function squarable(A::Matrix)

    A squarable matrix is one where 𝐀² is defined.
    Unit (dimensional) range and domain are parallel.
    Key for solving difference and differential equations.
    Have eigenstructure. 
"""
squarable(A::UnitfulMatrix) = (unitdomain(A) ∥ unitrange(A))
function squarable(A::AbstractMatrix)
    B = UnitfulMatrix(A)
    isnothing(B) ? false : squarable(B) # fallback
end

"""
    function unit_symmetric(A::Matrix)

    `UnitSymmetricMatrix`s have units that are symmetric about the main diagonal and define weighted norms. 
    Definition: inverse dimensional range and dimensional domain are parallel.
    Called "dimensionally symmetric" by Hart, 1995.
"""
unit_symmetric(A::UnitfulMatrix) = (unitrange(A) ∥ unitdomain(A).^-1)
function unit_symmetric(A::AbstractMatrix)
    B = UnitfulMatrix(A)
    isnothing(B) ? false : unit_symmetric(B) # fallback
end

"""
    function invdimension

    Dimensional inverse
      
    pp. 64, Hart, `a~` in his notation
"""
#invdimension(a) = dimension.(1 ./ a)
invdimension(a) = dimension.(a).^-1

"""
    function dottable(a,b)

    Are two quantities dimensionally compatible
    to take a dot product?
"""
dottable(a,b) = parallel(a, 1 ./ b)

"""
    function convert_unitdomain(A, newdomain)

    When using the geometric interpretation of matrices,
    it is useful to convert the dimensional domain of the
    matrix to match the expected vectors during multiplication.
    Here we set the matrix to `exact=true` after this step.
"""
function convert_unitdomain(A::AbstractUnitfulVecOrMat, newdomain::Units) 
    if unitdomain(A) ∥ newdomain
        #shift = newdomain./unitdomain(A)
        #newrange = unitrange(A).*shift
        newrange = Units(unitrange(A).*(newdomain[1]/unitdomain(A)[1]))
        return rebuild(A, parent(A), (newrange, newdomain), true)
        #B = BestMultipliableMatrix(A.numbers,newrange,newdomain,exact=true)
    else
        error("New unit domain not parallel to unit domain of Multipliable Matrix")
    end
end

# """
#     function convert_unitdomain!(A, newdomain)

#     In-place conversion of unit (dimensional) domain.
#     Matrix Type not permitted to change.
# """
# function convert_unitdomain!(A::AbstractMultipliableMatrix, newdomain::Vector)
#     if unitdomain(A) ∥ newdomain
#         shift = newdomain[1]./unitdomain(A)[1]
#         # caution: not all matrices have this attribute
#         if hasproperty(A,:unitdomain)
#             for (ii,vv) in enumerate(A.unitdomain)
#                 A.unitdomain[ii] *= shift
#             end
#         end
#         if hasproperty(A,:unitrange)
#             for (ii,vv) in enumerate(A.unitrange)
#                 A.unitrange[ii] *= shift
#             end
#         end
#         # make sure the matrix is now exact
#         #A.exact = true # immutable so not possible
#     else
#         error("New domain not parallel to domain of Multipliable Matrix")
#     end
# end

"""
    function convert_unitrange(A, newrange)

    When using the geometric interpretation of matrices,
    it is useful to convert the dimensional range of the
    matrix to match the desired output of multiplication.
    Here we set the matrix to `exact=true` after this step.
    Permits MatrixType to change.
"""
function convert_unitrange(A::AbstractUnitfulMatrix, newrange::Units) 
    if unitrange(A) ∥ newrange
        #shift = newdomain./unitdomain(A)
        #newrange = unitrange(A).*shift
        newdomain = Units(unitdomain(A).*(newrange[1]/unitrange(A)[1]))
        B = rebuild(A, parent(A), (newrange, newdomain))
    else
        error("New unit domain not parallel to unit domain of Multipliable Matrix")
    end
end

# """
#     function convert_unitrange!(A, newrange)

#     In-place conversion of unit (dimensional) range.
#     Matrix Type not permitted to change.
# """
# function convert_unitrange!(A::AbstractMultipliableMatrix, newrange::Vector)
#     if unitrange(A) ∥ newrange
#         shift = newrange[1]./unitrange(A)[1]
#         # caution: not all matrices have this attribute
#         if hasproperty(A,:unitdomain)
#             for (ii,vv) in enumerate(A.unitdomain)
#                 A.unitdomain[ii] *= shift
#             end
#         end
#         if hasproperty(A,:unitrange)
#             for (ii,vv) in enumerate(A.unitrange)
#                 A.unitrange[ii] *= shift
#             end
#         end
#         #A.exact = true , immutable
#      else
#          error("New range not parallel to range of Multipliable Matrix")
#      end
# end
# function convert_unitrange!(A::AbstractUnitfulMatrix, newrange::Vector)
#     if unitrange(A) ∥ newrange
#         shift = newrange[1]./unitrange(A)[1]
#         for (ii,vv) in enumerate(A.dims[2])
#             A.dims[2][ii] *= shift
#         end
#         for (ii,vv) in enumerate(A.dims[1])
#             A.dims[1][ii] *= shift
#         end
#      else
#          error("New range not parallel to range of Unitful Matrix")
#      end
# end

"""
    function exact(A)

-    `exact=true`: geometric interpretation of unitdomain and unitrange
-    `exact=false`: algebraic interpretation
"""
exact(A::UnitfulMatrix) = A.exact

# these dummy functions are needed for the interface for DimensionalData. They are important for matrix slices. Would be nice if they were not needed.
DimensionalData.name(A::AbstractUnitfulVecOrMat) = ()
DimensionalData.metadata(A::AbstractUnitfulVecOrMat) = NoMetadata()
DimensionalData.refdims(A::AbstractUnitfulVecOrMat) = ()

# convert(::Type{AbstractMatrix{T}}, A::AbstractMultipliableMatrix) where {T<:Number} = convert(AbstractMultipliableMatrix{T}, A)
# convert(::Type{AbstractArray{T}}, A::AbstractMultipliableMatrix) where {T<:Number} = convert(AbstractMultipliableMatrix{T}, A)
# #convert(::Type{AbstractArray{T}}, S::AbstractToeplitz) where {T<:Number} = convert(AbstractToeplitz{T}, S)

"""
    function unitdomain(A)

    Find the dimensional (unit) domain of a matrix
"""
unitdomain(A::AbstractUnitfulMatrix) = last(dims(A))
# this line may affect matrix multiplication
unitdomain(A::AbstractUnitfulVector) = Units([unit(1.0)]) # kludge for a nondimensional scalar 
#unitdomain(A::AbstractUnitfulVector) = Unitful.FreeUnits{(), NoDims, nothing}() # nondimensional scalar 

"""
    function unitrange(A)

    Find the dimensional (unit) range of a matrix
"""
unitrange(A::AbstractUnitfulVecOrMat) = first(dims(A))
#unitrange(A::AbstractUnitfulMatrix) = first(dims(A))

"""
    function transpose

    Defined by condition `A[i,j] = transpose(A)[j,i]`.
    Not analogous to function for dimensionless matrices.

    Hart, pp. 205.
"""
# Had to redefine tranpose or it was incorrect based on AbstractArray interface!
transpose(A::AbstractUnitfulMatrix) = rebuild(A,transpose(parent(A)),(Units(unitdomain(A).^-1), Units(unitrange(A).^-1)))
transpose(a::AbstractUnitfulVector) = rebuild(a,transpose(parent(a)),(Units([unit(1.0)]), Units(unitrange(a).^-1))) # kludge for unitrange of row vector

"""
    function identitymatrix(dimrange)

    Input: dimensional (unit) range.
    `A + I` only defined when `endomorphic(A)=true`
    When accounting for units, there are many identity matrices.
    This function returns a particular identity matrix
    defined by its dimensional range.
    Hart, pp. 200.                             
"""
identitymatrix(dimrange) = UnitfulMatrix(I(length(dimrange)),(dimrange,dimrange),exact=true)

"""
     function inv

     Inverse of Multipliable Matrix.
     Only defined for nonsingular matrices.
     Inverse reverses mapping from unitdomain to range.
     Is `exact` if input is exact.

    Hart, pp. 205. 
"""
inv(A::AbstractUnitfulMatrix) = ~singular(A) ? rebuild(A,inv(parent(A)),(unitdomain(A),unitrange(A))) : error("matrix is singular")


"""
    function det

    Unitful matrix determinant.
"""
function det(A::AbstractUnitfulMatrix) 
    if square(A)
        detunit = prod([unitrange(A)[i]/unitdomain(A)[i] for i = 1:size(A)[1]])
        return Quantity(det(parent(A)),detunit)
    else
        error("Determinant requires square matrix")
    end
end

"""
    function singular(A)

    Is a square matrix singular? If no, then it is invertible.
"""
singular(A::AbstractUnitfulMatrix) = iszero(ustrip(det(A)))

"""
    function trace(A)

    Trace = sum of diagonal elements of a square matrix
"""
trace(A::AbstractUnitfulMatrix) = sum(diag(parent(A))).*(unitrange(A)[1]/unitdomain(A)[1])

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
function eigen(A::AbstractUnitfulMatrix;permute::Bool=true, scale::Bool=true, sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) 
    if squarable(A) 
        F = LinearAlgebra.eigen(parent(A), permute=permute, scale=scale, sortby=sortby)
        return Eigen(F.values.*(unitrange(A)[1]/unitdomain(A)[1]), rebuild(A,F.vectors,(unitdomain(A),Units(fill(unit(1.0),size(A,2))))))
    else
        error("UnitfulLinearAlgebra: Eigenvalue decomposition doesn't exist for for non-squarable matrices")
    end
end

"""
   Extend `isposdef` for Eigen factorizations of `MultipliableMatrix`s.
    Should the units be stripped out of the function?
    Only defined for matrices with uniform units (pp. 101, Hart, 1995). 
"""
isposdef(A::Eigen{T,V,S,U}) where {U<: AbstractVector, S<:AbstractUnitfulMatrix, V, T <: Number} = (uniform(A.vectors) && isreal(A.values)) && all(x -> x > 0, ustrip.(A.values))

"""
   Extend `inv` for Eigen factorizations of `MultipliableMatrix`s.
    Only defined for matrices with uniform units (pp. 101, Hart, 1995). 
"""
function inv(A::Eigen{T,V,S,U}) where {U <: AbstractVector, S <: AbstractUnitfulMatrix, V, T <: Number}

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
function svd(A::AbstractUnitfulMatrix;full=false,alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(parent(A))) 
    if uniform(A) 
        F = svd(parent(A), full=full, alg=alg)
        return SVD(F.U,F.S * unitrange(A)[1]./unitdomain(A)[1],F.Vt)
    else
        error("UnitfulLinearAlgebra: SVD doesn't exist for non-uniform matrices")
    end
end

# Dimensional (Unitful) Singular Value Decomposition, following Singular Value Decomposition from Julia LinearAlgebra.jl
"""
    DSVD <: Factorization

Matrix factorization type of the dimensional singular value decomposition (DSVD) of a matrix `A`.
This is the return type of [`dsvd(_)`](@ref), the corresponding matrix factorization function.

If `F::DSVD` is the factorization object, `U`, `S`, `V` and `V⁻¹` can be obtained
via `F.U`, `F.S`, `F.V` and `F.V⁻¹`, such that `A = U * Diagonal(S) * V⁻¹`.
The singular values in `S` are sorted in descending order.

Iterating the decomposition produces the components `U`, `S`, and `V`.

Differences from SVD struct: Vt -> V⁻¹, U and V can have different types.

Functions available for DSVD: `size`, `dsvdvals`, `inv`. 
Function available for SVD that would be good to have to DSVD: `ldiv!`, `transpose`. 
```
"""
struct DSVD{T,Tr,MU<:AbstractUnitfulMatrix{T},MV<:AbstractUnitfulMatrix{T},MQY<:AbstractUnitfulMatrix{T},MQX<:AbstractUnitfulMatrix{T},C<:AbstractVector{Tr}} <: Factorization{T}
    U′::MU
    S::C
    V′⁻¹::MV
    Qy::MQY
    Qx::MQX
    function DSVD{T,Tr,MU,MV,MQY,MQX,C}(U′, S, V′⁻¹,Qy,Qx) where {T,Tr,MU<:AbstractUnitfulMatrix{T},MV<:AbstractUnitfulMatrix{T},MQY<:AbstractUnitfulMatrix{T},MQX<:AbstractUnitfulMatrix{T},C<:AbstractVector{Tr}}
        LinearAlgebra.require_one_based_indexing(U′, S, V′⁻¹,Qy,Qx)
        new{T,Tr,MU,MV,MQY,MQX,C}(U′, S, V′⁻¹,Qy,Qx)
    end
end

DSVD(U′::AbstractArray{T}, S::AbstractVector{Tr}, V′⁻¹::AbstractArray{T}, Qy::AbstractArray{T}, Qx::AbstractArray{T}) where {T,Tr} =
    DSVD{T,Tr,typeof(U′),typeof(V′⁻¹),typeof(Qy),typeof(Qx),typeof(S)}(U′, S, V′⁻¹,Qy,Qx)
DSVD{T}(U′::AbstractArray, S::AbstractVector{Tr}, V′⁻¹::AbstractArray,Qy::AbstractArray, Qx::AbstractArray) where {T,Tr} =
    DSVD(convert(AbstractArray{T}, U′),
        convert(AbstractVector{Tr}, S),
         convert(AbstractArray{T}, V′⁻¹),
         convert(AbstractArray{T}, Qy),
         convert(AbstractArray{T}, Qx))
# backwards-compatible constructors (remove with Julia 2.0)
@deprecate(DSVD{T,Tr,MU,MV,MQY,MQX}(U′::AbstractArray{T}, S::AbstractVector{Tr}, V′⁻¹::AbstractArray{T}, Qy::AbstractArray{T}, Qx::AbstractArray{T}) where {T,Tr,MU,MV,MQY,MQX},
           DSVD{T,Tr,MU,MV,MQY,MQX,typeof(S)}(U′, S, V′⁻¹,Qy,Qx))

DSVD{T}(F::DSVD) where {T} = DSVD(
    convert(AbstractMatrix{T}, F.U′),
    convert(AbstractVector{real(T)}, F.S),
    convert(AbstractMatrix{T}, F.V′⁻¹),
    convert(AbstractMatrix{T}, F.Qy),
    convert(AbstractMatrix{T}, F.Qx))

Factorization{T}(F::DSVD) where {T} = DSVD{T}(F)

# iteration for destructuring into components
Base.iterate(S::DSVD) = (S.U, Val(:S))
Base.iterate(S::DSVD, ::Val{:S}) = (S.S, Val(:V))
Base.iterate(S::DSVD, ::Val{:V}) = (S.V, Val(:done))
Base.iterate(S::DSVD, ::Val{:done}) = nothing

function getproperty(F::DSVD, d::Symbol)
    if d === :U
        return F.Qy\F.U′
    elseif d === :U⁻¹
        return transpose(F.U′)*F.Qy
    elseif d === :V⁻¹
        return F.V′⁻¹*F.Qx
    elseif d === :V
        return F.Qx\transpose(F.V′⁻¹)
    else
        return getfield(F, d)
    end
end

Base.propertynames(F::DSVD, private::Bool=false) =
    private ? (:U, :U⁻¹, :V, :V⁻¹,  fieldnames(typeof(F))...) : (:U, :U⁻¹, :S, :V, :V⁻¹)

"""
    function dsvd(A::AbstractMultipliableMatrix,Prange::UnitSymmetricMatrix,Pdomain::UnitSymmetricMatrix;full=false,alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(A.numbers)) 

    Dimensional singular value decomposition (DSVD).
    Appropriate version of SVD for non-uniform matrices.
    `svd` can be computed for `Number`s, `Adjoint`s, `Tranpose`s, and `Integers`; `dsvd` doesn't yet implement these.
# Input
- `A::AbstractMultipliableMatrix`
- `Pr::UnitSymmetricMatrix`: square matrix defining norm of range
- `Pd::UnitSymmetricMatrix`: square matrix defining norm of domain
- `full=false`: optional argument
- `alg`: optional argument for algorithm
# Output:
- `F::DSVD`: Dimensional SVD object with units that can be deconstructed
"""
function dsvd(A::AbstractUnitfulMatrix,Py::AbstractUnitfulMatrix,Px::AbstractUnitfulMatrix;full=false,alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(parent(A))) 

    unit_symmetric(Py) ? Qy = getproperty(cholesky(Py),:U) : error("norm matrix for range not unit symmetric")
    unit_symmetric(Px) ? Qx = getproperty(cholesky(Px),:U) : error("norm matrix for domain not unit symmetric")

    # must be more efficient way
    #A′ = Qr*(A*inv(Qd))
    # still inefficient with copy
    A′ =   copy(transpose(transpose(Qx)\transpose(Qy*A)))
    ~dimensionless(A′) && error("A′ should be dimensionless to implement `LinearAlgebra.svd`")
    F = svd(parent(A′), full=full, alg=alg)

    # matrix slices cause unit domain and range to become ambiguous.
    # output of DSVD cannot be exact.
    # if exact(A)
    #     U = convert_unitdomain(UnitfulMatrix(F.U),Units(fill(unit(1.0),size(F.U,2))))
    #     Vt = convert_unitrange(UnitfulMatrix(F.Vt),Units(fill(unit(1.0),size(F.Vt,1))))
    #     return DSVD(U,F.S,Vt,Qy,Qx)
    # else
        return DSVD(UnitfulMatrix(F.U),F.S,UnitfulMatrix(F.Vt),Qy,Qx)
    #end
end

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::DSVD{<:Any,<:Any,<:AbstractArray,<:AbstractArray,<:AbstractArray,<:AbstractArray,<:AbstractVector})
    #summary(io, F); println(io)
    println(io, "U (left singular vectors):")
    show(io, mime, F.U)
    println(io, "\nsingular values:")
    show(io, mime, F.S)
    println(io, "\nV (right singular vectors):")
    show(io, mime, F.V)
end

dsvdvals(S::DSVD{<:Any,T}) where {T} = (S.S)::Vector{T}

function inv(F::DSVD{T}) where T
    @inbounds for i in eachindex(F.S)
        iszero(F.S[i]) && throw(SingularException(i))
    end
    k = searchsortedlast(F.S, eps(real(T))*F.S[1], rev=true)
    # adapted from `svd.jl`
    #@views (F.S[1:k] .\ F.V[:,1:k, :]) #* F.U⁻¹[1:k,:]

    # make sure it is inexact
    Σ⁻¹ = UnitfulMatrix(Diagonal(F.S[1:k].^-1))
    # a less efficient matrix way to do it.
    #Σ⁻¹ = Diagonal(F.S[1:k].^-1,fill(unit(1.0),k),fill(unit(1.0),k))
    # Σ⁻¹ = Diagonal(F.S[1:k].^-1,unitdomain(F.V[:,1:k]),unitrange(F.U⁻¹[1:k,:]))
    #    Σ⁻¹ = Diagonal(F.S[1:k].^-1,unitdomain(F.V)[1:k],unitrange(F.U⁻¹)[1:k])
    #println(exact(F.V[:,1:k]))
    return F.V[:,1:k]*Σ⁻¹*F.U⁻¹[1:k,:]
end

### DSVD least squares ### Not implemented
# function ldiv!(A::SVD{T}, B::StridedVecOrMat) where T
#     m, n = size(A)
#     k = searchsortedlast(A.S, eps(real(T))*A.S[1], rev=true)
#     mul!(view(B, 1:n, :), view(A.Vt, 1:k, :)', view(A.S, 1:k) .\ (view(A.U, :, 1:k)' * _cut_B(B, 1:m)))
#     return B
# end

size(A::DSVD, dim::Integer) = dim == 1 ? size(A.U, dim) : size(A.V⁻¹, dim)
size(A::DSVD) = (size(A, 1), size(A, 2))

# adjoint not yet defined for AbstractMultipliableMatrix
#function adjoint(F::DSVD)
#    return SVD(F.V⁻¹', F.S, F.U')
#end

"""
    function diagm(v::AbstractVector,r::Unitful.Unitlike,d::Unitful.Unitlike; exact = false)

    Construct diagonal matrix with units where the diagonal has elements `v`.
    If `v` has units, check that they conform with dimensional unit range `r`
     and dimensional unit domain `d`. Works for square or non-square matrices.
"""
diagm(v::AbstractVector,r::Units,d::Units; exact = false) = UnitfulMatrix(spdiagm(length(r),length(d),ustrip.(v)),(r,d); exact=exact)    

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
Diagonal(v::AbstractVector,r::AbstractVector,d::AbstractVector; exact = false) = ((length(r) == length(d)) && (length(v) == length(d))) ? UnitfulMatrix(LinearAlgebra.Diagonal(ustrip.(v)),(r,d); exact=exact) : error("unit range and domain do not define a square matrix")   
Diagonal(v::AbstractVector,r::Units,d::Units; exact = false) = ((length(r) == length(d)) && (length(v) == length(d))) ? UnitfulMatrix(LinearAlgebra.Diagonal(ustrip.(v)),(r,d); exact=exact) : error("unit range and domain do not define a square matrix")   

# """
#     function vcat(A,B)

#     Modeled after function `VERTICAL` (pp. 203, Hart, 1995).
# """
# function vcat(A::AbstractMultipliableMatrix,B::AbstractMultipliableMatrix)

#     numbers = vcat(A.numbers,B.numbers)
#     shift = unitdomain(A)[1]./unitdomain(B)[1]
#     ur = vcat(unitrange(A),unitrange(B).*shift)
#     bothexact = (exact(A) && exact(B))
#     return BestMultipliableMatrix(numbers,ur,unitdomain(A),exact=bothexact)
# end

# """
#     function hcat(A,B)

#     Modeled after function `HORIZONTAL` (pp. 202, Hart, 1995).
# """
# function hcat(A::AbstractMultipliableMatrix,B::AbstractMultipliableMatrix)

#     numbers = hcat(A.numbers,B.numbers)
#     shift = unitrange(A)[1]./unitrange(B)[1]
#     ud = vcat(unitdomain(A),unitdomain(B).*shift)
#     bothexact = (exact(A) && exact(B))
#     return BestMultipliableMatrix(numbers,unitrange(A),ud,exact=bothexact)
# end

end
