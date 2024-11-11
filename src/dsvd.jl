# Dimensioned (Unitful) Singular Value Decomposition, following Singular Value Decomposition from Julia LinearAlgebra.jl
"""
    DSVD <: Factorization

Matrix factorization type of the dimensioned singular value decomposition (DSVD) of a matrix `A`.
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

function Base.getproperty(F::DSVD, d::Symbol)
    if d === :U
        #return inv(F.U⁻¹) # short-term workaround
        # would be better to handle with multiplication of non-UnitfulMatrix F.U′
        return F.Qy\convert_unitrange(F.U′,unitrange(F.Qy))
        #return F.Qy\F.U′
    elseif d === :U⁻¹
        return transpose(F.U′)*F.Qy
    elseif d === :V⁻¹
        return F.V′⁻¹*F.Qx
    elseif d === :V
        #return inv(F.V⁻¹) # short-term workaround
        #return F.Qx\transpose(F.V′⁻¹)
        return F.Qx\convert_unitrange(transpose(F.V′⁻¹),unitrange(F.Qx))
    else
        return getfield(F, d)
    end
end

Base.propertynames(F::DSVD, private::Bool=false) =
    private ? (:U, :U⁻¹, :V, :V⁻¹,  fieldnames(typeof(F))...) : (:U, :U⁻¹, :S, :V, :V⁻¹)

"""
    function dsvd(A::AbstractMultipliableMatrix,Prange::UnitSymmetricMatrix,Pdomain::UnitSymmetricMatrix;full=false,alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(A.numbers)) 

    Dimensioned singular value decomposition (DSVD).
    Appropriate version of SVD for non-uniform matrices.
    `svd` can be computed for `Number`s, `Adjoint`s, `Transpose`s, and `Integers`; `dsvd` doesn't yet implement these.
# Input
- `A::AbstractMultipliableMatrix`
- `Pr::UnitSymmetricMatrix`: square matrix defining norm of range
- `Pd::UnitSymmetricMatrix`: square matrix defining norm of domain
- `full=false`: optional argument
- `alg`: optional argument for algorithm
# Output:
- `F::DSVD`: Dimensioned SVD object with units that can be deconstructed
"""
function dsvd(A::AbstractUnitfulMatrix,Py::AbstractUnitfulMatrix,Px::AbstractUnitfulMatrix;full=false,alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(parent(A))) 

    unit_symmetric(Py) ? Qy = getproperty(cholesky(Py),:U) : error("norm matrix for range not unit symmetric")
    unit_symmetric(Px) ? Qx = getproperty(cholesky(Px),:U) : error("norm matrix for domain not unit symmetric")

    # must be more efficient way
    #A′ = Qr*(A*inv(Qd))
    # still inefficient with copy
    A′ =   copy(transpose(transpose(Qx)\transpose(Qy*A)))
    !dimensionless(A′) && error("A′ should be dimensionless to implement `LinearAlgebra.svd`")
    F = svd(parent(A′), full=full, alg=alg)

    # matrix slices cause unit domain and range to become ambiguous.
    # output of DSVD cannot be exact.
    return DSVD(UnitfulMatrix(F.U),F.S,UnitfulMatrix(F.Vt),Qy,Qx)

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
    # make sure it is inexact
    Σ⁻¹ = UnitfulMatrix(Diagonal(F.S[1:k].^-1))
    return F.V[:,1:k]*Σ⁻¹*F.U⁻¹[1:k,:]
end

Base.size(A::DSVD, dim::Integer) = dim == 1 ? size(A.U, dim) : size(A.V⁻¹, dim)
Base.size(A::DSVD) = (size(A, 1), size(A, 2))

### DSVD least squares ### Not implemented

# adjoint not yet defined for AbstractMultipliableMatrix
#function adjoint(F::DSVD)
#    return SVD(F.V⁻¹', F.S, F.U')
#end
