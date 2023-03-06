# UnitfulDimMatrix type and constructors
abstract type AbstractUnitfulDimVecOrMat{T,N,UD<:Tuple,D<:Tuple,A} <: AbstractUnitfulType{T,N,D,A} end

const AbstractUnitfulDimVector{T<:Number} = AbstractUnitfulDimVecOrMat{T,1} where T
const AbstractUnitfulDimMatrix{T<:Number} = AbstractUnitfulDimVecOrMat{T,2} where T

"""
    struct UnitfulDimMatrix

    Built on DimensionalData.DimArray.
    Add `unitdims` for unit dimensions (range and domain).
    Add `exact::Bool` which is true for geometric interpretation.
"""
struct UnitfulDimMatrix{T,N,UD<:Tuple,D<:Tuple,R<:Tuple,A<:AbstractArray{T,N},Na,Me} <: AbstractUnitfulDimVecOrMat{T,N,UD,D,A}
    data::A
    unitdims::UD
    dims::D
    refdims::R
    name::Na
    metadata::Me
    exact::Bool
end

# 2 arg version: required input: numerical values, unitdims, (axis) dims
UnitfulDimMatrix(data::AbstractArray, unitdims; kw...) = UnitfulDimMatrix(data, (unitdims,); kw...)
function UnitfulDimMatrix(data::AbstractArray, unitdims::Union{Tuple,NamedTuple}; dims=(),
    refdims=(), name=DimensionalData.NoName(), metadata=DimensionalData.NoMetadata(), exact = true)
    if eltype(unitdims) <: Vector
        return UnitfulDimMatrix(data, format(Units.(unitdims), data), format(dims,data), refdims, name, metadata, exact)
    elseif eltype(unitdims) <: Units
        return UnitfulDimMatrix(data, format(unitdims, data), format(dims,data), refdims, name, metadata, exact)
    end        
end
# back consistency with MMatrix
function UnitfulDimMatrix(data::AbstractArray, unitrange, unitdomain; 
    dims=(), refdims=(), name=DimensionalData.NoName(), metadata=DimensionalData.NoMetadata(), exact = true)
    return UnitfulDimMatrix(data, format((Units(unitrange),Units(unitdomain)), data), format(dims, data), refdims, name, metadata, exact)
end

"""
    rebuild(A::UnitfulDimMatrix, data, [dims, refdims, name, metadata]) => UnitfulMatrix
    rebuild(A::UnitfulDimMatrix; kw...) => UnitfulMatrix

Rebuild a `UnitfulDimMatrix` with some field changes. All types
that inherit from `UnitfulMatrix` must define this method if they
have any additional fields or alternate field order.

Implementations can discard arguments like `refdims`, `name` and `metadata`.

This method can also be used with keyword arguments in place of regular arguments.
"""
@inline function DimensionalData.rebuild(
    A::UnitfulDimMatrix, data, unitdims::Tuple=unitdims(A), dims::Tuple=dims(A), refdims=refdims(A), name=name(A))
    DimensionalData.rebuild(A, data, unitdims, dims, refdims, name, metadata(A), exact(A))
end

@inline function DimensionalData.rebuild(
    A::UnitfulDimMatrix, data::AbstractArray, unitdims::Tuple, dims::Tuple, refdims::Tuple, name, metadata, exact
)
    UnitfulDimMatrix(data, unitdims, dims, refdims, name, metadata,exact)
end

#@inline function rebuild(
#     A::UnitfulMatrix, data; dims::Tuple=dims(A), refdims=refdims(A), name=name(A))
#     DimensionalData.rebuild(A, data, dims, refdims, name, metadata(A),exact(A))
# end

"""
    rebuild(A::UnitfulMatrix, data, dims, refdims, name, metadata,exactflag) => UnitfulMatrix
    rebuild(A::UnitfulMatrix; kw...) => UnitfulMatrix

Rebuild a `UnitfulMatrix` with new fields. Handling partial field
update is dealt with in `rebuild` for `AbstractDimArray` (still true?).
"""
#@inline function rebuild(
#    A::UnitfulMatrix, data::AbstractArray, dims::Tuple, refdims::Tuple, name, metadata, exactflag
#)
#    UnitfulMatrix(data, dims, refdims, name, metadata, exactflag)
#end

"""
    rebuild(A::UnitfulMatrix, data, dims, exact) => UnitfulMatrix
    rebuild(A::UnitfulMatrix; kw...) => UnitfulMatrix

Rebuild a `UnitfulMatrix` with new fields. Handling partial field
update is dealt with in `rebuild` for `AbstractDimArray` (still true?).
"""
@inline DimensionalData.rebuildsliced(A::AbstractUnitfulDimVecOrMat, args...) = DimensionalData.rebuildsliced(getindex, A, args...)
# WARNING: kludge here, slicedims returns Tuple(Tuple())) which causes problems, Insert [1], needs a fix
@inline function DimensionalData.rebuildsliced(f::Function, A::AbstractUnitfulDimVecOrMat, data::AbstractArray, I::Tuple; exact= exact(A))

    urange = unitrange(A)[I[1]]
    udomain = unitdomain(A)[I[2]]

    if (udomain isa Unitful.FreeUnits || urange isa Unitful.FreeUnits )
        # case of column vector, row vector, scalar
        # scalar appears to be overridden by getindex
        newunitrange = slicedvector(urange,udomain)

        ## NEED TO SLICE DIMS ALSO
        return UnitfulDimMatrix(data, newunitrange)
        #return UnitfulMatrix(data, newunitrange, newunitdomain)
    else
        newunitrange, newunitdomain = slicedmatrix(urange,udomain)
        # unit range and domain of a sliced matrix are ambiguous.
        # It must be exact=false
        ##  NEED TO SLICE (axis) DIMS ALSO
        return UnitfulDimMatrix(data, newunitrange, newunitdomain, exact=false)
    end
end

"""
    function UnitfulDimMatrix(A::AbstractMatrix)

    Constructor to make inexact UnitfulDimMatrix.
    Satisfies algebraic interpretation of multipliable
    matrices. Doesn't add any metadata of a DimArray.
"""
function UnitfulDimMatrix(A::AbstractMatrix)
    numbers = ustrip.(A)
    M,N = size(numbers)
    unitdomain = Vector{Unitful.FreeUnits}(undef,N)
    unitrange = Vector{Unitful.FreeUnits}(undef,M)

    for i = 1:M
        unitrange[i] = unit(A[i,1])
    end
    
    for j = 1:N
        unitdomain[j] = unit(A[1,1])/unit(A[1,j])
    end

    B = UnitfulDimMatrix(numbers,unitrange,unitdomain)
    # if the array is not multipliable, return nothing
    if Matrix(B) == A
        return B
    else
        return nothing
    end
end
function UnitfulDimMatrix(A::AbstractVector) # should be called UnitfulVector?
    numbers = ustrip.(A)
    M = size(numbers)
    unitrange = Vector{Unitful.FreeUnits}(undef,M)

    unitrange = unit.(A)
    B = UnitfulDimMatrix(numbers,unitrange)
    # if the array is not multipliable, return nothing
    if Matrix(B) == A
        return B
    else
        return nothing
    end
end
                            
function DimensionalData._rebuildmul(A::AbstractUnitfulDimMatrix, B::AbstractUnitfulDimMatrix)
    # compare unitdims
    DimensionalData.comparedims(last(unitdims(A)), first(unitdims(B)); val=true)

    # compare regular (axis) dims
    DimensionalData.comparedims(last(dims(A)), first(dims(B)); val=true)
    
    rebuild(A, parent(A) * parent(B), (first(unitdims(A)),last(unitdims(B))), (first(dims(A)),last(dims(B))))
end

DimensionalData._rebuildmul(A::AbstractUnitfulDimMatrix, b::Number) = rebuild(A, parent(A).*b, (unitrange(A), unitdomain(A)))
DimensionalData._rebuildmul(a::AbstractUnitfulDimVector, b::Number) = rebuild(a, parent(a).*b, (unitrange(a)))
DimensionalData._rebuildmul(b::Number, A::AbstractUnitfulDimVecOrMat) = A*b
