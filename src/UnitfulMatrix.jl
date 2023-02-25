
abstract type AbstractUnitfulVecOrMat{T,N,D<:Tuple,A} <: AbstractDimArray{T,N,D,A} end

# should be a subtype, not an actual type
const AbstractUnitfulVector{T<:Number} = AbstractUnitfulVecOrMat{T,1} where T
const AbstractUnitfulMatrix{T<:Number} = AbstractUnitfulVecOrMat{T,2} where T

# Concrete implementation ######################################################
"""
    struct UnitfulMatrix

    Take DimArray and use dimensions for units
"""
struct UnitfulMatrix{T,N,D<:Tuple,A<:AbstractArray{T,N}} <: AbstractUnitfulVecOrMat{T,N,D,A}
    data::A
    dims::D
    #refdims::R
    exact::Bool
end

# 2 arg version
UnitfulMatrix(data::AbstractArray, dims; kw...) = UnitfulMatrix(data, (dims,); kw...)
function UnitfulMatrix(data::AbstractArray, dims::Union{Tuple,NamedTuple}; exact = false)
    if eltype(dims) <: Vector
        return UnitfulMatrix(data, format(Units.(dims), data), exact)
    elseif eltype(dims) <: Units
        return UnitfulMatrix(data, format(dims, data), exact)
    end        
end
# back consistency with MMatrix
function UnitfulMatrix(data::AbstractArray, unitrange::Vector, unitdomain::Vector; exact = true)
    return UnitfulMatrix(data, format((Units(unitrange),Units(unitdomain)), data), exact)
end

"""
    rebuild(A::UnitfulMatrix, data, [dims, exact]) => UnitfulMatrix
    rebuild(A::UnitfulMatrix; kw...) => UnitfulMatrix

Rebuild a `UnitfulMatrix` with some field changes. All types
that inherit from `UnitfulMatrix` must define this method if they
have any additional fields or alternate field order.

This method can also be used with keyword arguments in place of regular arguments.
"""
@inline function DimensionalData.rebuild(
    A::AbstractUnitfulVecOrMat, data, dims::Tuple=dims(A))
    DimensionalData.rebuild(A, data, dims, exact(A))
end

@inline function DimensionalData.rebuild(
    A::AbstractUnitfulVecOrMat, data::AbstractArray, dims::Tuple, exact::Bool)
    UnitfulMatrix(data, dims, exact)
end

"""
    rebuild(A::UnitfulMatrix, data, dims, exact) => UnitfulMatrix
    rebuild(A::UnitfulMatrix; kw...) => UnitfulMatrix

Rebuild a `UnitfulMatrix` with new fields. Handling partial field
update is dealt with in `rebuild` for `AbstractDimArray` (still true?).
"""
@inline DimensionalData.rebuildsliced(A::AbstractUnitfulVecOrMat, args...) = rebuildsliced(getindex, A, args...)
# WARNING: kludge here, slicedims returns Tuple(Tuple())) which causes problems, Insert [1], needs a fix
@inline DimensionalData.rebuildsliced(f::Function, A::AbstractUnitfulVecOrMat, data::AbstractArray, I::Tuple; exact= exact(A)) =
    DimensionalData.rebuild(A, data, DimensionalData.slicedims(f, A, I)[1],exact) 

function Base.show(io::IO, mime::MIME"text/plain", A::UnitfulMatrix{T,N}) where {T,N}
    lines = 0
    summary(io, A)
    #print_name(io, name(A))
    #lines += Dimensions.print_dims(io, mime, dims(A))
    #!(isempty(dims(A)) || isempty(refdims(A))) && println(io)
    #lines += Dimensions.print_refdims(io, mime, refdims(A))
    println(io)

    # DELETED THIS OPTIONAL PART HERE
    # Printing the array data is optional, subtypes can 
    # show other things here instead.
    ds = displaysize(io)
    ioctx = IOContext(io, :displaysize => (ds[1] - lines, ds[2]))
    #println("show after")
    #DimensionalData.show_after(ioctx, mime, Matrix(A))

    #function print_array(io::IO, mime, A::AbstractDimArray{T,2}) where T
    T2 = eltype(A)
    Base.print_matrix(DimensionalData._print_array_ctx(ioctx, T2), Matrix(A))

    return nothing
end

"""
    function UnitfulMatrix(A::AbstractMatrix)

    Constructor to make inexact UnitfulMatrix.
    Satisfies algebraic interpretation of multipliable
    matrices.
"""
function UnitfulMatrix(A::AbstractMatrix)
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

    B = UnitfulMatrix(numbers,unitrange,unitdomain,exact=false)
    # if the array is not multipliable, return nothing
    if Matrix(B) == A
        return B
    else
        return nothing
    end
end
# function UnitfulMatrix(A::AbstractVector) # should be called UnitfulVector?
#     numbers = ustrip.(A)
#     #M = size(numbers)
#     #unitrange = Vector{Unitful.FreeUnits}(undef,M)

#     unitrange = Units(unit.(A))
#     B = UnitfulMatrix(numbers,unitrange,exact=false)
#     # if the array is not multipliable, return nothing
#     if Matrix(B) == A
#         return B
#     else
#         return nothing
#     end
# end
function UnitfulMatrix(a::AbstractVector) # should be called UnitfulVector?
    numbers = ustrip.(a)
    M = size(numbers)
    unitrange = Vector{Unitful.FreeUnits}(undef,M)

    unitrange = unit.(a)
    b = UnitfulMatrix(numbers,unitrange,exact=false)
    # if the array is not multipliable, return nothing
    if Matrix(b) == a
        return b
    else
        println("warning: vector not multipliable")
        return nothing
    end
end
