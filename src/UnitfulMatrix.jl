# UnitfulMatrix type and constructors
abstract type AbstractUnitfulVecOrMat{T,N,D<:Tuple,A} <: AbstractUnitfulType{T,N,D,A} end

# should be a subtype, not an actual type
const AbstractUnitfulVector{T<:Number} = AbstractUnitfulVecOrMat{T,1} where T
const AbstractUnitfulMatrix{T<:Number} = AbstractUnitfulVecOrMat{T,2} where T

# Concrete implementation ######################################################
"""
    struct UnitfulMatrix

Extend `DimArray` to use dimensions for units, also add `exact` boolean flag

struct UnitfulMatrix{T,N,D<:Tuple,A<:AbstractArray{T,N}} <: AbstractUnitfulVecOrMat{T,N,D,A}
    data::A
    dims::D
    exact::Bool
end
"""
struct UnitfulMatrix{T,N,D<:Tuple,A<:AbstractArray{T,N}} <: AbstractUnitfulVecOrMat{T,N,D,A}
    data::A
    dims::D
    exact::Bool
    # inner constructor: do not allow units into `data`
    # must be better way to incorporate types into `new` function
    UnitfulMatrix(data,dims,exact) = (eltype(parent(data)) <: Quantity) ? error("units not allowed in data field") : new{eltype(data),ndims(data),typeof(dims),typeof(data)}(data,dims,exact)
end

# 2 arg version
UnitfulMatrix(data::AbstractArray, dims; kw...) = UnitfulMatrix(data, (dims,); kw...)
function UnitfulMatrix(data::AbstractArray, dims::Union{Tuple,NamedTuple}; exact = false)
    if eltype(dims) <: Vector
        return UnitfulMatrix(data, format(Units.(dims), data), exact)
    elseif eltype(dims) <: Units
        return UnitfulMatrix(data, format(dims, data), exact)
    else
        error("something unexpected has happened! Please report it to the developer at https://github.com/ggebbie/UnitfulLinearAlgebra.jl/issues/new")
    end        
end
# back consistency with MultipliableMatrices.MMatrix
UnitfulMatrix(data::AbstractArray, unitrange::AbstractVector, unitdomain::AbstractVector; exact = true) = UnitfulMatrix(data, format((Units(unitrange),Units(unitdomain)), data), exact)

UnitfulMatrix(data::AbstractArray, unitrange::AbstractVector, unitdomain::Units; exact = true) = UnitfulMatrix(data, format((Units(unitrange),unitdomain), data), exact)

UnitfulMatrix(data::AbstractArray, unitrange::Units, unitdomain::AbstractVector; exact = true) = UnitfulMatrix(data, format((unitrange,Units(unitdomain)), data), exact)

UnitfulMatrix(data::AbstractArray, unitrange::Units, unitdomain::Units; exact = true) = UnitfulMatrix(data, format((unitrange,unitdomain), data), exact)

# lose the meta data
UnitfulMatrix(data::DimArray) = UnitfulMatrix(parent(data))

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

# for inner products, data may become a scalar quantity
# Choice: return UnitfulMatrix or return scalar quantity?
# Here: return scalar quantity
@inline function DimensionalData.rebuild(
    A::AbstractUnitfulVecOrMat, data::Number, dims::Tuple, exact::Bool)
    #UnitfulMatrix([data], dims, exact)
    Quantity(data,first(dims)[1])
end

"""
    rebuild(A::UnitfulMatrix, data, dims, exact) => UnitfulMatrix
    rebuild(A::UnitfulMatrix; kw...) => UnitfulMatrix

Rebuild a `UnitfulMatrix` with new fields. Handling partial field
update is dealt with in `rebuild` for `AbstractDimArray` (still true?).
"""
@inline DimensionalData.rebuildsliced(A::AbstractUnitfulVecOrMat, args...) = DimensionalData.rebuildsliced(getindex, A, args...)
# WARNING: kludge here, slicedims returns Tuple(Tuple())) which causes problems, Insert [1], needs a fix
@inline function DimensionalData.rebuildsliced(f::Function, A::AbstractUnitfulVecOrMat, data::AbstractArray, I::Tuple; exact= exact(A))

    urange = unitrange(A)[I[1]]
    udomain = unitdomain(A)[I[2]]

    # something strange with two types of `Units` type
    if (udomain isa Unitful.Units && urange isa UnitfulLinearAlgebra.Units )
        # case of column vector
        newunitrange = slicedvector(parent(urange),udomain)
        return UnitfulMatrix(data, newunitrange,exact=false)
    elseif (udomain isa UnitfulLinearAlgebra.Units && urange isa Unitful.Units )
        # case of row vector
        newunitrange = slicedvector(urange,parent(udomain))
        return UnitfulMatrix(data, newunitrange,exact=false)
    else
        # matrix case
        newunitrange, newunitdomain = slicedmatrix(urange,udomain)
        # unit range and domain of a sliced matrix are ambiguous.
        # It must be exact=false
        return UnitfulMatrix(data, newunitrange, newunitdomain, exact=false)
    end
end

slicedvector(urange,udomain) = urange./udomain
function slicedmatrix(urange,udomain) 
    unt = Array{Unitful.Units}(undef,length(urange),length(udomain))
    for m in 1:length(urange)
        for n in 1:length(udomain)
            unt[m,n] = urange[m]./udomain[n]
        end
    end
    # determine new range/domain
    newunitrange = unt[:,1]
    newunitdomain = unt[1,1]./unt[1,:]
    return newunitrange,newunitdomain
end

"""
    function UnitfulMatrix(A::AbstractMatrix)

    Constructor to make inexact UnitfulMatrix.
    Satisfies algebraic interpretation of multipliable
    matrices.
"""
function UnitfulMatrix(A::AbstractMatrix)
    numbers = ustrip.(A)

    if unit.(A[:,1]) isa Vector
        unitrange = unit.(A[:,1])
        unitdomain = unit(A[1,1])./unit.(A[1,:])
    else # code for StaticArrays (and other types?)
        M,N = size(numbers)
        unitdomain = Vector{Unitful.Units}(undef,N)
        unitrange = Vector{Unitful.Units}(undef,M)

        for i = 1:M
            unitrange[i] = unit(A[i,1])
        end

        for j = 1:N
           unitdomain[j] = unit(A[1,1])/unit(A[1,j])
        end
    end

    B = UnitfulMatrix(numbers,unitrange,unitdomain,exact=false)
    # if the array is not multipliable, return nothing
    if Matrix(B) == A
        return B
    else
        return nothing
    end
end
function UnitfulMatrix(a::AbstractVector) # should be called UnitfulVector?
    ur = unit.(a)
    if ur isa Vector
        b = UnitfulMatrix(ustrip.(a),ur,exact=false)
    else # code for StaticArrays (and other types?)
        M, = size(ur)
        unitrange = Vector{Unitful.Units}(undef,M)
        for i = 1:M
            unitrange[i] = unit(a[i])
        end
        b = UnitfulMatrix(ustrip.(a),unitrange,exact=false)
    end
    
    # if the array is not multipliable, return nothing
    if Matrix(b) == a
        return b
    else
        println("UnitfulLinearAlgebra warning: vector not multipliable (strange: vectors should always be multipliable)")
        return nothing
    end
end
