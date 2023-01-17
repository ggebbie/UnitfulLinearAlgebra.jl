module UnitfulLinearAlgebra

using Unitful
using LinearAlgebra
using SparseArrays
using DimensionalData
#using DimensionalData: @dim, dims, DimArray, AbstractDimArray, NoName, NoMetadata, format, print_name

export AbstractMultipliableMatrix, DimMatrix
export MultipliableMatrix, EndomorphicMatrix
export SquarableMatrix, UniformMatrix
export LeftUniformMatrix, RightUniformMatrix
export UnitSymmetricMatrix, DSVD
export BestMultipliableMatrix
export similarity, ∥, parallel
export uniform, left_uniform, right_uniform
export invdimension, dottable
export getindex, setindex!, size, similar
export convert_unitrange, convert_unitdomain
export convert_unitrange!, convert_unitdomain!
export exact, multipliable, dimensionless, endomorphic
export svd, dsvd
export eigen, isposdef, inv, transpose
export unitrange, unitdomain
export square, squarable, singular, unit_symmetric
export lu, det, trace, diag, diagm
export Diagonal, (\), cholesky
export identitymatrix, show, vcat, hcat, rebuild

import LinearAlgebra: inv, det, lu,
    svd, getproperty, eigen, isposdef,
    diag, diagm, Diagonal, cholesky
import Base:(~), (*), (+), (-), (\), getindex, setindex!,
    size, range, transpose, similar, show, vcat, hcat

import DimensionalData: rebuild, @dim, dims, DimArray, AbstractDimArray, NoName, NoMetadata, format, print_name

@dim Units "units"

abstract type AbstractMultipliableMatrix{T<:Number} <: AbstractMatrix{T} end

#struct DimMatrix{T<:Number,N<:Integer} <: AbstractDimArray{T,N}
    
#end

# struct DimMatrix{T,N,D<:Tuple,A<:AbstractArray{T,N}} <: AbstractDimArray{T,N,D,A}
#     data::A
#     dims::D
# end

struct DimMatrix{T,N,D<:Tuple,R<:Tuple,A<:AbstractArray{T,N},Na,Me} <: AbstractDimArray{T,N,D,A}
    data::A
    dims::D
    refdims::R
    name::Na
    metadata::Me
    exact::Bool
end

# 2 arg version
DimMatrix(data::AbstractArray, dims; kw...) = DimMatrix(data, (dims,); kw...)
function DimMatrix(data::AbstractArray, dims::Union{Tuple,NamedTuple}; 
    refdims=(), name=DimensionalData.NoName(), metadata=DimensionalData.NoMetadata(), exact = false
)
    DimMatrix(data, format(dims, data), refdims, name, metadata, exact)
end
# back consistency with MMatrix
function DimMatrix(data::AbstractArray, unitrange, unitdomain; 
    refdims=(), name=DimensionalData.NoName(), metadata=DimensionalData.NoMetadata(), exact = false
                   )
    
    DimMatrix(data, format((Units(unitrange),Units(unitdomain)), data), refdims, name, metadata, exact)
end

#function BestMultipliableMatrix(numbers::AbstractMatrix,unitrange::AbstractVector,unitdomain::AbstractVector;exact=false)::AbstractMultipliableMatrix
#                V = DimMatrix(Unum,(Units(p),Units(q̃)))

"""
    rebuild(A::DimMatrix, data, [dims, refdims, name, metadata]) => DimMatrix
    rebuild(A::DimMatrix; kw...) => DimMatrix

Rebuild a `DimMatrix` with some field changes. All types
that inherit from `DimMatrix` must define this method if they
have any additional fields or alternate field order.

Implementations can discard arguments like `refdims`, `name` and `metadata`.

This method can also be used with keyword arguments in place of regular arguments.
"""
@inline function rebuild(
    A::DimMatrix, data, dims::Tuple=dims(A), refdims=refdims(A), name=name(A))
    rebuild(A, data, dims, refdims, name, metadata(A), exact(A))
end

@inline function rebuild(
    A::DimMatrix, data::AbstractArray, dims::Tuple, refdims::Tuple, name, metadata, exactflag
)
    DimMatrix(data, dims, refdims, name, metadata,exactflag)
end

#@inline function rebuild(
#     A::DimMatrix, data; dims::Tuple=dims(A), refdims=refdims(A), name=name(A))
#     DimensionalData.rebuild(A, data, dims, refdims, name, metadata(A),exact(A))
# end

"""
    rebuild(A::DimMatrix, data, dims, refdims, name, metadata,exactflag) => DimMatrix
    rebuild(A::DimMatrix; kw...) => DimMatrix

Rebuild a `DimMatrix` with new fields. Handling partial field
update is dealt with in `rebuild` for `AbstractDimArray` (still true?).
"""
#@inline function rebuild(
#    A::DimMatrix, data::AbstractArray, dims::Tuple, refdims::Tuple, name, metadata, exactflag
#)
#    DimMatrix(data, dims, refdims, name, metadata, exactflag)
#end

function Base.show(io::IO, mime::MIME"text/plain", A::DimMatrix{T,N}) where {T,N}
    lines = 0
    summary(io, A)
    print_name(io, name(A))
    lines += Dimensions.print_dims(io, mime, dims(A))
    !(isempty(dims(A)) || isempty(refdims(A))) && println(io)
    lines += Dimensions.print_refdims(io, mime, refdims(A))
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
    struct MultipliableMatrix

    Matrices with units that are physically reasonable,
    i.e., more than just an array of values with units.

    Units are consistent with many linear algebraic manipulations, including multiplication.

    Hart (1995) suggests that these matrices simply be called "matrices", and that matrices with dimensional values that cannot be multiplied should be called "arrays."

# Fields
- `numbers`: numerical (dimensionless) matrix
- `unitrange`: dimensional range in terms of units
- `unitdomain`: dimensional domain in terms of units
- `exact`: geometric (`true`) or algebraic (`false`) interpretation
"""
struct MultipliableMatrix{T <:Number} <: AbstractMultipliableMatrix{T}
    numbers::AbstractMatrix{T}
    unitrange::Vector{N1} where N1 <: Unitful.Unitlike
    unitdomain::Vector{N2} where N2 <: Unitful.Unitlike
    #unitrange::AbstractVector
    #unitdomain::AbstractVector
    exact::Bool
end

"""
    struct EndomorphicMatrix

    Maps dimensioned vector space to itself.
    Equivalent unit (dimensional) range and domain.

# Fields
- `numbers`: numerical (dimensionless) matrix
- `unitrange`: unit (dimensional) range in terms of units, and also equal the unit (dimensional) domain
- `exact`: geometric (`true`) or algebraic (`false`) interpretation
"""
struct EndomorphicMatrix{T<:Number} <: AbstractMultipliableMatrix{T} 
    numbers::AbstractMatrix{T}
    #unitrange::Vector
    unitrange::Vector{N1} where N1 <: Unitful.Unitlike
    exact::Bool
end

"""
    struct SquarableMatrix

    An squarable matrix is one where 𝐀² is defined.
    Unit (dimensional) range and domain are parallel.
    Key for solving difference and differential equations.
    Have eigenstructure. 

# Fields
- `numbers`: numerical (dimensionless) matrix
- `unitrange`: unit (dimensional) range
- `Δunitdomain`: shift to range that gives the domain
- `exact`: geometric (`true`) or algebraic (`false`) interpretation
"""
struct SquarableMatrix{T<:Number} <: AbstractMultipliableMatrix{T} 
    numbers::AbstractMatrix{T}
    #unitrange::Vector
    unitrange::Vector{N} where N <: Unitful.Unitlike
    Δunitdomain
    exact::Bool
end

"""
    struct UnitSymmetricMatrix

    `UnitSymmetricMatrix`s have units that are symmetric about the main diagonal and define weighted norms. 
    Definition: inverse dimensional range and dimensional domain are parallel.
    Called "dimensionally symmetric" by Hart, 1995.

# Fields
- `numbers`: numerical (dimensionless) matrix
- `unitrange`: dimensional range in terms of units, this is also the domain
- `Δunitdomain`: shift to range that gives the domain
- `exact`: geometric (`true`) or algebraic (`false`) interpretation
"""
struct UnitSymmetricMatrix{T<:Number} <: AbstractMultipliableMatrix{T} 
    numbers::AbstractMatrix{T}
    unitrange::Vector{N1} where N1 <: Unitful.Unitlike
    #unitrange::Vector
    Δunitdomain
    exact::Bool
end

"""
    struct UniformMatrix

    Uniform matrix: All entries have the same units

# Attributes
- `numbers`: numerical (dimensionless) matrix
- `unitrange`:  uniform unit (dimensional) range expressed as a single unit
- `unitdomain`: uniform unit (dimensional) domain expressed as a single unit
- `exact`: geometric (`true`) or algebraic (`false`) interpretation of matrix
"""
struct UniformMatrix{T<:Number} <: AbstractMultipliableMatrix{T}
    numbers::AbstractMatrix{T}
#    unitrange::AbstractVector # just one entry
    #    unitdomain::AbstractVector # just one entry
    unitrange::Vector{N1} where N1 <: Unitful.Unitlike
    unitdomain::Vector{N2} where N2 <: Unitful.Unitlike
    #unitrange::Vector{Unitful.FreeUnits} # just one entry
    #unitdomain::Vector{Unitful.FreeUnits} # just one entry
    exact::Bool
end
# struct UniformMatrix{T,R,D} <: AbstractMultipliableMatrix where {T <: Number} where {R,D <: Unitful.Unitlike}
#     numbers::Matrix{T}
#     range::R
#     domain::D
#     exact::Bool
# end

"""
    struct LeftUniformMatrix

    Left uniform matrix: output of matrix has uniform units

# Fields
- `numbers`: numerical (dimensionless) matrix
- `unitrange`:  uniform dimensional range expressed as a single unit
- `unitdomain`: dimensional domain (Vector)
- `exact`: geometric (`true`) or algebraic (`false`) interpretation
"""
struct LeftUniformMatrix{T<:Number} <: AbstractMultipliableMatrix{T}
    numbers::AbstractMatrix{T}
    unitrange::Vector{Unitful.FreeUnits} # usually just one entry, so that the type is not too exact and changeable with convert_unitdomain!
    unitdomain::AbstractVector
    exact::Bool
end

"""
    struct RightUniformMatrix

    Right uniform matrix: input of matrix must have uniform units

# Fields
- `numbers`: numerical (dimensionless) matrix
- `unitrange::Vector`:  unit (dimensional) range
- `unitdomain`: uniform dimensional domain expressed as a single unit
- `exact`: geometric (`true`) or algebraic (`false`) interpretation
"""
struct RightUniformMatrix{T<:Number} <: AbstractMultipliableMatrix{T}
    numbers::AbstractMatrix{T}
    unitrange::AbstractVector
    unitdomain::Vector{Unitful.FreeUnits} # usually just one entry
    exact::Bool
end

"""
    MultipliableMatrix(numbers,unitrange,unitdomain;exact=false)

    Constructor where `exact` is a keyword argument. One may construct a MultipliableMatrix without specifying exact, in which case it defaults to `false`. 
"""
MultipliableMatrix(numbers,unitrange,unitdomain;exact=false) =
    MultipliableMatrix(numbers,unitrange,unitdomain,exact)

"""
     BestMultipliableMatrix(A::Matrix)

    Transform array to a type <: AbstractMultipliableMatrix.
    Finds best representation amongst
    UniformMatrix, EndomorphicMatrix, or MultipliableMatrix.
    Assumes `exact=false`
"""
function BestMultipliableMatrix(A::Matrix)::AbstractMultipliableMatrix

    numbers = ustrip.(A)
    M,N = size(numbers)
    #U = typeof(unit(A[1,1]))
    #U = eltype(unit.(A))
    # unitdomain = Vector{U}(undef,N)
    # unitrange = Vector{U}(undef,M)
    unitdomain = Vector{Unitful.FreeUnits}(undef,N)
    unitrange = Vector{Unitful.FreeUnits}(undef,M)

    for i = 1:M
        unitrange[i] = unit(A[i,1])
    end
    
    for j = 1:N
        unitdomain[j] = unit(A[1,1])/unit(A[1,j])
    end

    B = BestMultipliableMatrix(numbers,unitrange,unitdomain)
    # if the array is not multipliable, return nothing
    if Matrix(B) == A
        return B
    else
        return nothing
    end
end

"""
    function BestMultipliableMatrix(numbers,unitrange,unitdomain;exact=false)

    What kind of Multipliable Matrix is the best representation?
"""
function BestMultipliableMatrix(numbers::AbstractMatrix,unitrange::AbstractVector,unitdomain::AbstractVector;exact=false)::AbstractMultipliableMatrix
    if uniform(unitrange) && uniform(unitdomain)
        ur = Base.convert(Vector{Unitful.FreeUnits},[unitrange[1]])
        ud = Base.convert(Vector{Unitful.FreeUnits},[unitdomain[1]])
        B = UniformMatrix(numbers,ur,ud,exact)
    elseif uniform(unitrange)
        ur = Base.convert(Vector{Unitful.FreeUnits},[unitrange[1]])
        B = LeftUniformMatrix(numbers,ur,unitdomain,exact)
    elseif uniform(unitdomain)
        ud = Base.convert(Vector{Unitful.FreeUnits},[unitdomain[1]])
        B = RightUniformMatrix(numbers,unitrange,ud,exact)
    elseif unitrange == unitdomain
        B = EndomorphicMatrix(numbers,unitrange,exact)
    elseif unitrange ∥ unitdomain
        Δunitdomain = unitdomain[1]./unitrange[1]
        B = SquarableMatrix(numbers,unitrange,Δunitdomain,exact)
    elseif unitrange ∥ 1 ./unitdomain
        Δunitdomain = unitdomain[1] * unitrange[1]
        B = UnitSymmetricMatrix(numbers,unitrange,Δunitdomain,exact)
    else
        B = MultipliableMatrix(numbers,unitrange,unitdomain,exact)
    end
end

"""
    MMatrix (Multipliable Matrix): shortcut for `BestMultipliableMatrix`
"""
MMatrix = BestMultipliableMatrix



"""
    function multipliable(A)::Bool

    Is an array multipliable?
    It requires a particular structure of the units/dimensions in the array. 
"""
multipliable(A::Matrix) = ~isnothing(BestMultipliableMatrix(A))
multipliable(A::T) where T <: AbstractMultipliableMatrix = true

"""
    function EndomorphicMatrix

    Constructor with keyword argument `exact=false`.
    If `exact` not specified, defaults to `false`.
"""
EndomorphicMatrix(numbers,unitrange;exact=false) =
    EndomorphicMatrix(numbers,unitrange,exact)

"""
     EndomorphicMatrix(A)

    Transform array to EndomorphicMatrix type
"""
function EndomorphicMatrix(A::Matrix)

    numbers = ustrip.(A)
    M,N = size(numbers)

    # must be square
    if M ≠ N
        return nothing
    end
    
    unitrange = Vector{Unitful.FreeUnits}(undef,M)
    for i = 1:M
        unitrange[i] = unit(A[i,1])
    end
    B = EndomorphicMatrix(numbers,unitrange)
    
    # if the array is not multipliable, return nothing
    if Matrix(B) == A
        return B
    else
        return nothing
    end
end

"""
    function EndomorphicMatrix(A::T) where T <: Number

    Special case of a scalar. Must be dimensionless.
"""
function EndomorphicMatrix(A::T) where T <: Number
    if dimensionless(A)
        numbers = Matrix{T}(undef,1,1)
        numbers[1,1] = A

        unitrange = Vector{Unitful.FreeUnits}(undef,1)
        unitrange[1] = unit(A)
        return EndomorphicMatrix(numbers,unitrange)
    else
        return nothing
    end
end


"""
    function endomorphic(A)::Bool

    Endomorphic matrices have a particular structure
     of the units/dimensions in the array. 
"""
endomorphic(A::Matrix) = ~isnothing(EndomorphicMatrix(A))
endomorphic(A::EndomorphicMatrix) = true
endomorphic(A::T) where T <: AbstractMultipliableMatrix = isequal(unitdomain(A),unitrange(A))
endomorphic(A::T) where T <: Number = dimensionless(A) # scalars must be dimensionless to be endomorphic

"""
    function UniformMatrix

    Constructor with keyword argument `exact=false`.
    If `exact` not specified, defaults to `false`.
"""
UniformMatrix(numbers,unitrange,unitdomain;exact=false) =
    UniformMatrix(numbers,unitrange,unitdomain,exact)

similar(A::AbstractMultipliableMatrix{T}) where T <: Number =  BestMultipliableMatrix(Matrix{T}(undef,size(A)),unitrange(A),unitdomain(A);exact=exact(A))
        
"""
    function getindex(A::AbstractMultipliableMatrix,i::Integer,j::Integer)

    Recover element (i,j) of a AbstractMultipliableMatrix.
    Part of the AbstractArray interface.
#Input
- `A::AbstractMultipliableMatrix`
- `i::Integer`: row index
- `j::Integer`: column index
#Output
- `Quantity`: numerical value and units (for vector)
- `AbstractMultipliableMatrix`: for matrix output
"""
getindex(A::T,i::Union{Colon,UnitRange},j::Int) where T <: AbstractMultipliableMatrix = Quantity.(A.numbers[i,j],unitrange(A)[i]./unitdomain(A)[j]) 

getindex(A::T,i::Int,j::Union{Colon,Int,UnitRange}) where T <: AbstractMultipliableMatrix = Quantity.(A.numbers[i,j],unitrange(A)[i]./unitdomain(A)[j]) 

#       getindex(::A{T,N}, ::Vararg{Int, N}) where {T,N} # if IndexCartesian()
getindex(A::T,i::Union{UnitRange,Colon},j::Union{UnitRange,Colon}) where T <: AbstractMultipliableMatrix = MMatrix(A.numbers[i,j],unitrange(A)[i],unitdomain(A)[j],exact=exact(A)) 

#getindex(A::T,i::Colon,j::UnitRange) where T <: AbstractMultipliableMatrix = MMatrix(A.numbers[i,j],unitrange(A)[i],unitdomain(A)[j],exact=exact(A)) 


"""
    function setindex!(A::MultipliableMatrix,v,i,j)

    Set element (i,j) of a MultipliableMatrix.
    Part of the AbstractArray interface.
#Input
- `A::AbstractMultipliableMatrix`
- `v`: new value
- `i::Integer`: row index
- `j::Integer`: column index
#Output
- `Quantity`: numerical value and units
"""
function setindex!(A::T,v,i::Int,j::Int) where T <: AbstractMultipliableMatrix

    if unit(v) == unitrange(A)[i]./unitdomain(A)[j]
        A.numbers[i,j] = ustrip(v)
    else error("new value has incompatible units")
    end
end
#function setindex!(A::T,v,i::Int,j::Int) where T <: AbstractMultipliableMatrix = A.numbers[i,j] = ustrip(v)) 

"""
    function Matrix(A::MultipliableMatrix)

    Expand A into array form
    Useful for tests, display
    pp. 193, Hart
"""
function Matrix(A::T) where T<: AbstractMultipliableMatrix

    M,N = size(A)
    #M = rangelength(A)
    #N = domainlength(A)
    if uniform(A)
        #B = A.numbers.*unit(getindex(A,1,1))
        B = A.numbers.*(unitrange(A)[1]/unitdomain(A)[1]) 
        return B
    else
        T2 = eltype(A.numbers)
        B = Matrix{Quantity{T2}}(undef,M,N)
        for m = 1:M
            for n = 1:N
                B[m,n] = getindex(A,m,n)
            end
        end
        return B
    end
end
function Matrix(A::T) where T<: DimMatrix

    M,N = size(A)
    T2 = eltype(parent(A))
    B = Matrix{Quantity{T2}}(undef,M,N)
    for m = 1:M
        for n = 1:N
            B[m,n] = Quantity.(getindex(A,m,n),unitrange(A)[m]./unitdomain(A)[n])
        end
    end
    return B
end

# """
#     function convert(AbstractMatrix,A::MultipliableMatrix)

#     Expand A into array form
#     Useful for tests, display
#     pp. 193, Hart
# """
# function convert(AbstractMatrix{T},A::AbstractMultipliableMatrix)  where T<: Number

#     M = rangelength(A)
#     N = domainlength(A)
#     T2 = eltype(A.numbers)
#     B = Matrix{Quantity{T2}}(undef,M,N)
#     for m = 1:M
#         for n = 1:N
#             B[m,n] = getindex(A,m,n)
#         end
#     end
#     return B
# end

"""
    function *(A::AbstractMultipliableMatrix,b)

    Matrix-vector multiplication with units/dimensions.
    Unitful also handles this case, but here there is added
    efficiency in the storage of units/dimensions by accounting
    for the necessary structure of the matrix. Check.
"""
function *(A::T,b::AbstractVector) where T<: AbstractMultipliableMatrix

    ur = unitrange(A)
    if dimension.(unitdomain(A)) == dimension.(b)
        #if unitdomain(A) ~ b
        # try column by column to reduce allocations
        return (A.numbers*ustrip.(b)).*unitrange(A) 

        #return Quantity.((A.numbers*ustrip.(b)),unitrange(A)) # slower
    elseif ~exact(A) && (unitdomain(A) ∥ b)
        #convert_unitdomain!(A,unit.(b)) # inefficient?

        shift = unit(b[1])/unitdomain(A)[1]
        return (A.numbers*ustrip.(b)).*(unitrange(A).*shift)
    else
        error("Dimensions of MultipliableMatrix and vector not compatible")
    end
end

"""
    function *(A::MultipliableMatrix,b)

    Matrix-scalar multiplication with units/dimensions.
    Must account for change in the unitrange when the
     scalar has units.
    Here, take product of dimension of the scalar and the unitrange.
    Alternatively, divide the domain by the dimension of the scalar. 
    Matrix-scalar multiplication is commutative.
    Result is `exact` if input matrix is exact and scalar is dimensionless. 
    Note: special matrix forms revert to a product that is a MultipliableMatrix.
"""
*(A::T1,b::T2) where T1 <: AbstractMultipliableMatrix where T2 <: Number = (exact(A) && dimensionless(b)) ?  BestMultipliableMatrix(A.numbers*ustrip(b),unitrange(A).*unit(b),unitdomain(A),exact = true) : BestMultipliableMatrix(A.numbers*ustrip(b),unitrange(A).*unit(b),unitdomain(A))
*(b::T2,A::T1) where T1 <: AbstractMultipliableMatrix where T2 <: Number = A*b

"""
    function *(A,B)

    Matrix-matrix multiplication with units/dimensions.
    A*B represents two successive transformations.
    Unitrange of B should equal domain of A in geometric interpretation.
    Unitrange of B should be parallel to unitdomain of A in algebraic interpretation.

    Note: special matrix forms revert to a product that is a MultipliableMatrix.
"""
function *(A::T1,B::T2) where T1<:AbstractMultipliableMatrix where T2<:AbstractMultipliableMatrix
    #if unitrange(B) ~ unitdomain(A) # should this be similarity()?
 
    bothexact = exact(A) && exact(B)
    if unitrange(B) == unitdomain(A) # should this be similarity()?
        return MMatrix(A.numbers*B.numbers,unitrange(A),unitdomain(B),exact=bothexact) 
    elseif unitrange(B) ∥ unitdomain(A) && ~bothexact
        #A2 = convert_unitdomain(A,unitrange(B)) 
        #convert_unitdomain!(A,unitrange(B))
        newrange = unitrange(A).*(unitrange(B)[1]/unitdomain(A)[1])

        return MMatrix(A.numbers*B.numbers,newrange,unitdomain(B),exact=bothexact)
    else
        error("matrix dimensional domain/unitrange not conformable")
    end
end

# special case: MultipliableMatrix * non-multipliable matrix
*(A::T1,B::T2) where T1<:AbstractMultipliableMatrix where T2<:AbstractMatrix = A*BestMultipliableMatrix(B)
*(A::T2,B::T1) where T1<:AbstractMultipliableMatrix where T2<:AbstractMatrix = BestMultipliableMatrix(A)*B

"""
    function +(A,B)

    Matrix-matrix addition with units/dimensions.
    A+B requires the two matrices to have dimensional similarity.
"""
function +(A::AbstractMultipliableMatrix{T1},B::AbstractMultipliableMatrix{T2}) where T1 where T2

    bothexact = exact(A) && exact(B)

    #if unitrange(A) ~ unitrange(B) && unitdomain(A) ~ unitdomain(B)
    if (unitrange(A) == unitrange(B) && unitdomain(A) == unitdomain(B)) ||
        ( unitrange(A) ∥ unitrange(B) && unitdomain(A) ∥ unitdomain(B) && ~bothexact)
        return MultipliableMatrix(A.numbers+B.numbers,unitrange(A),unitdomain(A),exact=bothexact) 
    else
        error("matrices not dimensionally conformable for addition")
    end
end

"""
    function -(A,B)

    Matrix-matrix subtraction with units/dimensions.
    A-B requires the two matrices to have dimensional similarity.
"""
function -(A::AbstractMultipliableMatrix{T1},B::AbstractMultipliableMatrix{T2}) where T1 where T2

    bothexact = exact(A) && exact(B)

    if (unitrange(A) == unitrange(B) && unitdomain(A) == unitdomain(B)) ||
       ( unitrange(A) ∥ unitrange(B) && unitdomain(A) ∥ unitdomain(B) && ~bothexact)
    #if unitrange(A) ~ unitrange(B) && unitdomain(A) ~ unitdomain(B)
    #if unitrange(A) == unitrange(B) && unitdomain(A) == unitdomain(B)
        return MMatrix(A.numbers-B.numbers,unitrange(A),unitdomain(A),exact=bothexact) 
    else
        error("matrices not dimensionally conformable for subtraction")
    end
end
-(A::AbstractMultipliableMatrix{T}) where T <: Number = 
MMatrix(-A.numbers,unitrange(A),unitdomain(A),exact=exact(A)) 

"""
    function lu(A::AbstractMultipliableMatrix{T})

    Extend `lu` factorization to AbstractMultipliableMatrix.
    Related to Gaussian elimination.
    Store dimensional domain and range in "factors" attribute
    even though this is not truly a MultipliableMatrix.
    Returns `LU` type in analogy with `lu` for unitless matrices.
    Based on LDU factorization, Hart, pp. 204.
"""
function lu(A::AbstractMultipliableMatrix{T}) where T <: Number
    F̂ = lu(A.numbers)
    factors = MMatrix(F̂.factors, unitrange(A), unitdomain(A), exact=exact(A))
    F = LU(factors,F̂.ipiv,F̂.info)
    return F
end

"""
    function getproperty(F::LU{T,<:AbstractMultipliableMatrix,Vector{Int64}}, d::Symbol) where T

    Extend LinearAlgebra.getproperty for AbstractMultipliableMatrix.

    LU factorization stores L and U together.
    Extract L and U while keeping consistent
    with dimensional domain and range.
"""
function getproperty(F::LU{T,<:AbstractMultipliableMatrix,Vector{Int64}}, d::Symbol) where T
    m, n = size(F)
    if d === :L
        mmatrix = getfield(F, :factors)
        numbers = getfield(mmatrix,:numbers)
        # add ustrip to get numerical values
        Lnum = tril!(numbers[1:m, 1:min(m,n)])
        for i = 1:min(m,n); Lnum[i,i] = one(T); end
        #L = EndomorphicMatrix(Lnum,unitrange(mmatrix),exact(mmatrix))
        L = BestMultipliableMatrix(Lnum,unitrange(mmatrix),unitrange(mmatrix),exact=exact(mmatrix))
        return L
    elseif d === :U
        mmatrix = getfield(F, :factors)
        numbers = getfield(mmatrix,:numbers)
        Unum = triu!(numbers[1:min(m,n), 1:n])
        U = BestMultipliableMatrix(Unum, unitrange(mmatrix), unitdomain(mmatrix), exact=exact(mmatrix))
        return U
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
 ~(a,b) = similarity(a,b)
#similarity(a,b) = isequal(dimension

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
            Δdim = dimension(a)./dimension(b)
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

"""
    function uniform(a)

    Is the dimension of this quantity uniform?

    There must be a way to inspect the Unitful type to answer this.
"""
uniform(a::T) where T <: Number = true # all scalars by default
function uniform(a::Vector) 
    dima = dimension.(a)
    for dd = 2:length(dima)
        if dima[dd] ≠ dima[1]
            return false
        end
    end
    return true
end
function uniform(A::Matrix)
    B = BestMultipliableMatrix(A)
    isnothing(B) ? false : uniform(B)
end
uniform(A::T) where T <: AbstractMultipliableMatrix = left_uniform(A) && right_uniform(A)
uniform(A::UniformMatrix) = true

"""
    function left_uniform(A)

    Definition: uniform unitrange of A
"""
left_uniform(A::Union{LeftUniformMatrix,UniformMatrix}) = true
left_uniform(A::T) where T<: AbstractMultipliableMatrix = uniform(unitrange(A)) ? true : false
function left_uniform(A::Matrix)
    B = BestMultipliableMatrix(A)
    isnothing(B) ? false : left_uniform(B)
end

"""
    function right_uniform(A)

    Does the unitdomain of A have uniform dimensions?
"""
right_uniform(A::Union{UniformMatrix,RightUniformMatrix}) = true
right_uniform(A::T) where T<:AbstractMultipliableMatrix = uniform(unitdomain(A)) ? true : false
function right_uniform(A::Matrix)
    B = BestMultipliableMatrix(A)
    isnothing(B) ? false : right_uniform(B)
end

"""
     function dimensionless(A)

     Not all dimensionless matrices have
     dimensionless domain and range.
"""
dimensionless(A::T) where T <: AbstractMultipliableMatrix = uniform(A) && unitrange(A)[1] == unitdomain(A)[1]
dimensionless(A::Matrix) = uniform(A) && dimension(A[1,1]) == NoDims
dimensionless(A::T) where T <: Number = (dimension(A) == NoDims)

square(A::T) where T <: AbstractMultipliableMatrix = (domainlength(A) == rangelength(A))
square(A::SquarableMatrix) = true
square(A::EndomorphicMatrix) = true

squarable(A::T) where T <: AbstractMultipliableMatrix = (unitdomain(A) ∥ unitrange(A))
squarable(A::SquarableMatrix) = true
squarable(A::EndomorphicMatrix) = true

unit_symmetric(A::AbstractMultipliableMatrix) = (unitrange(A) ∥ 1 ./unitdomain(A))
unit_symmetric(A::UnitSymmetricMatrix) = true

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
function convert_unitdomain(A::AbstractMultipliableMatrix, newdomain::Vector) 
    if unitdomain(A) ∥ newdomain
        #shift = newdomain./unitdomain(A)
        #newrange = unitrange(A).*shift
        newrange = unitrange(A).*(newdomain[1]/unitdomain(A)[1])

        B = BestMultipliableMatrix(A.numbers,newrange,newdomain,exact=true)
    else
        error("New unitdomain not parallel to unitdomain of Multipliable Matrix")
    end
end

"""
    function convert_unitdomain!(A, newdomain)

    In-place conversion of unit (dimensional) domain.
    Matrix Type not permitted to change.
"""
function convert_unitdomain!(A::AbstractMultipliableMatrix, newdomain::Vector)
    if unitdomain(A) ∥ newdomain
        shift = newdomain[1]./unitdomain(A)[1]
        # caution: not all matrices have this attribute
        if hasproperty(A,:unitdomain)
            for (ii,vv) in enumerate(A.unitdomain)
                A.unitdomain[ii] *= shift
            end
        end
        if hasproperty(A,:unitrange)
            for (ii,vv) in enumerate(A.unitrange)
                A.unitrange[ii] *= shift
            end
        end
        # make sure the matrix is now exact
        #A.exact = true # immutable so not possible
    else
        error("New domain not parallel to domain of Multipliable Matrix")
    end
end

"""
    function convert_unitrange(A, newrange)

    When using the geometric interpretation of matrices,
    it is useful to convert the dimensional range of the
    matrix to match the desired output of multiplication.
    Here we set the matrix to `exact=true` after this step.
    Permits MatrixType to change.
"""
function convert_unitrange(A::AbstractMultipliableMatrix, newrange::Vector)
    if unitrange(A) ∥ newrange
        #shift = newrange[1]./unitrange(A)[1]
        #newdomain = unitdomain(A).*shift
        newdomain = unitdomain(A).*(newrange[1]./unitrange(A)[1])
        B = BestMultipliableMatrix(A.numbers,newrange,newdomain,exact=true)
    else
        error("New unitrange not parallel to unitrange of Multipliable Matrix")
    end
end

"""
    function convert_unitrange!(A, newrange)

    In-place conversion of unit (dimensional) range.
    Matrix Type not permitted to change.
"""
function convert_unitrange!(A::AbstractMultipliableMatrix, newrange::Vector)
    if unitrange(A) ∥ newrange
        shift = newrange[1]./unitrange(A)[1]
        # caution: not all matrices have this attribute
        if hasproperty(A,:unitdomain)
            for (ii,vv) in enumerate(A.unitdomain)
                A.unitdomain[ii] *= shift
            end
        end
        if hasproperty(A,:unitrange)
            for (ii,vv) in enumerate(A.unitrange)
                A.unitrange[ii] *= shift
            end
        end
        #A.exact = true , immutable
     else
         error("New range not parallel to range of Multipliable Matrix")
     end
end

"""
    function exact(A)

-    `exact=true`: geometric interpretation of unitdomain and unitrange
-    `exact=false`: algebraic interpretation
"""
exact(A::T) where T <: AbstractMultipliableMatrix = A.exact
exact(A::DimMatrix) = A.exact
"""
    function rangelength(A::MultipliableMatrix)

    Numerical dimension (length or size) of unitrange
"""
#rangelength(A::T) where T <: AbstractMultipliableMatrix = length(unitrange(A))
rangelength(A::T) where T <: AbstractMultipliableMatrix = size(A)[1]

"""
    function domainlength(A::MultipliableMatrix)

    Numerical dimension (length or size) of unitdomain of A
"""
#domainlength(A::T) where T <: AbstractMultipliableMatrix = length(unitdomain(A))
domainlength(A::T) where T <: AbstractMultipliableMatrix = size(A)[2]

#size(A::AbstractMultipliableMatrix) = (rangelength(A), domainlength(A))
size(A::AbstractMultipliableMatrix) = size(A.numbers)

convert(::Type{AbstractMatrix{T}}, A::AbstractMultipliableMatrix) where {T<:Number} = convert(AbstractMultipliableMatrix{T}, A)
convert(::Type{AbstractArray{T}}, A::AbstractMultipliableMatrix) where {T<:Number} = convert(AbstractMultipliableMatrix{T}, A)
#convert(::Type{AbstractArray{T}}, S::AbstractToeplitz) where {T<:Number} = convert(AbstractToeplitz{T}, S)

unitdomain(A::T) where T <: AbstractMultipliableMatrix = A.unitdomain
unitdomain(A::SquarableMatrix) = A.unitrange.*A.Δunitdomain
#unitdomain(A::UnitSymmetricMatrix) = unit.(1 ./ A.unitrange).*A.Δunitdomain
unitdomain(A::UnitSymmetricMatrix) =  A.Δunitdomain./A.unitrange
unitdomain(A::EndomorphicMatrix) = A.unitrange # unitdomain not saved and must be reconstructed
unitdomain(A::Union{UniformMatrix,RightUniformMatrix}) = fill(A.unitdomain[1],size(A.numbers)[2])
unitdomain(A::DimMatrix{T,2}) where T <: Number = dims(A)[2]

unitrange(A::T) where T <: AbstractMultipliableMatrix = A.unitrange
unitrange(A::Union{UniformMatrix,LeftUniformMatrix}) = fill(A.unitrange[1],size(A.numbers)[1])
unitrange(A::DimMatrix{T,2}) where T <: Number = dims(A)[1]

"""
    function transpose

    Defined by condition `A[i,j] = transpose(A)[j,i]`.
    Not analogous to function for dimensionless matrices.

    Hart, pp. 205.
"""
transpose(A::AbstractMultipliableMatrix) = BestMultipliableMatrix(transpose(A.numbers),unitdomain(A).^-1, unitrange(A).^-1,exact=exact(A)) 
transpose(A::EndomorphicMatrix{T}) where T = EndomorphicMatrix(transpose(A.numbers),unitrange(A).^-1, exact(A)) 
transpose(A::UniformMatrix) = UniformMatrix(transpose(A.numbers),
                                            Base.convert(Vector{Unitful.FreeUnits},[unitdomain(A)[1]^-1]),
                                            Base.convert(Vector{Unitful.FreeUnits},[unitrange(A)[1]^-1]),
                                            exact(A)) 

"""
    function identitymatrix(dimrange)

    Input: dimensional (unit) range.
    `A + I` only defined when `endomorphic(A)=true`
    When accounting for units, there are many identity matrices.
    This function returns a particular identity matrix
    defined by its dimensional range.
    Hart, pp. 200.                             
"""
identitymatrix(dimrange) = EndomorphicMatrix(I(length(dimrange)),dimrange;exact=false)

"""
     function inv

     Inverse of Multipliable Matrix.
     Only defined for nonsingular matrices.
     Inverse reverses mapping from unitdomain to range.
     Is `exact` if input is exact.

    Hart, pp. 205. 
"""
inv(A::T) where T <: AbstractMultipliableMatrix = ~singular(A) ? BestMultipliableMatrix(inv(A.numbers),unitdomain(A),unitrange(A),exact=exact(A)) : error("matrix is singular")

"""
     function left divide

     Left divide of Multipliable Matrix.
     Reverse mapping from unitdomain to range.
     Is `exact` if input is exact.
"""
function (\)(A::AbstractMultipliableMatrix,b::AbstractVector)
    # unit.(range(b)) == range(A) ?  BestMultipliableMatrix(A.numbers\ustrip.(b),unitdomain(A),range(A),exact(A)) : error("matrix and vector units don't match")
    if dimension(unitrange(A)) == dimension(b)
    #if unitrange(A) ~ b
        return (A.numbers\ustrip.(b)).*unitdomain(A)
    elseif ~exact(A) && (unitrange(A) ∥ b)
        Anew = convert_unitrange(A,unit.(b)) # inefficient?
        return (Anew.numbers\ustrip.(b)).*unitdomain(Anew)
    else
        error("UnitfulLinearAlgebra.mldivide: Dimensions of MultipliableMatrix and vector not compatible")
    end
end

function (\)(A::AbstractMultipliableMatrix,B::AbstractMultipliableMatrix)
    if unitrange(A) == unitrange(B)
        #return (A.numbers\B.numbers).*unitdomain(A)
        return BestMultipliableMatrix(A.numbers\B.numbers,unitdomain(A),unitdomain(B),exact = (exact(A) && exact(B)))
    elseif ~exact(A) && (unitrange(A) ∥ unitrange(B))
        convert_unitrange!(A,unitrange(B)) # inefficient?
        return BestMultipliableMatrix(A.numbers\B.numbers,unitdomain(A),unitdomain(B),exact = (exact(A)&&exact(B)))
    else
        error("UnitfulLinearAlgebra.mldivide: Dimensions of Multipliable Matrices A and B not compatible")
    end
end

#function (\)(F::LU{T,AbstractMultipliableMatrix{T},Vector{Int64}}, B::AbstractVector) where T<:Number
"""
    function ldiv(F::LU{T,MultipliableMatrix{T},Vector{Int64}}, B::AbstractVector) where T<:Number

    Perform matrix left divide on LU factorization object,
    where LU object contains unit information.
    Doesn't require LeftUniformMatrix. 
"""
function (\)(F::LU{T,<: AbstractMultipliableMatrix{T},Vector{Int64}}, B::AbstractVector) where T<:Number

    # UnitfulLinearAlgebra: F - > F.factors
    LinearAlgebra.require_one_based_indexing(B)
    m, n = size(F)

    # UnitfulLinearAlgebra: check units
    if dimension(unitrange(F.factors)) == dimension(B)
         # pass without any issues
    elseif dimension(unitrange(F.factors)) ∥ dimension(b)
        # convert_range of F.factors
        # is allocating, will affect performance
        convert_unitrange!(F.factors,unit.(b))
    else
        error("units of F, B, are not conformable")
    end
    
    if m != size(B, 1)
        throw(DimensionMismatch("arguments must have the same number of rows"))
    end

    TFB = typeof(oneunit(eltype(ustrip.(B))) / oneunit(eltype(F.factors.numbers)))

    # does this line incur a lot of allocations?
    # it demotes Unitful LU struct to Numeric LU struct
    FF = LinearAlgebra.Factorization{TFB}(LU(F.factors.numbers,F.ipiv,F.info))

    # For wide problem we (often) compute a minimum norm solution. The solution
    # is larger than the right hand side so we use size(F, 2).
    BB = LinearAlgebra._zeros(TFB, B, n)

    if n > size(B, 1)
        # Underdetermined
        # Does "ustrip" cause performance issues?
        LinearAlgebra.copyto!(view(BB, 1:m, :), ustrip.(B))
    else
        LinearAlgebra.copyto!(BB, ustrip.(B))
    end

    LinearAlgebra.ldiv!(FF, BB)

    # For tall problems, we compute a least squares solution so only part
    # of the rhs should be returned from \ while ldiv! uses (and returns)
    # the complete rhs
    # UnitfulLinearAlgebra: add units
    return LinearAlgebra._cut_B(BB, 1:n).*unitdomain(F.factors)
end

"""
     function ldiv!

     In-place left division by a Multipliable Matrix.
     Reverse mapping from unitdomain to range.
     Is `exact` if input is exact.

    Problem: b changes type unless endomorphic
"""
function ldiv!(A::AbstractMultipliableMatrix,b::AbstractVector)
    ~endomorphic(A) && error("A not endomorphic, b changes type, ldiv! not available")
    
    if dimension(unitrange(A)) == dimension(b)
        #if unitrange(A) ~ b

        # seems to go against the point
        #b = copy((A.numbers\ustrip.(b)).*unitdomain(A))
        btmp = (A.numbers\ustrip.(b)).*unitdomain(A)
        for bb = 1:length(btmp)
            b[bb] = btmp[bb]
        end
        
    elseif ~exact(A) && (unitrange(A) ∥ b)
        Anew = convert_unitrange(A,unit.(b)) # inefficient?
        btmp = (Anew.numbers\ustrip.(b)).*unitdomain(Anew)
        for bb = 1:length(btmp)
            b[bb] = btmp[bb]
        end

    else
        error("UnitfulLinearAlgebra.ldiv!: Dimensions of MultipliableMatrix and vector not compatible")
    end
    
end

"""
    function det
"""
function det(A::T) where T<: AbstractMultipliableMatrix

    if square(A)
        # detunit = Vector{eltype(domain(A))}(undef,domainlength(A))
        # for i = 1:domainlength(A)
        # end
        detunit = prod([unitrange(A)[i]/unitdomain(A)[i] for i = 1:domainlength(A)])

        return Quantity(det(A.numbers),detunit)
    else
        error("Determinant requires square matrix")
    end
end

singular(A::T) where T <: AbstractMultipliableMatrix = iszero(ustrip(det(A)))

trace(A::T) where T<: AbstractMultipliableMatrix = sum(diag(A.numbers)).*(unitrange(A)[1]/unitdomain(A)[1])

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
function eigen(A::T;permute::Bool=true, scale::Bool=true, sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) where T <: AbstractMultipliableMatrix

    if squarable(A) 
        F = LinearAlgebra.eigen(A.numbers, permute=permute, scale=scale, sortby=sortby)
        return Eigen(F.values.*(unitrange(A)[1]/unitdomain(A)[1]), BestMultipliableMatrix(F.vectors,unitdomain(A),fill(unit(1.0),size(A,2))))
    else
        error("UnitfulLinearAlgebra: Eigenvalue decomposition doesn't exist for for non-squarable matrices")
    end
end

"""
   Extend `isposdef` for Eigen factorizations of `MultipliableMatrix`s.
    Should the units be stripped out of the function?
    Only defined for matrices with uniform units (pp. 101, Hart, 1995). 
"""
isposdef(A::Eigen{T,V,S,U}) where {U<: AbstractVector, S<:AbstractMultipliableMatrix, V, T <: Number} = (uniform(A.vectors) && isreal(A.values)) && all(x -> x > 0, ustrip.(A.values))

"""
   Extend `inv` for Eigen factorizations of `MultipliableMatrix`s.
    Only defined for matrices with uniform units (pp. 101, Hart, 1995). 
"""
function inv(A::Eigen{T,V,S,U}) where {U <: AbstractVector, S <: AbstractMultipliableMatrix, V, T <: Number}

    if (uniform(A.vectors) && isreal(A.values))
        ur = unitrange(A.vectors)
        ud = unit.(A.values)
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
#function svd(A::AbstractMultipliableMatrix;full=false) where T <: AbstractMultipliableMatrix
function svd(A::AbstractMultipliableMatrix;full=false,alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(A.numbers)) 
    if uniform(A) 
        F = svd(A.numbers, full=full, alg=alg)
        # U,V just regular matrices: return that way?
        # They are also Uniform and Endomorphic
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
struct DSVD{T,Tr,MU<:AbstractMultipliableMatrix{T},MV<:AbstractMultipliableMatrix{T},MQY<:AbstractMultipliableMatrix{T},MQX<:AbstractMultipliableMatrix{T},C<:AbstractVector{Tr}} <: Factorization{T}
    U′::MU
    S::C
    V′⁻¹::MV
    Qy::MQY
    Qx::MQX
    function DSVD{T,Tr,MU,MV,MQY,MQX,C}(U′, S, V′⁻¹,Qy,Qx) where {T,Tr,MU<:AbstractMultipliableMatrix{T},MV<:AbstractMultipliableMatrix{T},MQY<:AbstractMultipliableMatrix{T},MQX<:AbstractMultipliableMatrix{T},C<:AbstractVector{Tr}}
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
function dsvd(A::AbstractMultipliableMatrix,Py::AbstractMultipliableMatrix,Px::AbstractMultipliableMatrix;full=false,alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(A.numbers)) 

    unit_symmetric(Py) ? Qy = getproperty(cholesky(Py),:U) : error("norm matrix for range not unit symmetric")
    unit_symmetric(Px) ? Qx = getproperty(cholesky(Px),:U) : error("norm matrix for domain not unit symmetric")

    # must be more efficient way
    #A′ = Qr*(A*inv(Qd))
    # still inefficient with copy
    A′ =   copy(transpose(transpose(Qx)\transpose(Qy*A)))

    ~dimensionless(A′) && error("A′ should be dimensionless to implement `LinearAlgebra.svd`")

    F = svd(A′.numbers, full=full, alg=alg)

    println(typeof(MMatrix(F.U)))
    return DSVD(MMatrix(F.U),F.S,MMatrix(F.Vt),Qy,Qx)
end

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::DSVD{<:Any,<:Any,<:AbstractArray,<:AbstractArray,<:AbstractArray,<:AbstractArray,<:AbstractVector})
    summary(io, F); println(io)
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
    # from `svd.jl`
    #@views (F.S[1:k] .\ F.Vt[1:k, :])' * F.U[:,1:k]'

    # a less efficient matrix way to do it.
#    Σ⁻¹ = Diagonal(F.S[1:k].^-1,fill(unit(1.0),k),fill(unit(1.0),k))
    Σ⁻¹ = Diagonal(F.S[1:k].^-1,unitdomain(F.V[:,1:k]),unitrange(F.U⁻¹[1:k,:]))
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
diagm(v::AbstractVector,r::AbstractVector,d::AbstractVector; exact = false) = BestMultipliableMatrix(spdiagm(length(r),length(d),ustrip.(v)),r,d; exact=exact)    
#Diagonal(v::AbstractVector,r::Unitful.Unitlike,d::Unitful.Unitlike; exact = false) = MultipliableMatrix(Diagonal(ustrip.(v)),r,d ; exact=exact)    
#end

"""
    function diag(A::AbstractMultipliableMatrix)

    Diagonal elements of matrix with units.

    Usual `LinearAlgebra.diag` function is not working due to different type elements on diagonal
 """
function diag(A::AbstractMultipliableMatrix{T}) where T <: Number

    m,n = size(A)
    ndiag = max(m,n)
    dimensionless(A) ? vdiag = Vector{T}(undef,ndiag) : vdiag = Vector{Quantity}(undef,ndiag)
    for nd in 1:ndiag
        vdiag[nd] = getindex(A,nd,nd)
    end
    return vdiag
end

"""
    function cholesky(A::AbstractMultipliableMatrix)

    Cholesky decomposition extended for matrices with units.
    Requires unit (or dimensionally) symmetric matrix.
    Functions available for LinearAlgebra.Cholesky objects: `size`, `\`, `inv`, `det`, `logdet` and `isposdef`.
    Functions available for UnitfulLinearAlgebra.Cholesky objects: `size`, `det`, and `isposdef`.
"""
function cholesky(A::AbstractMultipliableMatrix)
    if unit_symmetric(A)
        C = LinearAlgebra.cholesky(A.numbers)
        factors = BestMultipliableMatrix(C.factors,unitdomain(A)./unitdomain(A),unitdomain(A),exact=exact(A))

        return Cholesky(factors,C.uplo,C.info)
    else
        error("requires unit symmetric matrix")
    end
    
end

function getproperty(C::Cholesky{T,<:AbstractMultipliableMatrix}, d::Symbol) where T 
    Cfactors = getfield(C, :factors)
    Cuplo    = getfield(C, :uplo)
    if d === :U
        numbers = UpperTriangular(Cuplo === LinearAlgebra.char_uplo(d) ? Cfactors.numbers : copy(Cfactors.numbers'))
        return BestMultipliableMatrix(numbers,unitrange(Cfactors),unitdomain(Cfactors),exact = Cfactors.exact)
    elseif d === :L
        numbers = LowerTriangular(Cuplo === LinearAlgebra.char_uplo(d) ? Cfactors.numbers : copy(Cfactors.numbers'))
        # use transpose to get units right
        return BestMultipliableMatrix(numbers,unitdomain(Cfactors).^-1,unitrange(Cfactors).^-1,exact = Cfactors.exact)
    elseif d === :UL
        return (Cuplo === 'U' ?        BestMultipliableMatrix(UpperTriangular(Cfactors.numbers),unitrange(Cfactors),unitdomain(Cfactors),exact = Cfactors.exact) : BestMultipliableMatrix(LowerTriangular(Cfactors.numbers),unitdomain(Cfactors).^-1,unitrange(Cfactors).^-1,exact = Cfactors.exact))
    else
        #println("caution: fallback not tested")
        return getfield(C, d)
    end
end

"""
    function Diagonal(v::AbstractVector,r::Unitful.Unitlike,d::Unitful.Unitlike; exact = false)

    Construct diagonal matrix with units where the diagonal has elements `v`.
    If `v` has units, check that they conform with dimensional unit range `r`
     and dimensional unit domain `d`.
    Like `LinearAlgebra.Diagonal`, this extension is restricted to square matrices.
"""
Diagonal(v::AbstractVector,r::AbstractVector,d::AbstractVector; exact = false) = ((length(r) == length(d)) && (length(v) == length(d))) ? BestMultipliableMatrix(LinearAlgebra.Diagonal(ustrip.(v)),r,d; exact=exact) : error("unit range and domain do not define a square matrix")   

"""
    function vcat(A,B)

    Modeled after function `VERTICAL` (pp. 203, Hart, 1995).
"""
function vcat(A::AbstractMultipliableMatrix,B::AbstractMultipliableMatrix)

    numbers = vcat(A.numbers,B.numbers)
    shift = unitdomain(A)[1]./unitdomain(B)[1]
    ur = vcat(unitrange(A),unitrange(B).*shift)
    bothexact = (exact(A) && exact(B))
    return BestMultipliableMatrix(numbers,ur,unitdomain(A),exact=bothexact)
end

"""
    function hcat(A,B)

    Modeled after function `HORIZONTAL` (pp. 202, Hart, 1995).
"""
function hcat(A::AbstractMultipliableMatrix,B::AbstractMultipliableMatrix)

    numbers = hcat(A.numbers,B.numbers)
    shift = unitrange(A)[1]./unitrange(B)[1]
    ud = vcat(unitdomain(A),unitdomain(B).*shift)
    bothexact = (exact(A) && exact(B))
    return BestMultipliableMatrix(numbers,unitrange(A),ud,exact=bothexact)
end

end
