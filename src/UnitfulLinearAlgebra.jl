module UnitfulLinearAlgebra

using Unitful, LinearAlgebra, SparseArrays

export AbstractMultipliableMatrix
export MultipliableMatrix, EndomorphicMatrix
export SquarableMatrix, UniformMatrix
export LeftUniformMatrix, RightUniformMatrix
export UnitSymmetricMatrix
export BestMultipliableMatrix
export similarity, ∥, parallel
export uniform, left_uniform, right_uniform
export invdimension, dottable
export getindex, setindex!, size, similar
export convert_unitrange, convert_unitdomain
export convert_unitrange!, convert_unitdomain!
export exact, multipliable, dimensionless, endomorphic
export svd, eigen, inv, transpose
export unitrange, unitdomain
export square, squarable, singular, unit_symmetric
export lu, det, trace, diag, diagm
export Diagonal, (\), cholesky
export identitymatrix

import LinearAlgebra: inv, det, lu,
    svd, getproperty, eigen,
    diag, diagm, Diagonal, cholesky
import Base:(~), (*), (+), (-), (\), getindex, setindex!,
    size, range, transpose, similar

abstract type AbstractMultipliableMatrix{T<:Number} <: AbstractMatrix{T} end

"""
    struct MultipliableMatrix

    Matrices with units that a physically reasonable,
    i.e., more than just an array of values with units.

    Multipliable matrices have dimensions that are consistent with many linear algebraic manipulations, including multiplication.

    Hart suggests that these matrices simply be called "matrices", and that matrices with dimensional values that cannot be multiplied should be called "arrays."

# Attributes
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

    An endomorphic matrix maps a dimensioned vector space
    to itself. The dimensional range and domain are the same.

# Attributes
- `numbers`: numerical (dimensionless) matrix
- `unitrange`: dimensional range in terms of units, and also equal the dimensional domain
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
    Definition: dimensional range and domain are parallel.

# Attributes
- `numbers`: numerical (dimensionless) matrix
- `unitrange`: dimensional range in terms of units, this is also the domain
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

    An squarable matrix is one where 𝐀² is defined.
    Definition: inverse dimensional range and dimensional domain are parallel.
    Called "dimensionally symmetric" by Hart.

# Attributes
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

    Uniform matrix

# Attributes
- `numbers`: numerical (dimensionless) matrix
- `unitrange`:  uniform dimensional range expressed as a single unit
- `unitdomain`: uniform dimensional domain expressed as a single unit
- `exact`: geometric (`true`) or algebraic (`false`) interpretation
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

    Left uniform matrix

# Attributes
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

    Right uniform matrix

# Attributes
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
- `Quantity`: numerical value and units
"""
getindex(A::T,i::Int,j::Int) where T <: AbstractMultipliableMatrix = Quantity(A.numbers[i,j],unitrange(A)[i]./unitdomain(A)[j]) 

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

        shift = ustrip(b[1])/unitdomain(A)[1]
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
        return BestMultipliableMatrix(A.numbers*B.numbers,unitrange(A),unitdomain(B),exact=bothexact) 
    elseif unitrange(B) ∥ unitdomain(A) && ~bothexact
        #A2 = convert_unitdomain(A,unitrange(B)) 
        convert_unitdomain!(A,unitrange(B)) 
        return BestMultipliableMatrix(A.numbers*B.numbers,unitrange(A),unitdomain(B),exact=bothexact)
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
        return MultipliableMatrix(A.numbers-B.numbers,unitrange(A),unitdomain(A),exact=bothexact) 
    else
        error("matrices not dimensionally conformable for subtraction")
    end
end

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
    factors = BestMultipliableMatrix(F̂.factors,unitrange(A),unitdomain(A),exact=exact(A))
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
        shift = newdomain./unitdomain(A)
        newrange = unitrange(A).*shift
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
        #A.exact = true # immutable
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
        shift = newrange[1]./unitrange(A)[1]
        newdomain = unitdomain(A).*shift
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

unitrange(A::T) where T <: AbstractMultipliableMatrix = A.unitrange
unitrange(A::Union{UniformMatrix,LeftUniformMatrix}) = fill(A.unitrange[1],size(A.numbers)[1])

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
# transpose(A::AbstractMultipliableMatrix) = MultipliableMatrix(transpose(A.numbers),unit.(1 ./unitdomain(A)), unit.(1 ./unitrange(A)),exact(A)) 
# transpose(A::EndomorphicMatrix{T}) where T = EndomorphicMatrix(transpose(A.numbers),unit.(1 ./unitrange(A)), exact(A)) 
# transpose(A::UniformMatrix) = UniformMatrix(transpose(A.numbers),unit.(1 ./unitdomain(A)[1]), unit.(1 ./unitrange(A)[1]), exact(A)) 

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
    elseif ~exact(A) && (unitrange(A) ∥ B)
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

trace(A::T) where T<: AbstractMultipliableMatrix = sum(A.numbers).*(unitrange(A)[1]/unitdomain(A)[1])

"""
    function eigen(A::T;permute::Bool=true, scale::Bool=true, sortby::Union{Function,Nothing}=eigsortby) where T <: AbstractMultipliableMatrix

    Thin wrapper for `UnitfulLinearAlgebra.eigen`.
    Keep same keyword arguments as `LinearAlgebra.eigen`.
    Ideally would simply work using AbstractArray interface,
    but there is an unsolved issue with Unitful conversions. 
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
    svd(A; full::Bool = false, alg::Algorithm = default_svd_alg(A)) -> SVD

Compute the singular value decomposition (SVD) of `A` and return an `SVD` object. Extended for MultipliableMatrix input.
"""
#function svd(A::AbstractMultipliableMatrix;full=false) where T <: AbstractMultipliableMatrix
function svd(A::AbstractMultipliableMatrix;full=false,alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(A.numbers)) 
    if uniform(A) 
        F = svd(A.numbers, full=full, alg=alg)
        # U,V just regular matrices: return that way?
        # They are also Uniform and Endomorphic
        return SVD(F.U,F.S * unitrange(A)[1]./unitdomain(A)[1],F.Vt)
    else
        error("SVD doesn't exist for non-uniform matrices")
    end
end

"""
    function svd(A::AbstractMultipliableMatrix,Pdomain::UnitSymmetricMatrix,Prange::UnitSymmetricMatrix;full=false,alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(A.numbers)) 

    Singular value decomposition for matrices with non-uniform units.
# Input
- `A::AbstractMultipliableMatrix`
- `Prange::UnitSymmetricMatrix`: square matrix defining norm of range
- `Pdomain::UnitSymmetricMatrix`: square matrix defining norm of domain
- `full=false`: optional argument
- `alg`: optional argument for algorithm
# Output:
- `F::SVD`: SVD object with units that can be deconstructed
"""
function svd(A::AbstractMultipliableMatrix,Pr::AbstractMultipliableMatrix,Pd::AbstractMultipliableMatrix;full=false,alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(A.numbers)) 

    unit_symmetric(Pr) ? Qr = getproperty(cholesky(Pr),:U) : error("norm matrix for range not unit symmetric")
    unit_symmetric(Pd) ? Qd = getproperty(cholesky(Pd),:U) : error("norm matrix for domain not unit symmetric")

    # must be more efficient way
    A′ = Qr*(A*inv(Qd))

    ~dimensionless(A′) && error("A′ should be dimensionless to implement `LinearAlgebra.svd`")

    F′ = svd(A′.numbers, full=full, alg=alg)

    # Issue: SVD structure calls for Vt, but it is really V⁻¹

    # Issue: SVD wants first and last matrices to have the same type.
    #return SVD( Qr\BestMultipliableMatrix(F′.U),F′.S,F′.Vt*Qd)

    U = Qr\BestMultipliableMatrix(F′.U)
    Û = MultipliableMatrix(U.numbers,unitrange(U),unitdomain(U),exact(U))

    Vt = F′.Vt*Qd
    V̂t = MultipliableMatrix(Vt.numbers,unitrange(Vt),unitdomain(Vt),exact(Vt))
    return SVD(Û,F′.S,V̂t)
    
end

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
function diag(A::AbstractMultipliableMatrix)

    m,n = size(A)
    ndiag = max(m,n)
    vdiag = Vector{Quantity}(undef,ndiag)
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
Diagonal(v::AbstractVector,r::AbstractVector,d::AbstractVector; exact = false) = (length(r) == length(d)) ? BestMultipliableMatrix(LinearAlgebra.Diagonal(ustrip.(v)),r,d; exact=exact) : error("unit range and domain do not define a square matrix")   


end
