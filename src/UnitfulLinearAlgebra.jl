module UnitfulLinearAlgebra

using Unitful, LinearAlgebra

export MultipliableMatrix, EndomorphicMatrix
export SquarableMatrix, UniformMatrix
export similar, ∥, parallel
export uniform, left_uniform, right_uniform
export invdimension, dottable
export getindex, array
export convert_range, convert_domain
#export convert_range!, convert_domain!
export exact, multipliable, dimensionless, endomorphic
export svd, inv 
export range, domain
export square, squarable, singular
export lu, det

import LinearAlgebra:inv, det, lu, svd
import Base:(~), (*), getindex
import Base.similar
import Base.range

abstract type MultipliableMatrices{T<:Number} <: AbstractMatrix{T} end

#abstract type AbstractToeplitz{T<:Number} <: AbstractMatrix{T} end

"""
    struct MultipliableMatrix

    Matrices with units that a physically reasonable,
    i.e., more than just an array of values with units.

    Multipliable matrices have dimensions that are consistent with many linear algebraic manipulations, including multiplication.

    Hart suggests that these matrices simply be called "matrices", and that matrices with dimensional values that cannot be multiplied should be called "arrays."

# Attributes
- `numbers`: numerical (dimensionless) matrix
- `range`: dimensional range in terms of units
- `domain`: dimensional domain in terms of units
- `exact`: geometric (`true`) or algebraic (`false`) interpretation
"""
struct MultipliableMatrix{T <:Number} <: MultipliableMatrices{T}
    numbers::Matrix{T}
    range::Vector
    domain::Vector
    exact::Bool
end

"""
    struct EndomorphicMatrix

    An endomorphic matrix maps a dimensioned vector space
    to itself. The dimensional range and domain are the same.

# Attributes
- `numbers`: numerical (dimensionless) matrix
- `range`: dimensional range in terms of units, and also equal the dimensional domain
- `exact`: geometric (`true`) or algebraic (`false`) interpretation
"""
struct EndomorphicMatrix{T<:Number} <: MultipliableMatrices{T} 
    numbers::Matrix{T}
    range::Vector
    exact::Bool
end

"""
    struct SquarableMatrix

    An squarable matrix is one where 𝐀² is defined.
    It is the case if the dimensional range and domain are parallel.

# Attributes
- `numbers`: numerical (dimensionless) matrix
- `range`: dimensional range in terms of units, this is also the domain
- `domainshift`: shift to range that gives the domain
- `exact`: geometric (`true`) or algebraic (`false`) interpretation
"""
struct SquarableMatrix{T<:Number} <: MultipliableMatrices{T} 
    numbers::Matrix{T}
    range::Vector
    domainshift
    exact::Bool
end

"""
    struct UniformMatrix

    Uniform matrix

# Attributes
- `numbers`: numerical (dimensionless) matrix
- `range`:  uniform dimensional range expressed as a single unit
- `domain`: uniform dimensional domain expressed as a single unit
- `exact`: geometric (`true`) or algebraic (`false`) interpretation
"""
struct UniformMatrix{T<:Number} <: MultipliableMatrices{T}
    numbers::Matrix{T}
    range
    domain
    exact::Bool
end
# struct UniformMatrix{T,R,D} <: MultipliableMatrices where {T <: Number} where {R,D <: Unitful.Unitlike}
#     numbers::Matrix{T}
#     range::R
#     domain::D
#     exact::Bool
# end

"""
    MultipliableMatrix(numbers,range,domain;exact=false)

    Constructor where `exact` is a keyword argument. One may construct a MultipliableMatrix without specifying exact, in which case it defaults to `false`. 
"""
MultipliableMatrix(numbers,range,domain;exact=false) =
    MultipliableMatrix(numbers,range,domain,exact)

"""
     MultipliableMatrix(A::Matrix)

    Transform array to MultipliableMatrix.
    Finds best representation amongst
    UniformMatrix, EndomorphicMatrix, or MultipliableMatrix.
"""
function MultipliableMatrix(A::Matrix)

    numbers = ustrip.(A)
    M,N = size(numbers)
    #U = typeof(unit(A[1,1]))
    #U = eltype(unit.(A))
    # domain = Vector{U}(undef,N)
    # range = Vector{U}(undef,M)
    domain = Vector{Unitful.FreeUnits}(undef,N)
    range = Vector{Unitful.FreeUnits}(undef,M)

    for i = 1:M
        range[i] = unit(A[i,1])
    end
    
    for j = 1:N
        domain[j] = unit(A[1,1])/unit(A[1,j])
    end

    #println(range)
    #println(domain)
    
    # what kind of matrix is the best representation?
    if uniform(range) && uniform(domain)
        B = UniformMatrix(numbers,range[1],domain[1])
    elseif range == domain
        B = EndomorphicMatrix(numbers,range)
    else
        B = MultipliableMatrix(numbers,range,domain)
    end
    
    # if the array is not multipliable, return nothing
    if Matrix(B) == A
        return B
    else
        return nothing
    end
end

"""
    function multipliable(A)::Bool

    Is an array multipliable?
    It requires a particular structure of the units/dimensions in the array. 
"""
multipliable(A::Matrix) = ~isnothing(MultipliableMatrix(A))
multipliable(A::T) where T <: MultipliableMatrices = true

"""
    function EndomorphicMatrix

    Constructor with keyword argument `exact=false`.
    If `exact` not specified, defaults to `false`.
"""
EndomorphicMatrix(numbers,range;exact=false) =
    EndomorphicMatrix(numbers,range,exact)

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
    
    range = Vector{Unitful.FreeUnits}(undef,M)
    for i = 1:M
        range[i] = unit(A[i,1])
    end
    B = EndomorphicMatrix(numbers,range)
    
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

        range = Vector{Unitful.FreeUnits}(undef,1)
        range[1] = unit(A)
        return EndomorphicMatrix(numbers,range)
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
endomorphic(A::T) where T <: MultipliableMatrices = isequal(domain(A),range(A))
endomorphic(A::T) where T <: Number = dimensionless(A) # scalars must be dimensionless to be endomorphic

"""
    function UniformMatrix

    Constructor with keyword argument `exact=false`.
    If `exact` not specified, defaults to `false`.
"""
UniformMatrix(numbers,range,domain;exact=false) =
    UniformMatrix(numbers,range,domain,exact)

"""
    function getindex(A::MultipliableMatrix,i::Integer,j::Integer)

    Recover element (i,j) of a MultipliableMatrix.
    Part of the AbstractArray interface.
#Input
- `A::MultipliableMatrix`
- `i::Integer`: row index
- `j::Integer`: column index
#Output
- `Quantity`: numerical value and units
"""
getindex(A::T,i::Integer,j::Integer) where T <: MultipliableMatrices = Quantity(A.numbers[i,j],range(A)[i]./domain(A)[j]) 

"""
    function Matrix(A::MultipliableMatrix)

    Expand A into array form
    Useful for tests, display
    pp. 193, Hart
"""
function Matrix(A::T) where T<: MultipliableMatrices

    M = rangelength(A)
    N = domainlength(A)
    T2 = eltype(A.numbers)
    B = Matrix{Quantity{T2}}(undef,M,N)
    for m = 1:M
        for n = 1:N
            B[m,n] = getindex(A,m,n)
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
# function convert(AbstractMatrix{T},A::MultipliableMatrices)  where T<: Number

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
    function *(A::MultipliableMatrix,b)

    Matrix-vector multiplication with units/dimensions.
    Unitful also handles this case, but here there is added
    efficiency in the storage of units/dimensions by accounting
    for the necessary structure of the matrix.
"""
function *(A::T,b::Vector) where T<: MultipliableMatrices

    if dimension(domain(A)) == dimension(b)
    #if domain(A) ~ b
        return (A.numbers*ustrip.(b)).*A.range
    elseif ~exact(A) && (domain(A) ∥ b)
        Anew = convert_domain(A,b) # inefficient?
        return (Anew.numbers*ustrip.(b)).*Anew.range
    else
        error("Dimensions of MultipliableMatrix and vector not compatible")
    end
end

"""
    function *(A::MultipliableMatrix,b)

    Matrix-scalar multiplication with units/dimensions.
    Must account for change in the range when the
     scalar has units.
    Here, take product of dimension of the scalar and the range.
    Alternatively, divide the domain by the dimension of the scalar. 
    Matrix-scalar multiplication is commutative.
    Result is `exact` if input matrix is exact and scalar is dimensionless. 
    Note: special matrix forms revert to a product that is a MultipliableMatrix.
"""
*(A::T1,b::T2) where T1 <: MultipliableMatrices where T2 <: Number = (exact(A) && dimensionless(b)) ?  MultipliableMatrix(A.numbers*ustrip(b),range(A).*unit(b),domain(A),exact = true) : MultipliableMatrix(A.numbers*ustrip(b),range(A).*unit(b),domain(A))
*(b::T2,A::T1) where T1 <: MultipliableMatrices where T2 <: Number = A*b

"""
    function *(A,B)

    Matrix-matrix multiplication with units/dimensions.
    A*B represents two successive transformations.
    Range of B should equal domain of A in geometric interpretation.
    Range of B should be parallel to domain of A in algebraic interpretation.

    Note: special matrix forms revert to a product that is a MultipliableMatrix.
"""
function *(A::T1,B::T2) where T1<:MultipliableMatrices where T2<:MultipliableMatrices
    #if range(B) ~ domain(A) # should this be similar()?

    if range(B) == domain(A) # should this be similar()?
        exactproduct = exact(A) && exact(B)
        return MultipliableMatrix(A.numbers*B.numbers,range(A),domain(B),exact=exactproduct) 
    elseif range(B) ∥ domain(A)
        return MultipliableMatrix(A.numbers*B.numbers,range(A),domain(B)./range(A))
    else
        error("matrix domain/range not conformable")
    end
end

#function lu(A::T) where T <: MultipliableMatrices
"""
    function lu(A::MultipliableMatrix{Float64})
"""
function lu(A::T) where T <: MultipliableMatrices

    F̂ = lu(A.numbers)

    F = ( 
    L = EndomorphicMatrix(F̂.L,range(A),exact(A)),
        U = MultipliableMatrix(F̂.U,range(A),domain(A),exact(A)),
    p = F̂.p)
    return F
end

"""
    function similar(a,b)::Bool

    Dimensional similarity of vectors, a binary relation
    Read "a has the same dimensional form as b"
    `a` and `b` may still have different units.
    A stronger condition than being parallel.
    pp. 184, Hart
"""
 similar(a,b)::Bool = isequal(dimension(a),dimension(b))
 ~(a,b) = similar(a,b)

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
    dima = dimension(a)
    for dd = 2:length(dima)
        if dima[dd] ≠ dima[1]
            return false
        end
    end
    return true
end
function uniform(A::Matrix)
    B = MultipliableMatrix(A)
    isnothing(B) ? false : uniform(B)
end
uniform(A::T) where T <: MultipliableMatrices = left_uniform(A) && right_uniform(A)
uniform(A::UniformMatrix) = true

"""
    function left_uniform(A)

    Does the range of A have uniform dimensions?
"""
left_uniform(A::T) where T<: MultipliableMatrices = uniform(range(A)) ? true : false
function left_uniform(A::Matrix)
    B = MultipliableMatrix(A)
    isnothing(B) ? false : left_uniform(B)
end

"""
    function right_uniform(A)

    Does the domain of A have uniform dimensions?
"""
right_uniform(A::T) where T<:MultipliableMatrices = uniform(domain(A)) ? true : false
function right_uniform(A::Matrix)
    B = MultipliableMatrix(A)
    isnothing(B) ? false : right_uniform(B)
end

"""
     function dimensionless(A)

     Not all dimensionless matrices have
     dimensionless domain and range.
"""
dimensionless(A::T) where T <: MultipliableMatrices = uniform(A) && range(A)[1] == domain(A)[1]
dimensionless(A::Matrix) = uniform(A) && dimension(A[1,1]) == NoDims
dimensionless(A::T) where T <: Number = (dimension(A) == NoDims)

square(A::T) where T <: MultipliableMatrices = (domainlength(A) == rangelength(A))

squarable(A::T) where T <: MultipliableMatrices = (domain(A) ∥ range(A))

"""
    function invdimension

    Dimensional inverse
      
    pp. 64, Hart, `a~` in his notation
"""
invdimension(a) = dimension(1 ./ a)

"""
    function dottable(a,b)

    Are two quantities dimensionally compatible
    to take a dot product?
"""
dottable(a,b) = parallel(a, 1 ./ b)

"""
    function convert_domain(A, newdomain)

    When using the geometric interpretation of matrices,
    it is useful to convert the dimensional domain of the
    matrix to match the expected vectors during multiplication.
    Here we set the matrix to `exact=true` after this step.
"""
function convert_domain(A::MultipliableMatrix, newdomain::Vector)::MultipliableMatrix
    if domain(A) ∥ newdomain
        shift = newdomain./domain(A)
        newrange = range(A).*shift
        B = MultipliableMatrix(A.numbers,newrange,newdomain,exact=true)
    else
        error("New domain not parallel to domain of Multipliable Matrix")
    end
end

#ERROR: setfield!: immutable struct of type MultipliableMatrix cannot be changed
# function convert_domain!(A::MultipliableMatrix, newdomain::Vector)::MultipliableMatrix
#     if A.domain ∥ newdomain
#         shift = newdomain./A.domain
#         newrange = A.range.*shift
#         A.domain = newdomain
#         A.range = newrange
#     else
#         error("New domain not parallel to domain of Multipliable Matrix")
#     end
# end

"""
    function convert_range(A, newrange)

    When using the geometric interpretation of matrices,
    it is useful to convert the dimensional range of the
    matrix to match the desired output of multiplication.
    Here we set the matrix to `exact=true` after this step.
"""
function convert_range(A::MultipliableMatrix, newrange::Vector)::MultipliableMatrix
    if A.range ∥ newrange
        shift = newrange./range(A)
        newdomain = domain(A).*shift
        B = MultipliableMatrix(A.numbers,newrange,newdomain,exact=true)
    else
        error("New range not parallel to range of Multipliable Matrix")
    end
end

#ERROR: setfield!: immutable struct of type MultipliableMatrix cannot be changed
# function convert_range!(A::MultipliableMatrix, newrange::Vector)::MultipliableMatrix
#     if A.range ∥ newrange
#         shift = newrange./A.range
#         newdomain = A.domain.*shift
#         A = MultipliableMatrix(A.numbers,newrange,newdomain,A.exact)
#     else
#         error("New range not parallel to range of Multipliable Matrix")
#     end
# end

"""
    function exact(A)

-    `exact=true`: geometric interpretation of domain and range
-    `exact=false`: algebraic interpretation
"""
exact(A::T) where T <: MultipliableMatrices = A.exact

"""
    function rangelength(A::MultipliableMatrix)

    Numerical dimension (length or size) of range
"""
rangelength(A::T) where T <: MultipliableMatrices = length(range(A))

"""
    function domainlength(A::MultipliableMatrix)

    Numerical dimension (length or size) of domain of A
"""
domainlength(A::T) where T <: MultipliableMatrices = length(domain(A))

size(A::MultipliableMatrices) = (rangelength(A), domainlength(A))

convert(::Type{AbstractMatrix{T}}, A::MultipliableMatrices) where {T<:Number} = convert(MultipliableMatrices{T}, A)
convert(::Type{AbstractArray{T}}, A::MultipliableMatrices) where {T<:Number} = convert(MultipliableMatrices{T}, A)
#convert(::Type{AbstractArray{T}}, S::AbstractToeplitz) where {T<:Number} = convert(AbstractToeplitz{T}, S)

domain(A::T) where T <: MultipliableMatrices = A.domain
domain(A::EndomorphicMatrix) = A.range # domain not saved
domain(A::UniformMatrix) = fill(A.domain,size(A.numbers)[2])

range(A::T) where T <: MultipliableMatrices = A.range
range(A::UniformMatrix) = fill(A.range,size(A.numbers)[1])



"""
     function inv
"""
inv(A::T) where T <: MultipliableMatrices = ~singular(A) ? MultipliableMatrix(inv(A.numbers),domain(A),range(A)) : error("matrix is singular")

"""
    function det
"""
function det(A::T) where T<: MultipliableMatrices

    if square(A)
        # detunit = Vector{eltype(domain(A))}(undef,domainlength(A))
        # for i = 1:domainlength(A)
        # end
        detunit = prod([range(A)[i]/domain(A)[i] for i = 1:domainlength(A)])

        return Quantity(det(A.numbers),detunit)
    else
        error("Determinant requires square matrix")
    end
end

singular(A::T) where T <: MultipliableMatrices = iszero(ustrip(det(A)))

"""
    svd(A; full::Bool = false, alg::Algorithm = default_svd_alg(A)) -> SVD

Compute the singular value decomposition (SVD) of `A` and return an `SVD` object. Extended for MultipliableMatrix input.
"""
#function svd(A::MultipliableMatrices;full=false) where T <: MultipliableMatrices
function svd(A::MultipliableMatrices;full=false) 
    if uniform(A) 
        F = svd(A.numbers, full=full)
        # U,V just regular matrices: return that way?
        # They are also Uniform and Endomorphic
        return SVD(F.U,F.S * range(A)[1]./domain(A)[1],F.Vt)
    else
        error("SVD not implemented for non-uniform matrices or non-full flag")
    end
    
    
end

end
