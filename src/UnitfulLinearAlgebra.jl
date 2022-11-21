module UnitfulLinearAlgebra

using Unitful, LinearAlgebra

export MultipliableMatrix, EndomorphicMatrix
export similar, ∥, parallel
export uniform, left_uniform, right_uniform
export invdimension, dottable
export element, array
export convert_range, convert_domain
#export convert_range!, convert_domain!
export exact, multipliable, dimensionless, endomorphic
export svd_unitful, inv, inv_unitful, diagonal_matrix 

import LinearAlgebra.inv
import Base:(~), (*)
import Base.similar

abstract type MultipliableMatrices end

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
struct MultipliableMatrix{T} <: MultipliableMatrices where T <: Number
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
- `range`: dimensional range in terms of units, this is also the domain
"""
struct EndomorphicMatrix{T} <: MultipliableMatrices where {T <: Number}
    numbers::Matrix{T}
    range::Vector
end

"""
    MultipliableMatrix(numbers,range,domain;exact=false)

    Constructor where `exact` is a keyword argument. One may construct a MultipliableMatrix without specifying exact, in which case it defaults to `false`. 
"""
MultipliableMatrix(numbers,range,domain;exact=false) =
    MultipliableMatrix(numbers,range,domain,exact)

"""
     MultipliableMatrix(array)

    Transform array to MultipliableMatrix
"""
function MultipliableMatrix(A::Matrix)

    numbers = ustrip.(A)
    M,N = size(numbers)
    #U = typeof(unit(A[1,1]))
    U = eltype(unit.(A))
    domain = Vector{U}(undef,N)
    range = Vector{U}(undef,M)

    for i = 1:M
        range[i] = unit(A[i,1])
    end
    
    for j = 1:N
        domain[j] = unit(A[1,1])/unit(A[1,j])
    end
    B = MultipliableMatrix(numbers,range,domain,exact=false)
    # if the array is not multipliable, return nothing
    if array(B) == A
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
    function endomorphic(A)::Bool

    Is an array endomorphic?
    It requires a particular structure of the units/dimensions in the array. 
"""
endomorphic(A::Matrix) = ~isnothing(EndomorphicMatrix(A))
endomorphic(A::EndomorphicMatrix) = true
endomorphic(A::MultipliableMatrix) = isequal(A.domain,A.range)
endomorphic(A::T) where T <: Number = dimensionless(A) # scalars must be dimensionless to be endomorphic

"""
    function element(A::MultipliableMatrix,i::Integer,j::Integer)

    Recover element (i,j) of a MultipliableMatrix.

#Input
- `A::MultipliableMatrix`
- `i::Integer`: row index
- `j::Integer`: column index

#Output
- `Quantity`: numerical value and units
"""
element(A::MultipliableMatrix,i::Integer,j::Integer) = Quantity(A.numbers[i,j],A.range[i]./A.domain[j])

"""
    function element(A::EndomorphicMatrix,i::Integer,j::Integer)

    Recover element (i,j) of a EndomorphicMatrix.

#Input
- `A::EndomorphicMatrix`
- `i::Integer`: row index
- `j::Integer`: column index

#Output
- `Quantity`: numerical value and units
"""
element(A::EndomorphicMatrix,i::Integer,j::Integer) = Quantity(A.numbers[i,j],A.range[i]./A.range[j])

"""
    function array(A::MultipliableMatrix)

    Expand A into array form
    Useful for tests, display
    pp. 193, Hart
"""
function array(A::T) where T<: MultipliableMatrices

    M = rangelength(A)
    N = domainlength(A)
    #M = length(A.range)
    #N = length(A.domain)
    #B = Matrix{Quantity}(undef,M,N)
    B = Matrix(undef,M,N)
    for m = 1:M
        for n = 1:N
            B[m,n] = element(A,m,n)
        end
    end
    return B
end

"""
    function *(A::MultipliableMatrix,b)

    Matrix-vector multiplication with units/dimensions

    Unitful also handles this case, but here there is added efficiency in the storage of units/dimensions by accounting for the necessary structure of the matrix.
"""
*(A::MultipliableMatrix,b) = dimension(A.domain) == dimension(b) ? c = (A.numbers*ustrip.(b)).*A.range : error("Dimensions of MultipliableMatrix and pair not compatible")

"""
    function similar(a,b)::Bool

    Dimensional similarity of vectors, a binary relation

    Read "a has the same dimensional form as b"

    `a` and `b` may still have different units.

    A stronger condition than being parallel.

    pp. 184, Hart
"""
 similar(a,b) = isequal(dimension(a),dimension(b))
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
uniform(A::MultipliableMatrix) = left_uniform(A) && right_uniform(A)

"""
    function left_uniform(A)

    Does the range of A have uniform dimensions?
"""
left_uniform(A::MultipliableMatrix) = uniform(A.range) ? true : false
function left_uniform(A::Matrix)
    B = MultipliableMatrix(A)
    isnothing(B) ? false : left_uniform(B)
end

"""
    function right_uniform(A)

    Does the domain of A have uniform dimensions?
"""
right_uniform(A::MultipliableMatrix) = uniform(A.domain) ? true : false
function right_uniform(A::Matrix)
    B = MultipliableMatrix(A)
    isnothing(B) ? false : right_uniform(B)
end

"""
     function dimensionless(A)

     Not all dimensionless matrices have
     dimensionless domain and range.
"""
dimensionless(A::MultipliableMatrix) = uniform(A) && A.range[1] == A.domain[1]
dimensionless(A::Matrix) = uniform(A) && dimension(A[1,1]) == NoDims
dimensionless(A::T) where T <: Number = (dimension(A) == NoDims)

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

function convert_domain(A::MultipliableMatrix, newdomain::Vector)::MultipliableMatrix
    if A.domain ∥ newdomain
        shift = newdomain./A.domain
        newrange = A.range.*shift
        B = MultipliableMatrix(A.numbers,newrange,newdomain,A.exact)
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

function convert_range(A::MultipliableMatrix, newrange::Vector)::MultipliableMatrix
    if A.range ∥ newrange
        shift = newrange./A.range
        newdomain = A.domain.*shift
        B = MultipliableMatrix(A.numbers,newrange,newdomain,A.exact)
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
    function exact(A::MultipliableMatrix) = A.exact

-    `exact=true`: geometric interpretation of domain and range
-    `exact=false`: algebraic interpretation
"""
exact(A::MultipliableMatrix) = A.exact

"""
    function rangelength(A::MultipliableMatrix)

    Numerical dimension (length or size) of range
"""
rangelength(A::T) where T <: MultipliableMatrices = length(A.range)

"""
    function domainlength(A::MultipliableMatrix)

    Numerical dimension (length or size) of domain of A
"""
domainlength(A::MultipliableMatrix) = length(A.domain)
domainlength(A::EndomorphicMatrix) = length(A.range) # domain not saved

"""
     EndomorphicMatrix(array)

    Transform array to EndomorphicMatrix
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
    if array(B) == A
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
    function diagonal_matrix(γ)

	       Take diagonal dimensional elements γ
	   Form a weighting or scaling matrix that has the proper units
	   Note that the off-diagonals are zero but they must have units
"""
function diagonal_matrix(s)
    #γ = .√γ2 # problematic statement; complex type error

    # get matrix of types/units
    S = 0.0 *s*s' # should use zeros or equivalent
    N = size(S,1)
    for i = 1:N
	for j = 1:N
	    #if i != j
		# make off-diagonals zero, but keep units
            #    S[i,j] -= S[i,j]
	    #end
            if i == j
                S[i,j] += s[i].^2
            end
	end
    end
    return S
end

# """
# 	function inv_unitful(E) 

# 	Take the inverse of a matrix and respect units
#     The second part of the equation string is a way to grab the inverse transpose units
# """
# inv_unitful(E::Matrix{Quantity{Float64}}) = inv(ustrip.(E)) .* unit.( 1 ./ E' ) 

"""
	function UnitfulLinearAlgebra.inv(E) 

	Take the inverse of a matrix and respect units
    The second part of the equation string is a way to grab the inverse transpose units
"""
#inv(E::Matrix{Quantity{Float64}}) = inv(ustrip.(E)) .* unit.( 1 ./ E' ) 
function inv(E::Matrix{Quantity{T}}) where T <: Real

    M,N = size(E)
    Einv = Matrix{Quantity{Float64}}(undef,M,N)
    Enumber = inv(ustrip.(E))
    Eunit = unit.( 1 ./ E' )
    for i in 1:M
        for j in 1:N
             if dimension(Eunit[i,j]) == NoDims
                Einv[i,j] = Quantity(Enumber[i,j]*100,u"percent")
                Einv[i,j] = Quantity(Enumber[i,j],Unitful.NoUnits)
            else
                Einv[i,j] = Quantity(Enumber[i,j],Eunit[i,j])
            end
        end
    end
    
    return Einv
end

"""
	function UnitfulLinearAlgebra.inv_unitful(E) 

	Take the inverse of a matrix and respect units
    The second part of the equation string is a way to grab the inverse transpose units

No multiple dispatch here, problem with Matrix{Any}
"""
#inv(E::Matrix{Quantity{Float64}}) = inv(ustrip.(E)) .* unit.( 1 ./ E' ) 
function inv_unitful(E) 

    M,N = size(E)
    Einv = Matrix{Quantity{Float64}}(undef,M,N)
    Enumber = inv(ustrip.(E))
    Eunit = unit.( 1 ./ E' )
    for i in 1:M
        for j in 1:N
             if dimension(Eunit[i,j]) == NoDims
                Einv[i,j] = Quantity(Enumber[i,j]*100,u"percent")
                Einv[i,j] = Quantity(Enumber[i,j],Unitful.NoUnits)
            else
                Einv[i,j] = Quantity(Enumber[i,j],Eunit[i,j])
            end
        end
    end
    
    return Einv
end


"""
	function svd_unitful(E)

		Extend the SVD to be used with unitful quantities.
		You will unlikely need to do this.
"""
function svd_unitful(E)
	U,L1,V = svd(ustrip.(E),full=true)
	# in simplest case, put units of E on L
	uE = unit.(diag(E))
	L = L1 .* uE
	return U,L,V
end

end
