module UnitfulLinearAlgebra

using Unitful, LinearAlgebra

export similar, parallel, ∥, uniform
export invdimension, dottable, MultipliableMatrix
export element, expand
export svd_unitful, inv, inv_unitful, diagonal_matrix 

import LinearAlgebra.inv
import Base:(~), (*)
import Base.similar

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
struct MultipliableMatrix
    numbers::Matrix
    range::Vector
    domain::Vector
    exact::Bool
end

"""
    MultipliableMatrix(numbers,range,domain;exact=false)

    Constructor where `exact` is a keyword argument. One may construct a MultipliableMatrix without specifying exact, in which case it defaults to `false`. 
"""
MultipliableMatrix(numbers,range,domain;exact=false) =
    MultipliableMatrix(numbers,range,domain,exact)
    
element(A::MultipliableMatrix,i::Integer,j::Integer) = Quantity(A.numbers[i,j],A.range[i]./A.domain[j])

"""
    function expand(A::MultipliableMatrix)

    Expand A into array form
    Useful for tests, display
    pp. 193, Hart
"""
function expand(A::MultipliableMatrix)

    M = length(A.range)
    N = length(A.domain)
    B = Matrix{Quantity}(undef,M,N)
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
*(A::MultipliableMatrix,b) = dimension(A.domain) == dimension(b) ? c = (A.numbers*ustrip.(b)).*A.range : error("Dimensions of A and b not compatible")

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
function uniform(a)
    # handle scalars
    if length(a) == 1
        return true # by default
    else
        dima = dimension(a)
        for dd = 2:length(dima)
            if dima[dd] ≠ dima[1]
                return false
            end
        end
    end
    return true
end

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
