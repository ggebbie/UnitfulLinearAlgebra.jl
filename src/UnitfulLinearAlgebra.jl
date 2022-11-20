module UnitfulLinearAlgebra

using Unitful, LinearAlgebra

export similar, parallel, ∥, uniform
export invdimension
export svd_unitful, inv, inv_unitful, diagonal_matrix 

import LinearAlgebra.inv
import Base:(~)
import Base.similar

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

# similar(a::T,b::T) where T <: Number = isequal(dimension(a),dimension(b))
# ~(a::T,b::T) where T <: Number = similar(a,b)

# similar(a::Quantity,b::Quantity) = isequal(dimension(a),dimension(b))
# ~(a::Quantity,b::Quantity) = similar(a,b)

# similar(a::Vector{Quantity{T}},b::Vector{Quantity{T}}) where T <: Number = isequal(dimension(a),dimension(b))

# ~(a::Vector{Quantity{T}},b::Vector{Quantity{T}}) where T <: Number = similar(a,b)

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
function parallel(a::Union{T,Quantity,Quantity{T},Vector{Quantity},Vector{Quantity{T}}},b::Union{T,Quantity,Quantity{T},Vector{Quantity},Vector{Quantity{T}}})::Bool where T <: Number 

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

∥(a::Union{T,Quantity,Quantity{T},Vector{Quantity},Vector{Quantity{T}}},b::Union{T,Quantity,Quantity{T},Vector{Quantity},Vector{Quantity{T}}}) where {T <: Number} = parallel(a,b)

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
invdimension(a::Union{Quantity,Vector{Quantity},Vector{Quantity{T}}}) where T <: Number = dimension(1 ./ a)

# uniform vectors and scalars are not dispatched to previous definition
invdimension(a) = uniform(a) ? dimension(1 ./ a) : error("inverse dimension not computable")

"""
    function dottable(a,b)

    Are two quantities dimensionally compatible
    to take a dot product?
"""
function dottable(a,b)

    parallel(a, 1 ./ b)
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
