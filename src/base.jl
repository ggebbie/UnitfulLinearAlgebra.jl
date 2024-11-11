# Extend Base methods

function Base.show(io::IO, mime::MIME"text/plain", B::AbstractUnitfulDimVecOrMat) 
    lines = 0
    summary(io, B)
    A = DimArray(B)
    print_name(io, name(A))
    lines += Dimensions.print_dims(io, mime, dims(A))
    !(isempty(dims(A)) || isempty(refdims(A))) && println(io)
    lines += Dimensions.print_refdims(io, mime, refdims(A))
    println(io)

    # Printing the array data is optional, subtypes can 
    # show other things here instead.
    ds = displaysize(io)
    ioctx = IOContext(io, :displaysize => (ds[1] - lines, ds[2]))
    DimensionalData.show_after(ioctx, mime, A)

    return nothing
end


function Base.show(io::IO, mime::MIME"text/plain", A::AbstractUnitfulVecOrMat)
    lines = 0
    summary(io, A)
    println(io)
    ds = displaysize(io)
    ioctx = IOContext(io, :displaysize => (ds[1] - lines, ds[2]))
    T2 = eltype(A)
    Base.print_matrix(DimensionalData._print_array_ctx(ioctx, T2), Matrix(A))
    return nothing
end

"""
    function *(A::AbstractUnitfulType,b)

    Matrix-scalar multiplication with units/dimensions.
    Must account for change in the unitrange when the
     scalar has units.
    Here, take product of dimension of the scalar and the unitrange.
    Alternatively, divide the domain by the dimension of the scalar. 
    Matrix-scalar multiplication is commutative.
    Result is `exact` if input matrix is exact and scalar is dimensionless. 

    function *(A,B)

    Matrix-matrix multiplication with units/dimensions.
    A*B represents two successive transformations.
    Unitrange of B should equal domain of A in geometric interpretation.
    Unitrange of B should be parallel to unitdomain of A in algebraic interpretation.
"""
Base.:*(A::AbstractUnitfulMatrix,b::Quantity) = DimensionalData.rebuild(A,parent(A)*ustrip(b),(Units(unitrange(A).*unit(b)),unitdomain(A)))
Base.:*(A::AbstractUnitfulMatrix,b::Unitful.Units) = DimensionalData.rebuild(A,parent(A),(Units(unitrange(A).*b),unitdomain(A)))
Base.:*(b::Union{Quantity,Unitful.Units},A::AbstractUnitfulMatrix) = A*b
Base.:*(A::AbstractUnitfulMatrix,b::Number) = DimensionalData.rebuild(A,parent(A)*b)
Base.:*(b::Number,A::AbstractUnitfulMatrix) = A*b
# could probably merge Matrix and Vector versions below
Base.:*(A::AbstractUnitfulMatrix, B::Matrix) = A * UnitfulMatrix(B)
Base.:*(A::Matrix, B::AbstractUnitfulMatrix) = UnitfulMatrix(A) * B
Base.:*(a::Vector, B::AbstractUnitfulMatrix) = UnitfulMatrix(a) * B

# matrix-vector multiplication, return vector of same type as input
# function uses two instances of transformation: slow?
Base.:*(A::AbstractUnitfulMatrix, b::Vector) = vec(A * UnitfulMatrix(b))

# vector-scalar multiplication: not recommended to use AbstractUnitfulVector,
# but if it is used, return variable has same type
Base.:*(a::AbstractUnitfulVector,b::Quantity) = DimensionalData.rebuild(a,parent(a)*ustrip(b),(Units(unitrange(a).*unit(b)),))
Base.:*(a::AbstractUnitfulVector,b::Unitful.Units) = DimensionalData.rebuild(a,parent(a),(Units(unitrange(a).*b),))
Base.:*(b::Union{Quantity,Unitful.Units},a::AbstractUnitfulVector) = a*b
# Need to test next line
Base.:*(a::Union{AbstractUnitfulVector,AbstractUnitfulDimVector},b::Number) = DimensionalData.rebuild(a,parent(a)*b)
Base.:*(b::Number,a::Union{AbstractUnitfulVector,AbstractUnitfulDimVector}) = a*b

# (matrix/vector)-(matrix/vector) multiplication when inexact handled here
function Base.:*(A::AbstractUnitfulVecOrMat,B::AbstractUnitfulVecOrMat)
    if exact(A) && exact(B)
        #return DimensionalData._rebuildmul(A,B) # uses strict checking

        # replace DimensionalData._rebuildmul(A,B) # uses strict checking
        # instead reproduce necessary part of DimensionalData here

        # from DimensionalData._comparedims_mul(A, B)
        DimensionalData.comparedims(last(dims(A)), first(dims(B)); 
            order=false, val=true, length=false
        )
        return rebuild(A, parent(A) * parent(B), (first(dims(A)), last(dims(B))))
        #rebuild(A, parent(A) * parent(B), (first(dims(A)),))

    elseif unitdomain(A) ∥ unitrange(B)
        return DimensionalData._rebuildmul(convert_unitdomain(A,unitrange(B)),B)

        Anew = convert_unitdomain(A,unitrange(B))
        DimensionalData.comparedims(last(dims(Anew)),
            first(dims(B)); 
            order=false, val=true, length=false
        )
        return rebuild(Anew, parent(Anew) * parent(B), (first(dims(Anew)), last(dims(B))))
    else
        error("UnitfulLinearAlgebra: unitdomain(A) and unitrange(B) not parallel")
    end
end

"""
    function +(A,B)

    Matrix-matrix addition with units/dimensions.
    A+B requires the two matrices to have dimensional similarity.
"""
function Base.:+(A::AbstractUnitfulVecOrMat,B::AbstractUnitfulVecOrMat) #where T1 where T2
    bothexact = exact(A) && exact(B)
    if (unitrange(A) == unitrange(B) && unitdomain(A) == unitdomain(B)) ||
        ( unitrange(A) ∥ unitrange(B) && unitdomain(A) ∥ unitdomain(B) && ~bothexact)
        return DimensionalData.rebuild(A,parent(A)+parent(B),(unitrange(A),unitdomain(A))) 
    else
        error("matrices not dimensionally conformable for addition")
    end
end

"""
    function -(A,B)

    Matrix-matrix subtraction with units/dimensions.
    A-B requires the two matrices to have dimensional similarity.
"""
function Base.:-(A::AbstractUnitfulVecOrMat{T1},B::AbstractUnitfulVecOrMat{T2}) where T1 where T2
    bothexact = exact(A) && exact(B)
    if (unitrange(A) == unitrange(B) && unitdomain(A) == unitdomain(B)) ||
       ( unitrange(A) ∥ unitrange(B) && unitdomain(A) ∥ unitdomain(B) && ~bothexact)
        return DimensionalData.rebuild(A,parent(A)-parent(B),(unitrange(A),unitdomain(A))) # takes exact(A) but should be bothexact 
    else
        error("matrices not dimensionally conformable for subtraction")
    end
end
# define negation
Base.:-(A::AbstractUnitfulType) = DimensionalData.rebuild(A,-parent(A)) 

"""
     function left divide

     Left divide of Multipliable Matrix.
     Reverse mapping from unitdomain to range.
     Is `exact` if input is exact.
"""
function Base.:(\ )(A::AbstractUnitfulMatrix,b::AbstractUnitfulVector)
    if exact(A)
        DimensionalData.comparedims(first(dims(A)), first(dims(b)); val=true)
        return rebuild(A,parent(A)\parent(b),(last(dims(A)),)) #,exact = (exact(A) && exact(B)))
    elseif ~exact(A) && (unitrange(A) ∥ unitrange(b))
        Anew = convert_unitrange(A,unitrange(b)) 
        return rebuild(Anew,parent(Anew)\parent(b),(last(dims(Anew)),))
    else
        error("UnitfulLinearAlgebra.mldivide: Dimensions of Unitful Matrices A and b not compatible")
    end
end
function Base.:(\ )(A::AbstractUnitfulMatrix,B::AbstractUnitfulMatrix)
    if exact(A)
        DimensionalData.comparedims(first(dims(A)), first(dims(B)); val=true)
        return rebuild(A,parent(A)\parent(B),(last(dims(A)),last(dims(B)))) #,exact = (exact(A) && exact(B)))
    elseif ~exact(A) && (unitrange(A) ∥ unitrange(B))
        Anew = convert_unitrange(A,unitrange(B)) 
        return rebuild(Anew,parent(Anew)\parent(B),(last(dims(Anew)),last(dims(B))))
    else
        error("UnitfulLinearAlgebra.matrix left divide): Dimensions of Unitful Matrices A and b not compatible")
    end
end
function Base.:(\ )(A::AbstractUnitfulDimMatrix,b::AbstractUnitfulDimVector)
    if exact(A)
        DimensionalData.comparedims(first(unitdims(A)), first(unitdims(b)); val=true)
        DimensionalData.comparedims(first(dims(A)), first(dims(b)); val=true)
        return rebuild(A,parent(A)\parent(b),(last(unitdims(A)),),(last(dims(A)),)) #,exact = (exact(A) && exact(B)))
    elseif ~exact(A) && (unitrange(A) ∥ unitrange(b))
        Anew = convert_unitrange(A,unitrange(b)) 
        return rebuild(Anew,parent(Anew)\parent(b),(last(unitdims(Anew)),),(last(dims(Anew)),))
    else
        error("UnitfulLinearAlgebra.mldivide: Dimensions of Unitful Matrices A and b not compatible")
    end
end
function Base.:(\ )(A::AbstractUnitfulDimMatrix,B::AbstractUnitfulDimMatrix)
    if exact(A)
        DimensionalData.comparedims(first(unitdims(A)), first(unitdims(B)); val=true)
        DimensionalData.comparedims(first(dims(A)), first(dims(B)); val=true)
        return rebuild(A,parent(A)\parent(B),(last(unitdims(A)),last(unitdims(B))),(last(dims(A)),last(dims(B)))) #,exact = (exact(A) && exact(B)))
    elseif ~exact(A) && (unitrange(A) ∥ unitrange(B))
        Anew = convert_unitrange(A,unitrange(B)) 
        return rebuild(Anew,parent(Anew)\parent(B),(last(unitdims(Anew)),last(unitdims(B))),(last(dims(Anew)),last(dims(B))))
    else
        error("UnitfulLinearAlgebra.(matrix left divide): Dimensions of Unitful Matrices A and b not compatible")
    end
end
# do what the investigator means -- convert to UnitfulType -- probably a promotion mechanism to do the same thing
Base.:(\ )(A::AbstractUnitfulType,b::Number) = A\UnitfulMatrix([b])
# this next one is quite an assumption
#Base.:\(A::AbstractUnitfulMatrix,b::AbstractVector) = A\UnitfulMatrix(vec(b)) #error("UnitfulLinearAlgebra: types not consistent")
Base.:(\ )(A::AbstractUnitfulMatrix,b::Vector) = vec(A\UnitfulMatrix(b)) # return something with same type as input `b`

"""
    function ldiv(F::LU{T,MultipliableMatrix{T},Vector{Int64}}, B::AbstractVector) where T<:Number

    Perform matrix left divide on LU factorization object,
    where LU object contains unit information.
    Doesn't require LeftUniformMatrix. 
"""
function (\)(F::LU{T,<: AbstractUnitfulMatrix,Vector{Int64}}, B::AbstractUnitfulVector) where T<:Number
    if unitrange(F.factors) == unitrange(B)
         # pass without any issues
    elseif unitrange(F.factors) ∥ unitrange(B)
        # convert_range of F.factors
        # is allocating, will affect performance
        convert_unitrange(F.factors,unitrange(B))
    else
        error("LU left divide: units of F, B, are not conformable")
    end
    LinearAlgebra.require_one_based_indexing(B)
    m, n = size(F)
    TFB = typeof(oneunit(eltype(parent(B))) / oneunit(eltype(parent(F.factors))))
    FF = LinearAlgebra.Factorization{TFB}(LU(parent(F.factors),F.ipiv,F.info))
    BB = LinearAlgebra._zeros(TFB, B, n)
    if n > size(B, 1)
        LinearAlgebra.copyto!(view(BB, 1:m, :), parent(B))
    else
        LinearAlgebra.copyto!(BB, parent(B))
    end
    LinearAlgebra.ldiv!(FF, BB)
    return rebuild(B,LinearAlgebra._cut_B(BB, 1:n),(unitdomain(F.factors),))
end

# """
#      function ldiv!

#      In-place left division by a Multipliable Matrix.
#      Reverse mapping from unitdomain to range.
#      Is `exact` if input is exact.

#     Problem: b changes type unless endomorphic
# """
# function ldiv!(A::AbstractMultipliableMatrix,b::AbstractVector)
#     ~endomorphic(A) && error("A not endomorphic, b changes type, ldiv! not available")
    
#     if dimension(unitrange(A)) == dimension(b)
#         #if unitrange(A) ~ b

#         # seems to go against the point
#         #b = copy((A.numbers\ustrip.(b)).*unitdomain(A))
#         btmp = (A.numbers\ustrip.(b)).*unitdomain(A)
#         for bb = 1:length(btmp)
#             b[bb] = btmp[bb]
#         end
        
#     elseif ~exact(A) && (unitrange(A) ∥ b)
#         Anew = convert_unitrange(A,unit.(b)) # inefficient?
#         btmp = (Anew.numbers\ustrip.(b)).*unitdomain(Anew)
#         for bb = 1:length(btmp)
#             b[bb] = btmp[bb]
#         end

#     else
#         error("UnitfulLinearAlgebra.ldiv!: Dimensions of MultipliableMatrix and vector not compatible")
#     end
    
# end

Base.:~(a,b) = similarity(a,b)

"""
    function transpose

    Defined by condition `A[i,j] = transpose(A)[j,i]`.
    Not analogous to function for dimensionless matrices.

    Hart, pp. 205.
"""
# A redefined tranpose that corrects error based on AbstractArray interface
#Base.transpose(A::AbstractUnitfulMatrix) = rebuild(A,transpose(parent(A)),(Units(unitdomain(A).^-1), Units(unitrange(A).^-1)))
# previous working version here
#Base.transpose(A::AbstractUnitfulMatrix) = rebuild(A,transpose(parent(A)),(Units(inv.(unitdomain(A))), Units(inv.(unitrange(A)))))
Base.transpose(A::AbstractUnitfulMatrix) = rebuild(A,transpose(parent(A)),(Units(parent(inv.(unitdomain(A)))), Units(parent(inv.(unitrange(A))))))
Base.transpose(a::AbstractUnitfulVector) = rebuild(a,transpose(parent(a)),(Units([NoUnits]), Units(unitrange(a).^-1))) # kludge for unitrange of row vector
Base.transpose(A::AbstractUnitfulDimMatrix) = rebuild(A,transpose(parent(A)),(Units(unitdomain(A).^-1), Units(unitrange(A).^-1)),(last(dims(A)),first(dims(A))))
Base.transpose(a::AbstractUnitfulDimVector) = rebuild(a,transpose(parent(a)),(Units([NoUnits]), Units(unitrange(a).^-1)),(:empty,first(dims(a))))

# adjoint follows transpose structure
Base.adjoint(A::AbstractUnitfulMatrix) = rebuild(A,adjoint(parent(A)),(Units(unitdomain(A).^-1), Units(unitrange(A).^-1)))
Base.adjoint(a::AbstractUnitfulVector) = rebuild(a,adjoint(parent(a)),(Units([NoUnits]), Units(unitrange(a).^-1))) # kludge for unitrange of row vector
Base.adjoint(A::AbstractUnitfulDimMatrix) = rebuild(A,adjoint(parent(A)),(Units(unitdomain(A).^-1), Units(unitrange(A).^-1)),(last(dims(A)),first(dims(A))))
Base.adjoint(a::AbstractUnitfulDimVector) = rebuild(a,adjoint(parent(a)),(Units([NoUnits]), Units(unitrange(a).^-1)),(:empty,first(dims(a))))

# Currently untested
Base.similar(A::AbstractUnitfulVecOrMat{T}) where T <: Number =
       DimensionalData.rebuild(A, zeros(A))
    #UnitfulMatrix(Matrix{T}(undef,size(A)),unitrange(A),unitdomain(A);exact=exact(A))

# NOTE: Base.getproperty is also expanded but stored next to relevant linear algebra functions.

# """
#     function vcat(A,B)

#     Modeled after function `VERTICAL` (pp. 203, Hart, 1995).
# """
# function Base.vcat(A::AbstractMultipliableMatrix,B::AbstractMultipliableMatrix)

#     numbers = vcat(A.numbers,B.numbers)
#     shift = unitdomain(A)[1]./unitdomain(B)[1]
#     ur = vcat(unitrange(A),unitrange(B).*shift)
#     bothexact = (exact(A) && exact(B))
#     return BestMultipliableMatrix(numbers,ur,unitdomain(A),exact=bothexact)
# end

# """
#     function hcat(A,B)

#     Modeled after function `HORIZONTAL` (pp. 202, Hart, 1995).
# """
# function Base.hcat(A::AbstractMultipliableMatrix,B::AbstractMultipliableMatrix)

#     numbers = hcat(A.numbers,B.numbers)
#     shift = unitrange(A)[1]./unitrange(B)[1]
#     ud = vcat(unitdomain(A),unitdomain(B).*shift)
#     bothexact = (exact(A) && exact(B))
#     return BestMultipliableMatrix(numbers,unitrange(A),ud,exact=bothexact)
# end

## start of UnitfulDimMatrix methods
Base.:*(A::AbstractUnitfulDimMatrix, B::AbstractUnitfulDimMatrix) = DimensionalData._rebuildmul(A,B)

function DimensionalData._rebuildmul(A::AbstractUnitfulDimMatrix, B::AbstractUnitfulDimVector)
    # compare unitdims
    DimensionalData.comparedims(last(unitdims(A)), first(unitdims(B)); val=true)

    # compare regular (axis) dims
    DimensionalData.comparedims(last(dims(A)), first(dims(B)); val=true)
    
    DimensionalData.rebuild(A, parent(A) * parent(B), (first(unitdims(A)),), (first(dims(A)),))
end
Base.:*(A::AbstractUnitfulDimMatrix, B::AbstractUnitfulDimVector) = DimensionalData._rebuildmul(A,B)

#copied from ULA.* 
DimensionalData._rebuildmul(A::AbstractUnitfulDimMatrix, b::Quantity) = rebuild(A,parent(A)*ustrip(b),(Units(unitrange(A).*unit(b)),unitdomain(A)))
Base.:*(A::AbstractUnitfulDimMatrix, b::Quantity) = DimensionalData._rebuildmul(A,b)
Base.:*(b::Quantity, A::AbstractUnitfulDimMatrix) = DimensionalData._rebuildmul(A,b)
Base.:*(A::AbstractUnitfulDimMatrix, b::Number) = DimensionalData._rebuildmul(A,b)
Base.:*(b::Number, A::AbstractUnitfulDimMatrix) = DimensionalData._rebuildmul(A,b)

#from ULA.+ 
function Base.:+(A::AbstractUnitfulDimVecOrMat,B::AbstractUnitfulDimVecOrMat) 
    
    # compare unitdims
    DimensionalData.comparedims(first(unitdims(A)), first(unitdims(B)); val=true)

    # compare regular (axis) dims
    DimensionalData.comparedims(last(dims(A)), last(dims(B)); val=true)
    
    bothexact = exact(A) && exact(B)
    if (unitrange(A) == unitrange(B) && unitdomain(A) == unitdomain(B)) ||
        ( unitrange(A) ∥ unitrange(B) && unitdomain(A) ∥ unitdomain(B) && ~bothexact)
        return rebuild(A,parent(A)+parent(B),(unitrange(A),unitdomain(A))) 
    else
        error("matrices not dimensionally conformable for addition")
    end
end

function Base.:-(A::AbstractUnitfulDimVecOrMat,B::AbstractUnitfulDimVecOrMat)
    
    # compare unitdims
    DimensionalData.comparedims(first(unitdims(A)), first(unitdims(B)); val=true)

    # compare regular (axis) dims
    DimensionalData.comparedims(last(dims(A)), last(dims(B)); val=true)
    
    bothexact = exact(A) && exact(B)
    if (unitrange(A) == unitrange(B) && unitdomain(A) == unitdomain(B)) ||
        ( unitrange(A) ∥ unitrange(B) && unitdomain(A) ∥ unitdomain(B) && ~bothexact)
        return rebuild(A,parent(A)-parent(B),(unitrange(A),unitdomain(A))) 
    else
        error("matrices not dimensionally conformable for subtraction")
    end
end

#this is probably bad - automatically broadcasts because I don't know how to override
#the dot syntax
function Base.:+(A::AbstractUnitfulDimVecOrMat,b::Quantity) 
    if unitrange(A)[1] == unit(b)
        println("broadcasting!")
        return rebuild(A, parent(A) .+ ustrip(b), (unitrange(A), unitdomain(A)))
    else
        error("matrix and scalar are not dimensionally conformable for subtraction")
    end
end

function Base.:-(A::AbstractUnitfulDimVecOrMat,b::Quantity) 
    if unitrange(A)[1] == unit(b)
        println("broadcasting!")
        return rebuild(A, parent(A) .- ustrip(b), (unitrange(A), unitdomain(A)))
    else
        error("matrix and scalar are not dimensionally conformable for subtraction")
    end
end

Base.sum(A::AbstractUnitfulType) = Base.sum(Matrix(A))

"""
    function vec(A::AbstractUnitfulType)

    return a Vector{Quantity}
    note ambiguity whether this function should return a Vector{Quantity} or an `AbstractUnitfulType` with one column

# Arguments
- `A::AbstractUnitfulType`: input matrix
"""
function Base.vec(A::AbstractUnitfulType)
    qn = vec(parent(A))
    ur = unitrange(A)
    ud = unitdomain(A)
    qu = vec([ur[i]/ud[j] for i in eachindex(ur), j in eachindex(ud)])
    return Quantity.(qn,qu)
end

Base.first(A::AbstractUnitfulType) = first(vec(A))
Base.last(A::AbstractUnitfulType) = last(vec(A))
