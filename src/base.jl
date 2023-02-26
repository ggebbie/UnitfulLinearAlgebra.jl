# Build off Base methods
"""
    function *(A::MultipliableMatrix,b)


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
Base.:*(A::AbstractUnitfulMatrix,b::Unitful.FreeUnits) = DimensionalData.rebuild(A,parent(A),(Units(unitrange(A).*b),unitdomain(A)))
Base.:*(b::Union{Quantity,Unitful.FreeUnits},A::AbstractUnitfulMatrix) = A*b
Base.:*(A::AbstractUnitfulMatrix,b::Number) = DimensionalData.rebuild(A,parent(A)*b)
Base.:*(b::Number,A::AbstractUnitfulMatrix) = A*b

# vector-scalar multiplication
Base.:*(a::AbstractUnitfulVector,b::Quantity) = DimensionalData.rebuild(a,parent(a)*ustrip(b),(Units(unitrange(a).*unit(b)),))
Base.:*(a::AbstractUnitfulVector,b::Unitful.FreeUnits) = DimensionalData.rebuild(a,parent(a),(Units(unitrange(a).*b),))
Base.:*(b::Union{Quantity,Unitful.FreeUnits},a::AbstractUnitfulVector) = a*b
# Need to test next line
#*(a::AbstractUnitfulVector,b::Number) = a*Quantity(b,unit(1.0))
Base.:*(a::AbstractUnitfulVector,b::Number) = DimensionalData.rebuild(a,parent(a)*b)
Base.:*(b::Number,a::AbstractUnitfulVector) = a*b

# (matrix/vector)-(matrix/vector) multiplication when inexact handled here
function Base.:*(A::AbstractUnitfulVecOrMat,B::AbstractUnitfulVecOrMat)
    if exact(A) && exact(B)
        return DimensionalData._rebuildmul(A,B)
    elseif unitdomain(A) ∥ unitrange(B)
        return DimensionalData._rebuildmul(convert_unitdomain(A,unitrange(B)),B)
    else
        error("unitdomain(A) and unitrange(B) not parallel")
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

"""
     function left divide

     Left divide of Multipliable Matrix.
     Reverse mapping from unitdomain to range.
     Is `exact` if input is exact.
"""
function (\)(A::AbstractUnitfulMatrix,b::AbstractUnitfulVector)
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
function (\)(A::AbstractUnitfulMatrix,B::AbstractUnitfulMatrix)
    if exact(A)
        DimensionalData.comparedims(first(dims(A)), first(dims(B)); val=true)
        return rebuild(A,parent(A)\parent(B),(last(dims(A)),last(dims(B)))) #,exact = (exact(A) && exact(B)))
    elseif ~exact(A) && (unitrange(A) ∥ unitrange(B))
        Anew = convert_unitrange(A,unitrange(B)) 
        return rebuild(Anew,parent(Anew)\parent(B),(last(dims(Anew)),last(dims(B))))
    else
        error("UnitfulLinearAlgebra.mldivide: Dimensions of Unitful Matrices A and b not compatible")
    end
end

#function (\)(F::LU{T,AbstractMultipliableMatrix{T},Vector{Int64}}, B::AbstractVector) where T<:Number
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
