module UnitfulLinearAlgebra

using Unitful
using LinearAlgebra
using SparseArrays
using DimensionalData

export UnitfulMatrix, AbstractUnitfulVecOrMat
export DSVD
export similarity, ∥, parallel
export uniform, left_uniform, right_uniform
export square, squarable, singular, unit_symmetric
export invdimension, dottable
export getindex, setindex!, size, similar
export convert_unitrange, convert_unitdomain
#export convert_unitrange!, convert_unitdomain!
export exact, multipliable, dimensionless, endomorphic
export svd, dsvd
export eigen, isposdef, inv, transpose
export unitrange, unitdomain
export lu, det, trace, diag, diagm
export Diagonal, (\), cholesky
export identitymatrix, show, vcat, hcat #, rebuild #, rebuildsliced
export describe

import LinearAlgebra: inv, det, lu,
    svd, getproperty, eigen, isposdef,
    diag, diagm, Diagonal, cholesky
import Base:(~), (*), (+), (-), (\), getindex, setindex!,
    size, range, transpose, similar, show, vcat, hcat

import DimensionalData: @dim, dims, DimArray, AbstractDimArray, NoName, NoMetadata, format, print_name

@dim Units "units"

include("UnitfulMatrix.jl")

"""
    function describe(A::UnitfulMatrix)

     Information regarding the type of multipliable matrix.
"""
function describe(A::UnitfulMatrix)
    matrixtype = ""
    
    dimensionless(A) && ( matrixtype *= "Dimensionless ")

    # check degree of uniformity
    #if uniform(unitrange) && uniform(unitdomain)
    if uniform(A)
        matrixtype *= "Uniform "
    #elseif uniform(unitrange)
    elseif left_uniform(A)
        matrixtype *= "Left Uniform "
    #elseif uniform(unitdomain)
    elseif right_uniform(A)
        matrixtype *= "Right Uniform "
    end

    if square(A)
        #if unitrange == unitdomain
        if endomorphic(A)
            matrixtype *= "Endomorphic "
        end
        
        #if unitrange ∥ unitdomain
        if squarable(A)
            matrixtype *= "Squarable "
        end
        
        #if unitrange ∥ 1 ./unitdomain
        if unit_symmetric(A)
            matrixtype *= "Unit Symmetric "
        end

        if ~endomorphic(A) && ~squarable(A) && ~unit_symmetric(A)
            matrixtype *= "Square " # a fallback description
        end
    end

    return matrixtype*"Matrix"
end

"""
    function multipliable(A)::Bool

    Is an array multipliable?
    It requires a particular structure of the units/dimensions in the array. 
"""
multipliable(A::Matrix) = ~isnothing(UnitfulMatrix(A))
multipliable(A::UnitfulMatrix) = true
    
"""
    function endomorphic(A)::Bool

    Endomorphic matrices have a particular structure
     of the units/dimensions in the array.
    It maps dimensioned vector space to itself.
    Equivalent unit (dimensional) range and domain.

"""
endomorphic(A::Matrix) = endomorphic(UnitfulMatrix(A))
endomorphic(A::UnitfulMatrix) = isequal(unitdomain(A),unitrange(A))
endomorphic(A::Number) = dimensionless(A) # scalars must be dimensionless to be endomorphic

# Need to test this
similar(A::AbstractUnitfulVecOrMat{T}) where T <: Number =
       DimensionalData.rebuild(A, zeros(A))
    #UnitfulMatrix(Matrix{T}(undef,size(A)),unitrange(A),unitdomain(A);exact=exact(A))

"""
    function getindexqty

    Get entry value of matrix including units.
"""
getindexqty(A::AbstractUnitfulMatrix,i::Int,j::Int) = Quantity.(parent(A)[i,j],unitrange(A)[i]./unitdomain(A)[j]) 

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
function setindex!(A::AbstractUnitfulMatrix,v::Quantity,i::Int,j::Int) 
    if unit(v) == unitrange(A)[i]./unitdomain(A)[j]
        A[i,j] = ustrip(v)
    else error("new value has incompatible units")
    end
end

"""
    function Matrix(A::MultipliableMatrix)

    Expand A into array form
    Useful for tests, display
    pp. 193, Hart
"""
function Matrix(A::T) where T<: AbstractUnitfulMatrix

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
function Matrix(a::AbstractUnitfulVector) 

    M, = size(a)
    T2 = eltype(parent(a))
    b = Vector{Quantity{T2}}(undef,M)
    for m = 1:M
        b[m] = Quantity.(getindex(a,m),unitrange(a)[m])
    #    b[m] = Quantity.(getindex(a,m),unitrange(a)[m])
    end
    #b= Quantity.(parent(a),unitrange(a)[1][:])
    return b
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
*(A::AbstractUnitfulMatrix,b::Quantity) = DimensionalData.rebuild(A,parent(A)*ustrip(b),(Units(unitrange(A).*unit(b)),unitdomain(A)))
*(A::AbstractUnitfulMatrix,b::Unitful.FreeUnits) = DimensionalData.rebuild(A,parent(A),(Units(unitrange(A).*b),unitdomain(A)))
*(b::Union{Quantity,Unitful.FreeUnits},A::AbstractUnitfulMatrix) = A*b
*(A::AbstractUnitfulMatrix,b::Number) = DimensionalData.rebuild(A,parent(A)*b)
*(b::Number,A::AbstractUnitfulMatrix) = A*b

# vector-scalar multiplication
*(a::AbstractUnitfulVector,b::Quantity) = DimensionalData.rebuild(a,parent(a)*ustrip(b),(Units(unitrange(a).*unit(b)),))
*(a::AbstractUnitfulVector,b::Unitful.FreeUnits) = DimensionalData.rebuild(a,parent(a),(Units(unitrange(a).*b),))
*(b::Union{Quantity,Unitful.FreeUnits},a::AbstractUnitfulVector) = a*b
# Need to test next line
#*(a::AbstractUnitfulVector,b::Number) = a*Quantity(b,unit(1.0))
*(a::AbstractUnitfulVector,b::Number) = DimensionalData.rebuild(a,parent(a)*b)
*(b::Number,a::AbstractUnitfulVector) = a*b

# (matrix/vector)-(matrix/vector) multiplication when inexact handled here
function *(A::AbstractUnitfulVecOrMat,B::AbstractUnitfulVecOrMat)
    if exact(A)
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
function +(A::AbstractUnitfulVecOrMat,B::AbstractUnitfulVecOrMat) #where T1 where T2
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
function -(A::AbstractUnitfulVecOrMat{T1},B::AbstractUnitfulVecOrMat{T2}) where T1 where T2
    bothexact = exact(A) && exact(B)
    if (unitrange(A) == unitrange(B) && unitdomain(A) == unitdomain(B)) ||
       ( unitrange(A) ∥ unitrange(B) && unitdomain(A) ∥ unitdomain(B) && ~bothexact)
        return DimensionalData.rebuild(A,parent(A)-parent(B),(unitrange(A),unitdomain(A))) # takes exact(A) but should be bothexact 
    else
        error("matrices not dimensionally conformable for subtraction")
    end
end

"""
    function lu(A::AbstractUnitfulVecOrMat{T})

    Extend `lu` factorization to AbstractMultipliableMatrix.
    Related to Gaussian elimination.
    Store dimensional domain and range in "factors" attribute
    even though this is not truly a MultipliableMatrix.
    Returns `LU` type in analogy with `lu` for unitless matrices.
    Based on LDU factorization, Hart, pp. 204.
"""
function lu(A::AbstractUnitfulVecOrMat)
    F̂ = lu(parent(A))
    factors = rebuild(A,parent(F̂.factors),(unitrange(A),unitdomain(A)))
    #factors = MMatrix(F̂.factors, unitrange(A), unitdomain(A), exact=exact(A))
    F = LU(factors,F̂.ipiv,F̂.info)
    return F
end

"""
    function getproperty(F::LU{T,<:AbstractMultipliableMatrix,Vector{Int64}}, d::Symbol) where T

    Extend LinearAlgebra.getproperty for AbstractUnitfulVecOrMat.

    LU factorization stores L and U together.
    Extract L and U while keeping consistent
    with dimensional domain and range.
"""
function getproperty(F::LU{T,<:AbstractUnitfulVecOrMat,Vector{Int64}}, d::Symbol) where T
    m, n = size(F)
    if d === :L
        mmatrix = getfield(F, :factors)
        numbers = parent(mmatrix)
        #numbers = getfield(mmatrix,:numbers)
        # add ustrip to get numerical values
        Lnum = tril!(numbers[1:m, 1:min(m,n)])
        for i = 1:min(m,n); Lnum[i,i] = one(T); end
        return rebuild(mmatrix,Lnum,(unitrange(mmatrix),unitrange(mmatrix)))
    elseif d === :U
        mmatrix = getfield(F, :factors)
        numbers = parent(mmatrix)
        Unum = triu!(numbers[1:min(m,n), 1:n])
        return rebuild(mmatrix,Unum,(unitrange(mmatrix),unitdomain(mmatrix)))
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
            Δdim = dimension.(a)./dimension.(b)
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
function parallel(a::AbstractUnitfulVector,b::AbstractUnitfulVector)::Bool
    if isequal(length(a),length(b))
        if length(a) == 1
            return true
        else
            Δdim = dimension(a)./dimension(b) # inconsistent function call
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

# not consistent in that it should be an element-wise function
Unitful.dimension(a::AbstractUnitfulVector) = dimension.(unitrange(a)) 

"""
    function uniform(a)

    Is the dimension of this quantity uniform?

    There must be a way to inspect the Unitful type to answer this.
    Uniform matrix: All entries have the same units
"""
uniform(a::T) where T <: Number = true # all scalars by default
function uniform(a::Union{Vector,<:DimensionalData.Dimension}) 
    dima = dimension.(a)
    for dd = 2:length(dima)
        if dima[dd] ≠ dima[1]
            return false
        end
    end
    return true
end
function uniform(A::Matrix)
    B = UnitfulMatrix(A)
    isnothing(B) ? false : uniform(B)
end
uniform(A::UnitfulMatrix) = left_uniform(A) && right_uniform(A)

"""
    function left_uniform(A)

    Definition: uniform unitrange of A
    Left uniform matrix: output of matrix has uniform units
"""
left_uniform(A::UnitfulMatrix) = uniform(unitrange(A)) ? true : false
function left_uniform(A::Matrix)
    B = UnitfulMatrix(A)
    isnothing(B) ? false : left_uniform(B)
end

"""
    function right_uniform(A)

    Does the unitdomain of A have uniform dimensions?
    Right uniform matrix: input of matrix must have uniform units
"""
right_uniform(A::UnitfulMatrix) = uniform(unitdomain(A)) ? true : false
function right_uniform(A::Matrix)
    B = UnitfulMatrix(A)
    isnothing(B) ? false : right_uniform(B)
end

"""
     function dimensionless(A)

     Not all dimensionless matrices have
     dimensionless domain and range.
"""
dimensionless(A::Union{Matrix,UnitfulMatrix}) = uniform(A) && dimension(A[1,1]) == NoDims
dimensionless(A::T) where T <: Number = (dimension(A) == NoDims)
function dimensionless(A::AbstractMatrix)
    B = UniformMatrix(A)
    isnothing(B) ? false : dimensionless(B) # fallback
end

"""
    function square(A)

    size(A)[1] == size(A)[2]
"""
square(A::T) where T <: AbstractMatrix = isequal(size(A)[1],size(A)[2]) 

"""
    function squarable(A::Matrix)

    A squarable matrix is one where 𝐀² is defined.
    Unit (dimensional) range and domain are parallel.
    Key for solving difference and differential equations.
    Have eigenstructure. 
"""
squarable(A::UnitfulMatrix) = (unitdomain(A) ∥ unitrange(A))
function squarable(A::AbstractMatrix)
    B = UnitfulMatrix(A)
    isnothing(B) ? false : squarable(B) # fallback
end

"""
    function unit_symmetric(A::Matrix)

    `UnitSymmetricMatrix`s have units that are symmetric about the main diagonal and define weighted norms. 
    Definition: inverse dimensional range and dimensional domain are parallel.
    Called "dimensionally symmetric" by Hart, 1995.
"""
unit_symmetric(A::UnitfulMatrix) = (unitrange(A) ∥ unitdomain(A).^-1)
function unit_symmetric(A::AbstractMatrix)
    B = UnitfulMatrix(A)
    isnothing(B) ? false : unit_symmetric(B) # fallback
end

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
function convert_unitdomain(A::AbstractUnitfulVecOrMat, newdomain::Units) 
    if unitdomain(A) ∥ newdomain
        #shift = newdomain./unitdomain(A)
        #newrange = unitrange(A).*shift
        newrange = Units(unitrange(A).*(newdomain[1]/unitdomain(A)[1]))
        return rebuild(A, parent(A), (newrange, newdomain))
        #B = BestMultipliableMatrix(A.numbers,newrange,newdomain,exact=true)
    else
        error("New unit domain not parallel to unit domain of Multipliable Matrix")
    end
end

# """
#     function convert_unitdomain!(A, newdomain)

#     In-place conversion of unit (dimensional) domain.
#     Matrix Type not permitted to change.
# """
# function convert_unitdomain!(A::AbstractMultipliableMatrix, newdomain::Vector)
#     if unitdomain(A) ∥ newdomain
#         shift = newdomain[1]./unitdomain(A)[1]
#         # caution: not all matrices have this attribute
#         if hasproperty(A,:unitdomain)
#             for (ii,vv) in enumerate(A.unitdomain)
#                 A.unitdomain[ii] *= shift
#             end
#         end
#         if hasproperty(A,:unitrange)
#             for (ii,vv) in enumerate(A.unitrange)
#                 A.unitrange[ii] *= shift
#             end
#         end
#         # make sure the matrix is now exact
#         #A.exact = true # immutable so not possible
#     else
#         error("New domain not parallel to domain of Multipliable Matrix")
#     end
# end

"""
    function convert_unitrange(A, newrange)

    When using the geometric interpretation of matrices,
    it is useful to convert the dimensional range of the
    matrix to match the desired output of multiplication.
    Here we set the matrix to `exact=true` after this step.
    Permits MatrixType to change.
"""
function convert_unitrange(A::AbstractUnitfulMatrix, newrange::Units) 
    if unitrange(A) ∥ newrange
        #shift = newdomain./unitdomain(A)
        #newrange = unitrange(A).*shift
        newdomain = Units(unitdomain(A).*(newrange[1]/unitrange(A)[1]))
        B = rebuild(A, parent(A), (newrange, newdomain))
    else
        error("New unit domain not parallel to unit domain of Multipliable Matrix")
    end
end

# """
#     function convert_unitrange!(A, newrange)

#     In-place conversion of unit (dimensional) range.
#     Matrix Type not permitted to change.
# """
# function convert_unitrange!(A::AbstractMultipliableMatrix, newrange::Vector)
#     if unitrange(A) ∥ newrange
#         shift = newrange[1]./unitrange(A)[1]
#         # caution: not all matrices have this attribute
#         if hasproperty(A,:unitdomain)
#             for (ii,vv) in enumerate(A.unitdomain)
#                 A.unitdomain[ii] *= shift
#             end
#         end
#         if hasproperty(A,:unitrange)
#             for (ii,vv) in enumerate(A.unitrange)
#                 A.unitrange[ii] *= shift
#             end
#         end
#         #A.exact = true , immutable
#      else
#          error("New range not parallel to range of Multipliable Matrix")
#      end
# end
# function convert_unitrange!(A::AbstractUnitfulMatrix, newrange::Vector)
#     if unitrange(A) ∥ newrange
#         shift = newrange[1]./unitrange(A)[1]
#         for (ii,vv) in enumerate(A.dims[2])
#             A.dims[2][ii] *= shift
#         end
#         for (ii,vv) in enumerate(A.dims[1])
#             A.dims[1][ii] *= shift
#         end
#      else
#          error("New range not parallel to range of Unitful Matrix")
#      end
# end

"""
    function exact(A)

-    `exact=true`: geometric interpretation of unitdomain and unitrange
-    `exact=false`: algebraic interpretation
"""
exact(A::UnitfulMatrix) = A.exact

DimensionalData.name(A::AbstractUnitfulVecOrMat) = ()
DimensionalData.metadata(A::AbstractUnitfulVecOrMat) = NoMetadata()
DimensionalData.refdims(A::AbstractUnitfulVecOrMat) = ()

# convert(::Type{AbstractMatrix{T}}, A::AbstractMultipliableMatrix) where {T<:Number} = convert(AbstractMultipliableMatrix{T}, A)
# convert(::Type{AbstractArray{T}}, A::AbstractMultipliableMatrix) where {T<:Number} = convert(AbstractMultipliableMatrix{T}, A)
# #convert(::Type{AbstractArray{T}}, S::AbstractToeplitz) where {T<:Number} = convert(AbstractToeplitz{T}, S)

"""
    function unitdomain(A)

    Find the dimensional (unit) domain of a matrix
"""
unitdomain(A::AbstractUnitfulMatrix) = last(dims(A))
# this line may affect matrix multiplication
unitdomain(A::AbstractUnitfulVector) = Units([unit(1.0)]) # kludge for a nondimensional scalar 
#unitdomain(A::AbstractUnitfulVector) = Unitful.FreeUnits{(), NoDims, nothing}() # nondimensional scalar 

"""
    function unitrange(A)

    Find the dimensional (unit) range of a matrix
"""
unitrange(A::AbstractUnitfulVecOrMat) = first(dims(A))
#unitrange(A::AbstractUnitfulMatrix) = first(dims(A))

"""
    function transpose

    Defined by condition `A[i,j] = transpose(A)[j,i]`.
    Not analogous to function for dimensionless matrices.

    Hart, pp. 205.
"""
# Had to redefine tranpose or it was incorrect based on AbstractArray interface!
transpose(A::AbstractUnitfulMatrix) = rebuild(A,transpose(parent(A)),(Units(unitdomain(A).^-1), Units(unitrange(A).^-1)))
transpose(a::AbstractUnitfulVector) = rebuild(a,transpose(parent(a)),(Units([unit(1.0)]), Units(unitrange(a).^-1))) # kludge for unitrange of row vector

"""
    function identitymatrix(dimrange)

    Input: dimensional (unit) range.
    `A + I` only defined when `endomorphic(A)=true`
    When accounting for units, there are many identity matrices.
    This function returns a particular identity matrix
    defined by its dimensional range.
    Hart, pp. 200.                             
"""
identitymatrix(dimrange) = UnitfulMatrix(I(length(dimrange)),(dimrange,dimrange),exact=true)

"""
     function inv

     Inverse of Multipliable Matrix.
     Only defined for nonsingular matrices.
     Inverse reverses mapping from unitdomain to range.
     Is `exact` if input is exact.

    Hart, pp. 205. 
"""
inv(A::AbstractUnitfulMatrix) = ~singular(A) ? rebuild(A,inv(parent(A)),(unitdomain(A),unitrange(A))) : error("matrix is singular")

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

"""
    function det

    Unitful matrix determinant.
"""
function det(A::AbstractUnitfulMatrix) 
    if square(A)
        detunit = prod([unitrange(A)[i]/unitdomain(A)[i] for i = 1:size(A)[1]])
        return Quantity(det(parent(A)),detunit)
    else
        error("Determinant requires square matrix")
    end
end

"""
    function singular(A)

    Is a square matrix singular? If no, then it is invertible.
"""
singular(A::AbstractUnitfulMatrix) = iszero(ustrip(det(A)))

"""
    function trace(A)

    Trace = sum of diagonal elements of a square matrix
"""
trace(A::AbstractUnitfulMatrix) = sum(diag(parent(A))).*(unitrange(A)[1]/unitdomain(A)[1])

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
function eigen(A::AbstractUnitfulMatrix;permute::Bool=true, scale::Bool=true, sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) 
    if squarable(A) 
        F = LinearAlgebra.eigen(parent(A), permute=permute, scale=scale, sortby=sortby)
        return Eigen(F.values.*(unitrange(A)[1]/unitdomain(A)[1]), rebuild(A,F.vectors,(unitdomain(A),Units(fill(unit(1.0),size(A,2))))))
    else
        error("UnitfulLinearAlgebra: Eigenvalue decomposition doesn't exist for for non-squarable matrices")
    end
end

"""
   Extend `isposdef` for Eigen factorizations of `MultipliableMatrix`s.
    Should the units be stripped out of the function?
    Only defined for matrices with uniform units (pp. 101, Hart, 1995). 
"""
isposdef(A::Eigen{T,V,S,U}) where {U<: AbstractVector, S<:AbstractUnitfulMatrix, V, T <: Number} = (uniform(A.vectors) && isreal(A.values)) && all(x -> x > 0, ustrip.(A.values))

"""
   Extend `inv` for Eigen factorizations of `MultipliableMatrix`s.
    Only defined for matrices with uniform units (pp. 101, Hart, 1995). 
"""
function inv(A::Eigen{T,V,S,U}) where {U <: AbstractVector, S <: AbstractUnitfulMatrix, V, T <: Number}

    if (uniform(A.vectors) && isreal(A.values))
        ur = unitrange(A.vectors)
        ud = Units(unit.(A.values))
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
function svd(A::AbstractUnitfulMatrix;full=false,alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(parent(A))) 
    if uniform(A) 
        F = svd(parent(A), full=full, alg=alg)
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
struct DSVD{T,Tr,MU<:AbstractUnitfulMatrix{T},MV<:AbstractUnitfulMatrix{T},MQY<:AbstractUnitfulMatrix{T},MQX<:AbstractUnitfulMatrix{T},C<:AbstractVector{Tr}} <: Factorization{T}
    U′::MU
    S::C
    V′⁻¹::MV
    Qy::MQY
    Qx::MQX
    function DSVD{T,Tr,MU,MV,MQY,MQX,C}(U′, S, V′⁻¹,Qy,Qx) where {T,Tr,MU<:AbstractUnitfulMatrix{T},MV<:AbstractUnitfulMatrix{T},MQY<:AbstractUnitfulMatrix{T},MQX<:AbstractUnitfulMatrix{T},C<:AbstractVector{Tr}}
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
function dsvd(A::AbstractUnitfulMatrix,Py::AbstractUnitfulMatrix,Px::AbstractUnitfulMatrix;full=false,alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(parent(A))) 

    unit_symmetric(Py) ? Qy = getproperty(cholesky(Py),:U) : error("norm matrix for range not unit symmetric")
    unit_symmetric(Px) ? Qx = getproperty(cholesky(Px),:U) : error("norm matrix for domain not unit symmetric")

    # must be more efficient way
    #A′ = Qr*(A*inv(Qd))
    # still inefficient with copy
    A′ =   copy(transpose(transpose(Qx)\transpose(Qy*A)))
    ~dimensionless(A′) && error("A′ should be dimensionless to implement `LinearAlgebra.svd`")
    F = svd(parent(A′), full=full, alg=alg)
    return DSVD(UnitfulMatrix(F.U),F.S,UnitfulMatrix(F.Vt),Qy,Qx)
end

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::DSVD{<:Any,<:Any,<:AbstractArray,<:AbstractArray,<:AbstractArray,<:AbstractArray,<:AbstractVector})
    #summary(io, F); println(io)
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
diagm(v::AbstractVector,r::Units,d::Units; exact = false) = UnitfulMatrix(spdiagm(length(r),length(d),ustrip.(v)),(r,d); exact=exact)    

# """
#     function diag(A::AbstractMultipliableMatrix)

#     Diagonal elements of matrix with units.

#     Usual `LinearAlgebra.diag` function is not working due to different type elements on diagonal
#  """
# function diag(A::AbstractMultipliableMatrix{T}) where T <: Number

#     m,n = size(A)
#     ndiag = max(m,n)
#     dimensionless(A) ? vdiag = Vector{T}(undef,ndiag) : vdiag = Vector{Quantity}(undef,ndiag)
#     for nd in 1:ndiag
#         vdiag[nd] = getindex(A,nd,nd)
#     end
#     return vdiag
# end

"""
    function cholesky(A::AbstractMultipliableMatrix)

    Cholesky decomposition extended for matrices with units.
    Requires unit (or dimensionally) symmetric matrix.
    Functions available for LinearAlgebra.Cholesky objects: `size`, `\`, `inv`, `det`, `logdet` and `isposdef`.
    Functions available for UnitfulLinearAlgebra.Cholesky objects: `size`, `det`, and `isposdef`.
"""
function cholesky(A::AbstractUnitfulMatrix)
    if unit_symmetric(A)
        C = LinearAlgebra.cholesky(parent(A))
        factors = rebuild(A,C.factors,(Units(unitdomain(A)./unitdomain(A)),unitdomain(A)))
        return Cholesky(factors,C.uplo,C.info)
    else
        error("requires unit symmetric matrix")
    end
end

function getproperty(C::Cholesky{T,<:AbstractUnitfulMatrix}, d::Symbol) where T 
    Cfactors = getfield(C, :factors)
    Cuplo    = getfield(C, :uplo)
    if d === :U
        numbers = UpperTriangular(Cuplo === LinearAlgebra.char_uplo(d) ? parent(Cfactors) : copy(transpose(parent(Cfactors))))
        return rebuild(Cfactors,numbers,(unitrange(Cfactors),unitdomain(Cfactors)))
    elseif d === :L
        numbers = LowerTriangular(Cuplo === LinearAlgebra.char_uplo(d) ? parent(Cfactors) : copy(transpose(parent(Cfactors))))
        # use transpose to get units right
        return rebuild(Cfactors,numbers,(Units(unitdomain(Cfactors).^-1),Units(unitrange(Cfactors).^-1)))
    elseif d === :UL
        (Cuplo === 'U') ? (return rebuild(Cfactors,UpperTriangular(parent(Cfactors)))) : (return rebuild(Cfactors,LowerTriangular(parent(Cfactors)),(unitdomain(Cfactors).^-1,unitrange(Cfactors).^-1)))
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
Diagonal(v::AbstractVector,r::AbstractVector,d::AbstractVector; exact = false) = ((length(r) == length(d)) && (length(v) == length(d))) ? UnitfulMatrix(LinearAlgebra.Diagonal(ustrip.(v)),(r,d); exact=exact) : error("unit range and domain do not define a square matrix")   
Diagonal(v::AbstractVector,r::Units,d::Units; exact = false) = ((length(r) == length(d)) && (length(v) == length(d))) ? UnitfulMatrix(LinearAlgebra.Diagonal(ustrip.(v)),(r,d); exact=exact) : error("unit range and domain do not define a square matrix")   

# """
#     function vcat(A,B)

#     Modeled after function `VERTICAL` (pp. 203, Hart, 1995).
# """
# function vcat(A::AbstractMultipliableMatrix,B::AbstractMultipliableMatrix)

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
# function hcat(A::AbstractMultipliableMatrix,B::AbstractMultipliableMatrix)

#     numbers = hcat(A.numbers,B.numbers)
#     shift = unitrange(A)[1]./unitrange(B)[1]
#     ud = vcat(unitdomain(A),unitdomain(B).*shift)
#     bothexact = (exact(A) && exact(B))
#     return BestMultipliableMatrix(numbers,unitrange(A),ud,exact=bothexact)
# end

end
