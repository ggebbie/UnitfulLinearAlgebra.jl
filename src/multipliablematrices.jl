
"""
    function similarity(a,b)::Bool

    Dimensional similarity of vectors, a binary relation
    Read "a has the same dimensional form as b"
    `a` and `b` may still have different units.
    A stronger condition than being parallel.
    pp. 184, Hart
"""
 similarity(a,b)::Bool = isequal(dimension.(a),dimension.(b))

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
        return rebuild(A, parent(A), (newrange, newdomain), true)
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

# these dummy functions are needed for the interface for DimensionalData. They are important for matrix slices. Would be nice if they were not needed.
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

"""
    function getindexqty

    Get entry value of matrix including units.
    Note: Calling B::UnitfulMatrix[i,j] doesn't currently return the units.
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
    function identitymatrix(dimrange)

    Input: dimensional (unit) range.
    `A + I` only defined when `endomorphic(A)=true`
    When accounting for units, there are many identity matrices.
    This function returns a particular identity matrix
    defined by its dimensional range.
    Hart, pp. 200.

    Maybe change the name to UnitfulI?
"""
identitymatrix(dimrange) = UnitfulMatrix(I(length(dimrange)),(dimrange,dimrange),exact=true)

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

## start of UnitfulDimMatrix methods

"""
    function singular
    was same as ULA.singular, but I was getting singular on matrices that aren't actually
"""
#singular(A::AbstractUnitfulDimMatrix) = iszero(ustrip(det(A)))
singular(A::AbstractUnitfulDimMatrix) = rank(parent(A)) == max(size(parent(A))...)
