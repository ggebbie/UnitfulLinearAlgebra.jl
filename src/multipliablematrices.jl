
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
