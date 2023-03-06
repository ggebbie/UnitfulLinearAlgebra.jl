module UnitfulLinearAlgebra

using Unitful
using LinearAlgebra
using SparseArrays
using DimensionalData

export UnitfulMatrix, UnitfulDimMatrix
export AbstractUnitfulVecOrMat, AbstractUnitfulDimVecOrMat
export DSVD
export similarity, ∥, parallel
export uniform, left_uniform, right_uniform
export square, squarable, singular, unit_symmetric
export invdimension, dottable
export getindexqty
export size, similar
export convert_unitrange, convert_unitdomain
export exact, multipliable, dimensionless, endomorphic
export svd, dsvd
export transpose
export unitrange, unitdomain, unitdims
export identitymatrix
export describe, trace

# imported some troublesome methods
import Base: (\)
import LinearAlgebra: inv
import DimensionalData: @dim, dims, DimArray, AbstractDimArray, NoName, NoMetadata, format, print_name

@dim Units "units"

abstract type AbstractUnitfulType{T,N,D,A} <: AbstractDimArray{T,N,D,A} end

# constructor for streamlined struct based on DimensionalData.DimArray
include("UnitfulMatrix.jl")

# constructor for expanded struct based on DimensionalData.DimArray
include("UnitfulDimMatrix.jl")

# new methods
include("multipliablematrices.jl")

# Extend Base methods
include("base.jl")

# Extend LinearAlgebra methods
# underscore to differentiate from this package
include("linear_algebra.jl") 

# a new struct and methods
# Dimensional SVD: an underappreciated concept
include("dsvd.jl") 

end
