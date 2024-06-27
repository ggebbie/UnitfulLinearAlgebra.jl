module UnitfulLinearAlgebra

using Unitful
using LinearAlgebra
using DimensionalData
using Statistics

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
export exact, multipliable, dimensionless, unitless, endomorphic
export svd, dsvd
export transpose
export unitrange, unitdomain, unitdims
export identitymatrix
export describe, trace
#export latexify

# imported some troublesome methods
import Base: (\)
import LinearAlgebra: inv
import DimensionalData: @dim, dims, DimArray, AbstractDimArray, NoName, NoMetadata, format, print_name
#import Latexify: latexify

@dim Units "units"

abstract type AbstractUnitfulType{T,N,D,A} <: AbstractDimArray{T,N,D,A} end

# constructor for streamlined struct based on DimensionalData.DimArray
include("UnitfulMatrix.jl")

# another constructor for expanded struct based on DimensionalData.DimArray
include("UnitfulDimMatrix.jl")

# new methods
include("multipliable_matrices.jl")

# Extend Base methods
include("base.jl")

# Extend Statistics methods
include("statistics.jl")

# Extend LinearAlgebra methods
# underscore to differentiate from this package
include("linear_algebra.jl") 

# a new struct and methods
# Dimensioned SVD: an underappreciated concept
include("dsvd.jl") 

end
