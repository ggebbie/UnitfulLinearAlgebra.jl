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
export getindexqty
#export getindex, setindex!,
export size, similar
export convert_unitrange, convert_unitdomain
#export convert_unitrange!, convert_unitdomain!
export exact, multipliable, dimensionless, endomorphic
export svd, dsvd
    export transpose
export unitrange, unitdomain
export identitymatrix
export describe, trace
#export lu, det, trace, diag, diagm
#export Diagonal, (\), cholesky
#export eigen, isposdef, inv
#export show, vcat, hcat #, rebuild #, rebuildsliced

# imported some troublesome methods
import Base: size, (\), getproperty
import LinearAlgebra: inv
#, det, lu,
#    svd, getproperty, eigen, isposdef,
#    diag, diagm, Diagonal, cholesky
#import Base:(~), (*), (+), (-), (\), getindex, setindex!,
#    size, range, transpose, similar, show, vcat, hcat

import DimensionalData: @dim, dims, DimArray, AbstractDimArray, NoName, NoMetadata, format, print_name

@dim Units "units"

 # constructor for streamlined struct based on DimensionalData.DimArray
include("UnitfulMatrix.jl")

 # constructor for expanded struct based on DimensionalData.DimArray
#include("UnitfulDimMatrix.jl")

include("multipliablematrices.jl") # new methods

# Extend Base methods
include("base.jl")

# Extend LinearAlgebra methods
# underscore to differentiate from this package
include("linear_algebra.jl") 

# a new struct and methods
# Dimensional SVD: an underappreciated concept
include("dsvd.jl") 

end
