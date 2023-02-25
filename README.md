# UnitfulLinearAlgebra

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ggebbie.github.io/UnitfulLinearAlgebra.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ggebbie.github.io/UnitfulLinearAlgebra.jl/dev/)
[![Build Status](https://github.com/ggebbie/UnitfulLinearAlgebra.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ggebbie/UnitfulLinearAlgebra.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ggebbie/UnitfulLinearAlgebra.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ggebbie/UnitfulLinearAlgebra.jl)

Low-cost linear algebra functions for matrices with units

## Usage: Algebraic interpretation of unitful matrices

1. Convert a matrix of quantities to a UnitfulMatrix using the `UnitfulMatrix` constructor.
2. Do linear algebra operations like `inv`, `svd`, `cholesky`, `\`, `eigen`, and more.

```julia
import Pkg; Pkg.add(url="https://github.com/ggebbie/UnitfulLinearAlgebra.jl")
ENV["UNITFUL_FANCY_EXPONENTS"] = true
using UnitfulLinearAlgebra
using Unitful
using LinearAlgebra
m = u"m"
s = u"s"
K = u"K"
m² = u"m^2"

# example: fit a polynomial of order k-1 to data points at t = 0,1,2
y = [3,4,5]m 
ỹ = y .+ randn()m # contaminated observations
y̆ = UnitfulMatrix(ỹ)
k = length(y)
t = [0,1,2]s

julia> # Vandermonde matrix with right units
       E = hcat(ones(k),t.*ones(k),t.^2 .*ones(k))
3×3 Matrix{Quantity{Float64}}:
 1.0  0.0 s  0.0 s²
 1.0  1.0 s  1.0 s²
 1.0  2.0 s  4.0 s²

julia> # use UnitfulMatrix object
       F = UnitfulMatrix(E)
3×3 UnitfulMatrix{Float64,2}
 1.0  0.0 s  0.0 s²
 1.0  1.0 s  1.0 s²
 1.0  2.0 s  4.0 s²

julia> describe(F)
"Left Uniform Square Matrix"

julia> x̃ = F\y̆
3-element UnitfulMatrix{Float64,1}
          2.14233 m
          1.0 m s⁻¹
 -2.22045e-16 m s⁻²

julia> inv(F)
3×3 UnitfulMatrix{Float64,2}
      1.0       0.0      -0.0
 -1.5 s⁻¹   2.0 s⁻¹  -0.5 s⁻¹
  0.5 s⁻²  -1.0 s⁻²   0.5 s⁻²

julia> describe(inv(F))
"Right Uniform Square Matrix"

julia> x̆ = inv(F)*y̆
3-element UnitfulMatrix{Float64,1}
 2.14233 m
 1.0 m s⁻¹
 0.0 m s⁻²
```

## Usage: Geometric interpretation of unitful matrices

1. Define a concrete type of an `UnitfulMatrix` by using the `UnitfulMatrix` constructor that requires information about the matrix's numerical values, unit range, and unit domain. This matrix is defined to have `exact=true`, it only operates on vectors with units that are identical the unit domain, and it produces output with units according to the unit range.
2. Do linear algebra operations like `inv`, `svd`, `cholesky`, `\`, `eigen`, and more.

## Motivation

Julia provides a great environment for defining quantities with units and doing calculations with those unitful quantities  (`Unitful.jl`), where plots (`UnitfulRecipes.jl`, now deprecated), LaTeX output (`UnitfulLatexify.jl`), and educational notebooks (`Pluto.jl`) can be automatically added to this unitful workflow. Common linear algebra functions, such as matrix left divide, do not appear to be fully implemented, however. This package aims to extend common functions like `inv`, `(\)`, `svd`, `lu`, `cholesky` and `eigen` to unitful matrices and vectors.

## Approach

George W. Hart lays it out in "Multidimensional Analysis: Algebras and Systems for Science and Engineering" (Springer-Verlag, 1995). His approach fits nicely into Julia's type system and multiple dispatch. This packages aims to return objects defined by the `LinearAlgebra` package but extended for use with `AbstractUnitfulVecOrMat`s. 

Due to Unitful quantities that change types, it is not always easy to properly compose UnitfulLinearAlgebra functions with Unitful and LinearAlgebra functions. Also, some LinearAlgebra functions like `eigen` are highly restricted with unitful matrices. The `SVD` factorization object also makes assumptions that do not hold for matrices with units. Some compromises and design choices are necessary.

## Performance

Including units on matrices would seem to require twice the overhead of a dimensionless (purely numerical) matrix. Matrices that arise in scientific and engineering problems typically have a specific structure of units that permits the matrix to be used in linear algebraic operations. Such "multipliable matrices" have at most $m+n-1$ degrees of dimensional freedom, rather than the $m×n$ degrees of numerical freedom. Conceptually it is possible to store this information in an efficient way and to keep the overhead in propagating units low. 

Benchmarks with a random $1000 × 1000$ dimensional (unitful) matrix show that the LU decomposition currently has very little overhead when units are included.

```julia
julia> A
1000×1000 UnitfulMatrix{Float64,2}
  0.345829 m² kg⁻¹   0.252412 m² J⁻²  …  0.908051 m² J⁻¹   0.41719 m² J⁻¹
   0.447032 K kg⁻¹    0.177407 K J⁻²      0.336089 K J⁻¹   0.663898 K J⁻¹
   0.752077 s kg⁻¹    0.921767 s J⁻²      0.065284 s J⁻¹   0.139397 s J⁻¹
   0.314864 m kg⁻¹    0.766019 m J⁻²      0.524938 m J⁻¹   0.101552 m J⁻¹
 0.0551535 m² kg⁻¹   0.808757 m² J⁻²      0.14722 m² J⁻¹  0.856775 m² J⁻¹
                 ⋮                    ⋱                   
   0.98085 m² kg⁻¹   0.153549 m² J⁻²  …  0.613259 m² J⁻¹  0.599306 m² J⁻¹
   0.630143 m kg⁻¹    0.727533 m J⁻²      0.511823 m J⁻¹  0.0876397 m J⁻¹
   0.628102 K kg⁻¹   0.0429165 K J⁻²      0.346418 K J⁻¹   0.798539 K J⁻¹
   0.542709 s kg⁻¹     0.95824 s J⁻²       0.55042 s J⁻¹   0.216106 s J⁻¹
   0.855815 s kg⁻¹    0.992885 s J⁻²       0.18751 s J⁻¹   0.203458 s J⁻¹

julia> Ap = parent(A);
julia> @btime lu(Ap)
  6.769 ms (4 allocations: 7.64 MiB)
LU{Float64, Matrix{Float64}, Vector{Int64}}
L factor:
1000×1000 Matrix{Float64}:
 1.0          0.0         0.0         …   0.0        0.0       0.0       0.0
 9.04985e-5   1.0         0.0             0.0        0.0       0.0       0.0
 0.20055     -0.147565    1.0             0.0        0.0       0.0       0.0
 0.681351    -0.44258    -0.0026182       0.0        0.0       0.0       0.0
 0.778278     0.190795   -0.371202        0.0        0.0       0.0       0.0
 ⋮                                    ⋱                                  
 0.712105    -0.13409    -0.233651    …   0.0        0.0       0.0       0.0
 0.334072     0.125922    0.106544        1.0        0.0       0.0       0.0
 0.171789     0.828351    0.109153       -0.417991   1.0       0.0       0.0
 0.0751325    0.283581   -0.142071       -0.624394  -0.800151  1.0       0.0
 0.503542    -0.346511    0.132579       -0.952701  -0.779225  0.635828  1.0
U factor:
1000×1000 Matrix{Float64}:
 0.996953  0.808888  0.68578   0.243008   …   0.156691    0.193018     0.942043
 0.0       0.96316   0.702714  0.297232       0.127027    0.318024     0.59576
 0.0       0.0       0.930526  0.0308314      0.491912    0.984531     0.396951
 0.0       0.0       0.0       0.934871       0.349586    0.934023     0.441365
 0.0       0.0       0.0       0.0            0.386726    0.404526    -0.893423
 ⋮                                        ⋱                           
 0.0       0.0       0.0       0.0        …   0.859994   -4.79467     -7.48615
 0.0       0.0       0.0       0.0           -2.32847    -7.30684     -7.91318
 0.0       0.0       0.0       0.0            2.62589    -0.246053    -1.07557
 0.0       0.0       0.0       0.0            0.0        -5.23589     -7.14022
 0.0       0.0       0.0       0.0            0.0         0.0          1.37779

julia> 

julia> @btime lu(A)
  6.739 ms (4 allocations: 7.64 MiB)
LU{Float64, UnitfulMatrix{Float64, 2, Tuple{UnitfulLinearAlgebra.Units{DimensionalData.Dimensions.LookupArrays.Categorical{Unitful.FreeUnits{N, D, nothing} where {N, D}, Vector{Unitful.FreeUnits{N, D, nothing} where {N, D}}, DimensionalData.Dimensions.LookupArrays.Unordered, DimensionalData.Dimensions.LookupArrays.NoMetadata}}, UnitfulLinearAlgebra.Units{DimensionalData.Dimensions.LookupArrays.Categorical{Unitful.FreeUnits{N, D, nothing} where {N, D}, Vector{Unitful.FreeUnits{N, D, nothing} where {N, D}}, DimensionalData.Dimensions.LookupArrays.Unordered, DimensionalData.Dimensions.LookupArrays.NoMetadata}}}, Tuple{}, Matrix{Float64}, DimensionalData.NoName, DimensionalData.Dimensions.LookupArrays.NoMetadata}, Vector{Int64}}
L factor:
1000×1000 UnitfulMatrix{Float64,2}
              1.0        0.0 m² K⁻¹        0.0 m² s⁻¹  …  0.0 m² s⁻¹  0.0 m² s⁻¹
 9.04985e-5 K m⁻²               1.0         0.0 K s⁻¹      0.0 K s⁻¹   0.0 K s⁻¹
    0.20055 s m⁻²   -0.147565 s K⁻¹               1.0            0.0         0.0
     0.681351 m⁻¹    -0.44258 m K⁻¹  -0.0026182 m s⁻¹      0.0 m s⁻¹   0.0 m s⁻¹
         0.778278   0.190795 m² K⁻¹  -0.371202 m² s⁻¹     0.0 m² s⁻¹  0.0 m² s⁻¹
                ⋮                                      ⋱              
         0.712105   -0.13409 m² K⁻¹  -0.233651 m² s⁻¹  …  0.0 m² s⁻¹  0.0 m² s⁻¹
     0.334072 m⁻¹    0.125922 m K⁻¹    0.106544 m s⁻¹      0.0 m s⁻¹   0.0 m s⁻¹
   0.171789 K m⁻²          0.828351    0.109153 K s⁻¹      0.0 K s⁻¹   0.0 K s⁻¹
  0.0751325 s m⁻²    0.283581 s K⁻¹         -0.142071            1.0         0.0
   0.503542 s m⁻²   -0.346511 s K⁻¹          0.132579       0.635828         1.0
U factor:
1000×1000 UnitfulMatrix{Float64,2}
 0.996953 m² kg⁻¹  0.808888 m² J⁻²  …    0.193018 m² J⁻¹   0.942043 m² J⁻¹
       0.0 K kg⁻¹    0.96316 K J⁻²        0.318024 K J⁻¹     0.59576 K J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²        0.984531 s J⁻¹    0.396951 s J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²        0.934023 m J⁻¹    0.441365 m J⁻¹
      0.0 m² kg⁻¹       0.0 m² J⁻²       0.404526 m² J⁻¹  -0.893423 m² J⁻¹
                ⋮                   ⋱                     
      0.0 m² kg⁻¹       0.0 m² J⁻²  …    -4.79467 m² J⁻¹   -7.48615 m² J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²        -7.30684 m J⁻¹    -7.91318 m J⁻¹
       0.0 K kg⁻¹        0.0 K J⁻²       -0.246053 K J⁻¹    -1.07557 K J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²        -5.23589 s J⁻¹    -7.14022 s J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²             0.0 s J⁻¹     1.37779 s J⁻¹
```

---

*This package was generated using PkgTemplates.jl.*
