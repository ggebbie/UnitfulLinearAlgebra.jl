# UnitfulLinearAlgebra

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ggebbie.github.io/UnitfulLinearAlgebra.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ggebbie.github.io/UnitfulLinearAlgebra.jl/dev/)
[![Build Status](https://github.com/ggebbie/UnitfulLinearAlgebra.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ggebbie/UnitfulLinearAlgebra.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ggebbie/UnitfulLinearAlgebra.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ggebbie/UnitfulLinearAlgebra.jl)

More linear algebra functions for matrices with units

## Usage

1. Define a concrete type of an `AbstractMultipliableMatrix` using the constructor or convert a matrix of quantities.
2. Do linear algebra operations like `inv`, `svd`, `cholesky`, `\`, `eigen`, and more.

```
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
k = length(y)
t = [0,1,2]s

julia> # Vandermonde matrix with right units
       E = hcat(ones(k),t.*ones(k),t.^2 .*ones(k))
3×3 Matrix{Quantity{Float64}}:
 1.0  0.0 s  0.0 s²
 1.0  1.0 s  1.0 s²
 1.0  2.0 s  4.0 s²

julia> # use AbstractMultipliableMatrix object
       F = BestMultipliableMatrix(E)
3×3 LeftUniformMatrix{Float64}:
 1.0  0.0 s  0.0 s²
 1.0  1.0 s  1.0 s²
 1.0  2.0 s  4.0 s²

julia> x̃ = F\ỹ
3-element Vector{Quantity{Float64}}:
 1.953135929373488 m
           1.0 m s⁻¹
          -0.0 m s⁻²

julia> inv(F)
3×3 RightUniformMatrix{Float64}:
        1.0         0.0        -0.0
 -1.5 s⁻¹     2.0 s⁻¹    -0.5 s⁻¹
  0.5 s⁻²    -1.0 s⁻²     0.5 s⁻²

julia> x̆ = inv(F)*ỹ
3-element Vector{Quantity{Float64}}:
 1.953135929373488 m
           1.0 m s⁻¹
           0.0 m s⁻²
```

## Motivation

Julia provides a great environment for defining quantities with units and doing calculations with those unitful quantities  (`Unitful.jl`), where plots (`UnitfulRecipes.jl`, now deprecated), LaTeX output (`UnitfulLatexify.jl`), and educational notebooks (`Pluto.jl`) can be automatically added to this unitful workflow. Common linear algebra functions, such as matrix left divide, do not appear to be fully implemented, however. This package aims to extend common functions like `inv`, `(\)`, `svd`, `lu`, `cholesky` and `eigen` to unitful matrices and vectors.

## Approach

George W. Hart lays it out in "Multidimensional Analysis: Algebras and Systems for Science and Engineering" (Springer-Verlag, 1995). His approach fits nicely into Julia's type system and multiple dispatch. This packages aims to return objects defined by the `LinearAlgebra` package but extended for use with `AbstractMultipliableMatrix`s. 

Due to Unitful quantities that change types, it is not always easy to properly compose UnitfulLinearAlgebra functions with Unitful and LinearAlgebra functions. Also, some LinearAlgebra functions like `eigen` are highly restricted with unitful matrices. The `SVD` factorization object also makes assumptions that do not hold for matrices with units. Some compromises and design choices are necessary.

## Performance

Including units on matrices would seem to require twice the overhead of a dimensionless (purely numerical) matrix. Matrices that arise in scientific and engineering problems typically have a specific structure of units that permits the matrix to be used in linear algebraic operations. Such "multipliable matrices" have at most m+n-1 degrees of dimensional freedom, rather than the m*n degrees of numerical freedom. Conceptually it is possible to store this information in a efficient way and to keep the overhead in propagating units low. 

Benchmarks with a random 1000 x 1000 dimensional (unitful) matrix show that the LU decomposition is currently about 20% slower when units are included. This slowdown could be due to the lack of optimization in matrix multiplication with unitful matrices, where matrix-vector multiplication is currently about 5x slower for this matrix. Matrix-matrix multiplication incurs less than 1% slowdown and is better optimized than matrix-vector multiplication.
```
julia> @btime lu(A.numbers)
  6.883 ms (5 allocations: 7.64 MiB)
LU{Float64, Matrix{Float64}, Vector{Int64}}
L factor:
1000×1000 Matrix{Float64}:
 1.0          0.0         0.0          0.0       …   0.0        0.0        0.0        0.0       0.0
 0.00907939   1.0         0.0          0.0           0.0        0.0        0.0        0.0       0.0
 0.864246    -0.783037    1.0          0.0           0.0        0.0        0.0        0.0       0.0
 0.0337761    0.95132    -0.506981     1.0           0.0        0.0        0.0        0.0       0.0
 0.417638    -0.383787    0.600295    -0.377183      0.0        0.0        0.0        0.0       0.0
 0.115178     0.786904   -0.411723     0.680925  …   0.0        0.0        0.0        0.0       0.0
 ⋮                                               ⋱   ⋮                                          
 0.404589     0.656569    0.193141     0.333635  …   1.0        0.0        0.0        0.0       0.0
 0.099852     0.932473   -0.382737     0.917706     -0.454004   1.0        0.0        0.0       0.0
 0.0106895    0.0779327   0.55105      0.191918      0.248199  -0.251414   1.0        0.0       0.0
 0.342518     0.509994   -0.00247973   0.370809      0.817806  -0.632452   0.292211   1.0       0.0
 0.645603     0.379993    0.100097     0.249332     -0.902868   0.868855  -0.310387  -0.332151  1.0
U factor:
1000×1000 Matrix{Float64}:
 0.998767  0.891597  0.067844  0.298412   0.87389   0.445026  …   0.881143    0.929906   0.480685   0.941497
 0.0       0.969096  0.927726  0.173529   0.329032  0.236743      0.478279    0.34787    0.87551    0.657213
 0.0       0.0       1.60458   0.610667  -0.457382  0.396333     -0.0466389   0.289336   0.418421   0.672108
 0.0       0.0       0.0       1.11197    0.417167  0.132312      0.355055   -0.192716   0.188781  -0.30909
 0.0       0.0       0.0       0.0        1.1632    0.45811       0.865155   -0.162761   0.610506   0.284356
 0.0       0.0       0.0       0.0        0.0       1.07541   …   0.697304    0.254628   0.129371   0.751698
 ⋮                                                  ⋮         ⋱                                    
 0.0       0.0       0.0       0.0        0.0       0.0       …  -0.623047    0.832163  -2.77629   -2.58172
 0.0       0.0       0.0       0.0        0.0       0.0           5.91525    -9.09798   -1.69333   -6.65463
 0.0       0.0       0.0       0.0        0.0       0.0           0.0        -9.41721   -0.794938  -2.39488
 0.0       0.0       0.0       0.0        0.0       0.0           0.0         0.0        2.88564    2.4034
 0.0       0.0       0.0       0.0        0.0       0.0           0.0         0.0        0.0        5.65349

julia> @btime lu(A)
  8.042 ms (4712 allocations: 7.89 MiB)
LU{Float64, MultipliableMatrix{Float64}, Vector{Int64}}
L factor:
1000×1000 EndomorphicMatrix{Float64}:
                 1.0              0.0 K m⁻²         …                 0.0    0.0 K m⁻¹    0.0 K m⁻²
 0.00907939 m² K⁻¹                        1.0                0.0 m² K⁻¹          0.0 m            0.0
    0.864246 s K⁻¹          -0.783037 s m⁻²                   0.0 s K⁻¹      0.0 s m⁻¹    0.0 s m⁻²
   0.0337761 s K⁻¹            0.95132 s m⁻²                   0.0 s K⁻¹      0.0 s m⁻¹    0.0 s m⁻²
    0.417638 m K⁻¹            -0.383787 m⁻¹                   0.0 m K⁻¹              0.0    0.0 m⁻¹
    0.115178 m K⁻¹             0.786904 m⁻¹         …         0.0 m K⁻¹              0.0    0.0 m⁻¹
                 ⋮                                  ⋱                                     
                 0.404589    0.656569 K m⁻²         …                 0.0    0.0 K m⁻¹    0.0 K m⁻²
   0.099852 m² K⁻¹                        0.932473           0.0 m² K⁻¹          0.0 m            0.0
                 0.0106895  0.0779327 K m⁻²                           1.0    0.0 K m⁻¹    0.0 K m⁻²
    0.342518 m K⁻¹             0.509994 m⁻¹              0.292211 m K⁻¹              1.0    0.0 m⁻¹
   0.645603 m² K⁻¹                        0.379993     -0.310387 m² K⁻¹    -0.332151 m            1.0
U factor:
1000×1000 MultipliableMatrix{Float64}:
          0.998767                0.891597  …   0.480685 K m⁻²         0.941497 K s⁻¹
 0.0 m² K⁻¹         0.969096 m² K⁻¹                          0.87551  0.657213 m² s⁻¹
  0.0 s K⁻¹               0.0 s K⁻¹             0.418421 s m⁻²                      0.672108
  0.0 s K⁻¹               0.0 s K⁻¹             0.188781 s m⁻²                     -0.30909
  0.0 m K⁻¹               0.0 m K⁻¹               0.610506 m⁻¹         0.284356 m s⁻¹
  0.0 m K⁻¹               0.0 m K⁻¹         …     0.129371 m⁻¹         0.751698 m s⁻¹
          ⋮                                 ⋱                         
          0.0                     0.0       …   -2.77629 K m⁻²         -2.58172 K s⁻¹
 0.0 m² K⁻¹              0.0 m² K⁻¹                         -1.69333  -6.65463 m² s⁻¹
          0.0                     0.0          -0.794938 K m⁻²         -2.39488 K s⁻¹
  0.0 m K⁻¹               0.0 m K⁻¹                2.88564 m⁻¹           2.4034 m s⁻¹
 0.0 m² K⁻¹              0.0 m² K⁻¹                          0.0       5.65349 m² s⁻¹
```
# How this Julia package was started

This package was generated using PkgTemplates.jl. 

Steps: 
1. Use PkgTemplates to make git repo.

Run the following Julia code

`using Revise, PkgTemplates`

`t = Template(; 
    user="ggebbie",
    dir="~/projects",
    authors="G Jake Gebbie <ggebbie@whoi.edu>",
    julia=v"1.8",
    plugins=[
        License(; name="MIT"),
        Git(; manifest=true, ssh=true),
        GitHubActions(; x86=false),
        Codecov(),
        Documenter{GitHubActions}(),
        Develop(),
    ],
             )`

`t("UnitfulLinearAlgebra.jl")`

2. Make a new empty repository on GitHub.
	
3. Then push this existing repository from the command line:\
    `git push -u origin main`

	Previously it required setting the remote and branch name via the following settings. Not anymore.
    `git remote add origin git@github.com:ggebbie/UnitfulLinearAlgebra.jl.git`\
   `git branch -M main`
 
  In magit, use the command `M a` to set origin, but it's not necessary anymore.
  
4. Use Documenter.jl and DocumenterTools to automatically deploy documentation following: https://m3g.github.io/JuliaNotes.jl/stable/publish_docs/ .
