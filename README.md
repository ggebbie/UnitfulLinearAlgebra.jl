# UnitfulLinearAlgebra

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ggebbie.github.io/UnitfulLinearAlgebra.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ggebbie.github.io/UnitfulLinearAlgebra.jl/dev/)
[![Build Status](https://github.com/ggebbie/UnitfulLinearAlgebra.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ggebbie/UnitfulLinearAlgebra.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ggebbie/UnitfulLinearAlgebra.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ggebbie/UnitfulLinearAlgebra.jl)

Low-cost linear algebra functions for matrices with units

## Usage: Algebraic interpretation of unitful matrices

1. Convert a matrix of quantities to a UnitfulMatrix using the `UnitfulMatrix` constructor.
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

George W. Hart lays it out in "Multidimensional Analysis: Algebras and Systems for Science and Engineering" (Springer-Verlag, 1995). His approach fits nicely into Julia's type system and multiple dispatch. This packages aims to return objects defined by the `LinearAlgebra` package but extended for use with `AbstractMultipliableMatrix`s. 

Due to Unitful quantities that change types, it is not always easy to properly compose UnitfulLinearAlgebra functions with Unitful and LinearAlgebra functions. Also, some LinearAlgebra functions like `eigen` are highly restricted with unitful matrices. The `SVD` factorization object also makes assumptions that do not hold for matrices with units. Some compromises and design choices are necessary.

## Performance

Including units on matrices would seem to require twice the overhead of a dimensionless (purely numerical) matrix. Matrices that arise in scientific and engineering problems typically have a specific structure of units that permits the matrix to be used in linear algebraic operations. Such "multipliable matrices" have at most m+n-1 degrees of dimensional freedom, rather than the m*n degrees of numerical freedom. Conceptually it is possible to store this information in a efficient way and to keep the overhead in propagating units low. 

Benchmarks with a random 1000 x 1000 dimensional (unitful) matrix show that the LU decomposition currently has very little overhead when units are included.
```

julia> A
1000×1000 UnitfulMatrix{Float64,2}
  0.345829 m² kg⁻¹   0.252412 m² J⁻²  …  0.908051 m² J⁻¹   0.41719 m² J⁻¹
   0.447032 K kg⁻¹    0.177407 K J⁻²      0.336089 K J⁻¹   0.663898 K J⁻¹
   0.752077 s kg⁻¹    0.921767 s J⁻²      0.065284 s J⁻¹   0.139397 s J⁻¹
   0.314864 m kg⁻¹    0.766019 m J⁻²      0.524938 m J⁻¹   0.101552 m J⁻¹
 0.0551535 m² kg⁻¹   0.808757 m² J⁻²      0.14722 m² J⁻¹  0.856775 m² J⁻¹
  0.312722 m² kg⁻¹   0.113635 m² J⁻²  …  0.959325 m² J⁻¹  0.464903 m² J⁻¹
   0.343496 m kg⁻¹    0.974459 m J⁻²     0.0359663 m J⁻¹     0.4117 m J⁻¹
    0.81385 s kg⁻¹      0.1449 s J⁻²      0.585997 s J⁻¹   0.548329 s J⁻¹
   0.458147 m kg⁻¹    0.343768 m J⁻²      0.489571 m J⁻¹   0.536626 m J⁻¹
    0.73875 s kg⁻¹    0.823937 s J⁻²      0.250287 s J⁻¹   0.307086 s J⁻¹
   0.308745 m kg⁻¹    0.229908 m J⁻²  …   0.919632 m J⁻¹   0.893832 m J⁻¹
    0.62709 s kg⁻¹    0.652873 s J⁻²     0.0162129 s J⁻¹   0.850169 s J⁻¹
   0.985917 s kg⁻¹   0.0224312 s J⁻²      0.219591 s J⁻¹    0.67089 s J⁻¹
  0.654275 m² kg⁻¹   0.634063 m² J⁻²     0.780903 m² J⁻¹  0.266933 m² J⁻¹
   0.410227 m kg⁻¹    0.858184 m J⁻²      0.812488 m J⁻¹   0.798148 m J⁻¹
   0.104349 s kg⁻¹    0.737835 s J⁻²  …   0.806745 s J⁻¹   0.981819 s J⁻¹
   0.353769 m kg⁻¹    0.113536 m J⁻²      0.873178 m J⁻¹   0.363516 m J⁻¹
   0.372091 s kg⁻¹    0.595399 s J⁻²      0.104094 s J⁻¹   0.904824 s J⁻¹
   0.854835 m kg⁻¹    0.158248 m J⁻²      0.828702 m J⁻¹   0.877189 m J⁻¹
   0.464166 K kg⁻¹    0.755879 K J⁻²      0.115354 K J⁻¹   0.614889 K J⁻¹
  0.545963 m² kg⁻¹   0.787392 m² J⁻²  …  0.738273 m² J⁻¹  0.291599 m² J⁻¹
    0.27814 s kg⁻¹    0.453383 s J⁻²       0.13306 s J⁻¹    0.48607 s J⁻¹
   0.597206 s kg⁻¹    0.456676 s J⁻²      0.877199 s J⁻¹   0.160307 s J⁻¹
   0.425208 s kg⁻¹    0.623509 s J⁻²       0.53201 s J⁻¹   0.219811 s J⁻¹
                 ⋮                    ⋱                   
   0.699746 s kg⁻¹     0.22339 s J⁻²      0.508766 s J⁻¹   0.681289 s J⁻¹
  0.347389 m² kg⁻¹   0.605815 m² J⁻²     0.143665 m² J⁻¹  0.520141 m² J⁻¹
   0.209188 s kg⁻¹    0.657064 s J⁻²      0.233745 s J⁻¹   0.769076 s J⁻¹
   0.666705 m kg⁻¹  0.00233093 m J⁻²  …   0.802076 m J⁻¹  0.0972221 m J⁻¹
  0.494535 m² kg⁻¹   0.246649 m² J⁻²     0.856264 m² J⁻¹  0.671578 m² J⁻¹
    0.84012 s kg⁻¹   0.0594737 s J⁻²      0.931054 s J⁻¹   0.389611 s J⁻¹
    0.24412 m kg⁻¹    0.695482 m J⁻²      0.677289 m J⁻¹   0.546676 m J⁻¹
    0.12495 K kg⁻¹    0.636562 K J⁻²      0.952064 K J⁻¹   0.300944 K J⁻¹
   0.471942 m kg⁻¹   0.0452257 m J⁻²  …   0.623432 m J⁻¹   0.616376 m J⁻¹
   0.116878 s kg⁻¹    0.457986 s J⁻²      0.810867 s J⁻¹   0.534763 s J⁻¹
   0.580407 s kg⁻¹    0.320276 s J⁻²      0.385704 s J⁻¹   0.479867 s J⁻¹
   0.592588 m kg⁻¹    0.575677 m J⁻²      0.675719 m J⁻¹   0.259864 m J⁻¹
 0.0343571 m² kg⁻¹   0.302703 m² J⁻²      0.33764 m² J⁻¹  0.124556 m² J⁻¹
   0.535253 m kg⁻¹    0.454372 m J⁻²  …   0.777769 m J⁻¹   0.948826 m J⁻¹
  0.385238 m² kg⁻¹   0.839471 m² J⁻²     0.620008 m² J⁻¹    0.2574 m² J⁻¹
   0.157761 m kg⁻¹  0.00907478 m J⁻²      0.893751 m J⁻¹    0.73984 m J⁻¹
  0.741017 m² kg⁻¹    0.99141 m² J⁻²     0.378723 m² J⁻¹   0.94669 m² J⁻¹
   0.164442 s kg⁻¹     0.98644 s J⁻²      0.388186 s J⁻¹    0.98279 s J⁻¹
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
 0.209687     0.616915    0.256849    …   0.0        0.0       0.0       0.0
 0.111998     0.0875346   0.901319        0.0        0.0       0.0       0.0
 0.154841     0.796062   -0.610422        0.0        0.0       0.0       0.0
 0.961859    -0.595748    0.759932        0.0        0.0       0.0       0.0
 0.962222    -0.624533   -0.220126        0.0        0.0       0.0       0.0
 0.483397    -0.379869    0.596833    …   0.0        0.0       0.0       0.0
 0.381865     0.434241   -0.495074        0.0        0.0       0.0       0.0
 0.743178     0.190155   -0.428709        0.0        0.0       0.0       0.0
 0.28545      0.742548   -0.272201        0.0        0.0       0.0       0.0
 0.116851     0.569374    0.42884         0.0        0.0       0.0       0.0
 0.715945     0.0816972   0.16389     …   0.0        0.0       0.0       0.0
 0.393529     0.346702   -0.410756        0.0        0.0       0.0       0.0
 0.476344    -0.302957    0.342149        0.0        0.0       0.0       0.0
 0.958423    -0.393821    0.195038        0.0        0.0       0.0       0.0
 0.599031    -0.0289398   0.28223         0.0        0.0       0.0       0.0
 0.600429     0.37012    -0.243307    …   0.0        0.0       0.0       0.0
 0.182312     0.433645    0.374141        0.0        0.0       0.0       0.0
 0.211982     0.716216   -0.209947        0.0        0.0       0.0       0.0
 0.989104    -0.809546   -0.107884        0.0        0.0       0.0       0.0
 ⋮                                    ⋱                                  
 0.512826     0.0264984   0.246176        0.0        0.0       0.0       0.0
 0.37336      0.372351    0.295457        0.0        0.0       0.0       0.0
 0.316677     0.129895   -0.160723        0.0        0.0       0.0       0.0
 0.846289    -0.295713    0.239619    …   0.0        0.0       0.0       0.0
 0.0625987    0.548431   -0.162655        0.0        0.0       0.0       0.0
 0.780319    -0.619193    0.589872        0.0        0.0       0.0       0.0
 0.276807     0.0490817   0.368939        0.0        0.0       0.0       0.0
 0.154859     0.480469   -0.1894          0.0        0.0       0.0       0.0
 0.209828     0.505977   -0.402603    …   0.0        0.0       0.0       0.0
 0.387426    -0.137937    0.480953        0.0        0.0       0.0       0.0
 0.246995     0.0116356   0.352963        0.0        0.0       0.0       0.0
 0.662657    -0.0271536   0.377424        0.0        0.0       0.0       0.0
 0.521297     0.517753   -0.592086        0.0        0.0       0.0       0.0
 0.98893     -0.807241    0.384052    …   0.0        0.0       0.0       0.0
 0.543371    -0.287529    0.358105        0.0        0.0       0.0       0.0
 0.701547    -0.117257    0.0823985       0.0        0.0       0.0       0.0
 0.967746    -0.508198   -0.038218        0.0        0.0       0.0       0.0
 0.973748    -0.434395    0.00133669      0.0        0.0       0.0       0.0
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
 0.0       0.0       0.0       0.0        …  -0.354137   -1.30196     -0.381887
 0.0       0.0       0.0       0.0            0.0250985  -1.28817      0.531988
 0.0       0.0       0.0       0.0           -0.0564498   0.0369018    0.541579
 0.0       0.0       0.0       0.0           -0.226146   -1.67922     -0.259339
 0.0       0.0       0.0       0.0           -0.0732161  -0.0270443    0.99567
 0.0       0.0       0.0       0.0        …  -0.232498   -0.0358469   -0.982856
 0.0       0.0       0.0       0.0            0.184651   -0.179846     0.242389
 0.0       0.0       0.0       0.0            0.269389   -0.902827     0.787479
 0.0       0.0       0.0       0.0            1.3721      0.359166     0.784885
 0.0       0.0       0.0       0.0           -0.230887   -0.114439     1.10386
 0.0       0.0       0.0       0.0        …   1.06452     0.40291     -1.05537
 0.0       0.0       0.0       0.0            0.359172    0.0713204    0.15933
 0.0       0.0       0.0       0.0            0.620923    0.0895117    0.69862
 0.0       0.0       0.0       0.0            0.6848     -0.298475     0.724537
 0.0       0.0       0.0       0.0            0.489652    0.464314    -0.398735
 0.0       0.0       0.0       0.0        …  -0.332809   -0.179032    -1.13743
 0.0       0.0       0.0       0.0           -0.4898     -0.795879    -1.28712
 0.0       0.0       0.0       0.0           -0.519553   -0.71827     -0.544395
 0.0       0.0       0.0       0.0            0.396354   -1.07844     -0.0664624
 ⋮                                        ⋱                           
 0.0       0.0       0.0       0.0            4.11849     1.10072      0.509372
 0.0       0.0       0.0       0.0            0.719612   -2.07723      0.122362
 0.0       0.0       0.0       0.0           -4.73062     3.09442     -0.0224289
 0.0       0.0       0.0       0.0        …   9.27877    -3.74826      2.93463
 0.0       0.0       0.0       0.0            4.24813    -6.28082      1.36704
 0.0       0.0       0.0       0.0           -1.25158     0.947221    -2.09166
 0.0       0.0       0.0       0.0           -0.756998   -0.345487     3.86687
 0.0       0.0       0.0       0.0            0.302949   -3.59309     -3.47415
 0.0       0.0       0.0       0.0        …  -2.09602    -0.110635    -3.1717
 0.0       0.0       0.0       0.0           -0.430598    2.03325     -1.71573
 0.0       0.0       0.0       0.0            8.46014    -0.400879    -0.930562
 0.0       0.0       0.0       0.0           -2.13157     4.08814      2.70373
 0.0       0.0       0.0       0.0           -5.77188    -6.85504     -9.62432
 0.0       0.0       0.0       0.0        …  -1.38646     0.0655749   -4.6869
 0.0       0.0       0.0       0.0            2.7413      0.896733    -0.870388
 0.0       0.0       0.0       0.0           -1.56067     1.40979      2.39723
 0.0       0.0       0.0       0.0           -3.53154     0.00738267  -1.81146
 0.0       0.0       0.0       0.0            1.36284    -2.43208     -2.21598
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
         0.209687   0.616915 m² K⁻¹   0.256849 m² s⁻¹  …  0.0 m² s⁻¹  0.0 m² s⁻¹
     0.111998 m⁻¹   0.0875346 m K⁻¹    0.901319 m s⁻¹      0.0 m s⁻¹   0.0 m s⁻¹
   0.154841 s m⁻²    0.796062 s K⁻¹         -0.610422            0.0         0.0
     0.961859 m⁻¹   -0.595748 m K⁻¹    0.759932 m s⁻¹      0.0 m s⁻¹   0.0 m s⁻¹
   0.962222 s m⁻²   -0.624533 s K⁻¹         -0.220126            0.0         0.0
     0.483397 m⁻¹   -0.379869 m K⁻¹    0.596833 m s⁻¹  …   0.0 m s⁻¹   0.0 m s⁻¹
   0.381865 s m⁻²    0.434241 s K⁻¹         -0.495074            0.0         0.0
   0.743178 s m⁻²    0.190155 s K⁻¹         -0.428709            0.0         0.0
          0.28545   0.742548 m² K⁻¹  -0.272201 m² s⁻¹     0.0 m² s⁻¹  0.0 m² s⁻¹
     0.116851 m⁻¹    0.569374 m K⁻¹     0.42884 m s⁻¹      0.0 m s⁻¹   0.0 m s⁻¹
   0.715945 s m⁻²   0.0816972 s K⁻¹           0.16389  …         0.0         0.0
     0.393529 m⁻¹    0.346702 m K⁻¹   -0.410756 m s⁻¹      0.0 m s⁻¹   0.0 m s⁻¹
   0.476344 s m⁻²   -0.302957 s K⁻¹          0.342149            0.0         0.0
     0.958423 m⁻¹   -0.393821 m K⁻¹    0.195038 m s⁻¹      0.0 m s⁻¹   0.0 m s⁻¹
   0.599031 K m⁻²        -0.0289398     0.28223 K s⁻¹      0.0 K s⁻¹   0.0 K s⁻¹
         0.600429    0.37012 m² K⁻¹  -0.243307 m² s⁻¹  …  0.0 m² s⁻¹  0.0 m² s⁻¹
   0.182312 s m⁻²    0.433645 s K⁻¹          0.374141            0.0         0.0
   0.211982 s m⁻²    0.716216 s K⁻¹         -0.209947            0.0         0.0
   0.989104 s m⁻²   -0.809546 s K⁻¹         -0.107884            0.0         0.0
                ⋮                                      ⋱              
   0.512826 s m⁻²   0.0264984 s K⁻¹          0.246176            0.0         0.0
          0.37336   0.372351 m² K⁻¹   0.295457 m² s⁻¹     0.0 m² s⁻¹  0.0 m² s⁻¹
   0.316677 s m⁻²    0.129895 s K⁻¹         -0.160723            0.0         0.0
     0.846289 m⁻¹   -0.295713 m K⁻¹    0.239619 m s⁻¹  …   0.0 m s⁻¹   0.0 m s⁻¹
        0.0625987   0.548431 m² K⁻¹  -0.162655 m² s⁻¹     0.0 m² s⁻¹  0.0 m² s⁻¹
   0.780319 s m⁻²   -0.619193 s K⁻¹          0.589872            0.0         0.0
     0.276807 m⁻¹   0.0490817 m K⁻¹    0.368939 m s⁻¹      0.0 m s⁻¹   0.0 m s⁻¹
   0.154859 K m⁻²          0.480469     -0.1894 K s⁻¹      0.0 K s⁻¹   0.0 K s⁻¹
     0.209828 m⁻¹    0.505977 m K⁻¹   -0.402603 m s⁻¹  …   0.0 m s⁻¹   0.0 m s⁻¹
   0.387426 s m⁻²   -0.137937 s K⁻¹          0.480953            0.0         0.0
   0.246995 s m⁻²   0.0116356 s K⁻¹          0.352963            0.0         0.0
     0.662657 m⁻¹  -0.0271536 m K⁻¹    0.377424 m s⁻¹      0.0 m s⁻¹   0.0 m s⁻¹
         0.521297   0.517753 m² K⁻¹  -0.592086 m² s⁻¹     0.0 m² s⁻¹  0.0 m² s⁻¹
      0.98893 m⁻¹   -0.807241 m K⁻¹    0.384052 m s⁻¹  …   0.0 m s⁻¹   0.0 m s⁻¹
         0.543371  -0.287529 m² K⁻¹   0.358105 m² s⁻¹     0.0 m² s⁻¹  0.0 m² s⁻¹
     0.701547 m⁻¹   -0.117257 m K⁻¹   0.0823985 m s⁻¹      0.0 m s⁻¹   0.0 m s⁻¹
         0.967746  -0.508198 m² K⁻¹  -0.038218 m² s⁻¹     0.0 m² s⁻¹  0.0 m² s⁻¹
   0.973748 s m⁻²   -0.434395 s K⁻¹        0.00133669            0.0         0.0
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
      0.0 m² kg⁻¹       0.0 m² J⁻²  …    -1.30196 m² J⁻¹  -0.381887 m² J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²        -1.28817 m J⁻¹    0.531988 m J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²       0.0369018 s J⁻¹    0.541579 s J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²        -1.67922 m J⁻¹   -0.259339 m J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²      -0.0270443 s J⁻¹     0.99567 s J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²  …   -0.0358469 m J⁻¹   -0.982856 m J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²       -0.179846 s J⁻¹    0.242389 s J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²       -0.902827 s J⁻¹    0.787479 s J⁻¹
      0.0 m² kg⁻¹       0.0 m² J⁻²       0.359166 m² J⁻¹   0.784885 m² J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²       -0.114439 m J⁻¹     1.10386 m J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²  …      0.40291 s J⁻¹    -1.05537 s J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²       0.0713204 m J⁻¹     0.15933 m J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²       0.0895117 s J⁻¹     0.69862 s J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²       -0.298475 m J⁻¹    0.724537 m J⁻¹
       0.0 K kg⁻¹        0.0 K J⁻²        0.464314 K J⁻¹   -0.398735 K J⁻¹
      0.0 m² kg⁻¹       0.0 m² J⁻²  …   -0.179032 m² J⁻¹   -1.13743 m² J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²       -0.795879 s J⁻¹    -1.28712 s J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²        -0.71827 s J⁻¹   -0.544395 s J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²        -1.07844 s J⁻¹  -0.0664624 s J⁻¹
                ⋮                   ⋱                     
       0.0 s kg⁻¹        0.0 s J⁻²         1.10072 s J⁻¹    0.509372 s J⁻¹
      0.0 m² kg⁻¹       0.0 m² J⁻²       -2.07723 m² J⁻¹   0.122362 m² J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²         3.09442 s J⁻¹  -0.0224289 s J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²  …     -3.74826 m J⁻¹     2.93463 m J⁻¹
      0.0 m² kg⁻¹       0.0 m² J⁻²       -6.28082 m² J⁻¹    1.36704 m² J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²        0.947221 s J⁻¹    -2.09166 s J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²       -0.345487 m J⁻¹     3.86687 m J⁻¹
       0.0 K kg⁻¹        0.0 K J⁻²        -3.59309 K J⁻¹    -3.47415 K J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²  …    -0.110635 m J⁻¹     -3.1717 m J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²         2.03325 s J⁻¹    -1.71573 s J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²       -0.400879 s J⁻¹   -0.930562 s J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²         4.08814 m J⁻¹     2.70373 m J⁻¹
      0.0 m² kg⁻¹       0.0 m² J⁻²       -6.85504 m² J⁻¹   -9.62432 m² J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²  …    0.0655749 m J⁻¹     -4.6869 m J⁻¹
      0.0 m² kg⁻¹       0.0 m² J⁻²       0.896733 m² J⁻¹  -0.870388 m² J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²         1.40979 m J⁻¹     2.39723 m J⁻¹
      0.0 m² kg⁻¹       0.0 m² J⁻²     0.00738267 m² J⁻¹   -1.81146 m² J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²        -2.43208 s J⁻¹    -2.21598 s J⁻¹
      0.0 m² kg⁻¹       0.0 m² J⁻²  …    -4.79467 m² J⁻¹   -7.48615 m² J⁻¹
       0.0 m kg⁻¹        0.0 m J⁻²        -7.30684 m J⁻¹    -7.91318 m J⁻¹
       0.0 K kg⁻¹        0.0 K J⁻²       -0.246053 K J⁻¹    -1.07557 K J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²        -5.23589 s J⁻¹    -7.14022 s J⁻¹
       0.0 s kg⁻¹        0.0 s J⁻²             0.0 s J⁻¹     1.37779 s J⁻¹

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
