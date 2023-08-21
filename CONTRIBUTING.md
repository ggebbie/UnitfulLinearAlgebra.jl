`UnitfulLinearAlgebra.jl` should be of interest to the wider community, but the current package falls short of extending everything available in Julia's built-in `LinearAlgebra` package. Contributions to make this package more complete are needed and welcome! 

A primary design goal of this package is to include units in linear algebra methods with low or negligible cost. It is hoped that this core goal can be retained in all contributions.

`UnitfulLinearAlgebra.jl` is designed by combining features of `Unitful.jl` and `DimensionalData.jl`. This design choice has consequences, such as using an AbstractArray type that makes many fallbacks work. The downside is that fallbacks can work even when they shouldn't, no error is produced, and the code continues by silently dropping the units. Discussion and guidance on these design issues is appreciated. 

One example is indexing a UnitfulMatrix; the usual convention of `A[1,1]` will return a unitless value. Use `getindexqty(A,1,1)` to return a `Quantity` with units. Currently, this behavior is kept in order to optimize performance.

`UnitfulLinearAlgebra.jl` has an ambitious name, as it would be very difficult to implement all `LinearAlgebra` features with units. Perhaps a more specific or narrower name would be more appropriate. 

Please do submit Issues on GitHub for any bugs or inconveniences, no matter how big or small. 
