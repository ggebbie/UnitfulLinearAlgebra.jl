ENV["UNITFUL_FANCY_EXPONENTS"] = true

using Revise
using UnitfulLinearAlgebra
using Unitful
using LinearAlgebra
using Test

m = u"m"
s = u"s"
K = u"K"
m² = u"m^2"

unts = [m,s,K,m²]

n = 1000
r = rand(unts,n)
d = rand(unts,n)
num = rand(n,n)
A = BestMultipliableMatrix(num,r,d)
xnd = rand(n)
x = xnd.*d
ynd = num*xnd
y = A*x
println("matrix multiplication")
@btime num*xnd;
@btime A*x;

println("matrix left divide")
@btime num\ynd;
@btime A\y;

println("LU factorization")
@btime lu(A.numbers)
@btime lu(A)

numlu = lu(num)
Alu = lu(A)

println("LU left divide")
@btime numlu\ynd;
@btime Alu\y;
