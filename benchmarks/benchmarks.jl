ENV["UNITFUL_FANCY_EXPONENTS"] = true

using Revise
using UnitfulLinearAlgebra
using Unitful
using LinearAlgebra
using Test
using BenchmarkTools

m = u"m"
s = u"s"
K = u"K"
m² = u"m^2"
J = u"J"
J² = u"J^2"
kg = u"kg"
N = u"N"

unts = [m,s,K,m²]
unts2 = [J,N,kg,J²]
n = 1000
r = rand(unts,n)
d = rand(unts2,n)
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
@btime lu(A.numbers);
@btime lu(A);

numlu = lu(num)
Alu = lu(A)

println("LU left divide")
@btime numlu\ynd;
@btime Alu\y;

