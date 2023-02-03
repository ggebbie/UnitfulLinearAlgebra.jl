ENV["UNITFUL_FANCY_EXPONENTS"] = true

using Revise
using UnitfulLinearAlgebra
using Unitful
using LinearAlgebra
using Test
using BenchmarkTools

const m = u"m"
const s = u"s"
const K = u"K"
const m² = u"m^2"
const J = u"J"
const J² = u"J^2"
const kg = u"kg"
const N = u"N"

unts = [m,s,K,m²]
unts2 = [J,N,kg,J²]
n = 1000
r = rand(unts,n)
d = rand(unts2,n)
num = rand(n,n)
#A = BestMultipliableMatrix(num,r,d,exact=true)
A = UnitfulMatrix(num,r,d,exact=true)
xnd = rand(n)
x = UnitfulMatrix(xnd,d,exact=true)
ynd = num*xnd
y = A*x

println("matrix-vector multiplication")
@btime num*xnd;
@btime A*x;
#@btime UnitfulLinearAlgebra.longmultiply(A,x);

println("matrix inversion")
@btime inv(num);
@btime inv(A);

println("matrix-matrix multiply")
numinv = inv(num)
Ainv = inv(A)
@btime numinv*num;
@btime Ainv*A;

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

# does it help to make y vector a multipliable matrix?
y2 = BestMultipliableMatrix(y[:,:]);
@btime Alu\y2 #fails

# very slow, just as a check
function longmultiply(A::T,b::AbstractVector) where T<: AbstractMultipliableMatrix

    if dimension.(unitdomain(A)) == dimension.(b)
        #if unitdomain(A) ~ b
        #Amat = Matrix(A);
        return Matrix(A)*b
        #return (A.numbers*ustrip.(b)).*unitrange(A)
        #return Quantity.((A.numbers*ustrip.(b)),unitrange(A)) # slower
    elseif ~exact(A) && (unitdomain(A) ∥ b)
        #Anew = convert_unitdomain(A,unit.(b)) # inefficient?
        convert_unitdomain!(A,unit.(b)) # inefficient?
        return (A.numbers*ustrip.(b)).*unitrange(A)
    else
        error("Dimensions of MultipliableMatrix and vector not compatible")
    end
end
