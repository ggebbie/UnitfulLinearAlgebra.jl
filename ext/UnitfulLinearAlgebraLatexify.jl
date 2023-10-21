module UnitfulLinearAlgebraLatexify

using UnitfulLinearAlgebra, Latexify

latexify(A::AbstractUnitfulType) = latexify(Matrix(A))

end
