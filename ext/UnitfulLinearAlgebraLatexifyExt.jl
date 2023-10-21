module UnitfulLinearAlgebraLatexifyExt

using UnitfulLinearAlgebra, Latexify

Latexify.latexify(A::UnitfulLinearAlgebra.AbstractUnitfulType) = Latexify.latexify(Matrix(A))

end
