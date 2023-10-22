module UnitfulLinearAlgebraLatexifyExt

using UnitfulLinearAlgebra, UnitfulLatexify, Latexify

#UnitfulLatexify.latexify(A::UnitfulLinearAlgebra.AbstractUnitfulType) = UnitfulLatexify.latexify(Matrix(A))
UnitfulLatexify.latexify(A::UnitfulLinearAlgebra.AbstractUnitfulType) = UnitfulLatexify.latexify(Matrix(A))

end
