
"""
    function cholesky(A::AbstractMultipliableMatrix)

    Cholesky decomposition extended for matrices with units.
    Requires unit (or dimensionally) symmetric matrix.
    Functions available for LinearAlgebra.Cholesky objects: `size`, `\`, `inv`, `det`, `logdet` and `isposdef`.
    Functions available for UnitfulLinearAlgebra.Cholesky objects: `size`, `det`, and `isposdef`.
"""
function cholesky(A::AbstractUnitfulMatrix)
    if unit_symmetric(A)
        C = LinearAlgebra.cholesky(parent(A))
        factors = rebuild(A,C.factors,(Units(unitdomain(A)./unitdomain(A)),unitdomain(A)))
        return Cholesky(factors,C.uplo,C.info)
    else
        error("requires unit symmetric matrix")
    end
end

function getproperty(C::Cholesky{T,<:AbstractUnitfulMatrix}, d::Symbol) where T 
    Cfactors = getfield(C, :factors)
    Cuplo    = getfield(C, :uplo)
    if d === :U
        numbers = UpperTriangular(Cuplo === LinearAlgebra.char_uplo(d) ? parent(Cfactors) : copy(transpose(parent(Cfactors))))
        return rebuild(Cfactors,numbers,(unitrange(Cfactors),unitdomain(Cfactors)))
    elseif d === :L
        numbers = LowerTriangular(Cuplo === LinearAlgebra.char_uplo(d) ? parent(Cfactors) : copy(transpose(parent(Cfactors))))
        # use transpose to get units right
        return rebuild(Cfactors,numbers,(Units(unitdomain(Cfactors).^-1),Units(unitrange(Cfactors).^-1)))
    elseif d === :UL
        (Cuplo === 'U') ? (return rebuild(Cfactors,UpperTriangular(parent(Cfactors)))) : (return rebuild(Cfactors,LowerTriangular(parent(Cfactors)),(unitdomain(Cfactors).^-1,unitrange(Cfactors).^-1)))
    else
        #println("caution: fallback not tested")
        return getfield(C, d)
    end
end

