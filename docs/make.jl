using UnitfulLinearAlgebra
using Documenter

DocMeta.setdocmeta!(UnitfulLinearAlgebra, :DocTestSetup, :(using UnitfulLinearAlgebra); recursive=true)

makedocs(;
    modules=[UnitfulLinearAlgebra],
    authors="G Jake Gebbie <ggebbie@whoi.edu>",
    repo="https://github.com/ggebbie/UnitfulLinearAlgebra.jl/blob/{commit}{path}#{line}",
    sitename="UnitfulLinearAlgebra.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ggebbie.github.io/UnitfulLinearAlgebra.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ggebbie/UnitfulLinearAlgebra.jl",
    devbranch="main",
)
