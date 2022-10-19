# UnitfulLinearAlgebra

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ggebbie.github.io/UnitfulLinearAlgebra.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ggebbie.github.io/UnitfulLinearAlgebra.jl/dev/)
[![Build Status](https://github.com/ggebbie/UnitfulLinearAlgebra.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ggebbie/UnitfulLinearAlgebra.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ggebbie/UnitfulLinearAlgebra.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ggebbie/UnitfulLinearAlgebra.jl)

Vectors and matrices with units. 

Approach: ad-hoc attempt to get simple operations to work

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

`t("GGplot.jl")`

2. Make a new empty repository on GitHub.
	
3. Then push this existing repository from the command line:\
    `git push -u origin main`

	Previously it required setting the remote and branch name via the following settings. Not anymore.
    `git remote add origin git@github.com:ggebbie/IsopycnalSurfaces.jl.git`\
   `git branch -M main`
 
  In magit, use the command `M a` to set origin, but it's not necessary anymore.
  
4. Use Documenter.jl and DocumenterTools to automatically deploy documentation following: https://m3g.github.io/JuliaNotes.jl/stable/publish_docs/ .
