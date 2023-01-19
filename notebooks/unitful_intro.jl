### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 548064ce-f0bc-466a-aee5-4a72c629d240
import Pkg; Pkg.activate(".")

# ╔═╡ a50d93d4-d3ab-4388-a23a-05a402679dc8
begin
	using Pluto, LinearAlgebra, Statistics, Plots, PlotThemes, Latexify, UnitfulLatexify, Unitful, Statistics, StatsBase, DelimitedFiles, UnitfulLinearAlgebra, Measurements
	set_default(fmt=FancyNumberFormatter(4),convert_unicode=false,cdot=false)
	ENV["UNITFUL_FANCY_EXPONENTS"] = true
	MMatrix = BestMultipliableMatrix
end

# ╔═╡ b35d0cd8-f20a-4e7c-9c21-5b1792a5e610
# define Sverdrups
module UnitfulOcean; using Unitful; @unit Sverdrup "Sv" Sverdrup (10^6)u"m^3/s" false; end

# ╔═╡ 7e655edb-0e76-4a17-aabf-a2418c255ddc
md""" # Introduction to [Unitful.jl](https://painterqubits.github.io/Unitful.jl/stable) and [UnitfulLinearAlgebra.jl](https://github.com/ggebbie/UnitfulLinearAlgebra.jl) """

# ╔═╡ 52e83b98-6331-4197-87e3-83b85642dba1
m = u"m"; s = u"s"; kg = u"kg";  d = u"d"; K = u"K"; m² = m*m; m³ = m²*m; cm = 1u"cm"

# ╔═╡ 6c31f3e5-fbd0-4095-8642-46e992bfca09
Tg = u"Tg" # knows about SI prefixes

# ╔═╡ fb9aca04-ed9d-470d-8a43-ddc57767f813
u"g"(1Tg)

# ╔═╡ 1dc11f46-54a0-43e9-8c15-bd7cead29f2e
1Tg |> u"g"

# ╔═╡ 35451b1b-3459-4f11-ac6e-2de39a1cb809
70.0u"inch" |> u"ft"

# ╔═╡ 31334308-f348-44b7-b648-e252ea59b21a
uconvert(m,179cm)

# ╔═╡ 24e7d217-20e8-4ad4-9869-5032b389ba25
u"mi"(1u"ft")

# ╔═╡ ecaa125a-7a5c-49b6-9aba-fc1247c944db
uconvert(u"inch",180cm)

# ╔═╡ 9f2a5ca9-abd9-4ebb-9f77-18776b700af1
uconvert(u"inch",180.0cm)

# ╔═╡ ca550636-0e21-4e89-8725-2aab89c92618
a =  5cm; b = 10cm; c = a+b

# ╔═╡ ea18ce2b-3d46-42d9-b713-ff1a3f88d4df
m²(a*b) # convert 

# ╔═╡ 6d6ca46f-c247-41b2-ac11-e65af9f89c9b
a*b

# ╔═╡ 37d25430-77a3-4e31-be86-66d45661a0c4
uconvert(m²,a*b) # convert to square meters

# ╔═╡ 63521051-d110-4375-9a94-15a008d5e394
f = 1.5u"yd"; a+f # convert on the fly

# ╔═╡ 382bcf67-e92a-4967-b081-1853e24ba532
(a*b) |> m²

# ╔═╡ f603a7fd-3398-40f9-a552-dcc4f43505ce
g = 1.5u"N" 

# ╔═╡ 73b4c5f2-314e-4dbd-89af-cfdeca74be2a
a + g # can't add apples and oranges

# ╔═╡ 4ff15442-398b-46a2-8ba6-e20c2ba8a10b
a*g # but you can multiply apples and oranges

# ╔═╡ e6a36882-07b2-49ed-a86b-c906294427d2
md""" ### Do dot products exist? """

# ╔═╡ 05b5143d-fa04-4348-a058-fb5134d4f016
v1 = [1,2,3,4]m ⋅ [5,6,7,8]m # ok if both vectors uniformly have the same units

# ╔═╡ a4e1f6f0-dbf9-48ed-8b0f-dd438263d9c7
v2 = [1,2,3,4]m ⋅ [5,6,7,8]s # ok if both vectors are independently uniform

# ╔═╡ 77116f3d-0082-4496-9c78-54d402bc0698
v3 = [1m, 2s, 3K, 4kg] ⋅ [5m^-1, 6s^-1, 7K^-1, 8kg^-1] # ok if vectors have inverse dimensions

# ╔═╡ 3ae4367e-110c-4465-a048-6ee4fcc18b42
v4 = [1m, 2s, 3K, 4kg]; v4 ⋅ v4 # can't even take dot product with itself

# ╔═╡ f6df4aba-3f3a-11ed-1852-efa05902f9f3
md"""
### Read, calculate, plot, and display unitful quantities

Download the Atlantic meridional overturning circulation (MOC) at 26.5°N, in monthly averages starting with April, 2004 (`moc24N_monavg.txt`). 
"""

# ╔═╡ 65b205d6-033b-4960-9efd-f221d83e4e1e
md""" ### Define your own units and dimensions """

# ╔═╡ 98db278f-ff36-452e-91d7-8c34671bf50e
Unitful.register(UnitfulOcean);

# ╔═╡ 86f83b51-2ea5-4e5b-9775-134b6be9470b
Sv = u"Sverdrup"

# ╔═╡ 83ffc859-2eb3-49b6-9a1f-ef5458124ded
permil = u"permille"

# ╔═╡ 526ec907-f30c-4c4d-af75-01820a8c5a4d
md""" How are sverdrups related to a mass flux?"""

# ╔═╡ a91012ed-ee11-4af5-9488-c3ca31198676
Sv(1000.0m³/s) # a test conversion

# ╔═╡ ddae4472-b930-4ce4-b6ae-caddf509468e
1000.0m³/s |> Sv # another way to convert

# ╔═╡ b47a39e5-1f88-4c70-97cf-38840cee6da4
@latexdefine ρ = 1035.0(kg/m³) # could automatically put it into LaTeX

# ╔═╡ 60bacb4e-6b5e-428e-8e90-be8dff980dca
ρ*1Sv |> Tg/s  # convert Sv to Tg/s

# ╔═╡ b7d5434d-b962-427e-a78f-2ca5d97c69ff
Ψ = readdlm("moc24N_monavg.txt")Sv

# ╔═╡ b76a3f03-7015-4e01-9f4a-2a47253f14fa
nΨ = length(Ψ)

# ╔═╡ bcf18441-869b-4b79-9d6d-845404232ec0
t = range(2004+4.5/12,step=1/12,length=nΨ)u"yr" # start= Apr. 2004

# ╔═╡ d8e0f433-3581-4527-a08b-8a63b08398a6
plot(t,Ψ,xlabel="Calendar years",ylabel="AMOC",label="RAPID-MOCHA array")

# ╔═╡ a40c9260-6cfe-431f-85a7-e1883aa6e44b
### Calculate a mean value through a least-squares problem

# ╔═╡ 3938becc-dc51-4678-9ecc-d837bf7359ff
@latexify Ψ̄ = 𝐄 * Ψ(t) + 𝐪(t)

# ╔═╡ a60459ee-662f-4b96-873a-b030606c0cec
urange = fill(Sv,nΨ); udomain = fill(Sv,1)

# ╔═╡ edc47ee4-0ed0-4208-94c4-d38419e3b321
𝐄 = MMatrix(fill(1,nΨ,1),urange,udomain,exact=true)

# ╔═╡ a06b8afd-bc1c-41e9-b473-5ac3adc0f63d
md""" make covariance matrix """

# ╔═╡ a1e9f7ad-defa-4028-bda4-d30445591455
begin
	# part a: assume obs are independent
	σₙ = 0.1Sv
	σₓ = 10Sv
	σ̃ₓ = var(Ψ) # estimated variance
	σq = √(σₓ^2+σₙ^2)
	Cqq = Diagonal(fill(ustrip.(σq.^2),nΨ),fill(Sv,nΨ),fill(Sv^-1,nΨ),exact=true)
    iCqq = inv(Cqq);
end

# ╔═╡ 0d8b5360-a24d-4c62-b703-4c22ad6f51af
Ψ̄̃ = (transpose(𝐄)*iCqq*𝐄)\(transpose(𝐄)*iCqq*Ψ); @latexdefine Ψ̄̃

# ╔═╡ 10c298f3-94ef-4ec2-9088-49f442956eb2
σΨ̃ = .√(diag(inv(transpose(𝐄)*iCqq*𝐄)))[1]; @latexdefine σΨ̃

# ╔═╡ 640917cb-e3d2-45c1-b2c0-8223f8ed6adc
solution = Ψ̄̃[1] ± σΨ̃

# ╔═╡ 7e34cce6-df00-4b58-8f3e-adf77b618485
# use Measurements.jl to get the significant digits right
md"""Mean value of AMOC is $solution """

# ╔═╡ 5fa09e7c-f190-4308-a39e-51a5507da344
md""" ### When is an array not a matrix? """

# ╔═╡ 3a144c6d-9072-47a3-ac0c-61179369e91c
ℋ = [1.0m 6.5s; 3.0K 5.0kg]

# ╔═╡ 0160792c-fb6a-4068-b0d1-62d7bb0732aa
j = [5.3m, 3.5kg]

# ╔═╡ dd26af62-3b23-4fda-aa93-bc1f231ad425
ℋ*j  # multiplication not possible

# ╔═╡ 84c4cda9-59fe-46a0-a6bd-397257d0b53a
M = 12; # obs

# ╔═╡ 0ebf2a2c-88af-4b4f-b3be-1b38a1609538
udomain1 = [K,K/s,K/s/s]; urange1 = fill(Sv,M)

# ╔═╡ 2b9d6d5e-3a88-44fa-a606-206ce0417b56
H = MMatrix(randn(3,3),urange1,udomain1,exact=true)

# ╔═╡ 5766a40a-2039-4192-8b20-66bd53daf964
k̃ = rand(3).*udomain1

# ╔═╡ 825876d2-4e3e-4e33-be7e-c859a2bb6328
H*k̃ # now you can do multiplication

# ╔═╡ 43d3823f-1b9b-4a6c-b4ba-8bc7cd508265
exact(H)

# ╔═╡ d053dd52-f862-49ff-86c6-9748440802a1
md""" to do matrix-vector multiplication, the unit domain of the matrix must match the units of the vector (or be parallel to those units) """

# ╔═╡ 83b06502-f0bc-43bf-a142-313c1909ddca
md""" ### Good news for people who like bad news: 

There is no eigenstructure or singular value decomposition for your favorite matrix """

# ╔═╡ f86dbe64-0a65-45fc-81e7-9b333373708a
md""" Try eigenvalues for uniform square matrix """

# ╔═╡ ccbbe3d2-7ead-4c6c-bc06-e2c8852346f5
md""" Try eigenvalues  for random square matrix """

# ╔═╡ e93ff3d9-9abf-450c-82db-5008313e9dc1
md""" Try SVD of uniform matrix """

# ╔═╡ d5acaef2-cca2-4519-b527-904d9bfa2f92
md""" Try SVD of Vandermond matrix"""

# ╔═╡ Cell order:
# ╠═548064ce-f0bc-466a-aee5-4a72c629d240
# ╠═a50d93d4-d3ab-4388-a23a-05a402679dc8
# ╟─7e655edb-0e76-4a17-aabf-a2418c255ddc
# ╠═52e83b98-6331-4197-87e3-83b85642dba1
# ╠═6c31f3e5-fbd0-4095-8642-46e992bfca09
# ╠═fb9aca04-ed9d-470d-8a43-ddc57767f813
# ╠═1dc11f46-54a0-43e9-8c15-bd7cead29f2e
# ╠═ea18ce2b-3d46-42d9-b713-ff1a3f88d4df
# ╠═35451b1b-3459-4f11-ac6e-2de39a1cb809
# ╠═31334308-f348-44b7-b648-e252ea59b21a
# ╠═24e7d217-20e8-4ad4-9869-5032b389ba25
# ╠═ecaa125a-7a5c-49b6-9aba-fc1247c944db
# ╠═9f2a5ca9-abd9-4ebb-9f77-18776b700af1
# ╠═ca550636-0e21-4e89-8725-2aab89c92618
# ╠═6d6ca46f-c247-41b2-ac11-e65af9f89c9b
# ╠═37d25430-77a3-4e31-be86-66d45661a0c4
# ╠═63521051-d110-4375-9a94-15a008d5e394
# ╠═382bcf67-e92a-4967-b081-1853e24ba532
# ╠═f603a7fd-3398-40f9-a552-dcc4f43505ce
# ╠═73b4c5f2-314e-4dbd-89af-cfdeca74be2a
# ╠═4ff15442-398b-46a2-8ba6-e20c2ba8a10b
# ╟─e6a36882-07b2-49ed-a86b-c906294427d2
# ╠═05b5143d-fa04-4348-a058-fb5134d4f016
# ╠═a4e1f6f0-dbf9-48ed-8b0f-dd438263d9c7
# ╠═77116f3d-0082-4496-9c78-54d402bc0698
# ╠═3ae4367e-110c-4465-a048-6ee4fcc18b42
# ╟─f6df4aba-3f3a-11ed-1852-efa05902f9f3
# ╟─65b205d6-033b-4960-9efd-f221d83e4e1e
# ╠═b35d0cd8-f20a-4e7c-9c21-5b1792a5e610
# ╠═98db278f-ff36-452e-91d7-8c34671bf50e
# ╠═86f83b51-2ea5-4e5b-9775-134b6be9470b
# ╠═83ffc859-2eb3-49b6-9a1f-ef5458124ded
# ╠═526ec907-f30c-4c4d-af75-01820a8c5a4d
# ╠═a91012ed-ee11-4af5-9488-c3ca31198676
# ╠═ddae4472-b930-4ce4-b6ae-caddf509468e
# ╠═b47a39e5-1f88-4c70-97cf-38840cee6da4
# ╠═60bacb4e-6b5e-428e-8e90-be8dff980dca
# ╠═b7d5434d-b962-427e-a78f-2ca5d97c69ff
# ╠═b76a3f03-7015-4e01-9f4a-2a47253f14fa
# ╠═bcf18441-869b-4b79-9d6d-845404232ec0
# ╠═d8e0f433-3581-4527-a08b-8a63b08398a6
# ╠═a40c9260-6cfe-431f-85a7-e1883aa6e44b
# ╟─3938becc-dc51-4678-9ecc-d837bf7359ff
# ╠═a60459ee-662f-4b96-873a-b030606c0cec
# ╠═edc47ee4-0ed0-4208-94c4-d38419e3b321
# ╟─a06b8afd-bc1c-41e9-b473-5ac3adc0f63d
# ╠═a1e9f7ad-defa-4028-bda4-d30445591455
# ╠═0d8b5360-a24d-4c62-b703-4c22ad6f51af
# ╠═10c298f3-94ef-4ec2-9088-49f442956eb2
# ╠═640917cb-e3d2-45c1-b2c0-8223f8ed6adc
# ╟─7e34cce6-df00-4b58-8f3e-adf77b618485
# ╠═5fa09e7c-f190-4308-a39e-51a5507da344
# ╠═3a144c6d-9072-47a3-ac0c-61179369e91c
# ╠═0160792c-fb6a-4068-b0d1-62d7bb0732aa
# ╠═dd26af62-3b23-4fda-aa93-bc1f231ad425
# ╠═84c4cda9-59fe-46a0-a6bd-397257d0b53a
# ╠═0ebf2a2c-88af-4b4f-b3be-1b38a1609538
# ╠═2b9d6d5e-3a88-44fa-a606-206ce0417b56
# ╠═5766a40a-2039-4192-8b20-66bd53daf964
# ╠═825876d2-4e3e-4e33-be7e-c859a2bb6328
# ╠═43d3823f-1b9b-4a6c-b4ba-8bc7cd508265
# ╟─d053dd52-f862-49ff-86c6-9748440802a1
# ╟─83b06502-f0bc-43bf-a142-313c1909ddca
# ╟─f86dbe64-0a65-45fc-81e7-9b333373708a
# ╟─ccbbe3d2-7ead-4c6c-bc06-e2c8852346f5
# ╟─e93ff3d9-9abf-450c-82db-5008313e9dc1
# ╠═d5acaef2-cca2-4519-b527-904d9bfa2f92
