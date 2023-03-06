@testset "scalars" begin
    K = u"K"
    @test dimensionless(1.0)
    @test ~dimensionless(1.0K)

    c = 1m
    d = 2m
    @test c~d
    @test similarity(c,d)
    @test rand() ~ rand()
    @test parallel(rand(),rand())
    @test rand() ∥ rand()
    @test uniform(rand())
    @test uniform((rand())K)
    @test isequal(invdimension(1.0),NoDims)
    #@test isequal(invdimension(1.0K),Symbol(𝚯^-1))
    invdimension(1.0K)

    f = 1m
    g = 1 ./ f
    @test dottable(f,g)
    f ⋅ g

    h = 12.0
    j = 1 ./ h
    @test dottable(h,j)
    h ⋅ j
end

@testset "vectors" begin


    # already implemented in Unitful?
    a = [1m, 1s, 10K]
    b = [10m, -1s, 4K]
    a + b
    @test similarity(a,b)
    @test a~b
    @test parallel(a,b)
    @test a ∥ b
    #a ⋅ b
    @test ~uniform(b)
    
    c = [1m, 1s, 10K]
    d = [10m², -1s, 4K]
    @test ~similarity(c,d)
    @test ~(c~d)
    @test ~(c∥d)
    #c ⋅ d

    # inverse dimension
    invdimension.(a)

    k = 1 ./ a
    a ⋅ k
    @test dottable(a,k)
    @test ~dottable(a,b)
end
