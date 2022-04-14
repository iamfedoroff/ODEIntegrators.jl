using CUDA
using ODEIntegrators
using Test


CUDA.allowscalar(false)

compare(u, uth, alg) = isapprox(u, uth)
compare(u, uth, alg::RK2) = isapprox(u, uth, rtol=1e-3)
compare(u, uth, alg::RK3) = isapprox(u, uth, rtol=1e-5)
compare(u, uth, alg::RK4) = isapprox(u, uth, rtol=1e-7)


# ------------------------------------------------------------------------------
a = 2.0
u0 = 10.0

Nr = 10
Nt = 100
tmin, tmax = 0.0, 5/a

algs = [RK2(), RK3(), RK4(), Tsit5(), ATsit5()]


# ------------------------------------------------------------------------------
t = range(tmin, tmax, length=Nt)

uth = @. u0 * exp(-a * t)

uths = zeros((Nr, Nt))
for i=1:Nr
    @. uths[i, :] = u0 * exp(-a * t)
end


# ------------------------------------------------------------------------------
@testset "in-place" begin
    include("iip.jl")
end

@testset "out-of-place" begin
    include("oop.jl")
end
