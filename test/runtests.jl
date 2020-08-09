import CUDA
import ODEIntegrators
using Test


CUDA.allowscalar(false)


function compare(u, uth, alg)
    if alg == "RK2"
        res = isapprox(u, uth, rtol=1e-3)
    elseif alg == "RK3"
        res = isapprox(u, uth, rtol=1e-5)
    elseif alg == "RK4"
        res = isapprox(u, uth, rtol=1e-7)
    else   # Tsit5 and ATsit5
        res = isapprox(u, uth)
    end
    return res
end


# ------------------------------------------------------------------------------
a = 2.0
u0 = 10.0

Nr = 10
Nt = 100
tmin, tmax = 0.0, 5/a

algs = ["RK2", "RK3", "RK4", "Tsit5", "ATsit5"]


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
