using CUDA
using ODEIntegrators
using Test


CUDA.allowscalar(false)

compare(u, uth, alg) = isapprox(u, uth)
compare(u, uth, alg::RK2) = isapprox(u, uth, rtol=1e-3)
compare(u, uth, alg::Union{RK3,SSPRK3,SSP4RK3}) = isapprox(u, uth, rtol=1e-5)
compare(u, uth, alg::RK4) = isapprox(u, uth, rtol=1e-6)


algs = [RK2(), RK3(), SSPRK3(), SSP4RK3(), RK4(), Tsit5(), ATsit5()]


@testset verbose=true "CPU" begin
    @testset "Single ODE" begin
        include("oop.jl")
    end

    @testset "Ensemble of single ODEs" begin
        include("oop_ensemble.jl")
    end

    @testset "Ensemble of single ODEs as the system of ODEs" begin
        include("oop_ensemble_system.jl")
    end

    @testset "System of ODEs" begin
        include("iip.jl")
    end

    @testset "Ensemble of systems of ODEs" begin
        include("iip_ensemble.jl")
    end

    @testset "Ensemble of systems of ODEs as the system of ODEs" begin
        include("iip_ensemble_system.jl")
    end
end


if CUDA.functional()
@testset verbose=true "CUDA" begin
    @testset "Ensemble of single ODEs" begin
        include("oop_ensemble.jl")
    end

    @testset "Ensemble of single ODEs as the system of ODEs" begin
        include("oop_ensemble_system_cuda.jl")
    end

    @testset "System of ODEs" begin
        include("iip_cuda.jl")
    end

    # @testset "Ensemble of systems of ODEs" begin
    #     In this case we have to deal with CuArray of CuArrays which is
    #     a troublesome combination:
    #     https://discourse.julialang.org/t/arrays-of-arrays-and-arrays-of-structures-in-cuda-kernels-cause-random-errors
    # end

    @testset "Ensemble of systems of ODEs as the system of ODEs" begin
        include("iip_ensemble_system_cuda.jl")
    end
end
end
