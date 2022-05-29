# Ensemble of single ODEs

function func(u, p, t)
    a, = p
    du = -a * u
    return du
end


function solve!(u, t, integs::CuVector)
    Na = size(u, 1)
    ckernel = @cuda launch=false solve_kernel!(u, t, integs)
    config = launch_configuration(ckernel.fun)
    nth = min(Na, config.threads)
    nbl = cld(Na, nth)
    ckernel(u, t, integs; threads=nth, blocks=nbl)
    return nothing
end


function solve_kernel!(u, t, integs)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    Na, Nt = size(u)
    dt = t[2] - t[1]
    for ia=id:stride:Na
        integ = integs[ia]
        u[ia,1] = integ.prob.u0
        for it=1:Nt-1
            u[ia,it+1] = ODEIntegrators.step(integ, u[ia,it], t[it], dt)
        end
    end
    return nothing
end


amin, amax, Na = 1f0, 2f0, 512
a = range(amin, amax, length=Na)

u0 = 10f0

tmin, tmax, Nt = 0f0, 5/amax, 100
t = range(tmin, tmax, length=Nt)

uth = zeros(Float32, (Na, Nt))
for ia=1:Na
    @. uth[ia,:] = u0 * exp(-a[ia] * t)
end

probs = [ODEIntegrators.Problem(func, u0, (a[ia],)) for ia=1:Na]

u = CUDA.zeros((Na, Nt))
for alg in algs
    integs = CuArray([ODEIntegrators.Integrator(probs[ia], alg) for ia=1:Na])
    CUDA.@allocated solve!(u, t, integs)
    @test (CUDA.@allocated solve!(u, t, integs)) == 0
    @test compare(collect(u), uth, alg)
end
