# System of ODEs as ensemble of single ODEs

function func(u, p, t)
    a, = p
    du = -a * u
    return du
end


function solve!(u, t, integ)
    Nu = size(u, 1)
    ckernel = @cuda launch=false solve_kernel!(u, t, integ)
    config = launch_configuration(ckernel.fun)
    nth = min(Nu, config.threads)
    nbl = cld(Nu, nth)
    ckernel(u, t, integ; threads=nth, blocks=nbl)
end


function solve_kernel!(u, t, integ)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    Nu, Nt = size(u)
    dt = t[2] - t[1]
    for iu=1:Nu
        u[iu,1] = integ.prob.u0[iu]
        for it=1:Nt-1
            u[iu,it+1] = ODEIntegrators.step(integ, u[iu,it], t[it], dt)
        end
    end
    return nothing
end


a = 2f0

u0min, u0max, Nu = 5f0, 10f0, 512
u0 = Vector(range(u0min, u0max, length=Nu))

tmin, tmax, Nt = 0f0, 5/a, 100
t = range(tmin, tmax, length=Nt)

uth = zeros((Nu, Nt))
for iu=1:Nu
    @. uth[iu,:] = u0[iu] * exp(-a * t)
end

p = (a, )
u0 = CuArray(u0)
prob = ODEIntegrators.Problem(func, u0, p)

u = CUDA.zeros((Nu, Nt))
for alg in algs
    integ = ODEIntegrators.Integrator(prob, alg)
    CUDA.@allocated solve!(u, t, integ)
    @test (CUDA.@allocated solve!(u, t, integ)) == 0
    @test compare(collect(u), uth, alg)
end
