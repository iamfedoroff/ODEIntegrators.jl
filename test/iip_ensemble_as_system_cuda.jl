# Ensemble of systems of ODEs as system of ODEs

function func!(du, u, p, t)
    Na, Nu = size(u)
    ckernel = @cuda launch=false func_kernel!(du, u, p, t)
    config = launch_configuration(ckernel.fun)
    nth = min(Na * Nu, config.threads)
    nbl = cld(Na * Nu, nth)
    ckernel(du, u, p, t; threads=nth, blocks=nbl)
end


function func_kernel!(du, u, p, t)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    a, = p
    Na, Nu = size(u)
    ci = CartesianIndices((Na, Nu))

    for ici=id:stride:Na*Nu
        ia = ci[ici][1]
        iu = ci[ici][2]
        du[ia,iu] = -a[ia] * u[ia,iu]
    end
    return nothing
end


function solve!(u, utmp, t, integ)
    Nt = length(t)
    dt = t[2] - t[1]
    @. utmp = integ.prob.u0
    @. u[:,:,1] = utmp
    for it=1:Nt-1
        ODEIntegrators.step!(integ, utmp, t[it], dt)
        @. u[:,:,it+1] = utmp
    end
    return nothing
end


amin, amax, Na = 1f0, 2f0, 128
a = range(amin, amax, length=Na)

u0min, u0max, Nu = 5f0, 10f0, 256
u0 = zeros((Na, Nu))
for ia=1:Na
    u0[ia,:] .= range(u0min, u0max, length=Nu)
end

tmin, tmax, Nt = 0f0, 5/amax, 100
t = range(tmin, tmax, length=Nt)

uth = zeros((Na, Nu, Nt))
for ia=1:Na
for iu=1:Nu
    @. uth[ia,iu,:] = u0[ia,iu] * exp(-a[ia] * t)
end
end

p = (a, )
u0 = CuArray(u0)
prob = ODEIntegrators.Problem(func!, u0, p)

u = CUDA.zeros((Na, Nu, Nt))
utmp = similar(u, (Na, Nu))
for alg in algs
    integ = ODEIntegrators.Integrator(prob, alg)
    if alg != ATsit5()   # ATsit5 uses sum function which allocates memory
        CUDA.@allocated solve!(u, utmp, t, integ)
        @test (CUDA.@allocated solve!(u, utmp, t, integ)) == 0
    else
        solve!(u, utmp, t, integ)
    end
    @test compare(collect(u), uth, alg)
end
