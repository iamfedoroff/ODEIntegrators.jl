# System of ODEs

function func!(du, u, p, t)
    a, = p
    @. du = -a * u
    return nothing
end


function solve!(u, utmp, t, integ)
    Nt = length(t)
    dt = t[2] - t[1]
    @. utmp = integ.prob.u0
    @. u[:,1] = utmp
    for it=1:Nt-1
        ODEIntegrators.step!(integ, utmp, t[it], dt)
        @. u[:,it+1] = utmp
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
prob = ODEIntegrators.Problem(func!, u0, p)

u = CUDA.zeros((Nu, Nt))
utmp = similar(u, Nu)
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
