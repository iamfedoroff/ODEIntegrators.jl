# Ensemble of single ODEs as system of ODEs

function func!(du, u, p, t)
    a, = p
    @. du = -a * u
    return nothing
end


function solve!(u, utmp, t, integ)
    Nt = length(t)
    dt = t[2] - t[1]
    @. utmp = integ.prob.u0
    @. u[:, 1] = utmp
    for i=1:Nt-1
        ODEIntegrators.step!(integ, utmp, t[i], dt)
        @. u[:, i+1] = utmp
    end
    return nothing
end


amin, amax, Na = 1f0, 2f0, 512
a = range(amin, amax, length=Na)

u0 = [10f0 for ia=1:Na]

tmin, tmax, Nt = 0f0, 5/amax, 100
t = range(tmin, tmax, length=Nt)

uth = zeros(Float32, (Na, Nt))
for ia=1:Na
    @. uth[ia,:] = u0[ia] * exp(-a[ia] * t)
end

p = (a, )
u0 = CuArray(u0)
prob = ODEIntegrators.Problem(func!, u0, p)

u = CUDA.zeros((Na, Nt))
utmp = similar(u, Na)
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
