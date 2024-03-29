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


a = 2.0

u0min, u0max, Nu = 5.0, 10.0, 6
u0 = range(u0min, u0max, length=Nu)

tmin, tmax, Nt = 0.0, 5/a, 100
t = range(tmin, tmax, length=Nt)

uth = zeros((Nu, Nt))
for iu=1:Nu
    @. uth[iu,:] = u0[iu] * exp(-a * t)
end

p = (a, )
u0 = collect(u0)
prob = ODEIntegrators.Problem(func!, u0, p)

u = zeros((Nu, Nt))
utmp = similar(u, Nu)
for alg in algs
    integ = ODEIntegrators.Integrator(prob, alg)
    @allocated solve!(u, utmp, t, integ)
    @test (@allocated solve!(u, utmp, t, integ)) == 0
    @test compare(u, uth, alg)
end
