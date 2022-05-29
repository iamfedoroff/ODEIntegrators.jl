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


amin, amax, Na = 1.0, 2.0, 11
a = range(amin, amax, length=Na)

u0 = [10.0 for ia=1:Na]

tmin, tmax, Nt = 0.0, 5/amax, 100
t = range(tmin, tmax, length=Nt)

uth = zeros((Na, Nt))
for ia=1:Na
    @. uth[ia,:] = u0[ia] * exp(-a[ia] * t)
end

p = (a, )
prob = ODEIntegrators.Problem(func!, u0, p)

u = zeros((Na, Nt))
utmp = similar(u, Na)
for alg in algs
    integ = ODEIntegrators.Integrator(prob, alg)
    @allocated solve!(u, utmp, t, integ)
    @test (@allocated solve!(u, utmp, t, integ)) == 0
    @test compare(u, uth, alg)
end
