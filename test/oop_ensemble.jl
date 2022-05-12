# Ensemble of single ODEs

function func(u, p, t)
    a, = p
    du = -a * u
    return du
end


function solve!(u, t, integs::Vector)
    Na, Nt = size(u)
    dt = t[2] - t[1]
    for ia=1:Na
        integ = integs[ia]
        u[ia,1] = integ.prob.u0
        for it=1:Nt-1
            u[ia,it+1] = ODEIntegrators.step(integ, u[ia,it], t[it], dt)
        end
    end
    return nothing
end


amin, amax, Na = 1.0, 2.0, 11
a = range(amin, amax, length=Na)

u0 = 10.0

tmin, tmax, Nt = 0.0, 5/amax, 100
t = range(tmin, tmax, length=Nt)

uth = zeros((Na, Nt))
for ia=1:Na
    @. uth[ia,:] = u0 * exp(-a[ia] * t)
end

probs = [ODEIntegrators.Problem(func, u0, (a[ia],)) for ia=1:Na]

u = zeros((Na, Nt))
for alg in algs
    integs = [ODEIntegrators.Integrator(probs[ia], alg) for ia=1:Na]
    @allocated solve!(u, t, integs)
    @test (@allocated solve!(u, t, integs)) == 0
    @test compare(u, uth, alg)
end
