# Single ODE

function func(u, p, t)
    a, = p
    du = -a * u
    return du
end


function solve!(u, t, integ)
    Nt = length(t)
    dt = t[2] - t[1]
    u[1] = integ.prob.u0
    for it=1:Nt-1
        u[it+1] = ODEIntegrators.step(integ, u[it], t[it], dt)
    end
    return nothing
end


a = 2.0

u0 = 10.0

tmin, tmax, Nt = 0.0, 5/a, 100
t = range(tmin, tmax, length=Nt)

uth = @. u0 * exp(-a * t)

p = (a, )
prob = ODEIntegrators.Problem(func, u0, p)

u = zeros(Nt)
for alg in algs
    integ = ODEIntegrators.Integrator(prob, alg)
    @allocated solve!(u, t, integ)
    @test (@allocated solve!(u, t, integ)) == 0
    @test compare(u, uth, alg)
end
