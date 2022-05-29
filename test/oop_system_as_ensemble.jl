# System of ODEs as ensemble of single ODEs

function func(u, p, t)
    a, = p
    du = -a * u
    return du
end


function solve!(u, t, integ)
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
prob = ODEIntegrators.Problem(func, u0, p)

u = zeros((Nu, Nt))
for alg in algs
    integ = ODEIntegrators.Integrator(prob, alg)
    @allocated solve!(u, t, integ)
    @test (@allocated solve!(u, t, integ)) == 0
    @test compare(u, uth, alg)
end
