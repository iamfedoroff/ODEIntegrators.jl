function func(u, p, t, args)
    a, = p
    du = -a * u
    return du
end


function solve!(u, t, integ)
    Nt = length(t)
    dt = t[2] - t[1]
    u[1] = integ.prob.u0
    for i=1:Nt-1
        args = ()
        u[i+1] = ODEIntegrators.step(integ, u[i], t[i], dt, args)
    end
    return nothing
end


p = (a, )
prob = ODEIntegrators.Problem(func, u0, p)

u = zeros(Nt)
for alg in algs
    integ = ODEIntegrators.Integrator(prob, alg)
    solve!(u, t, integ)
    @test compare(u, uth, alg)
end
