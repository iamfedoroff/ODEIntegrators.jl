function func(du, u, p, t)
    a, = p
    @. du = -a * u
    return nothing
end


function solve!(u, t, integ)
    Nr, Nt = size(u)
    dt = t[2] - t[1]
    utmp = similar(u, Nr)
    @. utmp = integ.prob.u0
    @. u[:, 1] = utmp
    for i=1:Nt-1
        ODEIntegrators.step!(integ, utmp, t[i], dt)
        @. u[:, i+1] = utmp
    end
    return nothing
end


# CPU --------------------------------------------------------------------------
u0s = u0 * ones(Nr)
as = a * ones(Nr)

p = (as, )
prob = ODEIntegrators.Problem(func, u0s, p)

u = zeros((Nr, Nt))
for alg in algs
    integ = ODEIntegrators.Integrator(prob, alg)
    solve!(u, t, integ)
    @test compare(u, uths, alg)
end


# GPU --------------------------------------------------------------------------
if CUDA.functional()
    u0s = CUDA.CuArray(u0s)
    as = tuple(as...)

    p = (as, )
    prob = ODEIntegrators.Problem(func, u0s, p)

    u_gpu = CUDA.CuArray(u)
    for alg in algs
        integ = ODEIntegrators.Integrator(prob, alg)
        solve!(u_gpu, t, integ)
        @test compare(CUDA.collect(u_gpu), uths, alg)
    end
end
