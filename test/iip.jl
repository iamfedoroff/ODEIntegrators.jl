function func(du, u, p, t)
    a, = p
    @. du = -a * u
    return nothing
end


function solve!(u, utmp, t, integ, )
    Nr, Nt = size(u)
    dt = t[2] - t[1]
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
utmp = similar(u, Nr)
for alg in algs
    integ = ODEIntegrators.Integrator(prob, alg)
    @allocated solve!(u, utmp, t, integ)
    @test (@allocated solve!(u, utmp, t, integ)) == 0
    @test compare(u, uths, alg)
end


# GPU --------------------------------------------------------------------------
if CUDA.functional()
    u0s = u0 * CUDA.ones(Nr)
    as = a * CUDA.ones(Nr)

    p = (as, )
    prob = ODEIntegrators.Problem(func, u0s, p)

    u_gpu = CUDA.CuArray(u)
    utmp_gpu = similar(u_gpu, Nr)
    for alg in algs
        integ = ODEIntegrators.Integrator(prob, alg)
        # CUDA.@allocated solve!(u_gpu, utmp_gpu, t, integ)
        # @test (CUDA.@allocated solve!(u_gpu, utmp_gpu, t, integ)) == 0
        solve!(u_gpu, utmp_gpu, t, integ)
        @test compare(CUDA.collect(u_gpu), uths, alg)
    end
end
