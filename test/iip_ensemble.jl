# Ensemble of systems of ODEs

function func!(du, u, p, t)
    a, = p
    @. du = -a * u
    return nothing
end


function solve!(u, utmp, t, integs::Vector)
    Na, Nu, Nt = size(u)
    dt = t[2] - t[1]
    for ia=1:Na
        integ = integs[ia]
        @. utmp = integ.prob.u0
        @. u[ia,:,1] = utmp
        for it=1:Nt-1
            ODEIntegrators.step!(integ, utmp, t[it], dt)
            @. u[ia,:,it+1] = utmp
        end
    end
    return nothing
end


amin, amax, Na = 1.0, 2.0, 11
a = range(amin, amax, length=Na)

u0min, u0max, Nu = 5.0, 10.0, 6
u0 = range(u0min, u0max, length=Nu)

tmin, tmax, Nt = 0.0, 5/amax, 100
t = range(tmin, tmax, length=Nt)

uth = zeros((Na, Nu, Nt))
for ia=1:Na
for iu=1:Nu
    @. uth[ia,iu,:] = u0[iu] * exp(-a[ia] * t)
end
end

u0 = collect(u0)
probs = [ODEIntegrators.Problem(func!, u0, (a[ia],)) for ia=1:Na]

u = zeros((Na, Nu, Nt))
utmp = similar(u, Nu)
for alg in algs
    integs = [ODEIntegrators.Integrator(probs[ia], alg) for ia=1:Na]
    @allocated solve!(u, utmp, t, integs)
    @test (@allocated solve!(u, utmp, t, integs)) == 0
    @test compare(u, uth, alg)
end
