# Ensemble of systems of ODEs as system of ODEs

function func!(du, u, p, t)
    a, = p
    Na, Nu = size(u)
    for iu=1:Nu
    for ia=1:Na
        du[ia,iu] = -a[ia] * u[ia,iu]
    end
    end
    return nothing
end


function solve!(u, utmp, t, integ)
    Nt = length(t)
    dt = t[2] - t[1]
    @. utmp = integ.prob.u0
    @. u[:,:,1] = utmp
    for it=1:Nt-1
        ODEIntegrators.step!(integ, utmp, t[it], dt)
        @. u[:,:,it+1] = utmp
    end
    return nothing
end


amin, amax, Na = 1.0, 2.0, 11
a = range(amin, amax, length=Na)

u0min, u0max, Nu = 5.0, 10.0, 6
u0 = zeros((Na, Nu))
for ia=1:Na
    u0[ia,:] .= range(u0min, u0max, length=Nu)
end

tmin, tmax, Nt = 0.0, 5/amax, 100
t = range(tmin, tmax, length=Nt)

uth = zeros((Na, Nu, Nt))
for ia=1:Na
for iu=1:Nu
    @. uth[ia,iu,:] = u0[ia,iu] * exp(-a[ia] * t)
end
end

p = (a, )
prob = ODEIntegrators.Problem(func!, u0, p)

u = zeros((Na, Nu, Nt))
utmp = similar(u, (Na, Nu))
for alg in algs
    integ = ODEIntegrators.Integrator(prob, alg)
    @allocated solve!(u, utmp, t, integ)
    @test (@allocated solve!(u, utmp, t, integ)) == 0
    @test compare(u, uth, alg)
end
