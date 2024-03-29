module ODEIntegrators

import Adapt: @adapt_structure
import CUDA: CuDeviceArray
import StaticArrays: SVector

export Problem, Integrator, step, step!,
       RK2, RK3, SSPRK3, SSP4RK3, RK4, Tsit5, ATsit5


abstract type Algorithm end

abstract type Integrator end


struct Problem{F, U, P}
    func :: F
    u0 :: U
    p :: P
end

@adapt_structure Problem


# ******************************************************************************
# RK2
# ******************************************************************************
struct RK2 <: Algorithm end


struct IntegratorRK2{F, U, P} <: Integrator
    prob :: Problem{F, U, P}
    k1 :: U
    k2 :: U
    utmp :: U
end

@adapt_structure IntegratorRK2


function Integrator(prob::Problem, alg::RK2)
    k1, k2, utmp = [zero(prob.u0) for i in 1:3]
    return IntegratorRK2(prob, k1, k2, utmp)
end


# out of place
function step(integ::IntegratorRK2, u, t, dt, args...)
    (; func, p) = integ.prob

    k1 = func(u, p, t, args...)

    utmp = u + dt * 2 * k1 / 3
    ttmp = t + 2 * dt / 3
    k2 = func(utmp, p, ttmp, args...)

    return u + dt * (k1 / 4 + 3 * k2 / 4)
end


# in place
function step!(integ::IntegratorRK2, u, t, dt, args...)
    (; func, p) = integ.prob
    (; k1, k2, utmp) = integ

    func(k1, u, p, t, args...)

    @. utmp = u + dt * 2 * k1 / 3
    ttmp = t + 2 * dt / 3
    func(k2, utmp, p, ttmp, args...)

    @. u = u + dt * (k1 / 4 + 3 * k2 / 4)
    return nothing
end


# in place for CUDA kernels
function step!(integ::IntegratorRK2, u::CuDeviceArray, t, dt, args...)
    (; func, p) = integ.prob
    (; k1, k2, utmp) = integ

    func(k1, u, p, t, args...)

    for i in eachindex(u)
        utmp[i] = u[i] + dt * 2 * k1[i] / 3
    end
    ttmp = t + 2 * dt / 3
    func(k2, utmp, p, ttmp, args...)

    for i in eachindex(u)
        u[i] = u[i] + dt * (k1[i] / 4 + 3 * k2[i] / 4)
    end
    return nothing
end


# ******************************************************************************
# RK3
# ******************************************************************************
struct RK3 <: Algorithm end


struct IntegratorRK3{F, U, P} <: Integrator
    prob :: Problem{F, U, P}
    k1 :: U
    k2:: U
    k3 :: U
    utmp :: U
end

@adapt_structure IntegratorRK3


function Integrator(prob::Problem, alg::RK3)
    k1, k2, k3, utmp = [zero(prob.u0) for i in 1:4]
    return IntegratorRK3(prob, k1, k2, k3, utmp)
end


# out of place
function step(integ::IntegratorRK3, u, t, dt, args...)
    (; func, p) = integ.prob

    k1 = func(u, p, t, args...)

    utmp = u + dt * k1 / 2
    ttmp = t + dt / 2
    k2 = func(utmp, p, ttmp, args...)

    utmp = u + dt * (-k1 + 2 * k2)
    ttmp = t + dt
    k3 = func(utmp, p, ttmp, args...)

    return u + dt * (k1 / 6 + 2 * k2 / 3 + k3 / 6)
end


# in place
function step!(integ::IntegratorRK3, u, t, dt, args...)
    (; func, p) = integ.prob
    (; k1, k2, k3, utmp) = integ

    func(k1, u, p, t, args...)

    @. utmp = u + dt * k1 / 2
    ttmp = t + dt / 2
    func(k2, utmp, p, ttmp, args...)

    @. utmp = u + dt * (-k1 + 2 * k2)
    ttmp = t + dt
    func(k3, utmp, p, ttmp, args...)

    @. u = u + dt * (k1 / 6 + 2 * k2 / 3 + k3 / 6)
    return nothing
end


# in place for CUDA kernels
function step!(integ::IntegratorRK3, u::CuDeviceArray, t, dt, args...)
    (; func, p) = integ.prob
    (; k1, k2, k3, utmp) = integ

    func(k1, u, p, t, args...)

    for i in eachindex(u)
        utmp[i] = u[i] + dt * k1[i] / 2
    end
    ttmp = t + dt / 2
    func(k2, utmp, p, ttmp, args...)

    for i in eachindex(u)
        utmp[i] = u[i] + dt * (-k1[i] + 2 * k2[i])
    end
    ttmp = t + dt
    func(k3, utmp, p, ttmp, args...)

    for i in eachindex(u)
        u[i] = u[i] + dt * (k1[i] / 6 + 2 * k2[i] / 3 + k3[i] / 6)
    end
    return nothing
end



# ******************************************************************************
# SSPRK3
# Dale R. Durran, “Numerical Methods for Fluid Dynamics”, Springer, 2nd ed.
# (2010), p. 56
# https://link.springer.com/book/10.1007/978-1-4419-6412-0
# ******************************************************************************
struct SSPRK3 <: Algorithm end


struct IntegratorSSPRK3{F, U, P} <: Integrator
    prob :: Problem{F, U, P}
    du :: U
    gg :: U
end

@adapt_structure IntegratorSSPRK3


function Integrator(prob::Problem, alg::SSPRK3)
    du, gg = zero(prob.u0), zero(prob.u0)
    return IntegratorSSPRK3(prob, du, gg)
end


# out of place
function step(integ::IntegratorSSPRK3, u, t, dt, args...)
    func = integ.prob.func
    p = integ.prob.p

    du = func(u, p, t, args...)
    gg = u + dt * du

    du = func(gg, p, t + dt, args...)
    gg = 3 * u / 4 + (gg + dt * du) / 4

    du = func(gg, p, t + dt / 2, args...)
    return u / 3 + 2 * (gg + dt * du) / 3
end


# in place
function step!(integ::IntegratorSSPRK3, u, t, dt, args...)
    func = integ.prob.func
    p = integ.prob.p

    du, gg = integ.du, integ.gg

    func(du, u, p, t, args...)
    @. gg = u + dt * du

    func(du, gg, p, t + dt, args...)
    @. gg = 3 * u / 4 + (gg + dt * du) / 4

    func(du, gg, p, t + dt / 2, args...)
    @. u = u / 3 + 2 * (gg + dt * du) / 3
    return nothing
end


# in place for CUDA kernels
function step!(integ::IntegratorSSPRK3, u::CuDeviceArray, t, dt, args...)
    func = integ.prob.func
    p = integ.prob.p

    du, gg = integ.du, integ.gg

    func(du, u, p, t, args...)
    for i in eachindex(u)
        gg[i] = u[i] + dt * du[i]
    end

    func(du, gg, p, t + dt, args...)
    for i in eachindex(u)
        gg[i] = 3 * u[i] / 4 + (gg[i] + dt * du[i]) / 4
    end

    func(du, gg, p, t + dt / 2, args...)
    for i in eachindex(u)
        u[i] = u[i] / 3 + 2 * (gg[i] + dt * du[i]) / 3
    end
    return nothing
end


# ******************************************************************************
# SSP4RK3 (four-stage SSPRK3)
# Dale R. Durran, “Numerical Methods for Fluid Dynamics”, Springer, 2nd ed.
# (2010), p. 56
# https://link.springer.com/book/10.1007/978-1-4419-6412-0
# ******************************************************************************
struct SSP4RK3 <: Algorithm end


struct IntegratorSSP4RK3{F, U, P} <: Integrator
    prob :: Problem{F, U, P}
    du :: U
    gg :: U
end

@adapt_structure IntegratorSSP4RK3


function Integrator(prob::Problem, alg::SSP4RK3)
    du, gg = zero(prob.u0), zero(prob.u0)
    return IntegratorSSP4RK3(prob, du, gg)
end


# out of place
function step(integ::IntegratorSSP4RK3, u, t, dt, args...)
    func = integ.prob.func
    p = integ.prob.p

    du, gg = integ.du, integ.gg

    du = func(u, p, t, args...)
    gg = u + dt / 2 * du

    du = func(gg, p, t + dt / 2, args...)
    gg = gg + dt / 2 * du

    du = func(gg, p, t + dt, args...)
    gg = 2 * u / 3 + gg / 3 + dt / 6 * du

    du = func(gg, p, t + dt / 2, args...)
    return gg + dt / 2 * du
end


# in place
function step!(integ::IntegratorSSP4RK3, u, t, dt, args...)
    func = integ.prob.func
    p = integ.prob.p

    du, gg = integ.du, integ.gg

    func(du, u, p, t, args...)
    @. gg = u + dt / 2 * du

    func(du, gg, p, t + dt / 2, args...)
    @. gg = gg + dt / 2 * du

    func(du, gg, p, t + dt, args...)
    @. gg = 2 * u / 3 + gg / 3 + dt / 6 * du

    func(du, gg, p, t + dt / 2, args...)
    @. u = gg + dt / 2 * du
    return nothing
end


# in place for CUDA kernels
function step!(integ::IntegratorSSP4RK3, u::CuDeviceArray, t, dt, args...)
    func = integ.prob.func
    p = integ.prob.p

    du, gg = integ.du, integ.gg

    func(du, u, p, t, args...)
    for i in eachindex(u)
        gg[i] = u[i] + dt / 2 * du[i]
    end

    func(du, gg, p, t + dt / 2, args...)
    for i in eachindex(u)
        gg[i] = gg[i] + dt / 2 * du[i]
    end

    func(du, gg, p, t + dt, args...)
    for i in eachindex(u)
        gg[i] = 2 * u[i] / 3 + gg[i] / 3 + dt / 6 * du[i]
    end

    func(du, gg, p, t + dt / 2, args...)
    for i in eachindex(u)
        u[i] = gg[i] + dt / 2 * du[i]
    end
    return nothing
end


# ******************************************************************************
# RK4
# ******************************************************************************
struct RK4 <: Algorithm end


struct IntegratorRK4{F, U, P} <: Integrator
    prob :: Problem{F, U, P}
    k1 :: U
    k2 :: U
    k3 :: U
    k4 :: U
    utmp :: U
end

@adapt_structure IntegratorRK4


function Integrator(prob::Problem, alg::RK4)
    k1, k2, k3, k4, utmp = [zero(prob.u0) for i in 1:5]
    return IntegratorRK4(prob, k1, k2, k3, k4, utmp)
end


# out of place
function step(integ::IntegratorRK4, u, t, dt, args...)
    (; func, p) = integ.prob

    k1 = func(u, p, t, args...)

    utmp = u + dt * k1 / 2
    ttmp = t + dt / 2
    k2 = func(utmp, p, ttmp, args...)

    utmp = u + dt * k2 / 2
    ttmp = t + dt / 2
    k3 = func(utmp, p, ttmp, args...)

    utmp = u + dt * k3
    ttmp = t + dt
    k4 = func(utmp, p, ttmp, args...)

    return u + dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
end


# in place
function step!(integ::IntegratorRK4, u, t, dt, args...)
    (; func, p) = integ.prob
    (; k1, k2, k3, k4, utmp) = integ

    func(k1, u, p, t, args...)

    @. utmp = u + dt * k1 / 2
    ttmp = t + dt / 2
    func(k2, utmp, p, ttmp, args...)

    @. utmp = u + dt * k2 / 2
    ttmp = t + dt / 2
    func(k3, utmp, p, ttmp, args...)

    @. utmp = u + dt * k3
    ttmp = t + dt
    func(k4, utmp, p, ttmp, args...)

    @. u = u + dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
    return nothing
end


# in place for CUDA kernels
function step!(integ::IntegratorRK4, u::CuDeviceArray, t, dt, args...)
    (; func, p) = integ.prob
    (; k1, k2, k3, k4, utmp) = integ

    func(k1, u, p, t, args...)

    for i in eachindex(u)
        utmp[i] = u[i] + dt * k1[i] / 2
    end
    ttmp = t + dt / 2
    func(k2, utmp, p, ttmp, args...)

    for i in eachindex(u)
        utmp[i] = u[i] + dt * k2[i] / 2
    end
    ttmp = t + dt / 2
    func(k3, utmp, p, ttmp, args...)

    for i in eachindex(u)
        utmp[i] = u[i] + dt * k3[i]
    end
    ttmp = t + dt
    func(k4, utmp, p, ttmp, args...)

    for i in eachindex(u)
        u[i] = u[i] + dt * (k1[i] / 6 + k2[i] / 3 + k3[i] / 3 + k4[i] / 6)
    end
    return nothing
end


# ******************************************************************************
# Tsit5
# ******************************************************************************
struct Tsit5 <: Algorithm end


function tableau_tsit5(T::Type)
    as = SVector{15, T}(
        0.161,   # a21
        -0.008480655492356989,   # a31
        0.335480655492357,   # a32
        2.8971530571054935,   # a41
        -6.359448489975075,   # a42
        4.3622954328695815,   # a43
        5.325864828439257,   # a51
        -11.748883564062828,   # a52
        7.4955393428898365,   # a53
        -0.09249506636175525,   # a54
        5.86145544294642,   # a61
        -12.92096931784711,   # a62
        8.159367898576159,   # a63
        -0.071584973281401,   # a64
        -0.028269050394068383,   # a65
    )
    bs = SVector{6, T}(
        0.09646076681806523,   # b1
        0.01,   # b2
        0.4798896504144996,   # b3
        1.379008574103742,   # b4
        -3.290069515436081,   # b5
        2.324710524099774,   # b6
    )
    cs = SVector{5, T}(0.161, 0.327, 0.9, 0.9800255409045097, 1)
    return as, bs, cs
end


struct IntegratorTsit5{F, U, P, T} <: Integrator
    prob :: Problem{F, U, P}
    as :: SVector{15, T}
    bs :: SVector{6, T}
    cs :: SVector{5, T}
    k1 :: U
    k2 :: U
    k3 :: U
    k4 :: U
    k5 :: U
    k6 :: U
    utmp :: U
end

@adapt_structure IntegratorTsit5


function Integrator(prob::Problem, alg::Tsit5)
    T = real(eltype(prob.u0))
    as, bs, cs = tableau_tsit5(T)
    k1, k2, k3, k4, k5, k6, utmp = [zero(prob.u0) for i in 1:7]
    return IntegratorTsit5(prob, as, bs, cs, k1, k2, k3, k4, k5, k6, utmp)
end


# out of place
function step(integ::IntegratorTsit5, u, t, dt, args...)
    (; func, p) = integ.prob
    (; as, bs, cs) = integ
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65 = as
    b1, b2, b3, b4, b5, b6 = bs
    c2, c3, c4, c5, c6 = cs

    k1 = func(u, p, t, args...)

    utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    k2 = func(utmp, p, ttmp, args...)

    utmp = u + dt * (a31 * k1 + a32 * k2)
    ttmp = t + c3 * dt
    k3 = func(utmp, p, ttmp, args...)

    utmp = u + dt * (a41 * k1 + a42 * k2 + a43 * k3)
    ttmp = t + c4 * dt
    k4 = func(utmp, p, ttmp, args...)

    utmp = u + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
    ttmp = t + c5 * dt
    k5 = func(utmp, p, ttmp, args...)

    utmp = u + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
    ttmp = t + c6 * dt
    k6 = func(utmp, p, ttmp, args...)

    return u + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
end


# in place
function step!(integ::IntegratorTsit5, u, t, dt, args...)
    (; func, p) = integ.prob
    (; as, bs, cs, k1, k2, k3, k4, k5, k6, utmp) = integ
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65 = as
    b1, b2, b3, b4, b5, b6 = bs
    c2, c3, c4, c5, c6 = cs

    func(k1, u, p, t, args...)

    @. utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    func(k2, utmp, p, ttmp, args...)

    @. utmp = u + dt * (a31 * k1 + a32 * k2)
    ttmp = t + c3 * dt
    func(k3, utmp, p, ttmp, args...)

    @. utmp = u + dt * (a41 * k1 + a42 * k2 + a43 * k3)
    ttmp = t + c4 * dt
    func(k4, utmp, p, ttmp, args...)

    @. utmp = u + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
    ttmp = t + c5 * dt
    func(k5, utmp, p, ttmp, args...)

    @. utmp = u + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
    ttmp = t + c6 * dt
    func(k6, utmp, p, ttmp, args...)

    @. u = u + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    return nothing
end


# in place for CUDA kernels
function step!(integ::IntegratorTsit5, u::CuDeviceArray, t, dt, args...)
    (; func, p) = integ.prob
    (; as, bs, cs, k1, k2, k3, k4, k5, k6, utmp) = integ
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65 = as
    b1, b2, b3, b4, b5, b6 = bs
    c2, c3, c4, c5, c6 = cs

    func(k1, u, p, t, args...)

    for i in eachindex(u)
        utmp[i] = u[i] + dt * a21 * k1[i]
    end
    ttmp = t + c2 * dt
    func(k2, utmp, p, ttmp, args...)

    for i in eachindex(u)
        utmp[i] = u[i] + dt * (a31 * k1[i] + a32 * k2[i])
    end
    ttmp = t + c3 * dt
    func(k3, utmp, p, ttmp, args...)

    for i in eachindex(u)
        utmp[i] = u[i] + dt * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i])
    end
    ttmp = t + c4 * dt
    func(k4, utmp, p, ttmp, args...)

    for i in eachindex(u)
        utmp[i] = u[i] + dt * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] +
                               a54 * k4[i])
    end
    ttmp = t + c5 * dt
    func(k5, utmp, p, ttmp, args...)

    for i in eachindex(u)
        utmp[i] = u[i] + dt * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] +
                               a64 * k4[i] + a65 * k5[i])
    end
    ttmp = t + c6 * dt
    func(k6, utmp, p, ttmp, args...)

    for i in eachindex(u)
        u[i] = u[i] + dt * (b1 * k1[i] + b2 * k2[i] + b3 * k3[i] + b4 * k4[i] +
                            b5 * k5[i] + b6 * k6[i])
    end
    return nothing
end


# ******************************************************************************
# ATsit5
# ******************************************************************************
struct ATsit5 <: Algorithm end


function tableau_atsit5(T::Type)
    as, bs, cs = tableau_tsit5(T)
    bhats = SVector{6, T}(
        0.00178001105222577714,   # bhat1
        0.0008164344596567469,   # bhat2
        -0.007880878010261995,   # bhat3
        0.1447110071732629,   # bhat4
        -0.5823571654525552,   # bhat5
        0.45808210592918697,   # bhat6
    )
    return as, bs, cs, bhats
end


struct IntegratorATsit5{F, U, P, T} <: Integrator
    prob :: Problem{F, U, P}
    as :: SVector{15, T}
    bs :: SVector{6, T}
    cs :: SVector{5, T}
    bhats :: SVector{6, T}
    k1 :: U
    k2 :: U
    k3 :: U
    k4 :: U
    k5 :: U
    k6 :: U
    utmp :: U
    uhat :: U
    etmp :: U
    atol :: T   # absolute tolerance
    rtol :: T   # relative tolerance
end

@adapt_structure IntegratorATsit5


function Integrator(prob::Problem, alg::ATsit5)
    T = real(eltype(prob.u0))
    as, bs, cs, bhats = tableau_atsit5(T)
    k1, k2, k3, k4, k5, k6, utmp, uhat, etmp = [zero(prob.u0) for i in 1:9]
    atol = convert(T, 1e-2)   # absolute tolerance
    rtol = convert(T, 1e-2)   # relative tolerance
    return IntegratorATsit5(
        prob, as, bs, cs, bhats, k1, k2, k3, k4, k5, k6, utmp, uhat, etmp,
        atol, rtol,
    )
end


# out of place
function substep(integ::IntegratorATsit5, u, t, dt, args...)
    (; func, p) = integ.prob
    (; as, bs, cs, bhats, atol, rtol) = integ
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65 = as
    b1, b2, b3, b4, b5, b6 = bs
    c2, c3, c4, c5, c6 = cs
    bhat1, bhat2, bhat3, bhat4, bhat5, bhat6 = bhats

    err = Inf
    utmp = zero(u)

    while err > 1
        k1 = func(u, p, t, args...)

        utmp = u + dt * a21 * k1
        ttmp = t + c2 * dt
        k2 = func(utmp, p, ttmp, args...)

        utmp = u + dt * (a31 * k1 + a32 * k2)
        ttmp = t + c3 * dt
        k3 = func(utmp, p, ttmp, args...)

        utmp = u + dt * (a41 * k1 + a42 * k2 + a43 * k3)
        ttmp = t + c4 * dt
        k4 = func(utmp, p, ttmp, args...)

        utmp = u + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        ttmp = t + c5 * dt
        k5 = func(utmp, p, ttmp, args...)

        utmp = u + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
        ttmp = t + c6 * dt
        k6 = func(utmp, p, ttmp, args...)

        utmp = u + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 +
                         b6 * k6)

        # Error estimation:
        uhat = u + dt * (bhat1 * k1 + bhat2 * k2 + bhat3 * k3 + bhat4 * k4 +
                         bhat5 * k5 + bhat6 * k6)

        err = abs(utmp - uhat) / (atol + rtol * max(abs(u), abs(utmp)))
        if err > 1
            # dt = 0.9 * dt / err^(1/5)
            dt = convert(typeof(dt), 0.9) * dt / err^convert(typeof(dt), 0.2)
        end
    end

    return utmp, t + dt
end


# out of place
function step(integ::IntegratorATsit5, u, t, dt, args...)
    tend = t + dt
    usub, tsub = substep(integ, u, t, dt, args...)
    while tsub < tend
        dtsub = tend - tsub
        usub, tsub = substep(integ, usub, tsub, dtsub, args...)
    end
    return usub
end


# in place
function substep!(integ::IntegratorATsit5, u, t, dt, args...)
    (; func, p) = integ.prob
    (; as, bs, cs, bhats, atol, rtol) = integ
    (; k1, k2, k3, k4, k5, k6, utmp, uhat, etmp) = integ
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65 = as
    b1, b2, b3, b4, b5, b6 = bs
    c2, c3, c4, c5, c6 = cs
    bhat1, bhat2, bhat3, bhat4, bhat5, bhat6 = bhats

    err = Inf

    while err > 1
        func(k1, u, p, t, args...)

        @. utmp = u + dt * a21 * k1
        ttmp = t + c2 * dt
        func(k2, utmp, p, ttmp, args...)

        @. utmp = u + dt * (a31 * k1 + a32 * k2)
        ttmp = t + c3 * dt
        func(k3, utmp, p, ttmp, args...)

        @. utmp = u + dt * (a41 * k1 + a42 * k2 + a43 * k3)
        ttmp = t + c4 * dt
        func(k4, utmp, p, ttmp, args...)

        @. utmp = u + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        ttmp = t + c5 * dt
        func(k5, utmp, p, ttmp, args...)

        @. utmp = u + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 +
                            a65 * k5)
        ttmp = t + c6 * dt
        func(k6, utmp, p, ttmp, args...)

        @. utmp = u + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 +
                            b6 * k6)

        # Error estimation:
        @. uhat = u + dt * (bhat1 * k1 + bhat2 * k2 + bhat3 * k3 + bhat4 * k4 +
                            bhat5 * k5 + bhat6 * k6)

        @. etmp = abs(utmp - uhat) / (atol + rtol * max(abs(u), abs(utmp)))
        err = sqrt(sum(abs2, etmp) / length(etmp))
        if err > 1
            # dt = 0.9 * dt / err^(1/5)
            dt = convert(typeof(dt), 0.9) * dt / err^convert(typeof(dt), 0.2)
        end
    end

    @. u = utmp
    return t + dt
end


# in place for CUDA kernels
function substep!(integ::IntegratorATsit5, u::CuDeviceArray, t, dt, args...)
    (; func, p) = integ.prob
    (; as, bs, cs, bhats, atol, rtol) = integ
    (; k1, k2, k3, k4, k5, k6, utmp, uhat, etmp) = integ
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65 = as
    b1, b2, b3, b4, b5, b6 = bs
    c2, c3, c4, c5, c6 = cs
    bhat1, bhat2, bhat3, bhat4, bhat5, bhat6 = bhats

    err = Inf

    while err > 1
        func(k1, u, p, t, args...)

        for i in eachindex(u)
            utmp[i] = u[i] + dt * a21 * k1[i]
        end
        ttmp = t + c2 * dt
        func(k2, utmp, p, ttmp, args...)

        for i in eachindex(u)
            utmp[i] = u[i] + dt * (a31 * k1[i] + a32 * k2[i])
        end
        ttmp = t + c3 * dt
        func(k3, utmp, p, ttmp, args...)

        for i in eachindex(u)
            utmp[i] = u[i] + dt * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i])
        end
        ttmp = t + c4 * dt
        func(k4, utmp, p, ttmp, args...)

        for i in eachindex(u)
            utmp[i] = u[i] + dt * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] +
                                   a54 * k4[i])
        end
        ttmp = t + c5 * dt
        func(k5, utmp, p, ttmp, args...)

        for i in eachindex(u)
            utmp[i] = u[i] + dt * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] +
                                   a64 * k4[i] + a65 * k5[i])
        end
        ttmp = t + c6 * dt
        func(k6, utmp, p, ttmp, args...)

        for i in eachindex(u)
            utmp[i] = u[i] + dt * (b1 * k1[i] + b2 * k2[i] + b3 * k3[i] +
                                   b4 * k4[i] + b5 * k5[i] + b6 * k6[i])
        end

        # Error estimation:
        for i in eachindex(u)
            uhat[i] = u[i] + dt * (bhat1 * k1[i] + bhat2 * k2[i] +
                                   bhat3 * k3[i] + bhat4 * k4[i] +
                                   bhat5 * k5[i] + bhat6 * k6[i])
        end

        for i in eachindex(u)
            etmp[i] = abs(utmp[i] - uhat[i]) /
                      (atol + rtol * max(abs(u[i]), abs(utmp[i])))
        end
        err = sqrt(sum(abs2, etmp) / length(etmp))
        if err > 1
            # dt = 0.9 * dt / err^(1/5)
            dt = convert(typeof(dt), 0.9) * dt / err^convert(typeof(dt), 0.2)
        end
    end

    for i in eachindex(u)
        u[i] = utmp[i]
    end
    return t + dt
end


# in place
function step!(integ::IntegratorATsit5, u, t, dt, args...)
    tend = t + dt
    tsub = substep!(integ, u, t, dt, args...)
    while tsub < tend
        dtsub = tend - tsub
        tsub = substep!(integ, u, tsub, dtsub, args...)
    end
    return nothing
end


end
