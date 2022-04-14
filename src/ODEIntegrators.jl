module ODEIntegrators

export Problem, Integrator, step, step!,
       RK2, RK3, SSPRK3, SSP4RK3, RK4, Tsit5, ATsit5

using StaticArrays: SVector


abstract type Algorithm end

abstract type Integrator end


struct Problem{F, U, P}
    func :: F
    u0 :: U
    p :: P
end


# ******************************************************************************
# RK2
# ******************************************************************************
struct RK2 <: Algorithm end


function _tableau_rk2(T::Type)
    as = SVector{1, T}(2/3)
    bs = SVector{2, T}(1/4, 3/4)
    cs = SVector{1, T}(2/3)
    return as, bs, cs
end


struct IntegratorRK2{F, U, P, T} <: Integrator
    prob :: Problem{F, U, P}
    as :: SVector{1, T}
    bs :: SVector{2, T}
    cs :: SVector{1, T}
    ks :: SVector{2, U}
    utmp :: U
end


function Integrator(prob::Problem{F, U, P}, alg::RK2) where {F, U, P}
    u0 = prob.u0
    T = real(eltype(u0))
    as, bs, cs = _tableau_rk2(T)
    ks = SVector{2, U}([zero(u0) for i in 1:2])
    utmp = zero(u0)
    return IntegratorRK2{F, U, P, T}(prob, as, bs, cs, ks, utmp)
end


# in place
function step!(
    integ::IntegratorRK2{F, U, P, T}, u::U, t::T, dt::T, args...,
) where {F, U, P, T}
    func = integ.prob.func
    p = integ.prob.p

    a21, = integ.as
    b1, b2 = integ.bs
    c2, = integ.cs
    k1, k2 = integ.ks
    utmp = integ.utmp

    func(k1, u, p, t, args...)

    @. utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    func(k2, utmp, p, ttmp, args...)

    @. u = u + dt * (b1 * k1 + b2 * k2)
    return nothing
end


# out of place
function step(
    integ::IntegratorRK2{F, U, P, T}, u::U, t::T, dt::T, args...,
) where {F, U, P, T}
    func = integ.prob.func
    p = integ.prob.p

    a21, = integ.as
    b1, b2 = integ.bs
    c2, = integ.cs

    k1 = func(u, p, t, args...)

    utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    k2 = func(utmp, p, ttmp, args...)

    utmp = u + dt * (b1 * k1 + b2 * k2)
    return utmp
end


# ******************************************************************************
# RK3
# ******************************************************************************
struct RK3 <: Algorithm end


function _tableau_rk3(T::Type)
    as = SVector{3, T}(0.5, -1, 2)   # a21, a31, a32
    bs = SVector{3, T}(1/6, 2/3, 1/6)   # b1, b2, b3
    cs = SVector{2, T}(0.5, 1)   # c2, c3
    return as, bs, cs
end


struct IntegratorRK3{F, U, P, T} <: Integrator
    prob :: Problem{F, U, P}
    as :: SVector{3, T}
    bs :: SVector{3, T}
    cs :: SVector{2, T}
    ks :: SVector{3, U}
    utmp :: U
end


function Integrator(prob::Problem{F, U, P}, alg::RK3) where {F, U, P}
    u0 = prob.u0
    T = real(eltype(u0))
    as, bs, cs = _tableau_rk3(T)
    ks = SVector{3, U}([zero(u0) for i in 1:3])
    utmp = zero(u0)
    return IntegratorRK3{F, U, P, T}(prob, as, bs, cs, ks, utmp)
end


# in place
function step!(
    integ::IntegratorRK3{F, U, P, T}, u::U, t::T, dt::T, args...,
) where {F, U, P, T}
    func = integ.prob.func
    p = integ.prob.p

    a21, a31, a32 = integ.as
    b1, b2, b3 = integ.bs
    c2, c3 = integ.cs
    k1, k2, k3 = integ.ks
    utmp = integ.utmp

    func(k1, u, p, t, args...)

    @. utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    func(k2, utmp, p, ttmp, args...)

    @. utmp = u + dt * (a31 * k1 + a32 * k2)
    ttmp = t + c3 * dt
    func(k3, utmp, p, ttmp, args...)

    @. u = u + dt * (b1 * k1 + b2 * k2 + b3 * k3)
    return nothing
end


# out of place
function step(
    integ::IntegratorRK3{F, U, P, T}, u::U, t::T, dt::T, args...,
) where {F, U, P, T}
    func = integ.prob.func
    p = integ.prob.p

    a21, a31, a32 = integ.as
    b1, b2, b3 = integ.bs
    c2, c3 = integ.cs

    k1 = func(u, p, t, args...)

    utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    k2 = func(utmp, p, ttmp, args...)

    utmp = u + dt * (a31 * k1 + a32 * k2)
    ttmp = t + c3 * dt
    k3 = func(utmp, p, ttmp, args...)

    utmp = u + dt * (b1 * k1 + b2 * k2 + b3 * k3)
    return utmp
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


function Integrator(prob::Problem{F, U, P}, alg::SSPRK3) where {F, U, P}
    du, gg = zero(prob.u0), zero(prob.u0)
    return IntegratorSSPRK3(prob, du, gg)
end


# in place
function step!(integ::IntegratorSSPRK3, u, t, dt, args...)
    func = integ.prob.func
    p = integ.prob.p

    du, gg = integ.du, integ.gg

    func(du, u, p, t, args...)
    @. gg = u + dt * du

    func(du, gg, p, t + dt, args...)
    @. gg = 3/4 * u + 1/4 * (gg + dt * du)

    func(du, gg, p, t + dt/2, args...)
    @. u = u / 3 + 2/3 * (gg + dt * du)
    return nothing
end


# out of place
function step(integ::IntegratorSSPRK3, u, t, dt, args...)
    func = integ.prob.func
    p = integ.prob.p

    du = func(u, p, t, args...)
    gg = u + dt * du

    du = func(gg, p, t + dt, args...)
    gg = 3/4 * u + 1/4 * (gg + dt * du)

    du = func(gg, p, t + dt/2, args...)
    return u / 3 + 2/3 * (gg + dt * du)
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


function Integrator(prob::Problem{F, U, P}, alg::SSP4RK3) where {F, U, P}
    du, gg = zero(prob.u0), zero(prob.u0)
    return IntegratorSSP4RK3(prob, du, gg)
end


# in place
function step!(integ::IntegratorSSP4RK3, u, t, dt, args...)
    func = integ.prob.func
    p = integ.prob.p

    du, gg = integ.du, integ.gg

    func(du, u, p, t, args...)
    @. gg = u + 1/2 * dt * du

    func(du, gg, p, t + 1/2 * dt, args...)
    @. gg = gg + 1/2 * dt * du

    func(du, gg, p, t + dt, args...)
    @. gg = 2/3 * u + 1/3 * gg + 1/6 * dt * du

    func(du, gg, p, t + 1/2 * dt, args...)
    @. u = gg + 1/2 * dt * du
    return nothing
end


# out of place
function step(integ::IntegratorSSP4RK3, u, t, dt, args...)
    func = integ.prob.func
    p = integ.prob.p

    du, gg = integ.du, integ.gg

    du = func(u, p, t, args...)
    gg = u + 1/2 * dt * du

    du = func(gg, p, t + 1/2 * dt, args...)
    gg = gg + 1/2 * dt * du

    du = func(gg, p, t + dt, args...)
    gg = 2/3 * u + 1/3 * gg + 1/6 * dt * du

    du = func(gg, p, t + 1/2 * dt, args...)
    return gg + 1/2 * dt * du
end


# ******************************************************************************
# RK4
# ******************************************************************************
struct RK4 <: Algorithm end


function _tableau_rk4(T::Type)
    as = SVector{6, T}(0.5, 0, 0.5, 0, 0, 1)
    bs = SVector{4, T}(1/6, 1/3, 1/3, 1/6)
    cs = SVector{3, T}(0.5, 0.5, 1)
    return as, bs, cs
end


struct IntegratorRK4{F, U, P, T} <: Integrator
    prob :: Problem{F, U, P}
    as :: SVector{6, T}
    bs :: SVector{4, T}
    cs :: SVector{3, T}
    ks :: SVector{4, U}
    utmp :: U
end


function Integrator(prob::Problem{F, U, P}, alg::RK4) where {F, U, P}
    u0 = prob.u0
    T = real(eltype(u0))
    as, bs, cs = _tableau_rk4(T)
    ks = SVector{4, U}([zero(u0) for i in 1:4])
    utmp = zero(u0)
    return IntegratorRK4{F, U, P, T}(prob, as, bs, cs, ks, utmp)
end


# in place
function step!(
    integ::IntegratorRK4{F, U, P, T}, u::U, t::T, dt::T, args...,
) where {F, U, P, T}
    func = integ.prob.func
    p = integ.prob.p

    a21, a31, a32, a41, a42, a43 = integ.as
    b1, b2, b3, b4 = integ.bs
    c2, c3, c4 = integ.cs
    k1, k2, k3, k4 = integ.ks
    utmp = integ.utmp

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

    @. u = u + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4)
    return nothing
end


# out of place
function step(
    integ::IntegratorRK4{F, U, P, T}, u::U, t::T, dt::T, args...,
) where {F, U, P, T}
    func = integ.prob.func
    p = integ.prob.p

    a21, a31, a32, a41, a42, a43 = integ.as
    b1, b2, b3, b4 = integ.bs
    c2, c3, c4 = integ.cs

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

    utmp = u + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4)
    return utmp
end


# ******************************************************************************
# Tsit5
# ******************************************************************************
struct Tsit5 <: Algorithm end


function _tableau_tsit5(T::Type)
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
    ks :: SVector{6, U}
    utmp :: U
end


function Integrator(prob::Problem{F, U, P}, alg::Tsit5) where {F, U, P}
    u0 = prob.u0
    T = real(eltype(u0))
    as, bs, cs = _tableau_tsit5(T)
    ks = SVector{6, U}([zero(u0) for i in 1:6])
    utmp = zero(u0)
    return IntegratorTsit5{F, U, P, T}(prob, as, bs, cs, ks, utmp)
end


# in place
function step!(
    integ::IntegratorTsit5{F, U, P, T}, u::U, t::T, dt::T, args...,
) where {F, U, P, T}
    func = integ.prob.func
    p = integ.prob.p

    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64,
    a65 = integ.as
    b1, b2, b3, b4, b5, b6 = integ.bs
    c2, c3, c4, c5, c6 = integ.cs
    k1, k2, k3, k4, k5, k6 = integ.ks
    utmp = integ.utmp

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


# out of place
function step(
    integ::IntegratorTsit5{F, U, P, T}, u::U, t::T, dt::T, args...,
) where {F, U, P, T}
    func = integ.prob.func
    p = integ.prob.p

    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64,
    a65 = integ.as
    b1, b2, b3, b4, b5, b6 = integ.bs
    c2, c3, c4, c5, c6 = integ.cs

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

    utmp = u + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    return utmp
end


# ******************************************************************************
# ATsit5
# ******************************************************************************
struct ATsit5 <: Algorithm end


function _tableau_atsit5(T::Type)
    as, bs, cs = _tableau_tsit5(T)
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
    ks :: SVector{6, U}
    utmp :: U
    uhat :: U
    etmp :: U
    atol :: T   # absolute tolerance
    rtol :: T   # relative tolerance
end


function Integrator(prob::Problem{F, U, P}, alg::ATsit5) where {F, U, P}
    u0 = prob.u0
    T = real(eltype(u0))
    as, bs, cs, bhats = _tableau_atsit5(T)
    ks = SVector{6, U}([zero(u0) for i in 1:6])
    utmp = zero(u0)
    uhat = zero(u0)
    etmp = zero(u0)
    atol = convert(T, 1e-2)   # absolute tolerance
    rtol = convert(T, 1e-2)   # relative tolerance
    return IntegratorATsit5{F, U, P, T}(
        prob, as, bs, cs, bhats, ks, utmp, uhat, etmp, atol, rtol,
    )
end


# in place
function substep!(
    integ::IntegratorATsit5{F, U, P, T}, u::U, t::T, dt::T, args...,
) where {F, U, P, T}
    func = integ.prob.func
    p = integ.prob.p

    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64,
    a65 = integ.as
    b1, b2, b3, b4, b5, b6 = integ.bs
    c2, c3, c4, c5, c6 = integ.cs
    bhat1, bhat2, bhat3, bhat4, bhat5, bhat6 = integ.bhats
    k1, k2, k3, k4, k5, k6 = integ.ks
    utmp = integ.utmp
    uhat = integ.uhat
    etmp = integ.etmp
    atol = integ.atol
    rtol = integ.rtol

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

        @. etmp = atol + rtol * max(abs(u), abs(utmp))
        @. etmp = abs(utmp - uhat) / etmp
        err = sqrt(sum(abs2, etmp) / length(etmp))
        if err > 1
            dt = convert(T, 0.9) * dt / err^convert(T, 1/5)
        end
    end

    @. u = utmp
    return t + dt
end


# in place
function step!(
    integ::IntegratorATsit5{F, U, P, T}, u::U, t::T, dt::T, args...,
) where {F, U, P, T}
    tend = t + dt

    tsub = substep!(integ, u, t, dt, args...)

    while tsub < tend
        dtsub = tend - tsub
        tsub = substep!(integ, u, tsub, dtsub, args...)
    end

    return nothing
end


# out of place
function substep(
    integ::IntegratorATsit5{F, U, P, T}, u::U, t::T, dt::T, args...,
) where {F, U, P, T}
    func = integ.prob.func
    p = integ.prob.p

    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64,
    a65 = integ.as
    b1, b2, b3, b4, b5, b6 = integ.bs
    c2, c3, c4, c5, c6 = integ.cs
    bhat1, bhat2, bhat3, bhat4, bhat5, bhat6 = integ.bhats
    atol = integ.atol
    rtol = integ.rtol

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

        etmp = @. atol + rtol * max(abs(u), abs(utmp))
        etmp = @. abs(utmp - uhat) / etmp
        err = sqrt(sum(abs2, etmp) / length(etmp))
        if err > 1
            dt = convert(T, 0.9) * dt / err^convert(T, 1/5)
        end
    end

    return utmp, t + dt
end


# out of place
function step(
    integ::IntegratorATsit5{F, U, P, T}, u::U, t::T, dt::T, args...,
) where {F, U, P, T}
    tend = t + dt

    usub, tsub = substep(integ, u, t, dt, args...)

    while tsub < tend
        dtsub = tend - tsub
        usub, tsub = substep(integ, usub, tsub, dtsub, args...)
    end

    return usub
end


end
