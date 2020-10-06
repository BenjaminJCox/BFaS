using DrWatson
using LinearAlgebra
using Distributions
using ForwardDiff
using Turing
using Optim

include("particle.jl")
include("kf.jl")

function d_energy_lGm(
    theta,
    A::Function,
    H::Function,
    Q::Function,
    R::Function,
    m::Vector{Float64},
    P::Matrix{Float64},
    y::Vector{Float64},
)
    At = A(theta)
    Ht = H(theta)
    Qt = Q(theta)
    Rt = R(theta)

    mkd = At * m
    Pkd = At * P * At' + Qt

    vk = y - Ht * mkd
    Sk = Ht * Pkd * Ht' + Rt
    Kk = Pkd * Ht' / Sk
    mk = mkd + Kk * vk
    Pk = Pkd - Kk * Sk * Kk'

    Δe::Float64 = 0.5 * (log(2π * det(Sk)) + vk' / Sk * vk)

    return (mk, Pk, Δe)
end

function sir_energy_approx(
    theta,
    A::Function,
    H::Function,
    Q::Function,
    R::Function,
    m::Function,
    P::Function,
    y::Matrix{Float64},
    N::Int64,
)
    T = size(y, 1)
    wts = zeros(N, T+1)

    At = A(theta)
    Ht = H(theta)
    Qt = Q(theta)
    Rt = R(theta)
    Pt = P(theta)
    mt = m(theta)

    prior_samples = bsf_draw_init(mt, Pt, N)
    xs = zeros(size(prior_samples, 1), size(prior_samples, 2), T+1)

    xs[:, :, 1] = prior_samples
    wts[:, 1] .= 1. / N
    v = zeros(N)

    slog = 0.
    l_est = zeros(T)

    yk = zeros(size(y, 2))

    for t = 1:T
        yk = y[t,:]
        importance_samples = bsf_redraw(xs[:,:,t], Qt, At)
        xs[:, :, t+1] = importance_samples
        for i = 1:N
            y_distn = MvNormal(Ht(importance_samples[i,:]), Rt)
            v[i] = pdf(y_distn, yk)
        end
        l_est[t] = sum(wts[:, t] .* v[:])

        wts[:, t+1] = wts[:, t] .* v[:]
        wts[:, t+1] ./= sum(wts[:, t+1])

        ninds = wsample(collect(1:N), wts[:, t+1], N)
        wts[:, t+1] = wts[ninds, t+1]
        rd_x = xs[ninds, :, t+1]
        xs[:, :, t+1] = rd_x
    end
    en_func = sum(log.(l_est))

    return en_func
end
