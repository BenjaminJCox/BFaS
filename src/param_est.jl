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

function rapid_mvn(x::Vector{Float64}, mu::Vector{Float64}, sigma::Matrix{Float64})
    @assert size(x) == size(mu)
    @assert size(mu, 1) == size(sigma, 1)
    @assert size(sigma, 1) == size(sigma, 2)

    k = size(mu, 1)

    t1 = (2π)^(-k / 2)
    t2 = 1.0 / sqrt(det(sigma))
    lt3 = transpose(x - mu) * inv(sigma) * (x - mu)
    t3 = exp(-0.5 * lt3)
    return (t1 * t2 * t3)
end

function rapid_mvn_prec(x::Vector{Float64}, mu::Vector{Float64}, i_sigma::Matrix{Float64}, isq_d_sigma::Float64)
    k = size(mu, 1)

    t1 = (2π)^(-k / 2)
    t2 = isq_d_sigma
    lt3 = transpose(x - mu) * i_sigma * (x - mu)
    t3 = exp(-0.5 * lt3)
    return (t1 * t2 * t3)
end

r_mvn_pdf = rapid_mvn_prec

function pgas_smooth(
    reference_x::Matrix{Float64},
    y::Matrix{Float64},
    s::Int64,
    psi::Function,
    H::Function,
    Q::Array{Float64},
    R::Array{Float64},
    prior_draws::Matrix{Float64},
)
    x_dim = size(reference_x, 1)
    T = size(reference_x, 2)

    Qq = Q
    if size(Q, 3) == 1
        Qq = repeat(Q, 1, 1, T)
    end

    Rr = R
    if size(R, 3) == 1
        Rr = repeat(R, 1, 1, T)
    end

    Qq_isig = zeros(size(Qq))
    Qq_sqds = zeros(T)
    Rr_isig = zeros(size(Rr))
    Rr_sqds = zeros(T)

    @simd for t = 1:T
        Qq_isig[:, :, t] = inv(Qq[:, :, t])
        Qq_sqds[t] = 1.0 / sqrt(det(Qq[:, :, t]))
        Rr_isig[:, :, t] = inv(Rr[:, :, t])
        Rr_sqds[t] = 1.0 / sqrt(det(Rr[:, :, t]))
    end

    trajectories = zeros(x_dim, s, T)
    x_tilde = zeros(x_dim, s, T)

    trajectories[:, 1:(s-1), 1] = prior_draws
    trajectories[:, s, 1] = reference_x[:, 1]

    weights = zeros(s)

    @simd for i = 1:s
        xi = trajectories[:, i, 1]
        xo = H(xi)
        weights[i] = pdf(MvNormal(xo, Rr[:, :, 1]), y[:, 1])
    end

    weights ./= sum(weights)

    for t = 2:T
        _Q1 = Qq_isig[:, :, t]
        _Q2 = Qq_sqds[t]
        _R1 = Rr_isig[:, :, t]
        _R2 = Rr_sqds[t]
        _y = y[:, t]

        x_tilde_indices = wsample(1:s, weights, s, replace = true)

        x_tilde = trajectories[:, x_tilde_indices, :]

        j_wts = zeros(s)

        @simd for i = 1:s
            j_wts[i] = weights[i] * r_mvn_pdf(reference_x[:, t], psi(trajectories[:, i, t-1]), _Q1, _Q2)
        end

        j_wts ./= sum(j_wts)

        x_tn_j = wsample(1:s, j_wts, 1)

        x_tilde[:, s, :] = trajectories[:, x_tn_j, :]

        new_x = zeros(x_dim, s)

        @simd for i = 1:(s-1)
            _x = trajectories[:, i, t-1]
            _xp = psi(_x)
            distn = MvNormal(_xp, Qq[:, :, t])
            new_x[:, i] = rand(distn)
        end

        new_x[:, s] = reference_x[:, t]
        trajectories = x_tilde
        trajectories[:, :, t] = new_x[:, :]

        @simd for i = 1:s
            _x = trajectories[:, i, t]
            _xo = H(_x)
            weights[i] = r_mvn_pdf(y[:, t], _xo, _R1, _R2)
        end
        weights ./= sum(weights)
    end


    mean_gibbs = zeros(size(reference_x))

    for i = 1:s
        mean_gibbs += weights[s] * trajectories[:, s, :]
    end

    return (trajectories, weights, prior_draws, mean_gibbs)
end
