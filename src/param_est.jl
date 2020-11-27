using DrWatson
using LinearAlgebra
using Distributions
using ForwardDiff
using Turing
using Optim
using HCubature

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
    wts = zeros(N, T + 1)

    At = A(theta)
    Ht = H(theta)
    Qt = Q(theta)
    Rt = R(theta)
    Pt = P(theta)
    mt = m(theta)

    prior_samples = bsf_draw_init(mt, Pt, N)
    xs = zeros(size(prior_samples, 1), size(prior_samples, 2), T + 1)

    xs[:, :, 1] = prior_samples
    wts[:, 1] .= 1.0 / N
    v = zeros(N)

    slog = 0.0
    l_est = zeros(T)

    yk = zeros(size(y, 2))

    for t = 1:T
        yk = y[t, :]
        importance_samples = bsf_redraw(xs[:, :, t], Qt, At)
        xs[:, :, t+1] = importance_samples
        for i = 1:N
            y_distn = MvNormal(Ht(importance_samples[i, :]), Rt)
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

function rapid_mvn_prec(
    x::Vector{Float64},
    mu::Vector{Float64},
    i_sigma::Matrix{Float64},
    isq_d_sigma::Float64,
)
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
            j_wts[i] =
                weights[i] *
                r_mvn_pdf(reference_x[:, t], psi(trajectories[:, i, t-1]), _Q1, _Q2)
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

function log_lh_weights(weights::Vector{Float64})
    N = length(weights)
    log_LH = log(sum(weights)) - log(N)
    return log_LH
end

function log_lh_weights(weights::Matrix{Float64})
    N = size(weights, 1)
    T = size(weights, 2)
    log_LH = 0.0
    for t = 1:T
        log_LH += log(sum(weights[:, t])) - log(N)
    end
    return log_LH
end

function rand_symmat(n::Int64)
    rv = randn(n, n)
    rv = Symmetric(rv)
    rv = Matrix(rv)
    return rv
end


function naive_pmmh_run(m0, P0, Q0, R0, y, psi, H, dr, iterations)
    T = size(y, 2)

    nxs = bsf_draw_init(m0, P0, dr)

    n = size(m0, 1)
    o_s = size(y, 1)

    kl_m = zeros(n, T)
    wts = zeros(dr, T)
    sds = zeros(dr, n, T)

    all_paths = zeros(n, dr, T)
    selected_paths = zeros(n, T, iterations)

    ll_list = zeros(iterations)
    # all_paths[:, :, 1] = nxs

    wv = zeros(dr)
    for k = 1:T
        m, P, nxs, wv, xpf = bsf_step(nxs, P0, Q0, y[:, k], R0, psi, H)
        kl_m[:, k] = m
        wts[:, k] = wv
        sds[:, :, k] = xpf
        all_paths[:, :, k] = nxs'
    end

    path_ind = wsample(1:dr, wv, 1)
    path = all_paths[:, path_ind, :][:, 1, :]

    m_list = zeros(n, iterations)
    P_list = zeros(n, n, iterations)
    Q_list = zeros(n, n, iterations)
    R_list = zeros(o_s, o_s, iterations)

    m_list[:, 1] = m0
    P_list[:, :, 1] = sqrt(P0)
    Q_list[:, :, 1] = sqrt(Q0)
    R_list[:, :, 1] = sqrt(R0)
    ll_list[1] = log_lh_weights(wts)
    selected_paths[:, :, 1] = path

    for i = 2:iterations
        # m_list[:, i] = m_list[:, i-1] + randn(n)
        # P_list[:, :, i] = P_list[:, :, i-1] + rand_symmat(n)
        # Q_list[:, :, i] = Q_list[:, :, i-1] + rand_symmat(n)
        # R_list[:, :, i] = R_list[:, :, i-1] + rand_symmat(o_s)
        ms = m_list[:, i-1] + randn(n)
        sPs = P_list[:, :, i-1] + rand_symmat(n)
        sQs = Q_list[:, :, i-1] + rand_symmat(n)
        sRs = R_list[:, :, i-1] + rand_symmat(o_s)

        Ps = sPs^2
        Qs = sQs^2
        Rs = sRs^2

        nxs = bsf_draw_init(ms, Ps, dr)

        for k = 1:T
            m, P, nxs, wv, xpf = bsf_step(nxs, Ps, Qs, y[:, k], Rs, psi, H)
            kl_m[:, k] = m
            wts[:, k] = wv
            sds[:, :, k] = xpf
            all_paths[:, :, k] = nxs'
        end

        path_ind = wsample(1:dr, wv, 1)
        path = all_paths[:, path_ind, :][:, 1, :]

        l_lh = log_lh_weights(wts)

        accept_prob = min(1.0, l_lh / ll_list[i-1])
        accept = (rand() <= accept_prob)

        if accept
            m_list[:, i] = ms
            P_list[:, :, i] = sPs
            Q_list[:, :, i] = sQs
            R_list[:, :, i] = sRs
            selected_paths[:, :, i] = path
        else
            m_list[:, i] = m_list[:, i-1]
            P_list[:, :, i] = P_list[:, :, i-1]
            Q_list[:, :, i] = Q_list[:, :, i-1]
            R_list[:, :, i] = R_list[:, :, i-1]
            selected_paths[:, :, i] = selected_paths[:, :, i-1]
        end
    end

    return @dict m_list P_list Q_list R_list selected_paths
end

#calculates p(y_k, x_k|y_k-1, θ) = p(y_k|x_k, θ)*p(x_k|y_k-1, θ), must be integrated over x_k to obtain marginalised
function state_llh(
    previous_state::Vector{Float64},
    current_state,
    current_obs::Vector{Float64},
    step_function::Function,
    obs_function::Function,
    process_cov::Matrix{Float64},
    obs_cov::Matrix{Float64},
)
    predictive_state = step_function(previous_state)
    predictive_obs = obs_function(predictive_state)

    current_state_dist = MvNormal(predictive_state, process_cov)
    obs_dist = MvNormal(predictive_obs, obs_cov)

    lh = logpdf(obs_dist, current_obs) + logpdf(current_state_dist, current_state)
    return lh
end

function marginal_lh(
    previous_state::Vector{Float64},
    current_obs::Vector{Float64},
    step_function::Function,
    obs_function::Function,
    process_cov::Matrix{Float64},
    obs_cov::Matrix{Float64},
    state_lower::Vector{Float64},
    state_upper::Vector{Float64},
)
    unmarg_lh(x) = exp(state_llh(
        previous_state,
        x,
        current_obs,
        step_function,
        obs_function,
        process_cov,
        obs_cov,
    ))

    marg_lh = hcubature(unmarg_lh, state_lower, state_upper, initdiv = 5)
    # @info("Cubature Complete")
    return marg_lh[1]
end
