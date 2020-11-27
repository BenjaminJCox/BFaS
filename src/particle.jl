#file containing code used in particle filtering

using LinearAlgebra
using Distributions
using Statistics
using Random
using Bootstrap
using StatsBase

# function bsf_draw(x::Vector{Float64}, Q::Matrix{Float64}, psi::Function = x -> x, n::Int64 = 5000; adapt::Int64 = 1_000, accept::Float64 = 0.65)
#     @model dyn_mod(x::Vector{Float64}, Q::Matrix{Float64}, psi::Function) = begin
#         xk ~ psi(x) + MvNormal(Q)
#     end
#     sps = Array(sample(dyn_mod(x, Q, psi), NUTS(adapt, accept), n + adapt))
#     return sps
# end

include(srcdir("kf.jl")) # for proposals

function bsf_draw_init(x::Vector{Float64}, P::Matrix{Float64}, n::Int64 = 5000)
    xk = zeros(n, size(x, 1))
    prior = MvNormal(x, P)
    for i = 1:n
        xk[i, :] = rand(prior)
    end
    return xk
end

function bsf_redraw(x::Matrix{Float64}, Q::Matrix{Float64}, psi::Function)
    n = size(x, 1)
    xp = mapslices(psi, x, dims = 2)
    prior = MvNormal(Q)
    @simd for i = 1:n
        xp[i, :] += rand(prior)
    end
    return xp
end

function bsf_weights(
    x::Matrix{Float64},
    y::Vector{Float64},
    H::Function,
    R::Matrix{Float64},
)
    x_obs_transform = mapslices(H, x, dims = 2)
    n = size(x_obs_transform, 1)
    weight_vector = zeros(size(x_obs_transform, 1))
    @simd for i = 1:n
        xi = @views x_obs_transform[i, :]
        weight_vector[i] = pdf(MvNormal(xi, R), y)
    end
    weight_vector ./= sum(weight_vector)
    return weight_vector
end

function eff_particles(weights::Vector{Float64})
    wsq = weights .^ 2
    ism = sum(wsq)
    return 1.0 / ism
end

function stratified_resample(x::Matrix{Float64}, weights::Vector{Float64})
    n = size(weights, 1)
    positions = (rand(n) + collect(0:(n-1))) / n
    indexes = zeros(Int64, n)
    cs = cumsum(weights)
    (i, j) = (1, 1)
    while i <= n
        if positions[i] < cs[j]
            indexes[i] = j
            i += 1
        else
            j += 1
        end
    end
    return x[indexes, :]
end

function resample(x::Matrix{Float64}, weights::Vector{Float64})
    n = size(weights, 1)
    inds = collect(1:n)
    ns = sample(inds, pweights(weights), n)
    return x[ns, :]
end

function bsf_step(
    x::Matrix{Float64},
    P::Matrix{Float64},
    Q::Matrix{Float64},
    y::Vector{Float64},
    R::Matrix{Float64},
    psi::Function,
    H::Function,
)
    n = size(x, 1)
    xpf = bsf_redraw(x, Q, psi)
    xob = mapslices(H, xpf, dims = 2)
    weight_vector = zeros(size(xob, 1))
    @simd for i = 1:n
        xi = @views xob[i, :]
        weight_vector[i] = pdf(MvNormal(xi, R), y)
    end
    weight_vector .+= eps(Float64) #for stability reasons
    #calculates approximation of p(y_n|y_1:n-1) by summing weights as on pg196 Sarkaa BFaS (previous weights equal)
    weight_sum = sum(weights) / num_particles
    weight_vector ./= sum(weight_vector)
    m = sum(weight_vector .* xpf, dims = 1)
    P = cov(xpf, dims = 1)
    ninds = wsample(collect(1:n), weight_vector, n, replace = true)
    nxs = xpf[ninds, :]
    return (m, P, nxs, weight_vector, xpf, ninds, weight_sum)
end

function apf_init(x::Vector{Float64}, P::Matrix{Float64}, n::Int64 = 5000)
    xk = zeros(n, size(x, 1))
    prior = MvNormal(x, P)
    for i = 1:n
        xk[i, :] = rand(prior)
    end
    wts = repeat([1 / n], n)
    return (xk, wts)
end

function apf_redraw(x::Matrix{Float64}, Q::Matrix{Float64}, psi::Function)
    n = size(x, 1)
    xp = mapslices(psi, x, dims = 2)
    prior = MvNormal(Q)
    @simd for i = 1:n
        xp[i, :] += rand(prior)
    end
    return xp
end

function apf_step(
    x::Matrix{Float64},
    Q::Matrix{Float64},
    y::Vector{Float64},
    R::Matrix{Float64},
    psi::Function,
    H::Function,
    prev_weights::Vector{Float64},
)
    n = size(x, 1)

    x_pred = mapslices(psi, x, dims = 2)
    x_obs = mapslices(H, x_pred, dims = 2)
    weight_vector = zeros(n)

    @simd for i = 1:n
        xi = x_obs[i, :]
        weight_vector[i] = pdf(MvNormal(xi, R), y) * prev_weights[i]
    end

    weight_vector ./= sum(weight_vector)
    index_sample = wsample(1:n, weight_vector, n)
    selected_x = x[index_sample, :]
    selected_x_sim = apf_redraw(selected_x, Q, psi)

    @simd for i = 1:n
        x_num = H(selected_x_sim[i, :])
        x_den = H(selected_x[i, :])
        weight_vector[i] = pdf(MvNormal(x_num, R), y) / pdf(MvNormal(x_den, R), y)
    end
    #calculates approximation of p(y_n|y_1:n-1) by summing weights as on pg196 Sarkaa BFaS
    weight_sum = sum(weights .* prev_weights)
    weight_vector ./= sum(weight_vector)

    m = sum(weight_vector .* selected_x_sim, dims = 1)
    P = cov(weight_vector .* selected_x_sim, dims = 1)

    return (m, P, selected_x_sim, weight_vector, selected_x_sim, index_sample, weight_sum)
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

function iapf_step(
    x::Matrix{Float64},
    Q::Matrix{Float64},
    y::Vector{Float64},
    R::Matrix{Float64},
    psi::Function,
    H::Function,
    prev_weights::Vector{Float64},
)
    n = size(x, 1)

    x_pred = mapslices(psi, x, dims = 2)
    x_obs = mapslices(H, x_pred, dims = 2)

    weight_vector = zeros(n)
    l_weight_vector = zeros(n)
    i_sigma = inv(Q)
    isq_d_sigma = 1.0 / sqrt(det(Q))
    for i = 1:n
        xi = x_obs[i, :]
        xp = x_pred[i, :]
        l_weight_vector[i] = pdf(MvNormal(xi, R), y)
        # helper(x) = pdf(MvNormal(x, Q), xp)
        pixty = zeros(n)
        # pixty = mapslices(helper, x_pred, dims = 2)
        @simd for j = 1:n
            # pixty[j] = pdf(MvNormal(x_pred[j, :], Q), xp)
            pixty[j] = rapid_mvn_prec(xp, x_pred[j, :], i_sigma, isq_d_sigma)
        end
        l_weight_vector[i] *= sum(pixty .* prev_weights) / sum(pixty)
    end

    l_weight_vector ./= sum(l_weight_vector)
    index_sample = wsample(1:n, l_weight_vector, n)
    selected_x = x[index_sample, :]
    selected_x_sim = apf_redraw(selected_x, Q, psi)

    for i = 1:n
        x_num = H(selected_x_sim[i, :])
        weight_vector[i] = pdf(MvNormal(x_num, R), y)
        # helper(x) = pdf(MvNormal(x, Q), selected_x_sim[i, :])
        # pixty = mapslices(helper, x_pred, dims = 2)
        pixty = zeros(n)
        @simd for j = 1:n
            # pixty[j] = pdf(MvNormal(x_pred[j, :], Q), selected_x_sim[i, :])
            pixty[j] =
                rapid_mvn_prec(selected_x_sim[i, :], x_pred[j, :], i_sigma, isq_d_sigma)
        end
        weight_vector[i] *= sum(pixty)
        weight_vector[i] /= sum(l_weight_vector .* pixty)
    end

    #calculates approximation of p(y_n|y_1:n-1) by summing weights as on pg196 Sarkaa BFaS
    weight_sum = sum(weight_vector .* prev_weights)
    weight_vector ./= sum(weight_vector)

    m = sum(weight_vector .* selected_x_sim, dims = 1)
    P = cov(weight_vector .* selected_x_sim, dims = 1)

    return (m, P, selected_x_sim, weight_vector, weight_sum)
end

#mh kernal for use on a per path basis
function mh_kernel(
    resampled_path::Array{Float64,2},
    obs_hist::Array{Float64,2},
    psi::Function,
    H_func::Function,
    Q::Matrix{Float64},
    R::Matrix{Float64},
    lag::Int64,
    mh_steps::Int64;
    e_cov,
)
    _xdim = size(resampled_path, 1)
    _T = size(resampled_path, 2)

    jittered_path = copy(resampled_path)
    # det_q = det(e_cov)
    # H_distn = MvNormal(Q)

    for m = 1:mh_steps
        # @info("Metropolis Step")
        H = 2. .* randn(_xdim, lag)
        # H = zeros(_xdim, lag)
        # for i = 1:lag
        #     H[:,i] = rand(H_distn)
        # end

        mh_proposal = copy(jittered_path)
        mh_proposal[:, (_T-lag+1):_T] .+= H
        l_acc_num = 0.0
        l_acc_den = 0.0

        for k = (_T-lag+1):_T
            f_num = MvNormal(psi(mh_proposal[:, k-1]), Q)
            f_den = MvNormal(psi(jittered_path[:, k-1]), Q)

            g_num = MvNormal(H_func(mh_proposal[:, k]), R)
            g_den = MvNormal(H_func(jittered_path[:, k]), R)

            l_acc_num += logpdf(f_num, mh_proposal[:, k]) + logpdf(g_num, obs_hist[:, k])
            l_acc_den += logpdf(f_den, jittered_path[:, k]) + logpdf(g_den, obs_hist[:, k])
        end

        l_acc_rat = l_acc_num - l_acc_den
        _a = rand()
        if log(_a) <= l_acc_rat
            jittered_path = copy(mh_proposal)
        end
    end

    return jittered_path
end

# busted
function resample_move_MH_pf_step(
    x::Matrix{Float64},
    P::Matrix{Float64},
    Q::Matrix{Float64},
    y::Vector{Float64},
    R::Matrix{Float64},
    psi::Function,
    H::Function,
    all_particles::Array{Float64,3},
    all_obs::Array{Float64,2},
    lag::Int64,
    mh_steps::Int64,
)

    t = size(all_particles, 3)
    num_particles = size(x, 1)
    sirres = sir_filter_ukfprop(x, P, Q, y, R, psi, H)
    p_mean = sirres[1]
    p_cov = sirres[2]
    samples = sirres[3]
    weights = sirres[4]
    weight_sum = sirres[6]
    select_indices = sirres[5]

    resampled_paths = cat(all_particles[select_indices, :, :], samples, dims = 3)

    if t <= lag
        return (sirres...,  resampled_paths)
    end


    obs_hist = cat(all_obs, y, dims = 2)
    for i = 1:num_particles
        # @info(resampled_paths[i,:,:])
        resampled_paths[i, :, :] =
            mh_kernel(resampled_paths[i, :, :], obs_hist, psi, H, Q, R, lag, mh_steps, e_cov = p_cov)
    end

    # p_mean = sum(inv(num_particles) .* resampled_paths[:, :, t], dims = 1)
    # p_cov = cov(inv(num_particles) .* resampled_paths[:, :, t], dims = 1)

    samples = resampled_paths[:, :, end]
    p_mean = sum(weights .* samples, dims = 1)
    p_cov = cov(weights .* samples, dims = 1)

    return (p_mean, p_cov, samples, weights, select_indices, weight_sum, resampled_paths)
end

function sir_filter_ukfprop(
    x::Matrix{Float64},
    P::Matrix{Float64},
    Q::Matrix{Float64},
    y::Vector{Float64},
    R::Matrix{Float64},
    psi::Function,
    H::Function;
)
    num_particles = size(x, 1)
    _xdim = size(x, 2)
    _ydim = size(y, 1)


    f_means = zeros(num_particles, _xdim)
    f_covs = zeros(num_particles, _xdim, _xdim)
    sqrtP = sqrt(P)

    @simd for i = 1:num_particles
        x_m = copy(x[i, :])
        predicted_mean, predicted_cov = ukf_predict_sqrt(x_m, sqrtP, psi, Q)
        filtered_mean, filtered_cov = ukf_update(predicted_mean, predicted_cov, y, H, R)
        filtered_cov = Matrix(Hermitian(filtered_cov))
        f_means[i, :] = filtered_mean
        f_covs[i, :, :] = filtered_cov
    end

    # proposal_samples = rand(proposal_dist, num_particles)
    # proposal_samples = Matrix(transpose(proposal_samples))
    proposal_samples = zeros(num_particles, _xdim)
    @simd for i = 1:num_particles
        proposal_dist = MvNormal(f_means[i, :], f_covs[i, :, :])
        proposal_samples[i, :] = rand(proposal_dist)
    end

    implied_obs = zeros(num_particles, _ydim)

    @simd for i = 1:num_particles
        implied_obs[i, :] = H(proposal_samples[i, :])
    end

    log_weights = zeros(num_particles)
    @simd for i = 1:num_particles
        @views _x = proposal_samples[i, :]
        @views _impY = implied_obs[i, :]
        model_dist = MvNormal(psi(x[i, :]), Q)
        obs_dist = MvNormal(_impY, R)
        proposal_dist = MvNormal(f_means[i, :], f_covs[i, :, :])
        log_weights[i] =
            logpdf(model_dist, _x) + logpdf(obs_dist, y) - logpdf(proposal_dist, _x)
    end
    # cst = maximum(log_weights)
    # weights = exp.(log_weights .- cst)
    weights = exp.(log_weights)
    # weights .+= eps(Float64)

    #calculates approximation of p(y_n|y_1:n-1) by summing weights as on pg196 Sarkaa BFaS (previous weights equal)
    weight_sum = sum(weights) / num_particles
    weights ./= sum(weights)

    sample_inds = wsample(1:num_particles, weights, num_particles)

    mn = sum(weights .* proposal_samples, dims = 1)
    Pn = cov(weights .* proposal_samples, dims = 1)

    selected_samples = proposal_samples[sample_inds, :]

    # mn = mean(proposal_samples, dims = 1)
    # Pn = cov(selected_samples, dims = 1)


    return (mn, Pn, selected_samples, weights, sample_inds, weight_sum)
end
