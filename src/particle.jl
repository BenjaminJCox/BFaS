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
    weight_vector .+= eps(Float64) #for starbility reasons
    weight_vector ./= sum(weight_vector)
    m = sum(weight_vector .* xpf, dims = 1)
    P = cov(xpf, dims = 1)
    ninds = wsample(collect(1:n), weight_vector, n, replace = true)
    nxs = xpf[ninds, :]
    return (m, P, nxs, weight_vector, xpf, ninds)
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

    weight_vector ./= sum(weight_vector)

    m = sum(weight_vector .* selected_x_sim, dims = 1)
    P = cov(weight_vector .* selected_x_sim, dims = 1)

    return (m, P, selected_x_sim, weight_vector, selected_x_sim, sample_inds)
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

    weight_vector ./= sum(weight_vector)

    m = sum(weight_vector .* selected_x_sim, dims = 1)
    P = cov(weight_vector .* selected_x_sim, dims = 1)

    return (m, P, selected_x_sim, weight_vector)
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
    mh_steps::Int64,
)
    _xdim = size(resampled_path, 1)
    _T = size(resampled_path, 2)

    jittered_path = resampled_path

    for m = 1:mh_steps
        H = randn(_xdim, lag)
        mh_proposal = jittered_path
        mh_proposal[:, (_T-lag+1):_T] .+= H
        println(mh_proposal[:, 3])
        println(H_func(mh_proposal[:, 3]))
        acc_num = 1.0
        acc_den = 1.0

        for k = (_T-lag+1):_T
            f_num = MvNormal(psi(mh_proposal[:, k-1]), Q)
            f_den = MvNormal(psi(jittered_path[:, k-1]), Q)

            g_num = MvNormal(H_func(mh_proposal[:, k]), R)
            g_den = MvNormal(H_func(jittered_path[:, k]), R)

            acc_num *= pdf(f_num, mh_proposal[:, k]) * pdf(g_num, obs_hist[:, k])
            acc_den *= pdf(f_den, jittered_path[:, k]) * pdf(g_den, obs_hist[:, k])
        end

        acc_rat = acc_num / acc_den
        _a = rand()
        if _a <= acc_rat
            jittered_path = mh_proposal
        end
    end

    return jittered_path
end

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

    sirres = sir_filter_ukfprop(x, P, Q, y, R, psi, H)
    p_mean = sirres[1]
    p_cov = sirres[2]
    samples = sirres[3]
    select_indices = sirres[6]

end

function sir_filter_ukfprop(
    x::Matrix{Float64},
    P::Matrix{Float64},
    Q::Matrix{Float64},
    y::Vector{Float64},
    R::Matrix{Float64},
    psi::Function,
    H::Function,
)
    num_particles = size(x, 1)
    _xdim = size(x, 2)
    _ydim = size(y, 1)


    f_means = zeros(num_particles, _xdim)
    f_covs = zeros(num_particles, _xdim, _xdim)
    for i = 1:num_particles
        x_m = x[i, :]
        predicted_mean, predicted_cov = ukf_predict(x_m, P, psi, Q)
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
    cst = maximum(log_weights)
    weights = exp.(log_weights .- cst)
    # weights .+= eps(Float64)
    weights ./= sum(weights)

    sample_inds = wsample(1:num_particles, weights, num_particles)

    mn = sum(weights .* proposal_samples, dims = 1)
    Pn = cov(weights .* proposal_samples, dims = 1)

    selected_samples = proposal_samples[sample_inds, :]

    # mn = mean(proposal_samples, dims = 1)
    # Pn = cov(selected_samples, dims = 1)

    return (mn, Pn, selected_samples, weights, selected_samples, sample_inds)
end
