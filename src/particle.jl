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

function bsf_weights(x::Matrix{Float64}, y::Vector{Float64}, H::Function, R::Matrix{Float64})
    x_obs_transform = mapslices(H, x, dims = 2)
    n = size(x_obs_transform, 1)
    weight_vector = zeros(size(x_obs_transform, 1))
    @simd for i = 1:n
        xi = x_obs_transform[i, :]
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
        xi = xob[i, :]
        weight_vector[i] = pdf(MvNormal(xi, R), y)
    end
    weight_vector .+= eps(Float64) #for starbility reasons
    weight_vector ./= sum(weight_vector)
    m = sum(weight_vector .* xpf, dims = 1)
    P = cov(xpf, dims = 1)
    ninds = wsample(collect(1:n), weight_vector, n, replace = true)
    nxs = xpf[ninds, :]
    return (m, P, nxs, weight_vector, xpf)
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

    return (m, P, selected_x_sim, weight_vector)
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
            pixty[j] = rapid_mvn_prec(selected_x_sim[i, :], x_pred[j, :], i_sigma, isq_d_sigma)
        end
        weight_vector[i] *= sum(pixty)
        weight_vector[i] /= sum(l_weight_vector .* pixty)
    end

    weight_vector ./= sum(weight_vector)

    m = sum(weight_vector .* selected_x_sim, dims = 1)
    P = cov(weight_vector .* selected_x_sim, dims = 1)

    return (m, P, selected_x_sim, weight_vector)
end
