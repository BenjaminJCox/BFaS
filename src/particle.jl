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
    weight_vector ./= sum(weight_vector)
    m = sum(weight_vector .* xpf, dims = 1)
    P = cov(xpf, dims = 1)
    ninds = wsample(collect(1:n), weight_vector, n)
    nxs = xpf[ninds, :]
    return (m, P, nxs, weight_vector, xpf)
end
