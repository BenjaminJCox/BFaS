# using Turing
using Random
using MCMCChains
using Distributions
using Statistics
using StatsBase

# @model dyn(x, P, psi) = begin
#     xk ~ MvNormal(psi(x), P)
# end

x = [1.0, 1.0]
P = [1.0 0.0; 0.0 2.0]
psi(x) = 1.01 * x .^ 0.99

# sps = sample(dyn(x, P, psi), NUTS(1000, 0.65), 3000)

# mkarr = Array(sps)

# function bsf_draw(x::Vector{Float64}, P::Matrix{Float64}, psi::Function = x -> x, n::Int64 = 5000)
#     @model dyn_mod(x::Vector{Float64}, P::Matrix{Float64}, psi::Function) = begin
#         xk ~ MvNormal(psi(x), P)
#     end
#     sps = Array(sample(dyn_mod(x, P, psi), NUTS(1000, 0.65), n + 1000))
# end
function bsf_draw(x::Vector{Float64}, Q::Matrix{Float64}, psi::Function, n::Int64 = 5000; adapt::Int64 = 1_000, accept::Float64 = 0.65)
    xk = zeros(n, size(x,1))
    prior = MvNormal(psi(x), Q)
    for i = 1:n
        xk[i, :] = rand(prior)
    end
    return xk
end

bsf = bsf_draw(x, P, psi)

H(x) = x[1] + 2.0 * x[2]

R = hcat([1.0])

ys = [H(x)]

x_obs_transform = mapslices(H, bsf, dims = 2)

function bsf_weights(x::Matrix{Float64}, y::Vector{Float64}, H, R::Matrix{Float64})
    x_obs_transform = mapslices(H, x, dims = 2)
    n = size(bsf, 1)
    weight_vector = zeros(size(x_obs_transform, 1))
    for i = 1:n
        xi = x_obs_transform[i, :]
        weight_vector[i] = pdf(MvNormal(xi, R), y)
    end
    weight_vector ./= sum(weight_vector)
    return weight_vector
end

wts = bsf_weights(bsf, ys, H, R)

wts_norm = wts ./ sum(wts)

function eff_particles(weights::Vector{Float64})
    wsq = weights .^ 2
    ism = sum(wsq)
    return 1.0 / ism
end

eff_particles(wts)

n = size(wts, 1)
positions = (rand(n) + collect(0:(n-1))) / n
cumsum(wts)
(i, j) = (0, 0)

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

ind = stratified_resample(bsf, wts)
ind = resample(bsf, wts)

xk = zeros(n, size(x,1))
