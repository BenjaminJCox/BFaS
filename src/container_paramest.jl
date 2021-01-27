using LinearAlgebra
using Distributions
using Statistics
using Random
using Bootstrap
using StatsBase

include(srcdir("kf.jl")) # for proposals
include(srcdir("filter_container.jl"))
include(srcdir("container_filters.jl"))

# Needs stupid long chains
function perform_PMMH_noises(
    filter::containers.add_gaussian_ssm_particle_filter_known_T,
    noise_dist::Function,
    noise_prior_llh::Function,
    noise_step_llh::Function,
    steps::Integer;
    filter_function::Function = BPF_kT_step!,
    num_particles::Integer = 100,
)
    observations = copy(filter.historic_observations)
    working_filter = filter
    working_ssm = filter.SSM

    θ_samples = Vector{Dict{Symbol,Any}}(undef, steps)
    θ = Dict(:P => working_ssm.P, :Q => working_ssm.Q, :R => working_ssm.R)
    θ_llh = noise_prior_llh(θ)
    obs_llh = sum(log.(filter.likelihood))

    T = filter.T
    θ_new = copy(θ)
    θ_new_llh = copy(θ_llh)
    obs_new_llh = copy(obs_llh)

    f = working_ssm.f
    g = working_ssm.g
    p0 = working_ssm.p0
    parameters = working_ssm.ssm_parameters
    state_dim = size(θ[:Q], 1)
    obs_dim = size(θ[:R], 1)

    make_ssm(θ) =
        make_add_gaussian_ssm(f, g, θ[:Q], θ[:R], p0, θ[:P], parameters, state_dim, obs_dim)
    for i = 1:steps
        θ_new = noise_dist(θ)
        θ_new_llh = noise_prior_llh(θ_new)

        working_ssm = make_ssm(θ_new)
        working_filter = init_PF_kT(working_ssm, T, num_particles = num_particles)
        for t = 1:T
            filter_function(working_filter, t, observations[:, t])
        end
        obs_new_llh = sum(log.(working_filter.likelihood))

        mh_ratio =
            θ_new_llh - θ_llh + obs_new_llh - obs_llh + noise_step_llh(θ, θ_new) -
            noise_step_llh(θ_new, θ)
        lr = log(rand())
        if lr < mh_ratio
            θ = θ_new
            θ_llh = θ_new_llh
            obs_llh = obs_new_llh
        end
        # @info(θ[:Q])
        θ_samples[i] = θ
    end
    return θ_samples
end
