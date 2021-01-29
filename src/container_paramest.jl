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
    filter_function::Function = QBPF_kT_step!,
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
        @info(θ[:Q])
        θ_samples[i] = θ
    end
    return θ_samples
end

function perform_PMMH_params(
    filter::containers.add_gaussian_ssm_particle_filter_known_T,
    param_dist::Function,
    param_prior_llh::Function,
    param_step_llh::Function,
    steps::Integer;
    filter_function::Function = QBPF_kT_step!,
    num_particles::Integer = 100,
)
    observations = copy(filter.historic_observations)
    working_filter = filter
    working_ssm = filter.SSM

    θ_samples = Vector{Dict{Symbol,Any}}(undef, steps)
    θ = filter.SSM.ssm_parameters
    θ_llh = param_prior_llh(θ)
    obs_llh = sum(log.(filter.likelihood))

    T = filter.T
    θ_new = copy(θ)
    θ_new_llh = copy(θ_llh)
    obs_new_llh = copy(obs_llh)

    f = working_ssm.f
    g = working_ssm.g
    p0 = working_ssm.p0
    P = working_ssm.P
    Q = working_ssm.Q
    R = working_ssm.R
    parameters = working_ssm.ssm_parameters
    state_dim = size(Q, 1)
    obs_dim = size(R, 1)

    make_ssm(θ) =
        make_add_gaussian_ssm(f, g, Q, R, p0, P, θ, state_dim, obs_dim)
    for i = 1:steps
        θ_new = param_dist(θ)
        θ_new_llh = param_prior_llh(θ_new)

        working_ssm = make_ssm(θ_new)
        working_filter = init_PF_kT(working_ssm, T, num_particles = num_particles)
        for t = 1:T
            filter_function(working_filter, t, observations[:, t])
        end
        obs_new_llh = sum(log.(working_filter.likelihood))

        mh_ratio =
            θ_new_llh - θ_llh + obs_new_llh - obs_llh + param_step_llh(θ, θ_new) -
            param_step_llh(θ_new, θ)
        lr = log(rand())
        if lr < mh_ratio
            θ = θ_new
            θ_llh = θ_new_llh
            obs_llh = obs_new_llh
        end
        @info(θ[:a])
        θ_samples[i] = θ
    end
    return θ_samples
end

# implicitely assumes static parameters
function CSMC_sampler_run(filter::containers.add_gaussian_ssm_particle_filter_known_T, fixed_trajectory::AbstractMatrix; num_particles::Integer = 100)
    T = filter.T

    state_dim = length(filter.SSM.p0)
    obs_dim = length(filter.current_observation)

    current_particles = Array{Float64,2}(undef, state_dim, num_particles)
    historic_particles = Array{Float64,3}(undef, state_dim, num_particles, T+1)

    ancestry = Array{Int64,2}(undef, num_particles, T+1)
    likelihood = Vector{Float64}(undef, T)

    historic_observations = filter.historic_observations
    current_weights = Vector{Float64}(undef, num_particles)

    prior_dist = MvNormal(filter.SSM.p0, Matrix(filter.SSM.P))
    pert_dist = MvNormal(Matrix(filter.SSM.Q))

    current_particles .= rand(prior_dist, num_particles)
    current_particles[:, 1] = fixed_trajectory[:, 1]
    current_weights .= inv(num_particles)

    historic_particles[:, :, 1] .= current_particles
    ancestry[:, 1] .= 1:num_particles

    selected_indices = copy(ancestry[:, 1])

    for t = 1:T
        current_particles[:, 1] = fixed_trajectory[:, t+1]
        observation = historic_observations[:, t]
        for i = 2:num_particles
            xp =
                filter.SSM.f(current_particles[:, i], filter.SSM.ssm_parameters) .+
                rand(pert_dist)
            current_particles[:, i] = xp
        end
        for i = 1:num_particles
            current_weights[i] = pdf(
                MvNormal(filter.SSM.g(current_particles[:, i], filter.SSM.ssm_parameters), Matrix(filter.SSM.R)),
                observation,
            )
        end
        # csmc resampling
        selected_indices = wsample(1:num_particles, current_weights, num_particles)
        selected_indices[1] = 1
        ancestry[:, t+1] .= selected_indices
        current_particles .= current_particles[:, selected_indices]
        historic_particles[:, :, t+1] = current_particles

        likelihood[t] = sum(current_weights) * inv(num_particles)
    end

    llh = sum(log.(likelihood))
    return @dict historic_particles ancestry likelihood llh
end

function pull_trajectory(filter, trajectory_index::Integer)
    T = filter.T
    state_dim = length(filter.SSM.p0)
    historic_particles = filter.historic_particles
    ancestry = filter.ancestry
    i_anc = -1

    trajectory = Array{Float64,2}(undef, state_dim, T+1)

    trajectory[:, T+1] .= historic_particles[:, trajectory_index, T+1]
    i_anc = ancestry[trajectory_index, T+1]

    for back in T:(-1):1
        trajectory[:, back] .= historic_particles[:, i_anc, back]
        i_anc = ancestry[trajectory_index, back]
    end
    return trajectory
end
