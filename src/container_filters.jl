using LinearAlgebra
using Distributions
using Statistics
using Random
using Bootstrap
using StatsBase

include(srcdir("kf.jl")) # for proposals
include(srcdir("filter_container.jl"))

function init_BPF_kT(SSM::containers.state_space_model_gaussian, T::Int64; num_particles = 100)
    filter_container = containers.make_pf_wT(SSM, T, Dict{Symbol,Any}(); num_particles = num_particles)
    prior_dist = MvNormal(SSM.p0, Matrix(SSM.P))
    for i = 1:num_particles
        filter_container.current_particles[:, i] = rand(prior_dist)
        filter_container.ancestry[i, 1] = i

    end
    filter_container.historic_particles[:, :, 1] = filter_container.current_particles
    filter_container.current_weights .= inv(num_particles)
    filter_container.historic_weights[:, 1] = filter_container.current_weights

    return filter_container
end

function BPF_kT_step!(filter::containers.gaussian_ssm_particle_filter_known_T, t, observation)
    filter.current_observation = observation
    filter.historic_observations[:, t] = observation
    pert_dist = MvNormal(0.0 .* filter.SSM.p0, Matrix(filter.SSM.Q))

    for i in 1:filter.num_particles
        xp = filter.SSM.f(filter.current_particles[:, i], filter.SSM.ssm_parameters) .+ rand(pert_dist)
        filter.current_particles[:, i] = xp
        filter.current_weights[i] = pdf(MvNormal(filter.SSM.g(xp, filter.SSM.ssm_parameters), Matrix(filter.SSM.R)), observation)
    end
    norm_weights = filter.current_weights/sum(filter.current_weights)
    selected_indices = wsample(1:filter.num_particles, norm_weights, filter.num_particles, replace = true)
    filter.ancestry[:, t+1] = selected_indices
    filter.current_particles = filter.current_particles[:, selected_indices]
    filter.historic_particles[:, :, t+1] = filter.current_particles
    return filter
end
