using LinearAlgebra
using Distributions
using Statistics
using Random
using Bootstrap
using StatsBase

include(srcdir("kf.jl")) # for proposals
include(srcdir("filter_container.jl"))

function init_BPF_kT(
    SSM::containers.state_space_model_gaussian,
    T::Int64;
    num_particles = 100,
)
    filter_container =
        containers.make_pf_wT(SSM, T, Dict{Symbol,Any}(); num_particles = num_particles)
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

function init_PF_kT(
    SSM::containers.state_space_model_gaussian,
    T::Int64;
    num_particles = 100,
)
    filter_container =
        containers.make_pf_wT(SSM, T, Dict{Symbol,Any}(); num_particles = num_particles)
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

function BPF_kT_step!(
    filter::containers.gaussian_ssm_particle_filter_known_T,
    t,
    observation,
)
    filter.current_observation = observation
    filter.historic_observations[:, t] = observation
    filter.t = t
    pert_dist = MvNormal(Matrix(filter.SSM.Q))

    for i = 1:filter.num_particles
        xp =
            filter.SSM.f(filter.current_particles[:, i], filter.SSM.ssm_parameters) .+
            rand(pert_dist)
        filter.current_particles[:, i] = xp
        filter.current_weights[i] = pdf(
            MvNormal(filter.SSM.g(xp, filter.SSM.ssm_parameters), Matrix(filter.SSM.R)),
            observation,
        )
    end
    norm_weights = filter.current_weights / sum(filter.current_weights)
    selected_indices =
        wsample(1:filter.num_particles, norm_weights, filter.num_particles, replace = true)
    filter.ancestry[:, t+1] = selected_indices
    filter.current_mean = sum(filter.current_particles .* norm_weights', dims = 2)[:, 1]
    filter.current_cov = cov(filter.current_particles .* norm_weights', dims = 2)
    filter.current_particles = filter.current_particles[:, selected_indices]
    filter.historic_particles[:, :, t+1] = filter.current_particles
    filter.historic_weights[:, t+1] = filter.current_weights
    filter.likelihood[t] = sum(filter.current_weights) ./ filter.num_particles
    return filter
end

function SIR_ExKF_kT_step!(
    filter::containers.gaussian_ssm_particle_filter_known_T,
    t,
    observation,
)
    filter.current_observation = observation
    filter.historic_observations[:, t] = observation
    filter.t = t

    # _xdim = length(filter.current_particles[:, 1])
    # _ydim = length(observation)

    f_map(x) = filter.SSM.f(x, filter.SSM.ssm_parameters)
    g_map(x) = filter.SSM.g(x, filter.SSM.ssm_parameters)

    @simd for i = 1:filter.num_particles
        # x_m = filter.current_particles[:, i]
        predicted_mean, predicted_cov = exkf_predict(filter.current_particles[:, i], Matrix(filter.SSM.P), f_map, Matrix(filter.SSM.Q))
        filtered_mean, filtered_cov = exkf_update(predicted_mean, predicted_cov, observation, g_map, Matrix(filter.SSM.R))
        filtered_cov = Matrix(Hermitian(filtered_cov))
        prop_dist = MvNormal(filtered_mean, filtered_cov)

        # imp_y = g_map(xp)
        model_dist = MvNormal(f_map(filter.current_particles[:,i]), Matrix(filter.SSM.Q))
        filter.current_particles[:, i] = rand(prop_dist)
        obs_dist = MvNormal(g_map(filter.current_particles[:, i]), Matrix(filter.SSM.R))
        filter.current_weights[i] = logpdf(model_dist, filter.current_particles[:, i]) + logpdf(obs_dist, observation) - logpdf(prop_dist, filter.current_particles[:, i])
    end

    filter.current_weights .= exp.(filter.current_weights)


    norm_weights = filter.current_weights / sum(filter.current_weights)
    selected_indices =
        wsample(1:filter.num_particles, norm_weights, filter.num_particles, replace = true)
    filter.ancestry[:, t+1] = selected_indices
    filter.current_mean = sum(filter.current_particles .* norm_weights', dims = 2)[:, 1]
    filter.current_cov = cov(filter.current_particles .* norm_weights', dims = 2)
    filter.current_particles = filter.current_particles[:, selected_indices]
    filter.historic_particles[:, :, t+1] = filter.current_particles
    filter.historic_weights[:, t+1] = filter.current_weights
    filter.likelihood[t] = sum(filter.current_weights) ./ filter.num_particles
    return filter
end

function SIR_UKF_kT_step!(
    filter::containers.gaussian_ssm_particle_filter_known_T,
    t,
    observation,
)
    filter.current_observation = observation
    filter.historic_observations[:, t] = observation
    filter.t = t

    sqrtP = Matrix(sqrt(filter.SSM.P))
    _xdim = length(filter.current_particles[:, 1])
    _ydim = length(observation)

    f_means = zeros(filter.num_particles, _xdim)
    f_covs = zeros(filter.num_particles, _xdim, _xdim)
    for i = 1:filter.num_particles
        x_m = copy(filter.current_particles[:, i])
        predicted_mean, predicted_cov = ukf_predict_sqrt_param(
            x_m,
            sqrtP,
            filter.SSM.f,
            Matrix(filter.SSM.Q),
            filter.SSM.ssm_parameters,
        )
        filtered_mean, filtered_cov = ukf_update_param(
            predicted_mean,
            predicted_cov,
            observation,
            filter.SSM.g,
            Matrix(filter.SSM.R),
            filter.SSM.ssm_parameters,
        )
        filtered_cov = Matrix(Hermitian(filtered_cov))
        f_means[i, :] = filtered_mean
        f_covs[i, :, :] = filtered_cov
    end

    proposal_samples = zeros(filter.num_particles, _xdim)
    @simd for i = 1:filter.num_particles
        proposal_dist = MvNormal(f_means[i, :], f_covs[i, :, :])
        proposal_samples[i, :] = rand(proposal_dist)
    end

    implied_obs = zeros(filter.num_particles, _ydim)

    @simd for i = 1:filter.num_particles
        implied_obs[i, :] = filter.SSM.g(proposal_samples[i, :], filter.SSM.ssm_parameters)
    end

    log_weights = zeros(filter.num_particles)
    @simd for i = 1:filter.num_particles
        @views _x = proposal_samples[i, :]
        @views _impY = implied_obs[i, :]
        model_dist = MvNormal(
            filter.SSM.f(filter.current_particles[:, i], filter.SSM.ssm_parameters),
            Matrix(filter.SSM.Q),
        )
        obs_dist = MvNormal(_impY, Matrix(filter.SSM.R))
        proposal_dist = MvNormal(f_means[i, :], f_covs[i, :, :])
        log_weights[i] =
            logpdf(model_dist, _x) + logpdf(obs_dist, observation) -
            logpdf(proposal_dist, _x)
    end
    filter.current_weights .= exp.(log_weights)
    norm_weights = filter.current_weights / sum(filter.current_weights)
    selected_indices =
        wsample(1:filter.num_particles, norm_weights, filter.num_particles, replace = true)
    filter.ancestry[:, t+1] = selected_indices
    filter.current_mean = vec(sum(norm_weights .* proposal_samples, dims = 1))
    filter.current_cov = cov(norm_weights .* proposal_samples, dims = 1)
    filter.current_particles = proposal_samples[selected_indices, :]'
    filter.historic_particles[:, :, t+1] = filter.current_particles
    filter.historic_weights[:, t+1] = filter.current_weights
    filter.likelihood[t] = sum(filter.current_weights) ./ filter.num_particles
    return filter
end

function APF_kT_step!(
    filter::containers.gaussian_ssm_particle_filter_known_T,
    t,
    observation,
)
    f_map(x) = filter.SSM.f(x, filter.SSM.ssm_parameters)
    g_map(x) = filter.SSM.g(x, filter.SSM.ssm_parameters)

    x_pred = mapslices(f_map, filter.current_particles, dims = 1)
    x_pred_obs = mapslices(g_map, x_pred, dims = 1)
    pert_dist = MvNormal(Matrix(filter.SSM.Q))

    for i = 1:filter.num_particles
        filter.current_weights[i] =
            pdf(MvNormal(x_pred_obs[:, i], Matrix(filter.SSM.R)), observation) *
            filter.historic_weights[i, t]
    end

    filter.current_weights ./= sum(filter.current_weights)
    selected_indices =
        wsample(1:filter.num_particles, filter.current_weights, filter.num_particles)
    selected_particles = filter.current_particles[:, selected_indices]

    for i = 1:filter.num_particles
        filter.current_particles[:, i] .= f_map(selected_particles[:, i]) .+ rand(pert_dist)
    end

    for i = 1:filter.num_particles
        x_num = g_map(filter.current_particles[:, i])
        x_den = g_map(selected_particles[:, i])
        filter.current_weights[i] =
            logpdf(MvNormal(x_num, Matrix(filter.SSM.R)), observation) -
            logpdf(MvNormal(x_den, Matrix(filter.SSM.R)), observation)
    end

    filter.current_weights .= exp.(filter.current_weights)
    norm_weights = filter.current_weights ./ sum(filter.current_weights)

    filter.ancestry[:, t+1] = selected_indices
    filter.current_mean = sum(filter.current_particles .* norm_weights', dims = 2)[:, 1]
    filter.current_cov = cov(filter.current_particles .* norm_weights', dims = 2)
    filter.historic_particles[:, :, t+1] = filter.current_particles
    filter.historic_weights[:, t+1] = filter.current_weights
    filter.likelihood[t] = sum(filter.current_weights .* filter.historic_weights[:, t]) ./ sum(filter.historic_weights[:, t])
    return filter
end

function approx_energy_function(filter::containers.gaussian_ssm_particle_filter_known_T)
    llh_a = log.(filter.likelihood)
    return (-1 .* cumsum(llh_a, dims = 1))
end
