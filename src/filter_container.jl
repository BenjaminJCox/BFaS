using Parameters
using Distributions
using LinearAlgebra
using Plots
using Turing
using Distributions
using ForwardDiff

@with_kw mutable struct state_space_model_add_gaussian
    # transition function E(x_t) = f(x_(t-1))
    f::Function = x -> x

    # observation function E(y_t) = g(x_t)
    g::Function = x -> x

    # is noise multiplicative for (state, observation), otherwise assume additive

    # noise covaraiance for state and observation respectively
    Q::Hermitian{Float64}
    R::Hermitian{Float64}

    # prior state distribution
    p0::Vector{Float64}
    P::Hermitian{Float64}

    # optional parameters to pass to observation and transition functions
    ssm_parameters::Dict{Symbol,Any}
end

@with_kw mutable struct state_space_model_gen_gaussian
    # transition function x_t = f(x_(t-1), q_(k-1))
    # of form a(x)*q + b(x) + v*q = mul_f + add_f
    add_f::Function
    mul_f::Function
    f::Function = x -> x

    # observation function y_t = g(x_t, r_k)
    # of form a(x)*r + b(x) + v*r = mul_g + add_g
    add_g::Function
    mul_g::Function
    g::Function = x -> x

    # is noise multiplicative for (state, observation), otherwise assume additive

    # noise covaraiance for state and observation respectively
    Q::Hermitian{Float64}
    R::Hermitian{Float64}

    # prior state distribution
    p0::Vector{Float64}
    P::Hermitian{Float64}

    # optional parameters to pass to observation and transition functions
    ssm_parameters::Dict{Symbol,Any}
end

function make_add_gaussian_ssm(f, g, Q, R, p0, P, parameters, state_dim, obs_dim)
    @assert size(Q, 1) == size(Q, 2)
    @assert size(P, 1) == size(P, 2)
    @assert size(R, 1) == size(R, 2)
    @assert size(Q, 1) == state_dim
    @assert size(P, 1) == state_dim
    @assert size(R, 1) == obs_dim
    @assert Hermitian(Q) == Q
    hQ = Hermitian(Q)
    @assert Hermitian(P) == P
    hP = Hermitian(P)
    @assert Hermitian(R) == R
    hR = Hermitian(R)
    @assert length(p0) == state_dim
    @assert length(f(p0, parameters)) == state_dim
    @assert length(g(p0, parameters)) == obs_dim

    return state_space_model_add_gaussian(
        f = f,
        g = g,
        Q = hQ,
        R = hR,
        p0 = p0,
        P = hP,
        ssm_parameters = parameters,
    )
end

function make_gen_gaussian_ssm(add_f, mul_f, add_g, mul_g, Q, R, p0, P, parameters, state_dim, obs_dim)
    @assert size(Q, 1) == size(Q, 2)
    @assert size(P, 1) == size(P, 2)
    @assert size(R, 1) == size(R, 2)
    @assert size(Q, 1) == state_dim
    @assert size(P, 1) == state_dim
    @assert size(R, 1) == obs_dim
    @assert Hermitian(Q) == Q
    hQ = Hermitian(Q)
    @assert Hermitian(P) == P
    hP = Hermitian(P)
    @assert Hermitian(R) == R
    hR = Hermitian(R)
    tv_sta = ones(state_dim)
    tv_obs = ones(obs_dim)
    @assert length(p0) == state_dim
    @assert length(add_f(p0, tv_sta, parameters)) == state_dim
    @assert length(add_g(p0, tv_obs, parameters)) == obs_dim
    @assert length(mul_f(p0, tv_sta, parameters)) == state_dim
    @assert length(mul_g(p0, tv_obs, parameters)) == obs_dim
    f(x, y, p) = add_f(x, y, p) .+ mul_f(x, y, p)
    g(x, y, p) = add_g(x, y, p) .+ mul_g(x, y, p)

    return state_space_model_gen_gaussian(
        f = f,
        g = g,
        add_f = add_f,
        add_g = add_g,
        mul_f = mul_f,
        mul_g = mul_g,
        Q = hQ,
        R = hR,
        p0 = p0,
        P = hP,
        ssm_parameters = parameters,
    )
end

@with_kw mutable struct add_gaussian_ssm_particle_filter
    # ssm that the filter is to operate on
    SSM::state_space_model_add_gaussian

    # current time
    t::Int64

    current_particles::Array{Float64,2}
    historic_particles::Array{Float64,3}

    particle_covariances::Array{Float64, 3}

    current_observation::Vector{Float64}
    historic_observations::Array{Float64,2}

    current_weights::Vector{Float64}
    historic_weights::Array{Float64,2}

    ancestry::Array{Int64,2}

    num_particles::Int64 = 100

    current_mean::Vector{Float64}
    current_cov::Matrix{Float64}
    likelihood::Vector{Float64}

    # eg for resample move give lag and mh kernel parameters
    filter_specific_parameters::Dict
end

@with_kw mutable struct add_gaussian_ssm_particle_filter_known_T
    # ssm that the filter is to operate on
    SSM::state_space_model_add_gaussian

    # current time
    t::Int64
    T::Int64

    current_particles::Array{Float64,2}
    historic_particles::Array{Float64,3}

    particle_covariances::Array{Float64, 3}

    current_observation::Vector{Float64}
    historic_observations::Array{Float64,2}

    current_weights::Vector{Float64}
    historic_weights::Array{Float64,2}

    ancestry::Array{Int64,2}

    num_particles::Int64 = 100

    current_mean::Vector{Float64}
    current_cov::Matrix{Float64}
    likelihood::Vector{Float64}

    # eg for resample move give lag and mh kernel parameters
    filter_specific_parameters::Dict
end

@with_kw mutable struct gen_gaussian_ssm_particle_filter
    # ssm that the filter is to operate on
    SSM::state_space_model_gen_gaussian

    # current time
    t::Int64

    current_particles::Array{Float64,2}
    historic_particles::Array{Float64,3}

    particle_covariances::Array{Float64, 3}

    current_observation::Vector{Float64}
    historic_observations::Array{Float64,2}

    current_weights::Vector{Float64}
    historic_weights::Array{Float64,2}

    ancestry::Array{Int64,2}

    num_particles::Int64 = 100

    current_mean::Vector{Float64}
    current_cov::Matrix{Float64}
    likelihood::Vector{Float64}

    # eg for resample move give lag and mh kernel parameters
    filter_specific_parameters::Dict
end

@with_kw mutable struct gen_gaussian_ssm_particle_filter_known_T
    # ssm that the filter is to operate on
    SSM::state_space_model_gen_gaussian

    # current time
    t::Int64
    T::Int64

    current_particles::Array{Float64,2}
    historic_particles::Array{Float64,3}

    particle_covariances::Array{Float64, 3}

    current_observation::Vector{Float64}
    historic_observations::Array{Float64,2}

    current_weights::Vector{Float64}
    historic_weights::Array{Float64,2}

    ancestry::Array{Int64,2}

    num_particles::Int64 = 100

    current_mean::Vector{Float64}
    current_cov::Matrix{Float64}
    likelihood::Vector{Float64}

    # eg for resample move give lag and mh kernel parameters
    filter_specific_parameters::Dict
end

function make_pf_wT(
    SSM::state_space_model_add_gaussian,
    T,
    filter_specific_parameters;
    num_particles::Int64,
)
    @unpack Q, R = SSM
    state_dim = size(Q, 1)
    obs_dim = size(R, 1)

    current_particles = Array{Float64,2}(undef, state_dim, num_particles)
    historic_particles = Array{Float64,3}(undef, state_dim, num_particles, T+1)

    particle_covariances = Array{Float64,3}(undef, state_dim, state_dim, num_particles)
    for i in 1:num_particles
        particle_covariances[:, :, i] .= SSM.P
    end

    current_weights = Array{Float64,1}(undef, num_particles)
    historic_weights = Array{Float64,2}(undef, num_particles, T+1)

    current_observation = Array{Float64,1}(undef, obs_dim)
    historic_observations = Array{Float64,2}(undef, obs_dim, T)

    ancestry = Array{Int64,2}(undef, num_particles, T+1)
    likelihood = Vector{Float64}(undef, T)

    return add_gaussian_ssm_particle_filter_known_T(
        SSM = SSM,
        t = 0,
        T = T,
        current_particles = current_particles,
        historic_particles = historic_particles,
        current_weights = current_weights,
        historic_weights = historic_weights,
        current_observation = current_observation,
        historic_observations = historic_observations,
        filter_specific_parameters = filter_specific_parameters,
        num_particles = num_particles,
        ancestry = ancestry,
        current_mean = SSM.p0,
        current_cov = SSM.P,
        likelihood = likelihood,
        particle_covariances = particle_covariances,
    )
end

function make_pf_wT(
    SSM::state_space_model_gen_gaussian,
    T,
    filter_specific_parameters;
    num_particles::Int64,
    )
        @unpack Q, R = SSM
        state_dim = size(Q, 1)
        obs_dim = size(R, 1)

        current_particles = Array{Float64,2}(undef, state_dim, num_particles)
        historic_particles = Array{Float64,3}(undef, state_dim, num_particles, T+1)

        particle_covariances = Array{Float64,3}(undef, state_dim, state_dim, num_particles)
        for i in 1:num_particles
            particle_covariances[:, :, i] .= SSM.P
        end

        current_weights = Array{Float64,1}(undef, num_particles)
        historic_weights = Array{Float64,2}(undef, num_particles, T+1)

        current_observation = Array{Float64,1}(undef, obs_dim)
        historic_observations = Array{Float64,2}(undef, obs_dim, T)

        ancestry = Array{Int64,2}(undef, num_particles, T+1)
        likelihood = Vector{Float64}(undef, T)

        return gen_gaussian_ssm_particle_filter_known_T(
            SSM = SSM,
            t = 0,
            T = T,
            current_particles = current_particles,
            historic_particles = historic_particles,
            current_weights = current_weights,
            historic_weights = historic_weights,
            current_observation = current_observation,
            historic_observations = historic_observations,
            filter_specific_parameters = filter_specific_parameters,
            num_particles = num_particles,
            ancestry = ancestry,
            current_mean = SSM.p0,
            current_cov = SSM.P,
            likelihood = likelihood,
            particle_covariances = particle_covariances,
        )
    end
