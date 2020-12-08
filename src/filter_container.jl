module containers
using Parameters
using Distributions
using LinearAlgebra
using Plots
using Turing
using Distributions
using ForwardDiff

@with_kw mutable struct state_space_model_gaussian
    # transition function E(x_t) = f(x_(t-1))
    f::Function = x -> x

    # observation function E(y_t) = g(x_t)
    g::Function = x -> x

    # is noise multiplicative for (state, observation), otherwise assume additive
    mult_noises::Tuple{Bool,Bool} = (false, false)

    # noise covaraiance for state and observation respectively
    Q::Hermitian{Float64}
    R::Hermitian{Float64}

    # prior state distribution
    p0::Vector{Float64}
    P::Hermitian{Float64}

    # optional parameters to pass to observation and transition functions
    ssm_parameters::Dict{Symbol,Any}
end

function make_gaussian_ssm(f, g, Q, R, p0, P, parameters, state_dim, obs_dim)
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

    return state_space_model_gaussian(
        f = f,
        g = g,
        mult_noises = (false, false),
        Q = hQ,
        R = hR,
        p0 = p0,
        P = hP,
        ssm_parameters = parameters,
    )
end

@with_kw mutable struct gaussian_ssm_particle_filter
    # ssm that the filter is to operate on
    SSM::state_space_model_gaussian

    # current time
    t::Int64

    current_particles::Array{Float64,2}
    historic_particles::Array{Float64,3}

    current_observation::Vector{Float64}
    historic_observations::Array{Float64,2}

    is_log_weights::Bool = false
    current_weights::Vector{Float64}
    historic_weights::Array{Float64,2}

    ancestry::Array{Int64,2}

    num_particles::Int64 = 100

    current_mean::Vector{Float64}
    current_cov::Matrix{Float64}

    # eg for resample move give lag and mh kernel parameters
    filter_specific_parameters::Dict
end

@with_kw mutable struct gaussian_ssm_particle_filter_known_T
    # ssm that the filter is to operate on
    SSM::state_space_model_gaussian

    # current time
    t::Int64
    T::Int64

    current_particles::Array{Float64,2}
    historic_particles::Array{Float64,3}

    current_observation::Vector{Float64}
    historic_observations::Array{Float64,2}

    is_log_weights::Bool = false
    current_weights::Vector{Float64}
    historic_weights::Array{Float64,2}

    ancestry::Array{Int64,2}

    num_particles::Int64 = 100

    current_mean::Vector{Float64}
    current_cov::Matrix{Float64}

    # eg for resample move give lag and mh kernel parameters
    filter_specific_parameters::Dict
end

function make_pf_wT(
    SSM,
    T,
    filter_specific_parameters;
    is_log_weights::Bool = false,
    num_particles::Int64,
)
    @unpack Q, R = SSM
    state_dim = size(Q, 1)
    obs_dim = size(R, 1)

    current_particles = Array{Float64,2}(undef, state_dim, num_particles)
    historic_particles = Array{Float64,3}(undef, state_dim, num_particles, T+1)

    current_weights = Array{Float64,1}(undef, num_particles)
    historic_weights = Array{Float64,2}(undef, num_particles, T+1)

    current_observation = Array{Float64,1}(undef, obs_dim)
    historic_observations = Array{Float64,2}(undef, obs_dim, T)

    ancestry = Array{Int64,2}(undef, num_particles, T+1)

    return gaussian_ssm_particle_filter_known_T(
        SSM = SSM,
        t = 0,
        T = T,
        current_particles = current_particles,
        historic_particles = historic_particles,
        current_weights = current_weights,
        historic_weights = historic_weights,
        current_observation = current_observation,
        historic_observations = historic_observations,
        is_log_weights = is_log_weights,
        filter_specific_parameters = filter_specific_parameters,
        num_particles = num_particles,
        ancestry = ancestry,
        current_mean = SSM.p0,
        current_cov = SSM.P,
    )
end

end
