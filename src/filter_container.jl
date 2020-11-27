using Parameters
using Distributions

@with_kw mutable struct state_space_model
    # transition function E(x_t) = f(x_(t-1))
    f::Function = x -> x

    # observation function E(y_t) = g(x_t)
    g::Function = x -> x

    # is noise multiplicative for (state, observation), otherwise assume additive
    mult_noises::Tuple{Bool, Bool} = (false, false)

    # noise distributions for state and observation respectively
    q::Distribution
    r::Distribution

    # prior state distribution
    p0::Distribution
end
