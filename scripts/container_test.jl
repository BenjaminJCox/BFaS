using DrWatson
using Distributions
using Plots

include(srcdir("kf.jl"))
include(srcdir("particle.jl"))
include(srcdir("smoothers.jl"))
include(srcdir("param_est.jl"))
include(srcdir("modulator.jl"))
using .containers

gr()


Random.seed!(012)
# gr()
# plotlyjs()
function container_test()
    function psi(x, p)
        nx = copy(x)
        nx[1] = x[1] + x[2]
        nx[2] = -cbrt(abs((x[1] - 10))) * sign(x[1] - 10) / (x[1]+2)
        return nx
    end
    H = [1.0 0.0]
    Q = [1.0/10^2 0.0; 0.0 1.0^2]
    R = hcat([2.0^2])

    T = 150

    Hf(x, p) = H * x

    m0 = [0.0, 0.0]
    P = [1.0 0.0; 0.0 1.0]

    x = zeros(2, T)
    y = zeros(T)

    process_rand = MvNormal([0.0, 0.0], Q)
    obs_rand = Normal(0.0, R[1]^0.5)

    x[:, 1] = m0
    y[1] = x[1, 1] .+ rand(obs_rand)

    for k = 2:T
        x[:, k] =  psi(x[:, k-1], nothing) + rand(process_rand)
        y[k] = (H*x[:, k])[1] .+ rand(obs_rand)
    end
    params = Dict{Symbol,Any}()
    ao_SSM = containers.make_add_gaussian_ssm(psi, Hf, Q, R, m0, P, params, 2, 1)

    filter = containers.init_PF_kT(ao_SSM, T, num_particles = 1000)
    kl_m = zeros(2, T)
    kl_V = zeros(2, T)
    for t in 1:T
        # containers.SIR_ExKF_kT_step!(filter, t, [y[t]])
        containers.BPF_kT_step!(filter, t, [y[t]])
        # containers.APF_kT_step!(filter, t, [y[t]])
        kl_m[:, t] = filter.current_mean
        kl_V[1, t] = filter.current_cov[1,1]
        kl_V[2, t] = filter.current_cov[2,2]
    end
    return @dict filter x y kl_m kl_V T
end

function mul_container_test()
    function add_f(x, q, p)
        nx = 0.91*x[1] + 1.0*q[1]
        return [nx]
    end
    function mul_f(x, q, p)
        return 0.0
    end
    function add_g(x, r, p)
        return 0.0
    end
    function mul_g(x, r, p)
        ox = 0.5 * exp(x[1]/2.0) * r[1]
        return [ox]
    end
    params = Dict{Symbol,Any}()

    Q = hcat([1.0])
    R = hcat([1.0])
    T = 200

    m0 = [0.0]
    P = hcat([2.0])

    x = zeros(T)
    y = zeros(T)

    process_rand = MvNormal(Q)
    obs_rand = MvNormal(R)
    x[1] = m0[1]
    y[1] = mul_g(x[1], rand(obs_rand), params)[1]

    for k = 2:T
        x[k] = add_f(x[k-1], rand(process_rand), params)[1]
        y[k] = mul_g(x[k], rand(obs_rand), params)[1]
    end

    sv_SSM = containers.make_gen_gaussian_ssm(add_f, mul_f, add_g, mul_g, Q, R, m0, P, params, 1, 1)
    filter = containers.init_PF_kT(sv_SSM, T, num_particles = 1000)
    kl_m = zeros(T)
    kl_V = zeros(T)
    for t in 1:T
        containers.SIR_ExKF_kT_step!(filter, t, [y[t]])
        # containers.BPF_kT_step!(filter, t, [y[t]])
        # containers.APF_kT_step!(filter, t, [y[t]])
        kl_m[t] = filter.current_mean[1]
        kl_V[t] = filter.current_cov[1, 1]
    end
    return @dict filter x y kl_m kl_V T
end


# op = container_test()
# x = op[:x]
# y = op[:y]
# km = op[:kl_m]
# kv = op[:kl_V]
# T = op[:T]
# filter = op[:filter]
#
# umf = rmse(x, km)
# @info("Filter RMSE:", umf)
# a_ef = containers.approx_energy_function(filter)
# @info("Approximate Energy: ", a_ef[end])
#
# plot(1:T, x[1, :], size = (750, 500), label = "Truth", legend=:outertopright)
# plot!(1:T, y, label = "Observations", st = :scatter)
# plot!(1:T, km[1, :], label = "Filter Mean", ribbon = sqrt.(kv[1,:]))

mul = mul_container_test()
x = mul[:x]
y = mul[:y]
km = mul[:kl_m]
kv = mul[:kl_V]
T = mul[:T]
filter = mul[:filter]

umf = rmse(x, km)
@info("Filter RMSE:", umf)
a_ef = containers.approx_energy_function(filter)
@info("Approximate Energy: ", a_ef[end])

p1 = plot(1:T, x, size = (750, 500), label = "Truth")
plot!(1:T, km, label = "Filter Mean", ribbon = sqrt.(kv))
p2 = plot(1:T, y, label = "Observations")
plot(p1, p2, layout = (2, 1), size = (1250, 500), link = :x, legend = false)
