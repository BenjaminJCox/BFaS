using DrWatson
using Distributions
using Plots
using DifferentialEquations

include(srcdir("kf.jl"))
include(srcdir("particle.jl"))
include(srcdir("smoothers.jl"))
include(srcdir("param_est.jl"))
include(srcdir("modulator.jl"))
using .containers

# gr()


Random.seed!(02)
# gr()
plotlyjs()
function container_test()
    function psi(x, p)
        nx = copy(x)
        nx[1] = x[1] + x[2]
        nx[2] = -cbrt(abs((x[1] - 10))) * sign(x[1] - 10) / (x[1] + 2)
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
        x[:, k] = psi(x[:, k-1], nothing) + rand(process_rand)
        y[k] = (H*x[:, k])[1] .+ rand(obs_rand)
    end
    params = Dict{Symbol,Any}()
    ao_SSM = containers.make_add_gaussian_ssm(psi, Hf, Q, R, m0, P, params, 2, 1)

    filter = containers.init_PF_kT(ao_SSM, T, num_particles = 1000)
    kl_m = zeros(2, T)
    kl_V = zeros(2, T)
    for t = 1:T
        # containers.SIR_ExKF_kT_step!(filter, t, [y[t]])
        containers.BPF_kT_step!(filter, t, [y[t]])
        # containers.APF_kT_step!(filter, t, [y[t]])
        kl_m[:, t] = filter.current_mean
        kl_V[1, t] = filter.current_cov[1, 1]
        kl_V[2, t] = filter.current_cov[2, 2]
    end
    return @dict filter x y kl_m kl_V T
end

function mul_container_test()
    function add_f(x, q, p)
        nx = 0.91 * x[1] + 1.0 * q[1]
        return [nx]
    end
    function mul_f(x, q, p)
        return 0.0
    end
    function add_g(x, r, p)
        return 0.0
    end
    function mul_g(x, r, p)
        ox = 0.5 * exp(x[1] / 2.0) * r[1]
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

    sv_SSM = containers.make_gen_gaussian_ssm(
        add_f,
        mul_f,
        add_g,
        mul_g,
        Q,
        R,
        m0,
        P,
        params,
        1,
        1,
    )
    filter = containers.init_PF_kT(sv_SSM, T, num_particles = 1000)
    kl_m = zeros(T)
    kl_V = zeros(T)
    for t = 1:T
        # containers.SIR_ExKF_kT_step!(filter, t, [y[t]])
        # containers.BPF_kT_step!(filter, t, [y[t]])
        containers.APF_kT_step!(filter, t, [y[t]])
        kl_m[t] = filter.current_mean[1]
        kl_V[t] = filter.current_cov[1, 1]
    end
    return @dict filter x y kl_m kl_V T
end

function l63_test()
    function l63_step(x)
        function lorenz!(du, u, p, t)
            du[1] = 10.0 * (u[2] - u[1])
            du[2] = u[1] * (28.0 - u[3]) - u[2]
            du[3] = u[1] * u[2] - (8 / 3) * u[3]
        end
        tspan = (0.0, 0.05)
        prob = ODEProblem(lorenz!, x, tspan)
        sol = solve(prob, saveat = [0.05])
        return sol[1]
    end
    T = 50
    x0 = [1.0, 0.0, 0.0]
    x = zeros(3, T)
    y = zeros(2, T)

    params = Dict{Symbol,Any}()
    H = [1.0 0.0 0.0; 0.0 1.0 0.0]

    step(x, p) = l63_step(x)
    obs(x, p) = H * x

    x[:, 1] = x0
    y[:, 1] = obs(x0, params)

    Q = 1.0 * Matrix(I(3))
    P = 0.5 * Matrix(I(3))
    R = 1.0 * Matrix(I(2))

    _prand = MvNormal(Q)
    _orand = MvNormal(R)

    for t = 2:T
        x[:, t] = step(x[:, t-1], params) .+ rand(_prand)
        y[:, t] = obs(x[:, t], params) .+ rand(_orand)
    end
    params = Dict{Symbol,Any}()
    l63_ssm = containers.make_add_gaussian_ssm(step, obs, Q, R, x0, P, params, 3, 2)

    filter = containers.init_PF_kT(l63_ssm, T, num_particles = 1000)
    kl_m = zeros(3, T)
    kl_V = zeros(3, T)
    for t = 1:T
        # containers.SIR_ExKF_kT_step!(filter, t, y[:, t])
        # containers.BPF_kT_step!(filter, t, y[:, t])
        containers.APF_kT_step!(filter, t, y[:, t])
        kl_m[:, t] = filter.current_mean
        kl_V[1, t] = filter.current_cov[1, 1]
        kl_V[2, t] = filter.current_cov[2, 2]
        kl_V[3, t] = filter.current_cov[3, 3]
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


# mul = mul_container_test()
# x = mul[:x]
# y = mul[:y]
# km = mul[:kl_m]
# kv = mul[:kl_V]
# T = mul[:T]
# filter = mul[:filter]
#
# umf = rmse(x, km)
# @info("Filter RMSE:", umf)
# a_ef = containers.approx_energy_function(filter)
# @info("Approximate Energy: ", a_ef[end])
#
# p1 = plot(1:T, x, size = (750, 500), label = "Truth")
# plot!(1:T, km, label = "Filter Mean", ribbon = sqrt.(kv))
# p2 = plot(1:T, y, label = "Observations")
# plot(p1, p2, layout = (2, 1), size = (1250, 500), link = :x, legend = false)


lorenz = l63_test()
x = lorenz[:x]
y = lorenz[:y]
# plot(x[1,:], x[2,:], x[3,:])
# plot!(y[1,:], y[2,:], y[3,:])
km = lorenz[:kl_m]
kv = lorenz[:kl_V]
T = lorenz[:T]
filter = lorenz[:filter]

umf = rmse(x, km)
@info("Filter RMSE:", umf)
a_ef = containers.approx_energy_function(filter)
@info("Approximate Energy: ", a_ef[end])

plts = Array{Any}(nothing, 3)
for k = 1:3
    kthsubplot = plot(1:T, x[k,:], label = "Truth")
    plot!(1:T, km[k,:], label = "Filter Mean", ribbon = sqrt.(kv[k,:]))
    plts[k] = kthsubplot
end
plot(plts..., layout = (3,1), size = (1000, 1000), legend = false)
# # plot!(1:T, y[1,:], label = "Observations", st = :scatter)
