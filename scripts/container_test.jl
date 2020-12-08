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


Random.seed!(3)
# gr()
plotlyjs()
function container_test()
    function psi(x, p)
        nx = copy(x)
        nx[1] = x[1] + x[2]
        nx[2] = -cbrt(abs((x[1] - 10))) * sign(x[1] - 10)
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
    ao_SSM = containers.make_gaussian_ssm(psi, Hf, Q, R, m0, P, params, 2, 1)

    filter = containers.init_BPF_kT(ao_SSM, T, num_particles = 1000)
    kl_m = zeros(2, T)
    for t in 1:T
        containers.BPF_kT_step!(filter, t, [y[t]])
        kl_m[:, t] = filter.current_mean
    end
    return @dict filter x y kl_m T
end

op = container_test()
x = op[:x]
y = op[:y]
km = op[:kl_m]
T = op[:T]
filter = op[:filter]

umf = rmse(x, km)
@info("Filter RMSE:", umf)
a_ef = containers.approx_energy_function(filter)[end]
@info("Approximate Energy: ", a_ef)

plot(1:T, x[1, :], size = (750, 500), label = "Truth", legend=:outertopright)
plot!(1:T, y, label = "Observations", st = :scatter)
plot!(1:T, km[1, :], label = "BPF Filter Mean")
