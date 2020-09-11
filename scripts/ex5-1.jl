using DrWatson
using LinearAlgebra
using Distributions
using Plots
using StaticArrays
include(srcdir("kf.jl"))

global const dt = 0.005
global const g = 9.81
global const seqlen = 1000

function ex5_1()
    function linearised_pendulum(x)
        xp = [x[1]+x[2]*dt, x[2]-g*sin(x[1])*dt]
        return xp
    end

    function obs_func(x)
        yk = sin(x[1])
        return [yk]
    end

    qc = 0.1

    Q = [qc * dt^3 / 3 qc * dt^2 / 2; qc * dt^2 / 2 qc * dt]

    R = 0.01


    x = zeros(2, seqlen)
    y = zeros(seqlen)

    m0 = Vector{Float64}([1., 1.])

    process_rand = MvNormal([0.0, 0.0], Q)
    obs_rand = Normal(0.0, R)

    x[:, 1] = m0
    y[1] = sin(x[1, 1]) + rand(obs_rand)

    for k = 2:seqlen
        x[:, k] = linearised_pendulum(x[:, k-1]) + rand(process_rand)
        y[k] = obs_func(x[:, k])[1] + rand(obs_rand)
    end

    m = m0
    P = Matrix{Float64}(I(2))

    kl_m = zeros(2, seqlen)


    for k = 1:seqlen
        # m, P = kf_predict(m, P, A, Q)
        # m, P = kf_update(m, P, [y[k]], H, hcat(R))
        m, P = exkf_predict(m, P, linearised_pendulum, Q)
        m, P = exkf_update(m, P, [y[k]], obs_func, hcat(R))
        kl_m[:, k] = m
    end
    return @dict x y kl_m

end

op = ex5_1()
x = op[:x]
y = op[:y]
km = op[:kl_m]

plot(1:seqlen, x[1, :])
plot!(1:seqlen, y)
plot!(1:seqlen, km[1, :])
