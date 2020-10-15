using DrWatson
using LinearAlgebra
using Distributions
using Plots
plotlyjs()
# gr()

include(srcdir("kf.jl"))
include(srcdir("particle.jl"))
include(srcdir("smoothers.jl"))

global const dt = 0.01
global const g = 9.81
global const seqlen = 500

function ex5_1()
    function linearised_pendulum(x)
        xp = [x[1] + x[2] * dt, x[2] - g * sin(x[1]) * dt]
        return xp
    end

    function obs_func(x)
        yk = sin(x[1])
        return [yk]
    end

    qc = 0.001

    Q = [qc * dt^3 / 3 qc * dt^2 / 2; qc * dt^2 / 2 qc * dt]

    R = 0.1


    x = zeros(2, seqlen)
    y = zeros(seqlen)

    m0 = Vector{Float64}([0.5, 0.0])

    process_rand = MvNormal(Q)
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
    kl_P = zeros(2, 2, seqlen)

    dr = 1000
    nxs = bsf_draw_init(m0, P, dr)
    wts = zeros(dr, seqlen)
    sds = zeros(dr, 2, seqlen)

    for k = 1:seqlen
        # m, P = kf_predict(m, P, A, Q)
        # m, P = kf_update(m, P, [y[k]], H, hcat(R))
        # m, P = exkf_predict(m, P, linearised_pendulum, Q)
        # m, P = exkf_update(m, P, [y[k]], obs_func, hcat(R))
        # m, P = ukf_predict(m, P, linearised_pendulum, Q)
        # m, P = ukf_update(m, P, [y[k]], obs_func, hcat(R))
        m, P, nxs, wv, xpf = bsf_step(nxs, P, Q, [y[k]], hcat(R), linearised_pendulum, obs_func)
        # m, P, nxs, wv = sir_filter_ukfprop(nxs, P, Q, [y[k]], hcat(R), linearised_pendulum, obs_func)
        kl_m[:, k] = m
        kl_P[:, :, k] = P
        wts[:, k] = wv
        # sds[:, :, k] = xpf
    end

    sm_m, sm_P = urts_smoother(kl_m, kl_P, linearised_pendulum, Q)
    # ps_m_alltraj = bsp_smoother(sds, wts, 100, linearised_pendulum, Q)

    return @dict x y kl_m sm_m #ps_m_alltraj
end

op = ex5_1()
x = op[:x]
y = op[:y]
km = op[:kl_m]
sm = op[:sm_m]
ts = collect(dt .* (1:seqlen))

plot(ts, x[1, :], size = (750, 500), label = "Truth")
plot!(ts, y, st = :scatter, label = "Observations")
plot!(ts, km[1, :], label = "Filter Mean")
plot!(ts, sm[1, :], label = "Smoother Mean")

# alltraj = op[:ps_m_alltraj]
# m_traj = mean(alltraj, dims = 3)
# plot!(ts, m_traj[1, :, :])
