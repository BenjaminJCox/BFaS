using DrWatson
using Distributions
using Plots



include(srcdir("kf.jl"))
include(srcdir("particle.jl"))
include(srcdir("smoothers.jl"))

# gr()
plotlyjs()
function ex3_1()


    A = [1.0 1.0; 0.0 1.0]
    H = [1.0 0.0]
    Q = [1.0 / 10^2 0.0; 0.0 1.0^2]
    R = [10.0^2]

    m0 = rand(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]))

    seqlen = 100

    x = zeros(2, seqlen)
    y = zeros(seqlen)

    process_rand = MvNormal([0.0, 0.0], Q)
    obs_rand = Normal(0.0, sqrt(R[1]))

    x[:, 1] = m0
    y[1] = x[1, 1] + rand(obs_rand)

    for k = 2:seqlen
        x[:, k] = A * x[:, k-1] + rand(process_rand)
        y[k] = (H*x[:, k])[1] + rand(obs_rand)
    end

    m = [0.0; 0.0]
    P = Matrix{Float64}(I(2))

    kl_m = zeros(2, seqlen)
    kl_P = zeros(2, 2, seqlen)

    psi(x) = A * x
    Hf(x) = H * x
    dr = 1000
    nxs = bsf_draw_init(m, P, dr)
    wts = zeros(dr, seqlen)
    sds = zeros(dr, 2, seqlen)
    for k = 1:100
        # m, P = kf_predict(m, P, A, Q)
        # m, P = kf_update(m, P, [y[k]], H, hcat(R))
        # m, P = exkf_predict(m, P, psi, Q)
        # m, P = exkf_update(m, P, [y[k]], H, hcat(R))
        # m, P = ukf_predict(m, P, psi, Q)
        # m, P = ukf_update(m, P, [y[k]], Hf, hcat(R))
        m, P, nxs, wv, xpf = bsf_step(nxs, P, Q, [y[k]], hcat(R), psi, Hf)
        # println(m)
        # println(P)
        kl_m[:, k] = m
        kl_P[:, :, k] = P
        wts[:, k] = wv
        sds[:, :, k] = xpf
    end

    sm_m, sm_P = urts_smoother(kl_m, kl_P, psi, Q)

    # ps_m_alltraj = bsp_smoother(sds, wts, 30, psi, Q)

    ntj = rs_bsp_smoother(sds, wts, 30, psi, Q)

    return @dict x y kl_m kl_P sm_m sm_P wts sds ntj
end

op = ex3_1()
x = op[:x]
y = op[:y]
km = op[:kl_m]
sm = op[:sm_m]

plot(1:100, x[1, :], size = (750, 500), label = "Truth")
plot!(1:100, y, label = "Observations", st = :scatter)
plot!(1:100, km[1, :], label = "Filter Mean")
plot!(1:100, sm[1, :], label = "Smoother Mean")

# alltraj = op[:ps_m_alltraj]
# m_traj = mean(alltraj, dims = 2)
# plot!(1:100, m_traj[1, 1, :], label = "BSS Mean")


ntj = op[:ntj]
ntg_traj = mean(ntj, dims = 2)
plot!(1:100, ntg_traj[1, 1, :], label = "RS BSS Mean")
