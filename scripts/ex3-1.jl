using DrWatson
using Distributions
using Plots



include(srcdir("kf.jl"))
include(srcdir("particle.jl"))

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

    psi(x) = A * x
    Hf(x) = H * x
    # dr = 1000
    # nxs = bsf_draw_init(m, P, dr)
    for k = 1:100
        # m, P = kf_predict(m, P, A, Q)
        # m, P = kf_update(m, P, [y[k]], H, hcat(R))
        # m, P = exkf_predict(m, P, psi, Q)
        # m, P = exkf_update(m, P, [y[k]], H, hcat(R))
        m, P = ukf_predict(m, P, psi, Q)
        m, P = ukf_update(m, P, [y[k]], Hf, hcat(R))
        #m, P, nxs = bsf_step(nxs, P, Q, [y[k]], hcat(R), psi, Hf)
        # println(m)
        # println(P)
        kl_m[:, k] = m
    end
    return @dict x y kl_m
end

op = ex3_1()
x = op[:x]
y = op[:y]
km = op[:kl_m]

plot(1:100, x[1, :])
plot!(1:100, y)
plot!(1:100, km[1, :])
