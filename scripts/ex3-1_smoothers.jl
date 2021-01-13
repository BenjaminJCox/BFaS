using DrWatson
using Distributions
using Plots

include(srcdir("kf.jl"))
include(srcdir("particle.jl"))
include(srcdir("smoothers.jl"))
include(srcdir("param_est.jl"))


Random.seed!(3)
# gr()
plotlyjs()
function ex3_1()


    # A = [1.0 1.0; -0.1 0.2]
    function psi(x)
        nx = copy(x)
        nx[1] = x[1] + x[2]
        nx[2] = -cbrt(abs((x[1] - 10))) * sign(x[1] - 10)
        return nx
    end
    H = [1.0 0.0]
    Q = [1.0/10^2 0.0; 0.0 1.0^2]
    R = [2.0^2]

    m0 = rand(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]))

    seqlen = 150

    x = zeros(2, seqlen)
    y = zeros(seqlen)

    process_rand = MvNormal([0.0, 0.0], Q)
    obs_rand = Normal(0.0, sqrt(R[1]))

    x[:, 1] = m0
    y[1] = x[1, 1] + rand(obs_rand)

    for k = 2:seqlen
        # x[:, k] = A * x[:, k-1] + rand(process_rand)
        x[:, k] =  psi(x[:, k-1]) + rand(process_rand)
        y[k] = (H*x[:, k])[1] + rand(obs_rand)
    end

    m = [0.0; 0.0]
    P = Matrix{Float64}(I(2))

    kl_m = zeros(2, seqlen)
    kl_P = zeros(2, 2, seqlen)

    kl_em = zeros(2, seqlen)
    kl_eP = zeros(2, 2, seqlen)

    em = m
    eP = P

    # psi(x) = A * x
    Hf(x) = H * x
    dr = 1000
    nxs = bsf_draw_init(m, P, dr)
    inits = nxs
    wts = zeros(dr, seqlen)
    wv = zeros(dr)
    wv .+= 1.0
    sds = zeros(dr, 2, seqlen)
    lh_arr = zeros(seqlen)
    # ef_arr = zeros(seqlen+1)
    # ef_arr[1] = 0.0
    large_num = 5.
    # Q .*= 1.2
    # R .*= 1.9

    y_arr = zeros(1, seqlen)
    y_arr[1,:] .= y[:]

    sds[:, :, 1] = nxs
    for k = 1:150
        # m, P = kf_predict(m, P, A, Q)
        # m, P = kf_update(m, P, [y[k]], H, hcat(R))
        # m, P = exkf_predict(m, P, psi, Q)
        # m, P = exkf_update(m, P, [y[k]], H, hcat(R))
        # lh_arr[k] = -log(marginal_lh(em, [y[k]], psi, Hf, Q, hcat(R), em.+[-large_num, -large_num], em.+[large_num, large_num]))
        # ef_arr[k+1] = ef_arr[k] + lh_arr[k]
        em, eP = ukf_predict(em, eP, psi, Q)
        em, eP = ukf_update(em, eP, [y[k]], Hf, hcat(R))
        # m, P, nxs, wv, xpf = bsf_step(nxs, P, Q, [y[k]], hcat(R), psi, Hf)
        # m, P, nxs, wv, xpf = apf_step(nxs, Q, [y[k]], hcat(R), psi, Hf, wv)
        ukf_return = sir_filter_ukfprop(nxs, P, Q, [y[k]], hcat(R), psi, Hf)
        # if k > 3
        # ukf_return = resample_move_MH_pf_step(nxs, P, Q, [y[k]], hcat(R), psi, Hf, sds[:,:,1:k-1], y_arr[:,1:k-1], 5, 2)
        # end
        m = ukf_return[1]
        P = ukf_return[2]
        nxs = ukf_return[3]
        wv = ukf_return[4]
        xpf = ukf_return[3]
        lh_arr[k] = ukf_return[6]
        s_inds = ukf_return[5]
        # rs_paths = ukf_return[7]
        # println(m)
        # println(P)

        kl_em[:, k] = em
        kl_eP[:, :, k] = eP
        kl_m[:, k] = m
        kl_P[:, :, k] = P
        wts[:, k] = wv
        sds[:, :, 1:k] = sds[s_inds, :, 1:k]
        # sds[:, :, 1:k] = rs_paths
        sds[:, :, k] = xpf
        # sds[:, :, k] = nxs
    end
    llh_arr = log.(lh_arr)
    ef_arr = -1. .* cumsum(llh_arr)

    # y_long = Matrix(transpose(hcat(y)))
    # @info("Filter Complete")
    # mhrso = mh_kernel(sds[1,:,:], y_long, psi, Hf, Q, hcat(R), 3, 50)
    # sm_m, sm_P = urts_smoother(kl_em, kl_eP, psi, Q)
    # @info("URTS Smoother Complete")

    # ps_m_alltraj = bsp_smoother(sds, wts, 30, psi, Q)
    # @info("Backward Simulation Particle Smoother Complete")

    # ntj = rs_bsp_smoother(sds, wts, 30, psi, Q)
    # @info("Rejection Sampling Backward Simulation Particle Smoother Complete")

    # sirp = sirp_smoother(
    #     Matrix(transpose(hcat(y))),
    #     psi,
    #     Hf,
    #     Q,
    #     hcat(R),
    #     Matrix(transpose(inits[1:30, :])),
    # )
    # @info("SIR Particle Smoother Complete")

    # gibbs = pgas_smooth(kl_m, Matrix(hcat(y)'), 1000, psi, Hf, Q, hcat(R), Matrix(inits[1:999, :]'))

    # pmmh = naive_pmmh_run(
    #     m0 .+ 5.0,
    #     P ./ 3,
    #     Q .* 2,
    #     hcat(R) .* 2,
    #     Matrix(transpose(hcat(y))),
    #     psi,
    #     Hf,
    #     100,
    #     50,
    # )
    # println("Particle Marginal Metropolis Hastings Complete")
    # rv_ef = ef_arr[2:end]

    # return @dict x y kl_m kl_P kl_em kl_eP sm_m sm_P wts sds sirp ntj ps_m_alltraj lh_arr rv_ef
    return @dict x y kl_m kl_em kl_P kl_eP ef_arr sds
end

op = ex3_1()
x = op[:x]
y = op[:y]
km = op[:kl_m]
ekm = op[:kl_em]
efun = op[:ef_arr]
sds = op[:sds]
klp = op[:kl_P]
# sm = op[:sm_m]
# wts = op[:wts]


umf = rmse(x, km)
emf = rmse(x, ekm)
@info("UKF RMSE:", emf)
@info("SIRUKF RMSE:", umf)
@info("Approximate Energy:", efun[end])


plot(1:150, x[1, :], size = (750, 500), label = "Truth", legend=:outertopright)
plot!(1:150, y, label = "Observations", st = :scatter)
plot!(1:150, km[1, :], label = "Filter Mean")
plot!(1:150, ekm[1, :], label = "UKF Mean")
# plot!(1:150, sm[1, :], label = "URTS Smoother Mean")
#
# sirp = op[:sirp]
# m_sirp = mean(sirp, dims = 2)[:, 1, :]
# plot!(1:150, m_sirp[1, :], label = "SIR Smoother Mean")
#
# alltraj = op[:ps_m_alltraj]
# m_traj = mean(alltraj, dims = 2)
# plot!(1:150, m_traj[1, 1, :], label = "BSS Mean")
#
#
#
# ntj = op[:ntj]
# ntg_traj = mean(ntj, dims = 2)
# plot!(1:150, ntg_traj[1, 1, :], label = "RS BSS Mean")

# y_lh = op[:lh_arr]
# p1 = plot(1:150, y_lh, label = "Observation Likelihood")
# y_ef = op[:rv_ef]
# p2 = plot(1:150, y_ef, label = "Energy Function")
# plot(p1, p2, layout = (2,1), size = (750, 500))


# pmmh_path = op[:pmmh][:selected_paths][:, :, 50]
# plot!(1:100, pmmh_path[1, :], label = "PMMH Path <br> (For θ inference)")


#
# gibbs = op[:gibbs]
# g_part = size(gibbs[2], 1)
# gind = wsample(1:g_part, gibbs[2], 1)
# g_path = gibbs[1][:, gind, :][:, 1, :]
# g_mean = sum(gibbs[2]' .* gibbs[1][:, :, :], dims = 2)[:, 1, :]
# plot!(1:100, g_path[1, :], label = "PGAS")
