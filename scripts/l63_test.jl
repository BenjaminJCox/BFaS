using DrWatson
using Distributions
using Plots
using DifferentialEquations

include(srcdir("kf.jl"))
include(srcdir("particle.jl"))
include(srcdir("smoothers.jl"))
include(srcdir("param_est.jl"))


Random.seed!(6)
# gr()
plotlyjs()

function l63_test()
    function lorenz(du, u, p, t)
        du[1] = 10.0(u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
    end
    function σ_lorenz(du, u, p, t)
        du[1] = 3.0
        du[2] = 3.0
        du[3] = 3.0
    end
    u0 = [1.0, 0.0, 0.0]
    t_ext = (0., 10.)
    time = range(t_ext[1], t_ext[2], step = 0.05)
    seq_len = length(time)
    prob_sde_lorenz = SDEProblem(lorenz, σ_lorenz, u0, t_ext, saveat = time)
    sol = solve(prob_sde_lorenz)

    prob_ode_lorenz = ODEProblem(lorenz, u0, t_ext)
    det_sol = solve(prob_ode_lorenz)
    return (sol, det_sol)
end

# s = l63_test()
#
# sde = plot(s[1], vars=(1,2,3))
# ode = plot(s[2], vars=(1,2,3))
#
# plot(sde, ode, layout = (1, 2), size = (1500, 750))

function lorenz(du, u, p, t)
    du[1] = 10.0(u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

u0 = [1.0, 0.0, 0.0]
t_ext = (0., 100.)

prob_ode_lorenz = ODEProblem(lorenz, u0, t_ext)
det_sol = solve(prob_ode_lorenz, BS5())

ode_plt = plot(det_sol, vars=(1,2,3))
plot(ode_plt, size = (750, 750))
