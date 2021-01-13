using DrWatson
using LinearAlgebra
using Cubature
using Distributions

g_pdf_un(x; mu = 0.0, sigma = 1.0) = exp(-0.5 * (x .- mu)' / (sigma) * (x .- mu))

igpdf = hcubature(g_pdf_un, [-100.0], [100.0])
igpdftv = sqrt(2π)
igpdferr = igpdf[1] - igpdftv

xp = [1.0, 0.0]
Pp = [1.0 0.0; 0.0 1.0]

global const dt = 0.01
global const g = 9.81

function linearised_pendulum(x)
    xp = [x[1] + x[2] * dt, x[2] - g * sin(x[1]) * dt]
    return xp
end

psi(x) = x * x

mvn = MvNormal(xp, Pp)
gpdf(x) = pdf(mvn, x)

g_pdf(x; mu = xp, sigma = Pp) = (2π)^(-size(x, 1) / 2.0) * det(sigma)^(-0.5) * exp(-0.5 * (x .- mu)' / (sigma) * (x .- mu))
n = size(x, 1)
big_number = 100.0
function ev(x)
    (psi(x) * g_pdf(x))
end
function ev(x, v)
    v[:] .= (psi(x) * gpdf(x))
end
idint = hquadrature(n, ev, -100.0, 100.0)
idint


###
#=
conclusion:
use specifically implemented quadratures (gauss hermite comes to mind)
tbh assumed gaussian filters perform about as well as unscented. not going to bother implementing, going to instead focus on ensemble and particle methods from now on
=#
