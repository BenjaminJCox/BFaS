using LinearAlgebra
using QuasiMonteCarlo

function box_muller_mvn(n, dims; sampler=LatinHypercubeSample, mean = zeros(dims), cov = Matrix(I(dims)))
    dims_rt = dims + 2
    upr = ones(dims_rt)
    lwr = zeros(dims_rt)
    bmised = zeros(dims_rt, n)
    quf_rv = QuasiMonteCarlo.sample(n, lwr, upr, sampler())
    bmised[1:2:end,:] .= sqrt.(-2.0 .* log.(quf_rv[1:2:end,:]))
    for i = 1:2:dims_rt
        bmised[i+1,:] .= bmised[i,:] .* cospi.(2.0 .* quf_rv[i+1,:])
        bmised[i,:] .*= sinpi.(2.0 .* quf_rv[i+1,:])
    end
    cov_chol = cholesky(cov)
    rcc(x) = mean .+ Matrix(cov_chol.U)' * x
    bmised[1:dims, :] = mapslices(rcc, bmised[1:dims, :], dims = 1)
    return bmised[1:dims, :]
end
