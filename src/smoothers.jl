using DrWatson
using LinearAlgebra
using Distributions
using ForwardDiff

include("kf.jl") #mostly for sigma points

#=
Input:
    M - NxK matrix of K mean estimates from Kalman filter
    P - NxNxK matrix of K state covariances from Kalman Filter
    A - NxN state transition matrix or NxNxK matrix of K state
       transition matrices for each step.
    Q - NxN process noise covariance matrix or NxNxK matrix
       of K state process noise covariance matrices for each step.

Out:
M - Smoothed state mena sequence
P - Smoothed state covariance sequence
D - Smoothed Gain Sequence
=#
function rts_smoother(M::Matrix{Float64}, P::Array{Float64,3}, A::Array, Q::Array)
    Ap = A
    Qp = Q
    Mp = M
    Pp = P
    if size(A, 3) == 1
        Ap = repeat(A, 1, 1, size(M, 2))
    end
    if size(Q, 3) == 1
        Qp = repeat(Q, 1, 1, size(M, 2))
    end

    D = zeros(size(M, 1), size(M, 1), size(M, 2))
    for k = (size(M, 2)-1):-1:1
        P_pred = A[:, :, k] * P[:, :, k] * A[:, :, k]' + Q[:, :, k]
        D[:, :, k] = P[:, :, k] * A[:, :, k]' / P_pred
        Mp[:, k] = M[:, k] + D[:, :, k] * (M[:, k+1] - A[:, :, k] * M[:, k])
        Pp[:, :, k] = P[:, :, k] + D[:, :, k] * (P[:, :, k+1] - P_pred) * D[:, :, k]'
    end

    return (Mp, Pp, D)
end


#=
Extended Rauch Tung Striebel Smoother
=#
function erts_smoother(M::Matrix{Float64}, P::Array{Float64,3}, A::Function, Q::Array)
    Qq = Q
    if size(Q, 3) == 1
        Qp = repeat(Q, 1, 1, size(M, 2))
    end
    M_arr = zeros(size(M))
    P_arr = zeros(size(P))
    G_arr = zeros(size(M, 1), size(M, 1), size(M, 2))

    M_arr[:, end] = M[:, end]
    P_arr[:, :, end] = P[:, :, end]
    for k = (size(M, 2)-1):-1:1
        mk = M[:, k]
        Pk = P[:, :, k]
        Qk = Qp[:, :, k]
        jac = ForwardDiff.jacobian(A, mk)

        m_pred = A(mk)
        P_pred = jac * Pk * jac' + Qk

        Gk = Pk * jac' / P_pred
        msk = mk + Gk * (M_arr[:, k+1] - m_pred)
        Psk = Pk + Gk * (P_arr[:, :, k+1] - P_pred) * Gk'

        M_arr[:, k] = msk
        P_arr[:, :, k] = Psk
        G_arr[:, :, k] = Gk
    end
    return (M_arr, P_arr, G_arr)
end

#=
Unscented Rauch Tung Striebel Smoother
=#
function ukf_dpred(orig_sigma_points, sigma_points, mk, m_pred, weight_dict)
    n = size(sigma_points, 1)
    wc = weight_dict[:weight_c_vector]
    Dk = zeros(size((orig_sigma_points[:, 1] - mk) * (sigma_points[:, 1] - m_pred)'))
    for i = 1:(2n+1)
        Dk += wc[i] * (orig_sigma_points[:, i] - mk) * (sigma_points[:, i] - m_pred)'
    end
    return Dk
end

function urts_smoother(
    M::Matrix{Float64},
    P::Array{Float64,3},
    A::Function,
    Q::Array;
    alpha::Float64 = 1.0,
    kappa::Float64 = 3.0 - size(M, 1),
    beta::Float64 = 0.0,
)
    Qq = Q
    if size(Q, 3) == 1
        Qp = repeat(Q, 1, 1, size(M, 2))
    end
    M_arr = zeros(size(M))
    P_arr = zeros(size(P))
    G_arr = zeros(size(M, 1), size(M, 1), size(M, 2))

    M_arr[:, end] = M[:, end]
    P_arr[:, :, end] = P[:, :, end]
    for k = (size(M, 2)-1):-1:1
        mk = M[:, k]
        Pk = P[:, :, k]
        Qk = Qp[:, :, k]

        spoints = ukf_generate_sigma_points(mk, Pk; alpha = alpha, kappa = kappa)
        mspoints = ukf_propagate_sigma_points(spoints, A)
        wts = ukf_weights(size(mk, 1); alpha = alpha, kappa = kappa, beta = beta)
        m_pred, P_pred = ukf_pred_mean_cv(mspoints, wts, Qk)
        Dk = ukf_dpred(spoints, mspoints, mk, m_pred, wts)

        Gk = Dk / P_pred
        msk = mk + Gk * (M_arr[:, k+1] - m_pred)
        Psk = Pk + Gk * (P_arr[:, :, k+1] - P_pred) * Gk'

        M_arr[:, k] = msk
        P_arr[:, :, k] = Psk
        G_arr[:, :, k] = Gk
    end
    return (M_arr, P_arr, G_arr)
end

#=
Backward simulation particle smoother
=#
function bsp_smoother(X::Array{Float64,3}, W::Array{Float64,2}, s::Int64, psi::Function, Q::Array{Float64})
    n = size(W, 1)
    T = size(W, 2)
    Qq = Q
    if size(Q, 3) == 1
        Qp = repeat(Q, 1, 1, T)
    end
    trajectories = zeros(size(X, 2), T, s)
    Threads.@threads for i = 1:s
        xtbarind = wsample(1:n, W[:, T], 1)
        wvp = zeros(n)
        xtbar = zeros(size(X, 2), T)
        xtbar[:, T] = X[xtbarind, :, T]
        for t = (T-1):-1:1
            @simd for k = 1:n
                wvp[k] = pdf(MvNormal(psi(X[k, :, t]), Qp[:, :, t]), xtbar[:, t+1])
            end
            wvp ./= sum(wvp)
            xtbar[:, t] = X[wsample(1:n, wvp, 1), :, t]
        end
        trajectories[:, :, i] = xtbar
    end
    return trajectories
end