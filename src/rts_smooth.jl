using LinearAlgebra

#=
Input:
M - NxK matrix of K mean estimates from Kalman filter
P - NxNxK array of K state covariances from Kalman filter
A - NxN state transition matrix or NxNxK array of K state transition matrices
Q - NxN process noise covariance matrix or NxNxK matrix of K state process noise covariace matrices

Output:
M - Smoothed state mean sequence
P - Smoothed state covariance sequence
D - Smoothed gain sequence
=#

function rts_smooth(M, P, A, Q)
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
    for k = (size(M, 2)):-1:1
        P_pred = A[:, :, k] * P[:, :, k] * A[:, :, k]' + Q[:, :, k]
        D[:, :, k] = P[:, :, k] * A[:, :, k]' / P_pred
        Mp[:, k] = M[:, k] + D[:, :, k] * (M[:, k+1] - A[:, :, k] * M[:, k])
        Pp[:, :, k] = P[:, :, k] + D[:, :, k] * (P[:, :, k+1] - P_pred) * D[:, :, k]'
    end

    return (Mp, Pp, D)
end
