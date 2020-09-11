#file containg code analogous to kf_update and kf_predict

using LinearAlgebra
using Distributions

#=
Input
X - Nx1 state mean of previous step
P - NxN state covariance estimate of pervious step
A - Transition matrix of discrete model
Q - Process noise of discrete model
B - Input effect matrix
U - Constant input

Output
X - Predicted state mean
P - Predicted state covariance
=#

function kf_predict(x::Vector{Float64}, P::Matrix, A = I(size(x, 1)), Q = zeros(size(x, 1), size(x, 1)), B = [], u = [])
    if (isempty(B) && !isempty(u))
        B = zeros(size(x, 1), size(u, 1))
        B[diagind(B)] .= 1.0
    end

    if isempty(u)
        xp = A * x
        Pp = A * P * A' + Q
    else
        xp = A * x + B * u
        Pp = A * P * A' + Q
    end
    return (xp, Pp)
end

#=
Input:
x - Nx1 state mean of previous step
P - NxN state covariance estimate of pervious step
y - Dx1 Measurement vector
H - Observation Matrix
R - Measurement noise covariance

Output:
x - Updated State Mean
P - Updated State Covariance
K - Kalman Gain
IM - Predictive mean of y
IS - Covariance of predictive mean of y
LH - Predictive likelihood of measurement
=#

function kf_update(x::Vector{Float64}, P::Matrix{Float64}, y::Vector{Float64}, H::Matrix{Float64}, R::Matrix{Float64})
    IM = H*x
    IS = R+H*P*H'
    K = P*H'/IS
    xp = x + K * (y-IM)
    Pp = P - K * IS * K'
    LH = pdf(MvNormal(IM, IS), y)
    return (xp, Pp, K, IM, IS, LH)
end
