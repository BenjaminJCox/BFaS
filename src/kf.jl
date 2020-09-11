#file containg code analogous to kf_update and kf_predict

using LinearAlgebra
using Distributions
using ForwardDiff

##exact kalman filter

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
    IM = H * x
    IS = R + H * P * H'
    K = P * H' / IS
    xp = x + K * (y - IM)
    Pp = P - K * IS * K'
    LH = pdf(MvNormal(IM, IS), y)
    return (xp, Pp, K, IM, IS, LH)
end

##approximate gaussian filters
#=
Inputs remain the same except@
psi: function mapping prior mean to model mean

Output no longer has predictive likelihood as this is not meaningful for approximate filters
=#

function threeDVAR_predict(x::Vector{Float64}, P::Matrix, psi = x -> x, Q = zeros(size(x, 1), size(x, 1)), B = [], u = [])
    if (isempty(B) && !isempty(u))
        B = zeros(size(x, 1), size(u, 1))
        B[diagind(B)] .= 1.0
    end

    if isempty(u)
        xp = psi(x)
        Pp = P
    else
        xp = psi(x) + B * u
        Pp = P
    end
    return (xp, Pp)
end

function threeDVAR_update(x::Vector{Float64}, P::Matrix{Float64}, y::Vector{Float64}, H::Matrix{Float64}, R::Matrix{Float64})
    IM = H * x
    IS = H * P * H' + R
    K = P * H' / IS
    xp = x + K * (y - IM)
    Pp = P
    return (xp, Pp, K, IM, IS)
end

#=
Because Julia is awesome we can take gradients of very general code, meaning that the extended Kalman filter remains useful for a rather
long time in this context
=#

function exkf_predict(x::Vector{Float64}, P::Matrix, psi = x -> x, Q = zeros(size(x, 1), size(x, 1)))
    xp = psi(x)
    jac = ForwardDiff.jacobian(psi, x)
    Pp = jac * P * jac' + Q
    return (xp, Pp)
end

function exkf_update(x::Vector{Float64}, P::Matrix{Float64}, y::Vector{Float64}, H::Matrix{Float64}, R::Matrix{Float64})
    IM = H * x
    IS = R + H * P * H' + R
    K = P * H' / IS
    xp = x + K * (y - IM)
    Pp = P - K * H * P
    return (xp, Pp, K, IM, IS)
end

function exkf_update(x::Vector{Float64}, P::Matrix{Float64}, y::Vector{Float64}, H, R::Matrix{Float64})
    H_jac = ForwardDiff.jacobian(H, x)
    v = y - H(x)
    IM = H(x)
    IS = H_jac * P * H_jac'
    K = P * H_jac' / IS
    xp = x + K * v
    Pp = P - K * IS * K'
    return (xp, Pp, K, IM, IS)
end
