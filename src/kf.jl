#file containing code analogous to kf_update and kf_predict

using LinearAlgebra
using Distributions

#calculating jacobians and hessians
using ForwardDiff

##exact kalman filter

#=
Input
X - Nx1 state mean of previous step
P - NxN state covariance estimate of previous step
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

#=
Input
X - Nx1 state mean of previous step
P - NxN state covariance estimate of pervious step
psi - Transition function of discrete model
Q - Process noise of discrete model
B - Input effect matrix
U - Constant input

Output
X - Predicted state mean
P - Predicted state covariance
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
=#

function threeDVAR_update(x::Vector{Float64}, P::Matrix{Float64}, y::Vector{Float64}, H::Matrix{Float64}, R::Matrix{Float64})
    IM = H * x
    IS = H * P * H' + R
    K = P * H' / IS
    xp = x + K * (y - IM)
    Pp = P
    return (xp, Pp, K, IM, IS)
end
##
#=
Because Julia is awesome we can take gradients of very general code, meaning that the extended Kalman filter remains useful for a rather
long time in this context

todo: store jacobian information so that calculation is MUCH MUCH faster, should be ok tbh but not a priority
=#

#=
Input
X - Nx1 state mean of previous step
P - NxN state covariance estimate of pervious step
psi - Transition function of discrete model
Q - Process noise of discrete model
B - Input effect matrix
U - Constant input

Output
X - Predicted state mean
P - Predicted state covariance
=#

function exkf_predict(x::Vector{Float64}, P::Matrix, psi = x -> x, Q = zeros(size(x, 1), size(x, 1)))
    xp = psi(x)
    jac = ForwardDiff.jacobian(psi, x)
    Pp = jac * P * jac' + Q
    return (xp, Pp)
end

function exkf_predict_nonadd(x, P, f, Q)
    _xdim = length(x)
    _pnz = zeros(_xdim)
    _f1(x) = f(x, _pnz)
    _f2(q) = f(x, q)
    Fx = ForwardDiff.jacobian(_f1, x)
    Fq = ForwardDiff.jacobian(_f2, _pnz)

    xp = f(x, _pnz)
    Pp = Fx * P * Fx' + Fq * Q * Fq'
    return (xp, Pp)
end
#=
Input:
x - Nx1 state mean of previous step
P - NxN state covariance estimate of pervious step
y - Dx1 Measurement vector
H - Observation Matrix/function
R - Measurement noise covariance

Output:
x - Updated State Mean
P - Updated State Covariance
K - Kalman Gain
IM - Predictive mean of y
IS - Covariance of predictive mean of y
=#

function exkf_update(x::Vector{Float64}, P::Matrix{Float64}, y::Vector{Float64}, H::Matrix{Float64}, R::Matrix{Float64})
    IM = H * x
    IS = R + H * P * H' + R
    K = P * H' / IS
    xp = x + K * (y - IM)
    Pp = P - K * H * P
    return (xp, Pp, K, IM, IS)
end

function exkf_update(x::Vector{Float64}, P::Matrix{Float64}, y::Vector{Float64}, H::Function, R::Matrix{Float64})
    H_jac = ForwardDiff.jacobian(H, x)
    v = y - H(x)
    IM = H(x)
    IS = H_jac * P * H_jac' + R
    K = P * H_jac' / IS
    xp = x + K * v
    Pp = P - K * IS * K'
    return (xp, Pp, K, IM, IS)
end

function exkf_update_nonadd(x, P, y, h, R)
    _ydim = length(y)
    _onz = zeros(_ydim)
    _h1(x) = h(x, _onz)
    _h2(r) = h(x, r)
    Hx = ForwardDiff.jacobian(_h1, x)
    Hr = ForwardDiff.jacobian(_h2, _onz)

    v = y .- h(x, _onz)
    S = Hx * P * Hx' + Hr * R * Hr'
    K = P * Hx' / S
    xp = x + K * v
    Pp = P - K * S * K'
    return (xp, Pp)
end
##
#=
This filter is less stanky than others. Because it is unscented. Am I humourous Father?!

DON'T USE THE FUNCTIONS BELOW EXCEPT THE PREDICT AND UPDATE ONES, THEY ARE JUST FOR CONVIENIENCE
=#

function ukf_generate_sigma_points(
    x::Vector{Float64},
    P::Matrix{Float64};
    alpha::Float64 = 1.0,
    kappa::Float64 = 3.0 - size(x, 1),
)
    n = size(x, 1)
    sigma_point_matrix = zeros(size(x, 1), 2 * size(x, 1) + 1)
    sigma_point_matrix[:, 1] = x
    srp = sqrt(P) # this is fine as P is guaranteed to be symmetric by virtue of being a covariance matrix
    lambda = alpha^2 * (n + kappa) - n
    for i = 1:n
        sigma_point_matrix[:, i+1] = x + sqrt(n + lambda) * srp[:, i]
        sigma_point_matrix[:, i+1+n] = x - sqrt(n + lambda) * srp[:, i]
    end
    return sigma_point_matrix
end

function ukf_generate_sigma_points_sqrt(
    x::Vector{Float64},
    sqrtP::Matrix{Float64};
    alpha::Float64 = 1.0,
    kappa::Float64 = 3.0 - size(x, 1),
)
    n = size(x, 1)
    sigma_point_matrix = zeros(size(x, 1), 2 * size(x, 1) + 1)
    sigma_point_matrix[:, 1] = x
    lambda = alpha^2 * (n + kappa) - n
    for i = 1:n
        sigma_point_matrix[:, i+1] = x + sqrt(n + lambda) * sqrtP[:, i]
        sigma_point_matrix[:, i+1+n] = x - sqrt(n + lambda) * sqrtP[:, i]
    end
    return sigma_point_matrix
end

#this should get inlined but idec
function ukf_propagate_sigma_points(sigma_point_matrix::Matrix{Float64}, f::Function = x -> x)
    mapped_sigma_points = mapslices(f, sigma_point_matrix, dims = 1)
    return mapped_sigma_points
end

function ukf_propagate_sigma_points_param(sigma_point_matrix::Matrix{Float64}, f::Function, parameters)
    f_p(x) = f(x, parameters)
    mapped_sigma_points = mapslices(f_p, sigma_point_matrix, dims = 1)
    return mapped_sigma_points
end

function ukf_weights(n::Int64; alpha::Float64 = 1.0, kappa::Float64 = 3.0 - size(x, 1), beta::Float64 = 0.0)
    weight_m_vector = ones(2n + 1)
    weight_c_vector = ones(2n + 1)

    lambda = alpha^2 * (n + kappa) - n

    wv = 1 / (2 * n + 2 * lambda)

    weight_m_vector .*= wv
    weight_c_vector .*= wv


    weight_m_vector[1] = lambda / (n + lambda)
    weight_c_vector[1] = lambda / (n + lambda) + (1 - alpha^2 + beta)

    return @dict weight_m_vector weight_c_vector
end

function ukf_pred_mean_cv(sigma_points::Matrix{Float64}, weight_dict, Q::Matrix{Float64})
    n = size(sigma_points, 1)
    mkd = zeros(n)

    weight_m_vector = weight_dict[:weight_m_vector]
    weight_c_vector = weight_dict[:weight_c_vector]

    for i = 1:(2n+1)
        mkd += weight_m_vector[i] * sigma_points[:, i]
    end

    Pkd = Q
    for i = 1:(2n+1)
        mcs = sigma_points[:, i] - mkd
        Pkd += weight_c_vector[i] * mcs * mcs'
    end

    return (mkd, Pkd)
end

function ukf_upda_mean_cvneas_ccv_statemeas(
    mkd::Vector{Float64},
    sigma_points::Matrix{Float64},
    measuremodel_sigma_points::Matrix{Float64},
    weight_dict,
    R::Matrix{Float64},
)
    n = size(sigma_points, 1)
    muk = zeros(size(measuremodel_sigma_points, 1))

    weight_m_vector = weight_dict[:weight_m_vector]
    weight_c_vector = weight_dict[:weight_c_vector]

    for i = 1:(2n+1)
        muk += weight_m_vector[i] * measuremodel_sigma_points[:, i]
    end

    Sk = R
    for i = 1:(2n+1)
        mcs = measuremodel_sigma_points[:, i] - muk
        Sk += weight_c_vector[i] * mcs * mcs'
    end

    Ck = zeros(size((sigma_points[:, 1] - mkd) * (measuremodel_sigma_points[:, 1] - muk)'))
    for i = 1:(2n+1)
        Ck += weight_c_vector[i] * (sigma_points[:, i] - mkd) * (measuremodel_sigma_points[:, i] - muk)'
    end
    return (muk, Sk, Ck)
end

#=
Input
X - Nx1 state mean of previous step
P - NxN state covariance estimate of pervious step
psi - Transition function of discrete model
Q - Process noise of discrete model
B - Input effect matrix
U - Constant input

Output
X - Predicted state mean
P - Predicted state covariance
=#

function ukf_predict(
    x::Vector{Float64},
    P::Matrix,
    psi = x -> x,
    Q = zeros(size(x, 1), size(x, 1));
    alpha::Float64 = 1.0,
    kappa::Float64 = 3.0 - size(x, 1),
    beta::Float64 = 0.0,
)
    sigma_points = ukf_generate_sigma_points(x, P; alpha = alpha, kappa = kappa)
    mapped_sigma_points = ukf_propagate_sigma_points(sigma_points, psi)
    weights = ukf_weights(size(x, 1); alpha = alpha, kappa = kappa, beta = beta)
    (mp, Pp) = ukf_pred_mean_cv(mapped_sigma_points, weights, Q)
    return (mp, Pp)
end

function ukf_predict_sqrt(
    x::Vector{Float64},
    sqrtP::Matrix,
    psi = x -> x,
    Q = zeros(size(x, 1), size(x, 1));
    alpha::Float64 = 1.0,
    kappa::Float64 = 3.0 - size(x, 1),
    beta::Float64 = 0.0,
)
    sigma_points = ukf_generate_sigma_points_sqrt(x, sqrtP; alpha = alpha, kappa = kappa)
    mapped_sigma_points = ukf_propagate_sigma_points(sigma_points, psi)
    weights = ukf_weights(size(x, 1); alpha = alpha, kappa = kappa, beta = beta)
    (mp, Pp) = ukf_pred_mean_cv(mapped_sigma_points, weights, Q)
    return (mp, Pp)
end

function ukf_predict_sqrt_param(
    x::Vector{Float64},
    sqrtP::Matrix,
    psi = x -> x,
    Q = zeros(size(x, 1), size(x, 1)),
    parameters = Dict{Symbol,Any}();
    alpha::Float64 = 1.0,
    kappa::Float64 = 3.0 - size(x, 1),
    beta::Float64 = 0.0,
)
    sigma_points = ukf_generate_sigma_points_sqrt(x, sqrtP; alpha = alpha, kappa = kappa)
    mapped_sigma_points = ukf_propagate_sigma_points_param(sigma_points, psi, parameters)
    weights = ukf_weights(size(x, 1); alpha = alpha, kappa = kappa, beta = beta)
    (mp, Pp) = ukf_pred_mean_cv(mapped_sigma_points, weights, Q)
    return (mp, Pp)
end

#=
Input:
x - Nx1 state mean of previous step
P - NxN state covariance estimate of pervious step
y - Dx1 Measurement vector
H - Observation function
R - Measurement noise covariance

Output (go by order not name screw you logic):
x - Updated State Mean
P - Updated State Covariance
K - Kalman Gain
IM - Predictive mean of y
IS - Covariance of predictive mean of y
=#

function ukf_update(
    x::Vector{Float64},
    P::Matrix{Float64},
    y::Vector{Float64},
    H,
    R::Matrix{Float64};
    alpha::Float64 = 1.0,
    kappa::Float64 = 3.0 - size(x, 1),
    beta::Float64 = 0.0,
)
    sigma_points = ukf_generate_sigma_points(x, P; alpha = alpha, kappa = kappa)
    mapped_sigma_points = ukf_propagate_sigma_points(sigma_points, H)
    weights = ukf_weights(size(x, 1); alpha = alpha, kappa = kappa, beta = beta)
    (muk, Sk, Ck) = ukf_upda_mean_cvneas_ccv_statemeas(x, sigma_points, mapped_sigma_points, weights, R)
    Kk = Ck / Sk
    mk = x + Kk * (y - muk)
    Pk = P - Kk * Sk * Kk'
    return (mk, Pk, Kk, muk, Sk)
end

function ukf_update_param(
    x::Vector{Float64},
    P::Matrix{Float64},
    y::Vector{Float64},
    H,
    R::Matrix{Float64},
    parameters;
    alpha::Float64 = 1.0,
    kappa::Float64 = 3.0 - size(x, 1),
    beta::Float64 = 0.0,
)
    sigma_points = ukf_generate_sigma_points(x, P; alpha = alpha, kappa = kappa)
    mapped_sigma_points = ukf_propagate_sigma_points_param(sigma_points, H, parameters)
    weights = ukf_weights(size(x, 1); alpha = alpha, kappa = kappa, beta = beta)
    (muk, Sk, Ck) = ukf_upda_mean_cvneas_ccv_statemeas(x, sigma_points, mapped_sigma_points, weights, R)
    Kk = Ck / Sk
    mk = x + Kk * (y - muk)
    Pk = P - Kk * Sk * Kk'
    return (mk, Pk, Kk, muk, Sk)
end

function ukf_update_sqrt(
    x::Vector{Float64},
    P::Matrix{Float64},
    sqrtP::Matrix{Float64},
    y::Vector{Float64},
    H,
    R::Matrix{Float64};
    alpha::Float64 = 1.0,
    kappa::Float64 = 3.0 - size(x, 1),
    beta::Float64 = 0.0,
)
    sigma_points = ukf_generate_sigma_points_sqrt(x, sqrtP; alpha = alpha, kappa = kappa)
    mapped_sigma_points = ukf_propagate_sigma_points(sigma_points, H)
    weights = ukf_weights(size(x, 1); alpha = alpha, kappa = kappa, beta = beta)
    (muk, Sk, Ck) = ukf_upda_mean_cvneas_ccv_statemeas(x, sigma_points, mapped_sigma_points, weights, R)
    Kk = Ck / Sk
    mk = x + Kk * (y - muk)
    Pk = P - Kk * Sk * Kk'
    return (mk, Pk, Kk, muk, Sk)
end
##



function rmse(truth::Matrix{Float64}, estimate::Matrix{Float64})
    T = size(truth, 2)
    @assert size(truth) == size(estimate)

    mse = 0.
    for i = 1:T
        sre = truth[:, i] - estimate[:, i]
        mse += sre' * sre
    end
    mse /= T
    rmse = sqrt(mse)
    return rmse
end

function rmse(truth::Vector{Float64}, estimate::Vector{Float64})
    T = size(truth, 2)
    @assert size(truth) == size(estimate)

    mse = 0.
    for i = 1:T
        sre = truth[i] - estimate[i]
        mse += sre' * sre
    end
    mse /= T
    rmse = sqrt(mse)
    return rmse
end
