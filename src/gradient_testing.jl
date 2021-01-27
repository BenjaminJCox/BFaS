using DrWatson
using Distributions
using Plots
using ForwardDiff

include(srcdir("kf.jl"))

function perform_kalman(observations, A, H, m0, P0, Q, R)
    m = copy(m0)
    P = copy(P0)
    v = observations[:, 1] - H * m
    S = H * P * H' + R
    K = P * H' / S
    T = size(observations, 2)
    _xd = length(m0)
    filtered_state = zeros(Real, length(m0), T)
    filtered_cov = zeros(Real, length(m0), length(m0), T)
    l_like_est = 0.0
    offness = 0.0
    for t = 1:T
        m = A * m
        P = A * P * transpose(A) + Q
        v = observations[:, t] - H * m
        S = H * P * transpose(H) + R
        offness += norm(S - Matrix(Hermitian(S)), 1)
        S = Matrix(Hermitian(S))
        K = (P * transpose(H)) * inv(S)
        l_like_est += logpdf(MvNormal(H * m, S), observations[:, t])
        # unstable, need to implement in sqrt form
        m = m + K * v
        P = (I(_xd) - K * H) * P * (I(_xd) - K * H)' + K * R * K'
        filtered_state[:, t] = m
        filtered_cov[:, :, t] = P
    end
    return (filtered_state, filtered_cov, l_like_est, offness)
end

function kalman_llh(observations, A, H, m0, P0, Q, R)
    return perform_kalman(observations, A, H, m0, P0, Q, R)[3]
end

A = [1.0 1.0; 0.0 1.0]
H = [1.0 0.0]

P = [0.1 0.0; 0.0 0.1]
Q = [0.1 0.0; 0.0 1.0]
R = hcat([1.0])

m0 = [0.1, 0.0]

T = 50

X = zeros(2, T)
Y = zeros(1, T)

prs_noise = MvNormal(Q)
obs_noise = MvNormal(R)
prior_state = MvNormal(m0, P)

X[:, 1] = rand(prior_state)
Y[:, 1] = H * X[:, 1] .+ rand(obs_noise)

for t in 2:T
    X[:, t] = A * X[:, t-1] .+ rand(prs_noise)
    Y[:, t] = H * X[:, t] .+ rand(obs_noise)
end

llh = kalman_llh(Y, A, H, m0, P, Q, R)
@info(llh)

kalman_abq(Q) = kalman_llh(Y, A, H, m0, P, Q, R)

Q_2 = Q .+ diagm(rand(2))
kalman_abq(Q)
kalman_abq(Q_2)

kalgrad = ForwardDiff.gradient(kalman_abq, Q)
@info(kalgrad)
