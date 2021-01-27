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

A = [1.0 1.0; 0.0 0.5]
H = [1.0 0.0]

P = [0.1 0.0; 0.0 0.1]
Q = [0.1 0.0; 0.0 0.1]
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

kalman_abA(A) = kalman_llh(Y, A, H, m0, P, Q, R)

A_2 = A .+ rand(2,2)
kalman_abA(A)
kalman_abA(A_2)

lr = 1e-3
for i = 1:1000
    kalgrad = ForwardDiff.gradient(kalman_abA, A_2)
    A_2 .+= lr * kalgrad
end

kalman_abq(A_2)
@info(A_2)

A_res = perform_kalman(Y, A, H, m0, P, Q, R)[1]
A2_res = perform_kalman(Y, A_2, H, m0, P, Q, R)[1]

p1 = plot(1:T, X[1,:])
plot!(1:T, A_res[1,:])
p2 = plot(1:T, X[1,:])
plot!(1:T, A2_res[1,:])

plot(p1, p2, layout = (2,1), size = (1000, 1000), legend = false)
