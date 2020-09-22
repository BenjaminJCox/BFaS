using LinearAlgebra

#=
Discretise linear time invariant ODE with gaussian noise

Input:
F - NxN feedback matrix
L - NxL noise effect matrix
Qc - LxL Diagonal Spectral Density
dt - Time step

Output:
A - Transition Matrix
Q - Discrete Process Covariance
=#

function lti_ode_disc(
    F::Matrix{Float64},
    L::Matrix{Float64} = I(size(F, 1)),
    Q::Matrix{Float64} = zeros(size(F, 1), size(F, 1)),
    dt::Float64 = 1.0,
)
    A = exp(F .* dt)

    n = size(F, 1)
    Phi = [F L * Q * L'; zeros(n, n) -F']
    AB = exp(Phi * dt) * [zeros(n, n); I(n)]
    Q = AB[1:n, :] / AB[(n+1):(2*n), :]
    return (A, Q)
end
