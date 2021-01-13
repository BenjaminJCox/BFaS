function rs_bsp_stopping_smoother(
    X::Array{Float64,3},
    W::Array{Float64,2},
    s::Int64,
    psi::Function,
    Q::Array{Float64};
    max_iterations::Int64 = 10,
)
    n = size(W, 1)
    T = size(W, 2)
    Qq = Q
    if size(Q, 3) == 1
        Qp = repeat(Q, 1, 1, T)
    end

    rho = zeros(T)
    for i = 1:T
        rho[i] = pdf(MvNormal(X[1, :, 1], Qp[:, :, i]), X[1, :, 1])
    end

    trajectories = zeros(s, size(X, 2), T)

    trajectory_index = zeros(Int64, s, T)

    trajectory_index[:, T] = wsample(1:n, W[:, T], s)

    trajectories[:, :, T] = X[trajectory_index[:, T], :, T]

    for t = (T-1):-1:1
        c_iter = 0
        L = collect(1:s)
        delta = []
        c_l = length(L)
        while ((c_iter < max_iterations) && (c_l > 0))
            println(c_iter)
            delta = []
            indices = wsample(1:n, W[:, t], c_l)
            unifs = rho[t] .* rand(c_l)
            for k = 1:c_l
                ik = indices[k]
                lk = L[k]
                if (unifs[k] <= pdf(MvNormal(psi(trajectories[lk, :, t]), Qp[:, :, t+1]), X[ik, :, t]))
                    println("bingpot")
                    trajectory_index[lk, t] = ik
                    delta = vcat(delta, lk)
                end
            end
            filter!(x -> !in(x, delta), L)
            c_l = length(L)
            c_iter += 1
            println(L)
            println(rho[t])
        end
        trajectories[:, :, t] = X[trajectory_index[:, t], :, t]
    end
    return trajectories
end

function rs_bsp_smoother(X::Array{Float64,3}, W::Array{Float64,2}, s::Int64, psi::Function, Q::Array{Float64})
    n = size(W, 1)
    T = size(W, 2)
    Qq = Q
    if size(Q, 3) == 1
        Qp = repeat(Q, 1, 1, T)
    end

    rho = zeros(T)
    for i = 1:T
        rho[i] = pdf(MvNormal(X[1, :, 1], Qp[:, :, i]), X[1, :, 1])
    end

    trajectories = zeros(s, size(X, 2), T)

    trajectory_index = zeros(Int64, s, T)

    trajectory_index[:, T] = wsample(1:n, W[:, T], s)

    trajectories[:, :, T] = X[trajectory_index[:, T], :, T]

    for t = (T-1):-1:1
        to_assign = s
        assign_index = 1
        while to_assign > 0
            indices = wsample(1:n, W[:, t], to_assign)
            unifs = rho[t] .* rand(to_assign)
            for k = 1:to_assign
                index = indices[k]
                if (unifs[k] <= pdf(MvNormal((trajectories[assign_index, :, t+1]), Qp[:, :, t+1]), psi(X[index, :, t+1])))
                    trajectory_index[assign_index, t] = index
                    assign_index += 1
                    to_assign -= 1
                end
            end
        end
        println(trajectory_index[:, t])
        trajectories[:, :, t] = X[trajectory_index[:, t], :, t]
    end
    return trajectories
end

function pgas_smoother(
    x::Matrix{Float64},
    Q::Array{Float64},
    y::Matrix{Float64},
    R::Array{Float64},
    psi::Function,
    H::Function,
    s::Int64,
    prior_mean::Vector{Float64},
    prior_cov::Matrix{Float64},
)
    n = size(x, 1)
    T = size(x, 2)

    Qq = Q
    if size(Q, 3) == 1
        Qq = repeat(Q, 1, 1, T)
    end

    Rr = R
    if size(R, 3) == 1
        Rr = repeat(R, 1, 1, T)
    end

    Qq_isig = zeros(size(Qq))
    Qq_sqds = zeros(T)
    Rr_isig = zeros(size(Rr))
    Rr_sqds = zeros(T)

    for t = 1:T
        Qq_isig[:, :, t] = inv(Qq[:, :, t])
        Qq_sqds[t] = 1.0 / sqrt(det(Qq[:, :, t]))
        Rr_isig[:, :, t] = inv(Rr[:, :, t])
        Rr_sqds[t] = 1.0 / sqrt(det(Rr[:, :, t]))
    end

    trajectories = zeros(size(x, 1), s, T)

    s_niv = s - 1

    @simd for i = 1:s_niv
        trajectories[:, i, 1] = rand(MvNormal(prior_mean, prior_cov))
    end
    trajectories[:, s, 1] = x[:, 1]

    weights = zeros(s)

    @simd for i = 1:s
        _x = trajectories[:, i, 1]
        _xo = H(_x)
        _R1 = Rr_isig[:, :, 1]
        _R2 = Rr_sqds[1]
        weights[i] = r_mvn_pdf(y[:, 1], _xo, _R1, _R2)
    end

    weights ./= sum(weights)

    for t = 2:T
        #resampling and ancestor sampling
        _Q1 = Qq_isig[:, :, t]
        _Q2 = Qq_sqds[t]
        _R1 = Rr_isig[:, :, t]
        _R2 = Rr_sqds[t]
        _y = y[:, t]

        x_tilde_indices = wsample(1:s, weights, s; replace = true)
        x_tilde = trajectories[:, x_tilde_indices, 1:(t-1)]

        j_wts = zeros(s)

        @simd for i = 1:s
            j_wts[i] = weights[i] * r_mvn_pdf(x[:, t], psi(trajectories[:, i, t-1]), _Q1, _Q2)
        end

        j_wts ./= sum(j_wts)

        x_tn_i = wsample(1:s, j_wts, 1)

        x_tilde[:, s, :] = trajectories[:, x_tn_i, 1:(t-1)]
        if t == 4
            println(x_tilde)
        end
        #propagation
        x_t = zeros(size(x, 1), s)
        @simd for i = 1:s_niv
            _x = trajectories[:, i, t-1]
            _xp = psi(_x)
            distn = MvNormal(_xp, Qq[:, :, t])
            x_t[:, i] = rand(distn)
        end
        x_t[:, s] = x[:, t]

        trajectories[:, :, 1:(t-1)] = x_tilde
        trajectories[:, :, t] = x_t

        @simd for i = 1:s
            _x = trajectories[:, i, t]
            _xo = H(_x)
            weights[i] = r_mvn_pdf(y[:, t], _xo, _R1, _R2)
        end
    end

    weights ./= sum(weights)

    return (trajectories, weights)
end
