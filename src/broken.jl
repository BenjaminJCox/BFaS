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
                println(unifs .- )
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
