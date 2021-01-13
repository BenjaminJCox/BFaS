using Turing, Plots, Random

gr()


function sv_x(x)
    return (0.999 * x)
end

function ob_x(x)
    return exp(x/2)
end

process_noise = Normal(0., 1.)
obs_noise = Normal(0., 1.)

seq_len = 200
truth_sequence = zeros(seq_len)
obs_sequence = zeros(seq_len)

for i = 2:seq_len
    truth_sequence[i] = sv_x(truth_sequence[i-1]) + rand(process_noise)
    obs_sequence[i] = ob_x(truth_sequence[i]) * rand(obs_noise)
end

ts = 1:seq_len

# p1 = plot(ts, truth_sequence, label = "Truth")
# p2 = plot(ts, obs_sequence, label = "Observed")
# plot(p1, p2, layout = (2,1), legend = false)

@model BayesVolatile(observed) = begin
    N = length(observed)
    state = tzeros(Real, N)

    observed[1] ~ Normal(0., 1.)
    state[1] ~ Normal(0., 1.)

    for i = 2:N
        state[i] ~ Normal(sv_x(state[i-1]), 1.)
        observed[i] ~ Normal(0., ob_x(state[i]))
    end
end

iterations = 1_000
chain = sample(BayesVolatile(obs_sequence), PG(10), iterations)
cndata = chain.value.data
predstates = cndata[:,3:(end-1),1]
m_states = mean(predstates, dims = 1)[:]

p1 = plot(ts, truth_sequence, label = "Truth")
p2 = plot!(ts, m_states, label = "Predicted State")
p3 = plot(ts, obs_sequence, label = "Observed")

plot(p2, p3, layout = (2,1), legend = false, size = (750, 500))
