using Plots
using Distributions
using LinearAlgebra
plotlyjs()
p = plot(rand(5), rand(5))
# display(p)

gr()
q = plot(rand(5), rand(5))
# display(q)

# -JF:\julia\projects\BFaS\sys_plots.so

# distn = MvNormal(rand(2), [1. 0.; 0. 1.])
# rn = rand(distn)
# rn2 = rand(distn, 30)
# pd = pdf(distn, rand(2))
# create_sysimage([:Plots, :PlotlyJS], sysimage_path="sys_plots.so", precompile_execution_file="sysimgmaker.jl")
