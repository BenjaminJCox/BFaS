module containers
using DrWatson
    include(srcdir("filter_container.jl"))
    include(srcdir("container_filters.jl"))
    include(srcdir("container_paramest.jl"))
end
