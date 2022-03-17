using RankRevealing
using Test
using Random

filepath(x) = joinpath(dirname(@__FILE__), x)

# Fix random number seed, for reproducibility
# Random.seed!(12342352154)

@testset "RankRevealing.jl" begin
  @info "Testing Rank-sensitive LU"
  include(filepath("lu.jl"))

  @info "Testing Generalized Rank Revealing"
  include(filepath("grr.jl"))
end

