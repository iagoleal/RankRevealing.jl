using RankRevealing
using Test
using Random

filepath(x) = joinpath(dirname(@__FILE__), x)

# Useful sampling
exactrand(n...) = big.(rand(Int64, n...) .// rand(Int64, n...))
cisrand(n...) = (R = randn(ComplexF64, n...); R ./ norm.(R))

rank_rand(m, n, r=min(m,n)) = randn(m, r) * randn(r, n)
rank_rand(T::Type, m, n, r=min(m,n)) = randn(T, m, r) * randn(T, r, n)


# Fix random number seed, for reproducibility
# Random.seed!(12342352154)

@testset "RankRevealing.jl" begin
  @info "Testing Rank-sensitive LU"
  include(filepath("lu.jl"))

  @info "Testing Generalized Rank Revealing"
  include(filepath("grr.jl"))
end

