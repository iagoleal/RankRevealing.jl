using GLU
using Test
using Random

@testset "LU decomposition" begin
  A = random(100, 50)
  P, L, M, U, V, Q = pluq(A)
  @test A == P * vcat(L, M) * hcat(U, V) * Q
end
