using GLU
using Test
using Random
using LinearAlgebra

O = 0*I

@testset "GLU" for i in 1:100
  A = random(100, i)
  B = random(100, i)
  X, Y, H, r1, r2, r3 = grr(A)
  @test A == X * [I O O ; O O I] * H
  @test B == Y * [O I O ; O O I] * H
end
