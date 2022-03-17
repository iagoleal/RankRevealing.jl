using RankRevealing
using Test
using Random
using LinearAlgebra

exactrand(n...) = big.(rand(Int64, n...) .// rand(Int64, n...))

rank_rand(m, n, r=min(m,n)) = exactrand(m, r) * exactrand(r, n)

@testset "Rank-revealing decompositions" begin
  for n in 0:3, m in 0:3
    for r in 0:min(n, m)
      A = exactrand(m, r) * exactrand(r, n)
      X, H = simple_rr(A)
      @test isapprox(A, X*H)
      X, H = right_rr(A)
      F = pluq(A)
      r = F.rank
      Z = zeros(Rational{BigInt}, r, n-r)
      @test A == X * [Z I(r)] * H
      X, H = left_rr(A)
      Z = zeros(Rational{BigInt}, m-r, r)
      @test A == X * [I(r) ; Z] * H
    end
  end
end


# FIXME: n > mA
#        n > mB
@testset "Generalized Rank-revealing" begin
  for ma = 1:3, mb = 1:3, n = 1:min(ma,mb)
    A = exactrand(ma, n)
    B = exactrand(mb, n)
    X, Y, H, r1, r2, r3 = grr(A, B)
    @test A ≈ X * [I(r1) zeros(Int64, r1, r1+r3) ; zeros(Int64, r3, r1+r2) I(r3)] * H atol=1e-6
  end
end


# TODO: reverse-engineer: sorteia X, Y, H e verifica se funciona
