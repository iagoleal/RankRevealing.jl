using RankRevealing
using Test
using Random
using LinearAlgebra

Z(m, n) = zeros(Int64, m, n)

function recomp(F :: GeneralizedRankRevealing)
  X, Y, H, r1, r2, r3 = F.X, F.Y, F.H, F.r1, F.r2, F.r3
  A = X * [I(r1) Z(r1, r2) Z(r1, r3) ; Z(r3, r1) Z(r3, r2) I(r3)] * H
  B = Y * [Z(r2, r1) I(r2) Z(r2, r3) ; Z(r3, r1) Z(r3, r2) I(r3)] * H
  return A, B
end

# Check that the decomposition can be multiplied into the right matrix
function correctness(A::W, B::W) where {K <: AbstractFloat, S <: Complex{K}, T <: Union{S, K}, W<:AbstractArray{T}}
  eps = 1e-7
  F = grr(A, B)
  Ar, Br = recomp(F)
  rA = rank(A)
  rB = rank(B)
  return isapprox(Ar, A, atol = eps) && isapprox(Br, B, atol = eps) && rA == F.r1 + F.r3 && rB == F.r2 + F.r3
end

correctness(A, B) = recomp(grr(A, B)) == (A, B)

@testset "Rank-revealing decompositions" begin
  for n in 0:7, m in 0:7
    for r0 in 0:min(n, m)
      A = exactrand(m, r0) * exactrand(r0, n)
      X, H = simple_rr(A)
      @test isapprox(A, X*H)
      X, H = right_rr(A)
      F = pluq(A)
      r = F.rank
      @test A == X * [Z(r, n-r) I(r)] * H
      X, H = left_rr(A)
      @test A == X * [I(r) ; Z(m-r, r)] * H
    end
  end
end

@testset "Generalized Rank-revealing" begin
  @testset "Exact Rational" begin
    for ma in 0:6, mb in 0:6, n in 0:6
      A = exactrand(ma, n)
      B = exactrand(mb, n)
      @test correctness(A, B)
      @testset "Complex with exact rational components" begin
        A = exactrand(ma, n) + im*exactrand(ma, n)
        B = exactrand(mb, n) + im*exactrand(mb, n)
        @test correctness(A, B)
      end
    end
  end

  @testset "Floating point" begin
    for ma in 0:6, mb in 0:6, n in 0:6
      @test correctness(randn(ma, n), randn(mb, n))
      @test correctness(rand(ma, n), rand(mb, n))
      @testset "Complex with float components" begin
        A = rand(Complex{Float64}, ma, n)
        B = rand(Complex{Float64}, mb, n)
        @test correctness(A, B)
        # Unit complex matrices
        @test correctness(cisrand(ma, n), cisrand(mb, n))
      end
    end
  end
end

@testset "Destructuring" begin
  ma = 5 ; mb = 7; n = 3
  A = exactrand(ma, n)
  B = exactrand(mb, n)
  F = grr(A, B)
  X, Y, H = F
  r1, r2, r3 = F.r1, F.r2, F.r3
  @test size(X) == (ma, r1 + r3)
  @test size(Y) == (mb, r2 + r3)
  @test size(H) == (r1+r2+r3, n)
  Z(m, n) = zeros(Int64, m, n)
  @test A == X * [I(r1) Z(r1, r2) Z(r1, r3) ; Z(r3, r1) Z(r3, r2) I(r3)] * H
  @test B == Y * [Z(r2, r1) I(r2) Z(r2, r3) ; Z(r3, r1) Z(r3, r2) I(r3)] * H
end

# TODO: reverse-engineer: sorteia X, Y, H e verifica se funciona
