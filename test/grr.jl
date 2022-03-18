using RankRevealing
using Test
using Random
using LinearAlgebra

function recomp(F :: GeneralizedRankRevealing)
  X, Y, H, r1, r2, r3 = F.X, F.Y, F.H, F.r1, F.r2, F.r3
  A = X * [I(r1) zeros(Int64, r1, r1+r3) ; zeros(Int64, r3, r1+r2) I(r3)] * H
  B = Y * [zeros(Int64, r2, r1) I(r2) zeros(Int64, r2, r3) ; zeros(Int64, r3, r1+r2) I(r3)] * H
  return A, B
end

# Check that the decomposition
# can be multiplied into the right matrix
function correctness(A::W, B::W) where {K <: AbstractFloat, S <: Complex{K}, T <: Union{S, K}, W<:AbstractArray{T}}
  eps = 1e-10
  Ar, Br = recomp(grr(A,B))
  return isapprox(Ar, A, atol = eps) && isapprox(Br, B, atol = 1e-10)
end
correctness(A, B) = recomp(grr(A, B)) == (A, B)

@testset "Rank-revealing decompositions" begin
  for n in 0:7, m in 0:7
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
@testset "Generalized Rank-revealing ($ma by $n), ($mb by $n)" for ma in 0:6, mb in 0:6, n in 0:min(ma,mb)
  @testset "Exact Rational" begin
    A = exactrand(ma, n)
    B = exactrand(mb, n)
    @test correctness(A, B)
  end

  @testset "Floating point" begin
    @test correctness(randn(ma, n), randn(mb, n))
    @test correctness(rand(ma, n), rand(mb, n))
  end

  @testset "Complex components" begin
    A = rand(Complex{Float64}, ma, n)
    B = rand(Complex{Float64}, mb, n)
    @test correctness(A, B)
    # Complex with exact rational components
    A = exactrand(ma, n) + im*exactrand(ma, n)
    B = exactrand(mb, n) + im*exactrand(mb, n)
    @test correctness(A, B)
    # Unit complex matrices
    @test correctness(cisrand(ma, n), cisrand(mb, n))
  end
end

@testset "Destructuring" begin
  ma = 5 ; mb = 7; n = 3
  A = exactrand(ma, n)
  B = exactrand(mb, n)
  X, Y, H, r1, r2, r3 = grr(A, B)
  @test size(X) == (ma, r1 + r3)
  @test size(Y) == (mb, r2 + r3)
  @test size(H) == (r1+r2+r3, n)
  @test A == X * [I(r1) zeros(Int64, r1, r1+r3) ; zeros(Int64, r3, r1+r2) I(r3)] * H
  @test B == Y * [zeros(Int64, r2, r1) I(r2) zeros(Int64, r2, r3) ; zeros(Int64, r3, r1+r2) I(r3)] * H
end


# TODO: reverse-engineer: sorteia X, Y, H e verifica se funciona
