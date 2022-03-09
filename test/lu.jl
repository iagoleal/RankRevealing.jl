using RankRevealing
using Test
using Random

recomp = RankRevealing.validation

# TODO matriz vazia

@testset "Base cases for PLUQ" begin
  # Zero matrix
  A = [0;;]
  @test recomp(pluq(A)...) == A
  # Column vector
  A = [0;0;;]
  @test recomp(pluq(A)...) == A
  # Row column vector
  A = [0 0;]
  @test recomp(pluq(A)...) == A
  # Scalar
  A = [3;;]
  @test recomp(pluq(A)...) == A
  # Column vector
  A = [1;2;;]
  @test recomp(pluq(A)...) == A
  A = [0;2;;]
  @test recomp(pluq(A)...) == A
  # Row column vector
  A = [1 2;]
  @test recomp(pluq(A)...) == A
  A = [0 2;]
  @test recomp(pluq(A)...) == A
  # A square matrix
  # A = rand(100, 50)
  # P, L, M, U, V, Q = pluq(A)
  # @test A == P * vcat(L, M) * hcat(U, V) * Q
end
