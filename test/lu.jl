using RankRevealing
using Test
using Random
using LinearAlgebra

# Takes a PLUQ object into the original matrix
recomp(x :: PLUQ) = ([x.L ; x.M] * [x.U x.V])[x.p, x.q]

# Check that the decomposition
# can be multiplied into the right matrix
function correctness(A::AbstractArray{T}) where {K <: AbstractFloat, S <: Complex{K}, T <: Union{S, K}}
  F = pluq(A)
  return isapprox(recomp(F), A, atol = 1e-7) && rank(F) == rank(A, atol = 1e-7)
end
correctness(A) = recomp(pluq(A)) == A

@testset "Base cases for PLUQ" begin
  # Degenerate cases (dimension zero)
  @test correctness(zeros(0,0))
  @test correctness(zeros(0,1))
  @test correctness(zeros(1,0))
  # Non-degenerate zero matrix
  @test correctness(zeros(1,1))   # Scalar zero
  @test correctness(zeros(1,2))   # Column zero vector
  @test correctness(zeros(2, 1))  # Row zero vector
  # Non-zero
  @test correctness([3;;])        # Scalar
  @test correctness([1;2;;])      # Column vector
  @test correctness([0;2;;])      # column with pivoting
  @test correctness([1 2;])       # Row vector
  @test correctness([0 2;])       # row with pivoting
end

@testset "Recursive cases" begin
  # Degenerate rows and columns
  @test correctness(zeros(0,7))
  @test correctness(zeros(7,0))
  # Now the real work
  A = Rational{Int64}[1 0; 0 1]
  @test correctness(A)
  A = Rational{Int64}[1 2; 3 4]
  @test correctness(A)
end

@testset "Random matrices: ($m by $n)" for m in 0:6, n in 0:6
  @testset "Floating point components" begin
    @test correctness(rand(m,n))   # Uniform components
    @test correctness(randn(m,n))  # Gaussian matrix
    @testset "Does it reveal the rank?" begin
      for r in 0:min(m, n)
        A = randn(m, r) * randn(r, n)  # Generate a random rank r matrix
        F = pluq(A)
        @test F.rank == r
      end
    end
  end

  @testset "Exact Rational components" begin
    # We want to use BigInt and Rational to ensure exact arithmetic
    A = exactrand(m, n)
    @test correctness(A)
  end

  @testset "Complex components" begin
    @test correctness(rand(Complex{Float64}, m, n))
    # Complex with exact rational components
    A = exactrand(m, n)
    B = exactrand(m, n)
    @test correctness(A + im*B)
    # Unit complex matrices
    @test correctness(cisrand(m, n))
  end
end

@testset "Destructuring" begin
  A = exactrand(5, 7)
  PLUQ = pluq(A)
  p, L, U, V, M, q = PLUQ
  # Using permutation vectors
  @test ([L ; M] * [U V])[p, q] == A
  # Using permutation matrices
  @test (PLUQ.P * [L ; M] * [U V] * PLUQ.Q) == A
end

@testset "Uniqueness of LU" begin
  A = exactrand(5, 5)
  L = tril(A, -1) + I   # Unit lower trianguler
  U = triu(A)           # Upper triangular
  C = L * U
  F = pluq(C)
  @test C == F.L * F.U
end
