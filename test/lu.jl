using RankRevealing
using Test
using Random

recomp = RankRevealing.validation

# Check that the decomposition
# can be multiplied into the right matrix
correctness(A::AbstractArray{Float64}) = isapprox(recomp(pluq(A)...), A, atol = 1e-10)
correctness(A) = recomp(pluq(A)...) == A

@testset "Base cases for PLUQ" begin
  # Degenerate cases (dimension zero)
  @test correctness(zeros(0,0))
  @test correctness(zeros(0,1))
  @test correctness(zeros(1,0))
  # Non-degenerate zero matrix
  @test correctness(zeros(1,1))   # Scalar
  @test correctness(zeros(1,2))   # Column zero vector
  @test correctness(zeros(2, 1))  # Row zero vector
  # Non-zero
  @test correctness([3;;])        # Scalar
  @test correctness([1;2;;])      # Column vector
  @test correctness([0;2;;])      # column with pivoting
  @test correctness([1 2;])       # Row column vector
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

@testset "Float matrices" for m in 0:9, n in 0:9
  @test correctness(rand(m,n))   # Uniform components
  @test correctness(randn(m,n))  # Gaussian matrix
end

@testset "Rational matrices" for m in 0:9, n in 0:9
  # We want to use BigInt and Rational to ensure exact arithmetic
  A = big.(rand(Int64, m, n) .// rand(Int64, m, n))
  @test correctness(A)
end
