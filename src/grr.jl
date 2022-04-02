###########################################
# Generalized Rank Revealing decomposition
###########################################

"""
    GeneralizedRankRevealing <: Factorization

Matrix factorization type for the Generalized Rank Reveling decomposition
of two matrices `A` and `B` with the same number of columns.
This is obtained via the function `grr(A,B)`.

By calling `d_+` the dimension of the sum of `A` and `B` row spaces,
this factorization has the decomposes the matrices as

                    | I 0 0 |
    | A | = | X 0 | | 0 0 I |
    | B |   | 0 Y | | 0 I 0 | H
                    | 0 0 I |

| Component | Description                              |
|:----------|:-----------------------------------------|
| `F.X`     | `m_A x r_A` full column rank matrix      |
| `F.Y`     | `m_B x r_B` full column rank matrix      |
| `F.H`     | `d_+ x n` full row rank matrix           |

Other useful fields are:
* `F.r1`: Rank of `A` minus minus the intersection
* `F.r2`: Rank of `B` minus minus the intersection
* `F.r3`: Rank of the intersection
"""
struct GeneralizedRankRevealing{T, S <: AbstractMatrix{T}} <: Factorization{T}
  X        :: S
  Y        :: S
  H        :: S
end

function GeneralizedRankRevealing(X :: S, Y :: S, H :: S) where {T, S <: AbstractMatrix{T}}
  GeneralizedRankRevealing{T, S}(X, Y, H)
end

function Base.getproperty(D::GeneralizedRankRevealing{T}, key::Symbol) where T
  # cols(X) == rank of A
  # cols(Y) == rank of B
  # rows(H) == dim of intersection
  if     key == :r1
    return cols(D.X) - D.r3
  elseif key == :r2
    return cols(D.Y) - D.r3
  elseif key == :r3
    return cols(D.X) + cols(D.Y) - rows(D.H)    # Dimension of R(A) ∩ R(B)
  else
    getfield(D, key)
  end
end

# Destructure as p, L, U, V, M, q
Base.iterate(S::GeneralizedRankRevealing)             = (S.X, Val(:Y))
Base.iterate(S::GeneralizedRankRevealing, ::Val{:Y})  = (S.Y, Val(:H))
Base.iterate(S::GeneralizedRankRevealing, ::Val{:H})  = (S.H, Val(:done))
Base.iterate(S::GeneralizedRankRevealing, ::Val{:done}) = nothing

# For the GLU algorithm, we want to swaps U and V on PLUQ.
# We achive this by noticing that if
# [U V] * Q == [V U] * (J * Q) where J = [0 I; I 0].
# In Julia
function pluq2(A)
  F = pluq(A)
  n = cols(A)
  r = F.rank
  j = vcat(((n-r+1) : n),  (1:n-r)) # Permutation vector for matrix [0 I; I 0]
  newq = Perm(j) * Perm(F.q)
  return PLUQ(F.p, newq, F.rank, F.factors)
end

"""
    simple_rr(A)

Decompose `A == X*H` where
X is full column rank and H is full row rank.
"""
function simple_rr(A)
  p, L, U, V, M, q = pluq2(A)
  return ([L ; M])[p, :], ([V U])[:, q]
end

# Right rank revealing
"""
    right_rr(A)

Decompose `A == X*[0 I]*H` where
X is full column rank and H is invertible.
"""
function right_rr(A::AbstractMatrix{T}) where T
  p, L, U, V, M, q = pluq2(A)
  d = cols(V)
  Z = zeros(Int64, cols(V), cols(U))
  X = ([L ; M])[p, :]
  H = ([I(d) Z; V U])[:, q]
  return X, Array(H)
end

# Left rank revealing
"""
    left_rr(A)

Decompose `A == X*[I ; 0]*H` where
X is invertible and H is full row rank.
"""
function left_rr(A::AbstractMatrix{T}) where T
  p, L, U, V, M, q = pluq2(A)
  r = rows(L)
  m = rows(A)
  Z = zeros(Int64, r, m - r)
  d = m - r
  X = ([L Z; M I(d)])[p, :]
  H = ([V U])[:, q]
  return Array(X), H
end


"""
    grr(A, B) -> GeneralizedRankRevealing

Compute the Generalized LU decomposition of matrices `A` and `B`.
There are matrices `X`, `Y` and `H` such that

                    | I 0 0 |
    | A | = | X 0 | | 0 0 I |
    | B |   | 0 Y | | 0 I 0 | H
                    | 0 0 I |
"""
function grr(A, B)
  @assert(cols(A) == cols(B))
  m_A      = rows(A)
  M, H1    = simple_rr([A ; B])
  A1, B1   = vsplit(M, m_A)
  X2, A2   = simple_rr(A1)
  Y2, H2   = right_rr(B1)
  A3       = A2 / H2
  A31, A32 = hsplit(A3, cols(A3) - cols(Y2))
  X4, H4   = left_rr(A31)
  A4       = X4 \ A32
  A41, A42 = vsplit(A4, rows(H4))
  X5, H5   = right_rr(A42)
  # Outputs
  dx = cols(X4) - rows(X5)
  X = X2 * X4 * dsum(I(dx), X5)
  Y = Y2 / H5
  Z3 = zeros(Int64, rows(H5), cols(H4))
  H = [H4 A41 ; Z3 H5] * H2 * H1
  return GeneralizedRankRevealing(X, Y, H)
end
