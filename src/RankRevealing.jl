module RankRevealing
export pluq, PLUQ, grr, GeneralizedRankRevealing, simple_rr, right_rr, left_rr

using LinearAlgebra

rows(A) = size(A)[1]
cols(A) = size(A)[2]


########################
# Permutations
########################

function permcompose(p, q)
  n = length(p)
  if n != length(q)
    error("Tried to compose permutations over different dimensions.")
  end
  return [p[q[k]] for k in 1:n]
end

function permtranspose(p)
  pt = similar(p)
  for (i, v) in enumerate(p)
    pt[v] = i
  end
  return pt
end

# Transposition permutation for indices i and j
function transposition(n, i, j)
  perm = collect(1:n)
  perm[i], perm[j] = perm[j], perm[i]
  return perm
end

function perm2matrix(T, p)
  A = zeros(T, length(p), length(p))
  for (i, v) in enumerate(p)
    A[i, v] = 1
  end
  return A
end

########################
# PLUQ type
########################

# Strongly based on LinearAlgebra.LU
"""
    PLUQ <: Factorizarion

Matrix factorization type for the full `LU` factorization
of any rectangular matrix `A` over an arbitrary numeric field.
This is the return type of [`pluq`](@ref).

The factorization satisfies

    P * [L ; M] * [U V] * Q == A


| Component | Description                              |
|:----------|:-----------------------------------------|
| `F.L`     | Full-rank unit lower triangular part     |
| `F.U`     | Full-rank upper triangular part          |
| `F.M`     | Remainder of the row dimension           |
| `F.V`     | Remainder of the column dimension        |
| `F.P`     | row-wise permutation `Matrix`            |
| `F.Q`     | column-wise permutation `Matrix`         |
| `F.p`     | row-wise permutation `Vector`            |
| `F.q`     | column-wise permutation `Matrix`         |

Iteration and destructuring produce the components in the order `p`, `L`, `U`, `V`, `M`, q`.

Both `L` and `U` are square full-rank matrices
with the same rank as `A`.
"""
struct PLUQ{T, S <: AbstractMatrix{T}} <: Factorization{T}
  p :: Vector{Int64}
  q :: Vector{Int64}
  rank     :: Int64
  factors  :: S
  function PLUQ{T,S}(p, q, rank, A) where {T,S<:AbstractMatrix{T}}
      new{T,S}(p, q, rank, A)
  end
end

function PLUQ(p, q, r, A::AbstractMatrix{T}) where {T}
  PLUQ{T,typeof(A)}(p, q, r, A)
end

function Base.getproperty(D::PLUQ{T}, key::Symbol) where T
  if key == :L
    r = getfield(D, :rank)
    L = tril!(getfield(D, :factors)[1:r, 1:r])
    for i in 1:r
      L[i, i] = one(T)
    end
    return L
  elseif key == :U
    r = getfield(D, :rank)
    return triu!(getfield(D, :factors)[1:r, 1:r])
  elseif key == :V
    r = getfield(D, :rank)
    return getfield(D, :factors)[1:r, (r+1):end]
  elseif key == :M
    r = getfield(D, :rank)
    return getfield(D, :factors)[(r+1):end, 1:r]
  elseif key == :P
    return perm2matrix(T, getfield(D, :p))
  elseif key == :Q
    return perm2matrix(T, getfield(D, :q))
  else
    getfield(D, key)
  end
end

Base.propertynames(::PLUQ) = (:P, :L, :U, :V, :M, :Q, :p, :q)

# Destructure as p, L, U, V, M, q
Base.iterate(S::PLUQ)            = (S.p, Val(:L))
Base.iterate(S::PLUQ, ::Val{:L}) = (S.L, Val(:U))
Base.iterate(S::PLUQ, ::Val{:U}) = (S.U, Val(:V))
Base.iterate(S::PLUQ, ::Val{:V}) = (S.V, Val(:M))
Base.iterate(S::PLUQ, ::Val{:M}) = (S.M, Val(:q))
Base.iterate(S::PLUQ, ::Val{:q}) = (S.q, Val(:done))
Base.iterate(S::PLUQ, ::Val{:done}) = nothing

# C <- C - AB
function mm!(C, A, B)
  C .= C - A*B
  return C
  # LinearAlgebra.mul!(C, A, B, -1, 1)
end

# Permute the columns of A inplace
# TODO: Rewrite this without copying
function permC!(A, p::AbstractVector)
  B = A[:, permtranspose(p)]
  A .= B
end

# Permute the rows of A inplace
function permR!(A, p::AbstractVector)
  A .= A[permtranspose(p), :]
end

# In-place matrix splitting into 4 blocks
function msplit(A, mh = fld(rows(A),2), nh = fld(cols(A), 2))
  m, n = size(A)
  return view(A, 1:mh,     1:nh),
         view(A, 1:mh,     (nh+1):n),
         view(A, (mh+1):m, 1:nh),
         view(A, (mh+1):m, (nh+1):n)
end

#  A -> [A1
#       ;A2]
function vsplit(X, k)
  m = rows(X)
  return view(X, 1:k, :), view(X, (k+1):m, :)
end

# A -> [A1 A2]
function hsplit(X, k)
  n = cols(X)
  return view(X, :, 1:k), view(X, :, (k+1):n)
end

function decomp(A, r)
  m, n = size(A)
  L = UnitLowerTriangular(view(A, 1:r, 1:r))
  U = UpperTriangular(view(A, 1:r, 1:r))
  V = view(A, 1:r, (r+1):n)
  M = view(A, (r+1):m, 1:r)
  return (L = L, U = U, V = V, M = M)
end


# Floating point numbers should not be compared by equality
function checkzero(x :: W) where {K <: AbstractFloat, S <: Complex{K}, T <: Union{S, K}, W <: Union{T, AbstractArray{T}}}
  return isapprox(x, zero(x), atol = 1e-12)
end

# Defaults to good ol' equality
checkzero(x) = iszero(x)


# In-place PLUQ factorization.
# This is supposed to be used only as an internal method.
# To accelerate the recursion, this Returns the "crude" output,
# without creating the intermediary PLUQ structure.
function pluq!(A)
  m, n = size(A)
  # Degenerate zero
  if m == 0 && n == 0
    return [], [], 0, A
  end
  if m == 1
    P = [1]
    if checkzero(A)
      Q = collect(1:n)    # n x n identity matrix
      r = 0
    else
      i = findfirst(!(checkzero), A)[2]        # column index of the first non-zero element of A
      Q = transposition(n, 1, i)
      r = 1
      A[1, i], A[1, 1] = A[1, 1], A[1, i]   # Pivoting
    end
    return P, Q, r, A
  end
  if n == 1
    Q = [1]
    if checkzero(A)
      P = collect(1:m)
      r = 0
    else
      i = findfirst(!(checkzero), A)[1] # row index of the first non-zero element of A
      # NOTE: Error in article
      P = transposition(m, 1, i)
      r = 1
      A[i, 1], A[1, 1] = A[1, 1], A[i, 1]
      local pivot = A[1,1]
      for j in (i + 1):m
        A[j, 1] = A[j, 1] / pivot
      end
    end
    return P, Q, r, A
  end
  # Now the recursion step
  A1, A2, A3, A4 = msplit(A)
  P1, Q1, r1, A1 = pluq!(A1) # Decompose upper-left quadrant
  L1, U1, V1, M1 = decomp(A1, r1)
  B1, B2 = vsplit(permR!(A2, P1), r1)
  C1, C2 = hsplit(permC!(A3, Q1), r1)
  D = ldiv!(L1, B1)
  E = rdiv!(C1, U1)
  F = mm!(B2, M1, D)
  G = mm!(C2, E, V1)
  H = mm!(A4, E, D)
  P2, Q2, r2, F = pluq!(F) # Decompose
  L2, U2, V2, M2 = decomp(F, r2)
  P3, Q3, r3, G = pluq!(G) # Decompose
  L3, U3, V3, M3 = decomp(G, r3)
  permR!(H, P3)
  permC!(H, Q2)
  # Split based on the previous recursions
  H1, H2, H3, H4 = msplit(H, r3, r2)
  E1,  E2  = vsplit(permR!(E, P3),  r3)
  M11, M12 = vsplit(permR!(M1, P2), r2)
  # Erro do artigo: permC e não permR
  D1,  D2  = hsplit(permC!(D, Q2),  r2)
  V11, V12 = hsplit(permC!(V1, Q3), r3)
  I = rdiv!(H1, U2)
  J = ldiv!(L3, I)
  K = rdiv!(H3, U2)
  N = ldiv!(L3, H2)
  O = mm!(N, J, V2)
  R = (mm!(H4, K, V2); mm!(H4, M3, O))
  P4, Q4, r4, R = pluq!(R) # Decompose lower-left corner
  L4, U4, V4, M4 = decomp(R, r4)
  permR!(E2, P4)
  permR!(M3, P4)
  permR!(K,  P4)
  E21, E22 = vsplit(E2, r4)
  M31, M32 = vsplit(M3, r4)
  K1,  K2  = vsplit(K,  r4)
  permC!(D2, Q4)
  permC!(V2, Q4)
  permC!(O,  Q4)
  D21, D22 = hsplit(D2, r4)
  V21, V22 = hsplit(V2, r4)
  O1,  O2  = hsplit(O,  r4)
  # Permutations
  k = rows(A1)
  S = vcat(range(1,                     length=(r1 + r2)),
           range(r1 + r2 + r3 + r4 + 1, length=(k-r1-r2)),
           range(r1 + r2 + 1,           length=(r3+r4)),
           range(r3 + r4 + k + 1,       length=(m-k-r3-r4)))
  kT = cols(A1)
  T = vcat(range(1,          length=r1),
           range(kT+1,       length=r2),
           range(r1+1,       length=r3),
           range(kT+r2+1,    length=r4),
           range(r1+r3+1,    length=(kT-r1-r3)),
           range(kT+r2+r4+1, length=(n-kT-r2-r4)))
  P_ = vcat(permcompose(P1, vcat(1:r1, map(x -> x + r1, P2))),
            map(x -> x + length(P1),
                permcompose(P3, vcat(1:r3, map(x -> x + r3, P4)))))
  Q_ = vcat(permcompose(vcat(1:r1, map(x -> x + r1, Q3)), Q1),
            map(x -> x + length(Q1),
                permcompose(vcat(1:r2, map(x -> x + r2, Q4)), Q2)))
  P = permcompose(P_, S)
  Q = permcompose(T, Q_)
  permR!(A, S)
  permC!(A, T)
  return P, Q, r1 + r2 + r3 + r4, A
end

# Non-destructive PLUQ decomposition
"""
    pluq(A) -> PLUQ

Perform a rank-sensitive LU factorization of `A`.
The factorization of an `n` by `m` matrix
is computed in `O(n m r^(ω-2))` steps,
where `r = rank(A)` and `ω` is the matrix multiplication complexity exponent.

This is an exact function who should work with
any type implementing the basic arithmetic operations
`+`, `-`, `*`, `/`, `one`, `zero`.

The resulting factorization `F` satisfies

    F.P * [F.L ; F.M] * [F.U F.V] * F.Q == A

See [the link](https://arxiv.org/pdf/1301.4438.pdf)
for the original description of the algorithm.
"""
function pluq end

function pluq(A::AbstractMatrix)
  P, Q, r, A = pluq!(copy(A))
  return PLUQ(P, Q, r, A)
end

pluq(A::PLUQ) = A

###########################################
# Generalized Rank Revealing decomposition
###########################################

struct GeneralizedRankRevealing{T, S <: AbstractMatrix{T}} <: Factorization{T}
  X        :: S
  Y        :: S
  H        :: S
  r1       :: Int64
  r2       :: Int64
  r3       :: Int64
end

function GeneralizedRankRevealing(X :: S, Y :: S, H :: S, r1, r2, r3) where {T, S <: AbstractMatrix{T}}
  GeneralizedRankRevealing{T, S}(X, Y, H, r1, r2, r3)
end

# Destructure as p, L, U, V, M, q
Base.iterate(S::GeneralizedRankRevealing)             = (S.X, Val(:Y))
Base.iterate(S::GeneralizedRankRevealing, ::Val{:Y})  = (S.Y, Val(:H))
Base.iterate(S::GeneralizedRankRevealing, ::Val{:H})  = (S.H, Val(:r1))
Base.iterate(S::GeneralizedRankRevealing, ::Val{:r1}) = (S.r1, Val(:r2))
Base.iterate(S::GeneralizedRankRevealing, ::Val{:r2}) = (S.r2, Val(:r3))
Base.iterate(S::GeneralizedRankRevealing, ::Val{:r3}) = (S.r3, Val(:done))
Base.iterate(S::GeneralizedRankRevealing, ::Val{:done}) = nothing

# Swaps U and V on PLUQ
#=
[U V] * Q == [V U] * (J * Q)

J = [ 0 I; I 0]

du dv
[1]
[2, 1]
[

(r+1: n)  <> (1:r)
=#
function pluq2(A)
  F = pluq(A)
  n = cols(A)
  r = F.rank
  j = vcat(((n-r+1) : n),  (1:n-r))
  newq = permcompose(j, F.q)
  return PLUQ(F.p, newq, F.rank, F.factors)
end

function simple_rr(A)
  p, L, U, V, M, q = pluq2(A)
  # Permutation to swap U and V
  return ([L ; M])[p, :], ([V U])[:, q]
end


# Right rank revealing
function right_rr(A::AbstractMatrix{T}) where T
  p, L, U, V, M, q = pluq2(A)
  d = cols(V)
  Z = zeros(Int64, cols(V), cols(U))
  X = ([L ; M])[p, :]
  H = ([I(d) Z; V U])[:, q]
  return X, Array(H)
end

# Left rank revealing
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
  Z1 = zeros(Int64, dx, cols(X5))
  Z2 = zeros(Int64, rows(X5), dx)
  X = X2 * X4 * [I(dx) Z1 ; Z2 X5]
  Y = Y2 / H5
  Z3 = zeros(Int64, rows(H5), cols(H4))
  H = [H4 A41 ; Z3 H5] * H2 * H1
  # Ranks
  rA = cols(X)                 # Rank of A
  rB = cols(Y)                 # Rank of B
  r_cap = rA + rB - rows(H)    # Dimension of R(A) ∩ R(B)
  return GeneralizedRankRevealing(X, Y, H, rA - r_cap, rB - r_cap, r_cap)
end

end # module
