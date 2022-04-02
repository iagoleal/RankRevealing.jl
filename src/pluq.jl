########################
# PLUQ decomposition
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

# Auxiliary function to `pluq!`.
# This assigns the right views for the components
# given the calculated rank.
# We use this in order to not need to generate a PLUQ struct on every recursive call.
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

LinearAlgebra.rank(A :: PLUQ) = A.rank
