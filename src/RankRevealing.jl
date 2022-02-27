module RankRevealing
export pluq, grr

using LinearAlgebra
using Debugger

# https://github.com/JuliaArrays/BlockArrays.jl
# https://arxiv.org/pdf/1301.4438.pdf

rows(A) = size(A)[1]
cols(A) = size(A)[2]

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

struct PLUQ{T, S <: AbstractMatrix{T}} <: Factorization{T}
  factors  :: S
  perm_pre :: Vector{Int64}
  perm_pos :: Vector{Int64}
  rank     :: Int64
  function PLUQ{T,S}(A, perm_pre, perm_pos, rank) where {T,S<:AbstractMatrix{T}}
      new{T,S}(A, perm_pre, perm_pos, rank)
  end
end

function PLUQ(A::AbstractMatrix{T}, p, q, r) where {T}
  LU{T,typeof(A)}(A, p, q, r)
end

function getproperty(D::PLUQ{T, <:AbstractMatrix}, key::Symbol) where T
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
  else
    getfield(D, key)
  end
end

mm!(C, A, B) = LinearAlgebra.mul!(C, A, B, -1, 1)

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

# Transposition permutation for indices i and j
function transposition(n, i, j)
  perm = collect(1:n)
  perm[i], perm[j] = perm[j], perm[i]
  return perm
end

# In-place matrix splitting into 4 blocks
function msplit(A)
  m, n = size(A)
  mh, nh = fld(m, 2), fld(n, 2)
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


function pluq!(A)
  m, n = size(A)
  if m == 1
    if iszero(A)
      P = [1]
      Q = collect(1:n)    # n x n identity matrix
      r = 0
    else
      i = findfirst(!(iszero), A).I[2] # column index of the first non-zero element of A
      P = [1]
      Q = transposition(n, 1, i)
      r = 1
      A[1, i], A[1, 1] = A[1, 1], A[1, i]
    end
    @bp
    return P, Q, r, A
  end
  if n == 1
    if iszero(A)
      P = collect(1:m)
      Q = [1]
      r = 0
    else
      i = findfirst(!(iszero), A).I[1] # row index of the first non-zero element of A
      P = [1]
      Q = transposition(m, 1, i)
      r = 1
      A[i, 1], A[1, 1] = A[1, 1], A[i, 1]
      local pivot = A[1,1]
      for j in (i + 1):m
        A[j, 1] = A[j, 1] / pivot
      end
    end
    @bp
    return P, Q, r, A
  end
  # Now the recursion step
  A1, A2, A3, A4 = msplit(A)
  P1, Q1, r1, A1 = pluq!(A1) # Decompose upper-left quadrant
  L1, U1, V1, M1 = decomp(A1, r1)
  @bp
  B1, B2 = vsplit(permR!(A2, P1), rows(A2) - rows(M1))
  C1, C2 = hsplit(permC!(A3, Q1), rows(A3) - cols(V1))
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
  H1, H2, H3, H4 = msplit(H)
  E1,  E2  = vsplit(permR!(E, P3),  rows(E)  - rows(H3))
  M11, M12 = vsplit(permR!(M1, P2), rows(M1) - rows(M2))
  D1,  D2  = hsplit(permR!(D, Q2),  cols(D)  - cols(V2))
  @bp
  V11, V12 = hsplit(permR!(V1, Q3), cols(V1) - cols(V3))
  I = rdiv!(H1, U2)
  J = ldiv!(L3, I)
  K = rdiv!(H3, U2)
  N = ldiv!(L3, H2)
  O = mm!(N, J, V2)
  R = (mm!(H4, K, V2); mm!(H4, M3, O))
  P4, Q4, r4, R = pluq!(R) # Decompose lower-left crner
  L4, U4, V4, M4 = decomp(R, r4)
  # TODO: is this inplace?
  bot = view(A, (rows(A)-rows(E2)+1):rows(A), 1:(cols(A2) + cols(K)))
  permR!(bot, P4)
  E21, E22 = vsplit(E2, rows(V4))
  M31, M32 = vsplit(M3, rows(V4))
  K1,  K2  = vsplit(K,  rows(V4))
  right = view(A, 1:(rows(A) - rows(R)), (cols(A)-cols(R) + 1):cols(A))
  permC!(right, Q4)
  D21, D22 = hsplit(D2, cols(M4))
  V21, V22 = hsplit(V2, cols(M4))
  O1,  O2  = hsplit(O,  cols(M4))
  # TODO: assign these variables
  S = vcat(range(1, length=(r1 + r2)),
           range(r1 + r2 + r3 + r4 + 1, length=(k-r1-r2)),
           range(r1 + r2 + 1, length=(r3+r4)),
           range(r3 + r4 + k+ 1, length=(m-k-r3-r4)))
  T = vcat(range(1,      length=r1),
           range(k+1,    length=r2),
           range(r1+1,   length=r3),
           range(k+r2+1, length=r4),
           range(r1+r3+1,   length=(k-r1-r3),
           range(k+r2+r4+1, length=(n-k-r2-r4))))
  P_ = vcat(permcompose(P1, vcat(1:r1, map(x -> x + r1, P2))),
            permcompose(P3, vcat(1:r3, map(x -> x + r3, P4))))
  Q_ = vcat(permcompose(vcat(1:r1, map(x -> x + r1, Q2)), Q1),
            permcompose(vcat(1:r3, map(x -> x + r3, Q4)), Q3))
  P = permcompose(P_, S)
  Q = permcompose(T, Q_)
  permR!(A, S)
  permC!(A, T)
  return P, Q, r1 + r2 + r3 + r4, A
end

# Non-destructive PLUQ decomposition
"""
    pluq(A) -> P [L ; M] [U V] Q

Perform a rank-sensitive LU factorizaton of `A`.
"""
pluq(A) = pluq!(copy(A))


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

function epimono(A)
  P, L, M, V, U, Q = pluq(A)
  return P*[L ; M], [U V] * Q
end

# Right rank revealing
function invmono(A)
  P, L, M, V, U, Q = pluq(A)
  return P * [L ; M],
         [I 0*I ; U V] * Q
end

# Left rank revealing
function epiinv(A)
  P, L, M, V, U, Q = pluq(A)
  return P * [L 0*I ; M I],
         [U V] * Q
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
  M, H1    = epimono([A ; B])
  A1, B1   = vsplit(M, m_A)
  X2, A2   = epimono(A1)
  Y2, H2   = invmono(B1)
  A3       = A2 / H2
  A31, A32 = hsplit(A3, cols(H2))
  X4, H4   = epiinv(A31)
  A4       = X4 \ A32
  A41, A42 = vsplit(A4, rows(H4))
  X5, H5   = invmono(A42)
  # Outputs
  local Zero = 0*I
  X = (local _X = X2*X4; [_X Zero ; Zero _X * X5])
  Y = Y2 / H5
  H = [H4 A41 ; Zero H5] * H2 * H1
  # Ranks
  rA = cols(X)
  rB = cols(Y)
  r_cap = rA + rB - cols(H)
  return X, Y, H, rA - r_cap, rB - r_cap, r_cap
end

end # module
