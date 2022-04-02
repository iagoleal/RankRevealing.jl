### Helper type for dealing with permutations
### on the main decomposition algorithms
struct Perm
  vec :: Vector{Int64}
  Perm(v) = new(v)
end

Perm(p :: Perm) = p

# Transposition permutation for indices i and j
function transposition(n, i, j) :: Perm
  perm = collect(1:n)
  perm[i], perm[j] = perm[j], perm[i]
  return Perm(perm)
end

idperm(n) = Perm(collect(1:n))

## Matrix Interface
Base.size(p :: Perm) = (length(p.vec), length(p.vec))
Base.length(p :: Perm) = length(p.vec)

function getindex(p :: Perm, i, j)
  return Int64(p.vec[i] == j)
end

# Convert a permutation into its matrix representation.
function Base.Matrix(T :: Type, p :: Perm)
  A = zeros(T, length(p), length(p))
  for (i, v) in enumerate(p.vec)
    A[i, v] = 1
  end
  return A
end

Base.Matrix(p :: Perm) = Base.Matrix(Int64, p)

function Base.adjoint(p :: Perm) :: Perm
  pt = similar(p.vec)
  for (i, v) in enumerate(p.vec)
    pt[v] = i
  end
  return Perm(pt)
end

Base.inv(p :: Perm) = Base.adjoint(p)

# Composition of permutations.
# Equivalent to multiplying their representation matrices.
function Base.:*(p :: Perm, q :: Perm) :: Perm
  n = length(p)
  if n != length(q)
    error("Tried to compose permutations over different dimensions.")
  end
  return Perm([p.vec[q.vec[k]] for k in 1:n])
end

Base.:*(p :: Perm, A :: AbstractMatrix)  = A[p.vec, :]

Base.:*(A :: AbstractMatrix, p :: Perm) = A[:, p.vec]

# Permute the columns of A inplace
# TODO: Rewrite this without copying
function permC!(A, q :: Perm)
  A .= A[:, q.vec]
end

# Permute the rows of A inplace
function permR!(A, p :: Perm)
  A .= A[p.vec, :]
end

function ⊕(p :: Perm, q :: Perm)
  pv, qv = p.vec, q.vec
  off = length(pv)
  return Perm(vcat(pv, map(x -> x + off, qv)))
end
