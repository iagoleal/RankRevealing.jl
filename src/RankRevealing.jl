module RankRevealing

using LinearAlgebra

#########################
# General Utilities
#########################

rows(A) = size(A, 1)
cols(A) = size(A, 2)

# Matrix direct sum
# dsum(A, B) -> [A 0; 0 B]
dsum(A...) = cat(A...; dims=(1,2))

# Convert a list of vectors into a matrix with those vectors as rows.
rowstomatrix(xs) = reduce(vcat, map(x -> hcat(x...), xs))

# Return list containing a matrix' rows as vectors.
matrixtorows(A) = [A[x,:] for x in 1:rows(A)]

# In-place vertical matrix splitting
#  A -> [A1
#       ;A2]
function vsplit(X, k)
  m = rows(X)
  return view(X, 1:k, :), view(X, (k+1):m, :)
end

# In-place horizontal matrix splitting
# A -> [A1 A2]
function hsplit(X, k)
  n = cols(X)
  return view(X, :, 1:k), view(X, :, (k+1):n)
end

# In-place matrix splitting into 4 blocks
function msplit(A, mh = fld(rows(A),2), nh = fld(cols(A), 2))
  m, n = size(A)
  return view(A, 1:mh,     1:nh),
         view(A, 1:mh,     (nh+1):n),
         view(A, (mh+1):m, 1:nh),
         view(A, (mh+1):m, (nh+1):n)
end

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

# PLUQ decomposition
include("pluq.jl")

export pluq, PLUQ

# Generalized Rank Revealing decomposition
include("grr.jl")

export grr, GeneralizedRankRevealing, simple_rr, right_rr, left_rr
export intersection, sumspace

end # module
