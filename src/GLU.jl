module GLU
export grr

using LinearAlgebra

# https://github.com/JuliaArrays/BlockArrays.jl
# https://arxiv.org/pdf/1301.4438.pdf

rows(A) = size(A)[1]
cols(A) = size(A)[2]


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


###########################################
# Generalized Rank Revealing decomposition
###########################################

function epimono(A)
  P, L, M, V, U, Q = pluq(X)
  return P*[L ; M], [U V] * Q
end

# Right rank revealing
function invmono(X)
  P, L, M, V, U, Q = pluq(X)
  return P * [L ; M],
         [I 0*I ; U V] * Q
end

# Left rank revealing
function epiinv(X)
  P, L, M, V, U, Q = pluq(X)
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
