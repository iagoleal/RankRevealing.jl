# Rank Revealing Decompositions

<img src="https://user-images.githubusercontent.com/5944492/160865394-2dae48fa-1200-4215-8706-a63055ee0ab7.svg" width="100%" height="100">

This package defines methods for exact linear algebra
over any numerical field.
Our focus is on matrix decompositions that are _rank-sensitive_.
That is, faster for low-rank matrices.

## Installation

This package is not registered on Julia's general registry yet.
But you can install it by passing the repositories' url to the package manager.

To instal, you have to enter `]` on the Julia REPL and write

    pkg> add https://github.com/iagoleal/RankRevealing.jl

## Main algorithms

### PLUQ - For a single matrix

This a version of the [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition)
that works for **rectangular** and possible **singular** matrices.
In block notation, it turns a matrix `A` into

    A = P * |L| * | U V | * Q
            |M|

where `P`, `Q` are permutations,
`L` is non-singular unit lower triangular,
and `U` is non-singular upper triangular.
Also, the dimensions of `L` and `U` are equal
to the rank of `A`.

If `A` is `m x n` and has rank `r`,
the PLUQ takes `O(m n r^(ω-2))` steps,
where `ω` is the exponent for
the square matrix multiplication algorithm used by Julia.

See [[1]](#1) for the paper this implementation is based.

### GRR - For a pair of matrices

Given two matrices `A` and `B` with the same number of columns,
this decomposition divides the space based on how their row spaces interact.
In block notation, it decomposes the matrices as

                    | I 0 0 |
    | A | = | X 0 | | 0 0 I |
    | B |   | 0 Y | | 0 I 0 | H
                    | 0 0 I |

Also, the rows of `H` form bases for

* `R(A)`, the row space of `A`,
* `R(B)`, the row space of `B`,
* `R(B) ∩ R(B)`, the row spaces intersection,
* `R(B) + R(B)`, the sum of their row spaces,

with the additional property that
whenever one of those spaces is contained in another,
the calculated bases share the necessary vectors.

If `A` is `m_A x n`,
`B` is `m_B x n`
and `d` is the dimension of their images' sum space
(`{ a + b | ∃x, y, a = xA, b = yB }`),
then the GRR decomposition takes `O((m_A + m_B) n d^(ω-2))` steps,
where `ω` is the exponent for
the square matrix multiplication algorithm used by Julia.

## References

- <a id="1">[1]</a> Jean-Guillaume Dumas, Clément Pernet, Ziad Sultan.
["Simultaneous computation of the row and column rank profiles"](https://hal.archives-ouvertes.fr/file/index/docid/778136/filename/pluq_report.pdf).
In: _Proceedings of the 38th International Symposium on Symbolic and Algebraic Computation_.
2013, pp. 181–188.

- <a id="2">[2]</a> Iago Leal de Freitas, João Paixão, Lucas Rufino, and Pawelł Sobocínski.
"Rank sensitive complexity to find the intersection between two subspaces".
2022 (upcoming)
