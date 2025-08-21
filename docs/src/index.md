# RankRevealing.jl

```@meta
CurrentModule = RankRevealing
```

This package defines methods for exact linear algebra
over any numerical field.
Our focus is on matrix decompositions that are _rank-sensitive_.
That is, faster for low-rank matrices.

```@index
```

## References

- [1] Jean-Guillaume Dumas, Clément Pernet, Ziad Sultan.
  ["Simultaneous computation of the row and column rank profiles"](https://hal.archives-ouvertes.fr/file/index/docid/778136/filename/pluq_report.pdf).
  In: _Proceedings of the 38th International Symposium on Symbolic and Algebraic Computation_.
  2013, pp. 181–188.
- [2] Iago Leal de Freitas, Júlia Mota, João Paixão, Lucas Rufino.
  "Generalizing the Invertible Matrix Theorem with Linear Relations using Graphical Linear Algebra".
  2025, preprint at [arXiv:2502.16783v2](https://doi.org/10.48550/arXiv.2502.16783).
