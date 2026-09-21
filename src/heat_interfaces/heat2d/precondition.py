"""The diagonal-dominance preconditioner of dissertation Appendix B (§5.4.4, E2.8).

A row's *diagonal dominance ratio* (DDR, Appendix B eq. 93) is the diagonal
weight's size over the sum of the other weights' sizes; interface rows sit
well below the standard rows' and the iterative solvers were found to suffer
for it. Appendix B restores dominance by adding to each row multiples of the
rows centred on its neighbours: proceeding outward from the diagonal, the
weight the combined row carries at neighbour ``j`` is cancelled by adding
``−r_j / a_jj`` times row ``j`` (eq. 94–97, the 1-D worked example with a
1 : 50 layer between two nodes: DDR 0.846 → 0.968 after three neighbours a
side). In 2-D the neighbours are the nearest 37 nodes in distance order, and
the case-3 preconditioning made one sweep out, one back in (farthest first)
and one out again. The rows combined are always the *original* rows, so the
result is a sparse ``P`` and the preconditioned problem is ``P A u = P f``
(eq. 99), built here on the reduced interior system of ``heat2d.solve``.

The sweep is vectorised over all target rows at once: at each of the
``sweeps × k`` steps every row cancels its weight at its ``t``-th neighbour
by one sparse product ``C @ a``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .domain import NodeSet
from .neighbors import knn

__all__ = [
    "NEIGHBOURS",
    "SWEEPS",
    "diagonal_dominance_ratio",
    "dominance_preconditioner",
    "neighbour_table",
]

NEIGHBOURS = 37
"""Neighbours cancelled per sweep in the case-3 preconditioning (Appendix B)."""

SWEEPS = 3
"""Outward, inward, outward: the three sweeps of the case-3 preconditioning."""


def diagonal_dominance_ratio(a: sp.sparray) -> np.ndarray:
    """``|a_ii| / Σ_{j≠i} |a_ij|`` per row (Appendix B eq. 93).

    ``inf`` for a row with no off-diagonal weight (an identity row).
    """
    a = sp.csr_array(a)
    if a.shape[0] != a.shape[1]:
        raise ValueError("the matrix must be square")
    diag = np.abs(a.diagonal())
    off = np.asarray(abs(a).sum(axis=1)).ravel() - diag
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(off > 0.0, diag / off, np.inf)


def neighbour_table(
    nodes: NodeSet, interior: np.ndarray, k: int = NEIGHBOURS
) -> np.ndarray:
    """``(len(interior), k)`` interior indices of each unknown's ``k`` nearest nodes.

    Nearest first, the node itself left out, ``−1`` where the neighbour is a
    Dirichlet node: on the reduced system its column is already on the
    right-hand side, so there is nothing to cancel (on the identity-row form
    that step would only have moved the boundary value across, eq. 99's
    remark on ``f``).
    """
    if not 1 <= k < nodes.n:
        raise ValueError(f"k = {k} must be between 1 and the node count minus one")
    index, _ = knn(nodes.xy, k + 1)
    to_interior = -np.ones(nodes.n, dtype=int)
    to_interior[interior] = np.arange(len(interior))
    return to_interior[index[interior, 1:]]


def dominance_preconditioner(
    a: sp.sparray,
    neighbours: np.ndarray,
    rows: np.ndarray | None = None,
    sweeps: int = SWEEPS,
) -> sp.csr_array:
    """Appendix B's ``P``: the identity with the target ``rows`` recombined.

    ``a`` is the (interior) matrix, ``neighbours`` an ``(n, k)`` table of row
    indices in distance order, ``−1`` to skip; ``rows`` the rows to
    precondition (every row by default, the dissertation's "each RBF-FD
    stencil"). Each sweep runs over the ``k`` neighbours in order, the even
    sweeps outward and the odd ones inward, cancelling the combined row's
    weight at the neighbour with a multiple of that neighbour's original row.
    A neighbour with a zero diagonal is skipped.
    """
    a = sp.csr_array(a)
    n = a.shape[0]
    if a.shape[1] != n:
        raise ValueError("the matrix must be square")
    neighbours = np.asarray(neighbours, dtype=int)
    if neighbours.ndim != 2 or neighbours.shape[0] != n:
        raise ValueError("neighbours must be (n, k) with one row per unknown")
    rows = np.arange(n) if rows is None else np.asarray(rows, dtype=int)
    if sweeps < 1:
        raise ValueError("sweeps must be positive")
    m, k = len(rows), neighbours.shape[1]
    ar = np.arange(m)
    diag = a.diagonal()
    combined = sp.csr_array(a[rows])
    p = sp.csr_array((np.ones(m), (ar, rows)), shape=(m, n))
    order = np.arange(k)
    for s in range(sweeps):
        for step in order if s % 2 == 0 else order[::-1]:
            j = neighbours[rows, step]
            ok = (j >= 0) & (diag[np.maximum(j, 0)] != 0.0)
            jj = np.where(ok, j, 0)
            weight = np.asarray(combined[ar, jj]).ravel()
            c = np.where(ok, -weight / diag[jj], 0.0)
            change = sp.csr_array((c, (ar, jj)), shape=(m, n))
            p = p + change
            combined = combined + change @ a
            combined.eliminate_zeros()
    p.eliminate_zeros()
    if m == n and np.array_equal(rows, np.arange(n)):
        return sp.csr_array(p)
    select = sp.csr_array((np.ones(m), (ar, rows)), shape=(m, n))
    full = sp.eye_array(n, format="csr") - select.T @ select + select.T @ p
    return sp.csr_array(full)
