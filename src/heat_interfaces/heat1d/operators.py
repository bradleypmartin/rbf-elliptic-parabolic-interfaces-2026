"""Discrete 1-D operators: FD4 ``Dx``, ``A = diag(alpha)``, and ``(alpha u_x)_x``.

The naive baseline is dissertation eq. 76, ``L = Dx A Dx`` (plan D3). The
direct stencil, ``alpha(x_i) Dxx + alpha'(x_i) Dx``, is the operator applied to
the polynomial basis at each node without any knowledge of the jump; on a
piecewise-constant alpha its equilibrium solution is the straight line
between the boundary values that §4.2 warns about. It is kept for the record
and for the test that pins that failure. The §4.1 method, ``jump_aware_operator``,
is the direct stencil everywhere except on the windows that straddle an
interface, where the weights are recomputed on the translated basis of
``interface.py``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..fd_weights import fornberg_weights
from .domain import Grid1D, Medium1D
from .interface import Jump, stencil_weights

STENCIL_WIDTH = 5
"""Five nodes: fourth order for the first derivative, the FD4 of the dissertation."""


def derivative_matrix(
    grid: Grid1D, order: int, width: int = STENCIL_WIDTH
) -> sp.csr_array:
    """``d^order/dx^order`` on ``width``-node stencils, centred where they fit.

    Rows within ``width // 2`` of an end use the first or last ``width``
    nodes with Fornberg weights at the row's own node (one-sided FD4).
    """
    n = grid.n
    if n < width:
        raise ValueError("fewer nodes than the stencil width")
    unit = np.arange(width, dtype=float)
    # On an equispaced grid only the position within the window matters.
    table = np.array([fornberg_weights(p, unit, order)[order] for p in range(width)])
    table /= grid.h**order
    i = np.arange(n)
    lo = np.clip(i - width // 2, 0, n - width)
    rows = np.repeat(i, width)
    cols = (lo[:, None] + np.arange(width)[None, :]).ravel()
    vals = table[i - lo].ravel()
    return sp.csr_array((vals, (rows, cols)), shape=(n, n))


def dx_matrix(grid: Grid1D) -> sp.csr_array:
    """The ``Dx`` of eq. 76: FD4 first derivative, one-sided at the ends."""
    return derivative_matrix(grid, 1)


def dxx_matrix(grid: Grid1D) -> sp.csr_array:
    """FD second derivative on the same five-point windows (fourth order inside)."""
    return derivative_matrix(grid, 2)


def alpha_matrix(grid: Grid1D, medium: Medium1D) -> sp.dia_array:
    """``A = diag(alpha(x_j))``, evaluated on the snapped nodes (``Grid1D.snapped``)."""
    return sp.diags_array(medium.alpha(grid.snapped(medium.interfaces)))


def naive_operator(grid: Grid1D, medium: Medium1D) -> sp.csr_array:
    """``L = Dx A Dx``, dissertation eq. 76 (plan D3); nine-point rows inside."""
    dx = dx_matrix(grid)
    return (dx @ alpha_matrix(grid, medium) @ dx).tocsr()


def direct_operator(
    grid: Grid1D, medium: Medium1D, width: int = STENCIL_WIDTH
) -> sp.csr_array:
    """``alpha(x_i) Dxx + alpha'(x_i) Dx``: ``(alpha u_x)_x`` on the polynomial basis.

    Correct to fourth order wherever alpha is smooth (on the default five-node
    windows) and blind to a jump: the stencils across an interface see only
    the owning piece's value and slope.
    """
    x = grid.snapped(medium.interfaces)
    a = sp.diags_array(medium.alpha(x))
    a_x = sp.diags_array(medium.alpha_x(x))
    dxx = derivative_matrix(grid, 2, width)
    dx = derivative_matrix(grid, 1, width)
    return (a @ dxx + a_x @ dx).tocsr()


def straddling_windows(
    grid: Grid1D, medium: Medium1D, width: int = STENCIL_WIDTH
) -> list[tuple[int, int, list[int]]]:
    """``(row, lo, interfaces)`` for every row whose window straddles an interface.

    A window straddles interface ``k`` when it has nodes strictly on both
    sides of it; a window that merely ends on an interface node does not,
    since the basis is continuous there. Three rows per interface sitting on
    a node, four per interface sitting mid-cell.
    """
    x = grid.snapped(medium.interfaces)
    n = grid.n
    lo = np.clip(np.arange(n) - width // 2, 0, n - width)
    xi = np.asarray(medium.interfaces, dtype=float)
    seen = (x[lo][:, None] < xi[None, :]) & (xi[None, :] < x[lo + width - 1][:, None])
    return [
        (int(i), int(lo[i]), [int(k) for k in np.flatnonzero(seen[i])])
        for i in np.flatnonzero(seen.any(axis=1))
    ]


def jump_aware_operator(
    grid: Grid1D, medium: Medium1D, degree: int = 4
) -> sp.csr_array:
    """The §4.1 operator: direct rows, translated-basis rows across interfaces.

    Every row whose ``degree + 1`` node window straddles an interface gets
    weights that reproduce ``D`` on the piecewise polynomials of eq. 74
    (Fig. 4-3 / EABE Fig. 4); windows that straddle two interfaces translate
    twice. The stencil algebra runs in units of ``h`` (positions ``x / h``,
    Taylor coefficients ``a_k h^k``) and the weights are scaled back by
    ``h^2``, so the small solves are as well conditioned at 1601 nodes as
    at 101.
    """
    width = degree + 1
    x = grid.snapped(medium.interfaces)
    h = grid.h
    powers = h ** np.arange(degree + 1)
    jumps = [
        Jump(
            xi / h,
            medium.taylor(k, "left", degree) * powers,
            medium.taylor(k, "right", degree) * powers,
        )
        for k, xi in enumerate(medium.interfaces)
    ]
    op = direct_operator(grid, medium, width).tolil()
    for i, lo, seen in straddling_windows(grid, medium, width):
        cols = np.arange(lo, lo + width)
        w = stencil_weights([jumps[k] for k in seen], x[cols] / h, x[i] / h)
        op[i, :] = 0.0
        op[i, cols] = w / h**2
    return op.tocsr()
