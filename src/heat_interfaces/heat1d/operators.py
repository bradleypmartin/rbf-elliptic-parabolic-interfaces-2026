"""Discrete 1-D operators: FD4 ``Dx``, ``A = diag(alpha)``, and ``(alpha u_x)_x`` twice.

The naive baseline is dissertation eq. 76, ``L = Dx A Dx`` (plan D3). The
direct stencil, ``alpha(x_i) Dxx + alpha'(x_i) Dx``, is the operator applied to
the polynomial basis at each node without any knowledge of the jump; on a
piecewise-constant alpha its equilibrium solution is the straight line
between the boundary values that §4.2 warns about. It is kept for the record
and for the test that pins that failure.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..fd_weights import fornberg_weights
from .domain import Grid1D, Medium1D

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
    """``A = diag(alpha(x_j))``; a node on an interface takes the owner's value."""
    return sp.diags_array(medium.alpha(grid.x))


def naive_operator(grid: Grid1D, medium: Medium1D) -> sp.csr_array:
    """``L = Dx A Dx``, dissertation eq. 76 (plan D3); nine-point rows inside."""
    dx = dx_matrix(grid)
    return (dx @ alpha_matrix(grid, medium) @ dx).tocsr()


def direct_operator(grid: Grid1D, medium: Medium1D) -> sp.csr_array:
    """``alpha(x_i) Dxx + alpha'(x_i) Dx``: ``(alpha u_x)_x`` on the polynomial basis.

    Correct to fourth order wherever alpha is smooth and blind to a jump: the
    stencils across an interface see only the owning piece's value and slope.
    """
    a = sp.diags_array(medium.alpha(grid.x))
    a_x = sp.diags_array(medium.alpha_x(grid.x))
    return (a @ dxx_matrix(grid) + a_x @ dx_matrix(grid)).tocsr()
