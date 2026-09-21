"""The Cartesian FD4 baseline of EABE Fig. 10 and 14 (dissertation Fig. 5-9, 5-11).

Fourth-order five-point differences on an equispaced grid of the strip: ``m``
columns at ``x = i/m`` (periodic, so every row of ``Dx`` is centred) and
``m + 1`` levels at ``y = j/m`` with one-sided rows within two of ``y = 0``
and ``y = 1`` (``heat1d.operators.derivative_matrix``), assembled as
``Dx A Dx + Dy A Dy`` with ``A = diag(α)`` at the grid points: dissertation
eq. 76 in 2-D (plan D3), the MATLAB ``FDheat1.m``'s
``Dy (k .* Dy u) + (k .* u Dx') Dx'``. The grid is a ``NodeSet`` whose top and
bottom levels are its Dirichlet rows, and whose points on or inside a
Dirichlet hole (case 3's cooling disc) are a staircase Dirichlet set, so the
equilibrium solve and the resampling of E2.6 apply to it unchanged. The
stencils see nothing of an interface: first order where the grid crosses one
(Fig. 10's top line), fourth where α is smooth, and blind to case 3's ring
until the spacing resolves it (Fig. 14's top line).
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import scipy.sparse as sp

from ..fd_weights import fornberg_weights
from ..heat1d.domain import Grid1D
from ..heat1d.operators import STENCIL_WIDTH, derivative_matrix
from .domain import DIRICHLET, FREE, STRIP, Domain, NodeSet, Row
from .operators import alpha_matrix


def grid_size(n: int) -> int:
    """``m`` with ``m (m + 1)`` nearest to ``n``: the columns of an ``n``-node grid."""
    m = int(round((sqrt(1.0 + 4.0 * n) - 1.0) / 2.0))
    return max(m, STENCIL_WIDTH)


def cartesian_grid(domain: Domain, n: int) -> NodeSet:
    """The equispaced grid of about ``n`` nodes: ``m (m + 1)`` at spacing ``1/m``.

    Node ``j m + i`` is ``(i/m, j/m)``; levels ``j = 0`` and ``j = m`` are the
    Dirichlet rows of the strip, which ``domain.dirichlet`` must start with.
    Any further Dirichlet curve must be one of ``domain.holes``, and every
    grid point on or inside it becomes a Dirichlet node of that curve: the
    staircase a Cartesian code makes of case 3's cooling disc, the boundary
    placed to within one spacing (a first-order error the ring's dwarfs at
    every count of Fig. 14). A hole too small to catch a grid point is
    invisible to that grid, as it would be to the code; the driver prints
    the count. Every other node is ``FREE`` and no row straddles anything:
    the grid ignores the interfaces, as FD4 does.
    """
    extra = tuple(domain.dirichlet[2:])
    if (
        tuple(domain.dirichlet[:2]) != STRIP
        or any(curve not in domain.holes for curve in extra)
        or any(hole not in extra for hole in domain.holes)
    ):
        raise NotImplementedError(
            "the FD4 grid covers the strip, with holes that are Dirichlet curves"
        )
    m = grid_size(n)
    h = 1.0 / m
    i, j = np.meshgrid(np.arange(m), np.arange(m + 1))
    x = (i / m).ravel()
    y = (j / m).ravel()
    kind = np.full(x.size, FREE, dtype=np.int8)
    boundary = np.full(x.size, -1, dtype=int)
    bottom, top = np.arange(m), np.arange(m * m, m * (m + 1))
    kind[bottom] = kind[top] = DIRICHLET
    boundary[bottom], boundary[top] = 0, 1
    rows = [Row(0, 0.0, h, bottom), Row(1, 0.0, h, top)]
    for ci, hole in enumerate(extra, start=2):
        inside = np.flatnonzero(hole.level(x, y) <= 0.0)
        kind[inside] = DIRICHLET
        boundary[inside] = ci
        rows.append(Row(ci, 0.0, h, inside))
    return NodeSet(x, y, kind, h, (), tuple(rows), boundary)


def fd4_dx(m: int, h: float) -> sp.csr_array:
    """Periodic centred five-point ``d/dx`` on ``m`` columns at spacing ``h``."""
    half = STENCIL_WIDTH // 2
    unit = np.arange(STENCIL_WIDTH, dtype=float)
    w = fornberg_weights(float(half), unit, 1)[1] / h
    i = np.arange(m)
    rows = np.repeat(i, STENCIL_WIDTH)
    cols = ((i[:, None] + np.arange(-half, half + 1)[None, :]) % m).ravel()
    return sp.csr_array((np.tile(w, m), (rows, cols)), shape=(m, m))


def fd4_dy(levels: int, h: float) -> sp.csr_array:
    """``d/dy`` on ``levels`` equispaced levels, one-sided within two of each end."""
    return derivative_matrix(Grid1D(levels, np.arange(levels) * h, h), 1)


def fd4_operator(grid: NodeSet, material) -> sp.csr_array:
    """``Dx A Dx + Dy A Dy`` in ``cartesian_grid``'s node order (eq. 76, plan D3).

    Every node has rows of ``Dx`` and ``Dy``, the Dirichlet levels included,
    as the RBF-FD naive operator does; ``dirichlet_system`` replaces the
    assembled operator's Dirichlet rows afterwards.
    """
    m = int(round(1.0 / grid.h))
    levels = grid.n // m
    if m * levels != grid.n or m < STENCIL_WIDTH or levels < STENCIL_WIDTH:
        raise ValueError("the node set is not a cartesian_grid")
    dx = sp.kron(sp.eye_array(levels), fd4_dx(m, grid.h), format="csr")
    dy = sp.kron(fd4_dy(levels, grid.h), sp.eye_array(m), format="csr")
    a = alpha_matrix(grid, material)
    return (dx @ a @ dx + dy @ a @ dy).tocsr()
