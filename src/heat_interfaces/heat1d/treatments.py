"""Coefficient treatments for a sub-grid edge: the comparators of E3.5 (#30).

Plan §3.4 and ``docs/stiff-diffusion.md`` §1.8: change the *medium* the
standard scheme samples, rather than the scheme. Each nodal treatment is a
``NodalAlpha``, a material given by its values at the grid's nodes, which
``operators.naive_operator`` samples exactly as it samples any other
``Medium1D``; the sweep then reads ``Dx A Dx`` on the treated material
against the true-δ reference.

- **T1**, ``harmonic_cells``: the harmonic mean of alpha over ``cells`` cells
  about each node, ``(b − a) / ∫_a^b dξ/alpha`` (the classic conductance rule
  of finite-volume heat conduction, Patankar's interface conductivity).
- **T2**, ``arithmetic_cells``: the arithmetic mean over the same window,
  what a careless discretisation does.
- **T0**, ``widened_edge``: the true profile with δ replaced by ``max(δ, m
  h)``, regularising alpha itself; a ``SmoothEdges`` again, so the references
  and the windows stay consistent (E3.2's breadcrumb).
- **T1-FV**, ``face_conductance_operator``: the three-point conservative
  scheme with the *exact* face conductances ``a_{i+½} = h / ∫_{x_i}^{x_{i+1}}
  dξ/alpha`` (Tikhonov & Samarskii 1962; for a piecewise-constant alpha the
  harmonic mean of the two cell values, Patankar 1980 ch. 4). Its discrete
  flux ``a_{i+½} (u_{i+1} − u_i)/h`` equals the exact flux ``B`` at the exact
  nodal values of the equilibrium, so it is exact there at every δ (§1.8
  item 3), which no nodal alpha under ``Dx A Dx`` is (§1.8's correction and
  ``tests/heat1d/test_operators.py``). It is an operator, not a medium.

T3, the band-limited alpha of the seismic literature, is not built here: the
ticket keeps it only if the literature pass (E5.2, #43) finds it in use for
diffusion.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .domain import X_MAX, X_MIN, Grid1D, Medium1D, Piece, Side, SmoothEdges
from .exact import alpha_integral, inverse_alpha_integral
from .stiff import edge_width


@dataclass(frozen=True)
class LinearPiece:
    """``a0 + a1 (x - x0)``: one cell of a ``NodalAlpha``, a ``Piece`` of its own."""

    x0: float
    a0: float
    a1: float

    def alpha(self, x: np.ndarray) -> np.ndarray:
        return self.a0 + self.a1 * (np.asarray(x, dtype=float) - self.x0)

    def alpha_x(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), self.a1)

    def taylor(self, x0: float, degree: int) -> np.ndarray:
        a = np.zeros(degree + 1)
        a[0] = self.a0 + self.a1 * (x0 - self.x0)
        if degree >= 1:
            a[1] = self.a1
        return a


@dataclass(frozen=True, repr=False)
class NodalAlpha:
    """A diffusivity given by its ``values`` at the nodes of ``grid``, linear between.

    The material of a nodal treatment: ``alpha`` returns ``values[j]`` at
    ``x_j`` to the bit (``np.interp`` on a hit) and the linear interpolant
    elsewhere, so the naive operator, which samples alpha at the nodes only,
    sees the table and nothing else. It has no interfaces (the treatment has
    replaced the edge by a table), so the straddling and seeded windows are
    empty on it and ``taylor`` has nothing to expand about; ``alpha_x`` is
    the interpolant's slope, read from the cell to the right of a node;
    ``elements`` cut at the nodes with a ``LinearPiece`` per cell, so the
    quadrature and Chebyshev references of the treated material are well
    defined should anyone want them. The ``repr``, which keys the parabolic
    reference cache, carries a hash of the whole table rather than numpy's
    summary of it, which elides the middle of any array past 1000 entries,
    where the edge is.
    """

    grid: Grid1D
    values: np.ndarray
    interfaces: tuple[float, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float)
        if values.shape != (self.grid.n,):
            raise ValueError("one value per node is needed")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("a diffusivity must be positive and finite")
        object.__setattr__(self, "values", values)

    def __repr__(self) -> str:
        digest = hashlib.sha1(self.values.tobytes()).hexdigest()
        return f"NodalAlpha(n={self.grid.n}, h={self.grid.h!r}, sha1={digest})"

    def alpha(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.interp(x, self.grid.x, self.values)

    def alpha_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        slopes = np.diff(self.values) / self.grid.h
        cell = np.clip(
            np.searchsorted(self.grid.x, x, side="right") - 1, 0, self.grid.n - 2
        )
        return slopes[cell]

    def taylor(self, i: int, side: Side, degree: int) -> np.ndarray:
        raise ValueError("a nodal treatment has no interfaces to expand about")

    def elements(self) -> tuple[np.ndarray, tuple[Piece, ...]]:
        slopes = np.diff(self.values) / self.grid.h
        pieces = tuple(
            LinearPiece(float(x0), float(a0), float(a1))
            for x0, a0, a1 in zip(
                self.grid.x[:-1], self.values[:-1], slopes, strict=True
            )
        )
        return self.grid.x.copy(), pieces


def cell_windows(grid: Grid1D, cells: float) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` of the window ``x_j ± cells h / 2`` per node, clipped to [-1, 1].

    One cell is the node's own cell, halfway to each neighbour; two cells
    reach the neighbours themselves. A window at a domain end is cut off
    there and the mean is taken over what remains.
    """
    if not (np.isfinite(cells) and cells > 0.0):
        raise ValueError("the window must be a positive number of cells")
    half = 0.5 * cells * grid.h
    return np.maximum(grid.x - half, X_MIN), np.minimum(grid.x + half, X_MAX)


def harmonic_cells(
    grid: Grid1D, medium: Medium1D, cells: float = 1.0, n_gauss: int = 24
) -> NodalAlpha:
    """T1: the harmonic mean ``(hi − lo) / (F(hi) − F(lo))`` over each node's window.

    ``F`` is ``inverse_alpha_integral``, cut at the medium's elements, so the
    mean across a jump or a smooth edge is exact to rounding. With the edge
    mid-cell and ``cells = 1`` the window ends *on* the jump and no sampled
    value changes at δ = 0 (§1.8): the one-cell rule only sees the tails.
    """
    lo, hi = cell_windows(grid, cells)
    f = inverse_alpha_integral(medium, np.stack([lo, hi]), n_gauss)
    return NodalAlpha(grid, (hi - lo) / (f[1] - f[0]))


def arithmetic_cells(
    grid: Grid1D, medium: Medium1D, cells: float = 1.0, n_gauss: int = 24
) -> NodalAlpha:
    """T2: the arithmetic mean ``(G(hi) − G(lo)) / (hi − lo)`` of ``alpha_integral``.

    What a careless discretisation does across an edge: the mean of alpha
    where the mean of ``1/alpha`` governs the flux, so on the MATLAB jump the
    treated value beside the edge is five times the harmonic one.
    """
    lo, hi = cell_windows(grid, cells)
    g = alpha_integral(medium, np.stack([lo, hi]), n_gauss)
    return NodalAlpha(grid, (g[1] - g[0]) / (hi - lo))


def widened_edge(grid: Grid1D, medium: Medium1D, m: float = 1.0) -> SmoothEdges:
    """T0: the medium's edges widened to ``max(δ, m h)``, a ``SmoothEdges``.

    A ``SmoothEdges`` keeps its jump; any other medium is taken as the jump
    (δ = 0). Below ``δ = m h`` the treatment does not depend on δ at all:
    its equilibrium sits ``c (m h − δ)`` from the true one on constant
    pieces (``edge_resistance_deficit``), first order in h.
    """
    if not (np.isfinite(m) and m > 0.0):
        raise ValueError("the widening factor m must be positive")
    jump = medium.jump if isinstance(medium, SmoothEdges) else medium
    return SmoothEdges(jump, max(edge_width(medium), m * grid.h))


def face_conductances(grid: Grid1D, medium: Medium1D, n_gauss: int = 24) -> np.ndarray:
    """``a_{i+½} = h / ∫_{x_i}^{x_{i+1}} dξ/alpha`` on the ``n − 1`` faces.

    The exact conductance of each cell: on two constants meeting mid-cell it
    is their harmonic mean, and on any profile the discrete flux ``a (u_{i+1}
    − u_i) / h`` of the exact equilibrium is the exact flux.
    """
    f = inverse_alpha_integral(medium, grid.x, n_gauss)
    return grid.h / np.diff(f)


def face_conductance_operator(
    grid: Grid1D, medium: Medium1D, n_gauss: int = 24
) -> sp.csr_array:
    """T1-FV: ``(L u)_i = [a_{i+½} (u_{i+1} − u_i) − a_{i−½} (u_i − u_{i−1})] / h²``.

    Three-point rows inside; the end rows are zero, the Dirichlet rows the
    solvers put there being the only ones ever used. Second order on a
    smooth alpha, exact on the equilibrium at every δ (§1.8 item 3).
    """
    a = face_conductances(grid, medium, n_gauss) / grid.h**2
    n = grid.n
    main, lower, upper = np.zeros(n), np.zeros(n - 1), np.zeros(n - 1)
    main[1:-1] = -(a[:-1] + a[1:])
    lower[: n - 2] = a[: n - 2]
    upper[1:] = a[1:]
    return sp.csr_array(sp.diags_array([lower, main, upper], offsets=[-1, 0, 1]))


__all__ = (
    "LinearPiece",
    "NodalAlpha",
    "arithmetic_cells",
    "cell_windows",
    "face_conductance_operator",
    "face_conductances",
    "harmonic_cells",
    "widened_edge",
)
