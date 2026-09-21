"""Piecewise-polynomial bases across 1-D interfaces (dissertation §4.1, EABE §2.2).

Everything here acts on coefficient vectors ``c`` of polynomials
``sum_k c_k (x - x0)^k`` in P_p about an interface at ``x0``: differentiation
(EABE eq. 13 / dissertation eq. 60), multiplication by a smooth alpha through
its Taylor coefficients (eq. 15 / 62–63), the operator ``D = Dx M Dx``
(eq. 10–11 / 58–59), the continuity matrices that collect the constant terms
of the time derivatives of temperature and heat flux (eq. 23 / 71), and the
translation ``u_L = C_L^{-1} C_R u_R`` (eq. 25 / 73) that carries a basis
across the interface. ``translated_basis`` chains translations across every
interface a stencil sees (§4.1's remark below Fig. 4-2, used in §5.4.3).

The algebra is unit-free: it takes whatever Taylor coefficients and positions
it is given. ``operators.jump_aware_operator`` passes positions in units of
``h`` and coefficients ``a_k h^k`` so the small stencil solves stay O(1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.linalg import toeplitz

from .domain import Side


def coefficient_dx(degree: int) -> np.ndarray:
    """``Dx`` on coefficient vectors in P_degree (eq. 13 / 60): ``Dx[k-1, k] = k``."""
    d = np.zeros((degree + 1, degree + 1))
    k = np.arange(1, degree + 1)
    d[k - 1, k] = k
    return d


def multiplication_matrix(a: np.ndarray) -> np.ndarray:
    """Multiplication by ``sum a_k (x - x0)^k``, truncated to P_degree (eq. 15 / 62).

    Lower-triangular Toeplitz in the Taylor coefficients ``a``; the product's
    terms above ``degree = len(a) - 1`` are dropped, as in eq. 66 / 18.
    """
    a = np.asarray(a, dtype=float)
    return toeplitz(a, np.zeros_like(a))


def coefficient_operator(a: np.ndarray) -> np.ndarray:
    """``D = Dx M Dx`` on coefficient vectors (eq. 10–11 / 58–59).

    Exact below degree ``p``; the degree-``p`` coefficient of ``D c`` would
    need ``a_{p+1}`` and is truncated with the rest.
    """
    dx = coefficient_dx(len(a) - 1)
    return dx @ multiplication_matrix(a) @ dx


def continuity_conditions(degree: int) -> list[tuple[str, int]]:
    """The ``degree + 1`` conditions in eq. 23's order.

    Time derivatives of temperature and of heat flux interleaved:
    ``(u, 0), (flux, 0), (u, 1), (flux, 1), (u, 2), ...`` — u through
    ``k = 2`` and flux through ``k = 1`` for the fourth-degree case.
    """
    conditions: list[tuple[str, int]] = []
    k = 0
    while len(conditions) < degree + 1:
        conditions.append(("u", k))
        if len(conditions) < degree + 1:
            conditions.append(("flux", k))
        k += 1
    return conditions


def continuity_matrix(a: np.ndarray) -> np.ndarray:
    """One side's ``C`` of ``C_L u_L = C_R u_R`` (eq. 24 / 72).

    Row ``(u, k)`` is the constant term of ``D^k u`` (eq. 6 / 54), row
    ``(flux, k)`` the constant term of ``alpha d/dx D^k u`` (eq. 7 / 55), both
    as linear forms on the coefficient vector. Lower triangular with diagonal
    ``1, a_0, 2 a_0, 6 a_0^2, 24 a_0^2, ...``, so invertible whenever
    ``alpha`` is nonzero at the interface.
    """
    a = np.asarray(a, dtype=float)
    dx = coefficient_dx(a.size - 1)
    m = multiplication_matrix(a)
    d = dx @ m @ dx
    flux = m @ dx
    rows = []
    for kind, k in continuity_conditions(a.size - 1):
        d_k = np.linalg.matrix_power(d, k)
        rows.append((d_k if kind == "u" else flux @ d_k)[0])
    return np.array(rows)


def translation_matrix(a_from: np.ndarray, a_to: np.ndarray) -> np.ndarray:
    """``C_to^{-1} C_from``: coefficients on the ``to`` side from the ``from`` side.

    Eq. 25 / 73 with ``from = R``, ``to = L`` gives the ``U_L`` of eq. 74.
    """
    return np.linalg.solve(continuity_matrix(a_to), continuity_matrix(a_from))


def shift_matrix(delta: float, degree: int) -> np.ndarray:
    """Re-centre: coefficients about ``x0 + delta`` from those about ``x0``.

    ``c'_m = sum_{k >= m} C(k, m) delta^(k - m) c_k``; the polynomial itself
    is unchanged. A stencil that sees two interfaces re-centres the basis
    from the first to the second before translating across it.
    """
    s = np.zeros((degree + 1, degree + 1))
    for k in range(degree + 1):
        for m in range(k + 1):
            s[m, k] = comb(k, m) * delta ** (k - m)
    return s


@dataclass(frozen=True)
class Jump:
    """An interface at ``position`` with alpha's Taylor coefficients about it.

    ``left`` and ``right`` are ``a_0 .. a_degree`` of the two smooth pieces
    (``Medium1D.taylor``), possibly rescaled to stencil units.
    """

    position: float
    left: np.ndarray
    right: np.ndarray

    @property
    def degree(self) -> int:
        return len(self.left) - 1


@dataclass(frozen=True)
class Region:
    """The basis on one smooth region of a stencil.

    Column ``j`` of ``coefficients`` is basis function ``j`` as a polynomial
    about ``jumps[jump].position``; the region lies on ``side`` of that jump.
    """

    jump: int
    side: Side
    coefficients: np.ndarray

    def evaluate(self, jumps: Sequence[Jump], x: np.ndarray) -> np.ndarray:
        """``phi_j(x)`` for every basis function: shape ``(len(x), degree + 1)``."""
        return _vandermonde(jumps, self, x) @ self.coefficients

    def operator_values(self, jumps: Sequence[Jump], x: np.ndarray) -> np.ndarray:
        """``(D phi_j)(x)`` with ``D = Dx M Dx`` built from this region's alpha."""
        jump = jumps[self.jump]
        a = jump.left if self.side == "left" else jump.right
        d = coefficient_operator(a) @ self.coefficients
        return _vandermonde(jumps, self, x) @ d


def _vandermonde(jumps: Sequence[Jump], region: Region, x: np.ndarray) -> np.ndarray:
    t = np.atleast_1d(np.asarray(x, dtype=float)) - jumps[region.jump].position
    return t[:, None] ** np.arange(region.coefficients.shape[0])[None, :]


def translated_basis(jumps: Sequence[Jump], anchor: int) -> list[Region]:
    """Piecewise polynomials continuous in u, alpha u_x and their time derivatives.

    Region ``r`` lies between ``jumps[r - 1]`` and ``jumps[r]`` (``jumps``
    increasing). Region ``anchor`` carries the standard monomials, as ``U_R``
    does in eq. 74; every other region's basis comes from translating across
    the jumps in between, re-centring at each one. Which region is anchored
    is arbitrary (§4.1): the span, and so the stencil weights, do not depend
    on it.
    """
    if not jumps:
        raise ValueError("at least one jump is needed")
    degree = jumps[0].degree
    m = len(jumps)
    if not 0 <= anchor <= m:
        raise ValueError(f"anchor {anchor} is not one of the {m + 1} regions")
    monomials = np.eye(degree + 1)
    regions = {
        anchor: Region(anchor - 1, "right", monomials)
        if anchor > 0
        else Region(0, "left", monomials)
    }

    def carried_to(region: Region, k: int) -> np.ndarray:
        """This region's basis re-centred at ``jumps[k]``."""
        delta = jumps[k].position - jumps[region.jump].position
        if delta == 0.0:
            return region.coefficients
        return shift_matrix(delta, degree) @ region.coefficients

    for r in range(anchor, m):
        across = translation_matrix(jumps[r].left, jumps[r].right)
        regions[r + 1] = Region(r, "right", across @ carried_to(regions[r], r))
    for r in range(anchor, 0, -1):
        across = translation_matrix(jumps[r - 1].right, jumps[r - 1].left)
        regions[r - 1] = Region(r - 1, "left", across @ carried_to(regions[r], r - 1))
    return [regions[r] for r in range(m + 1)]


def region_index(jumps: Sequence[Jump], x: np.ndarray) -> np.ndarray:
    """Which region each ``x`` lies in; a point on a jump counts as its right side.

    Every basis function is continuous across a jump, and so is ``D`` applied
    to it (the ``(u, 1)`` condition), so the side chosen for such a point
    changes nothing but rounding.
    """
    positions = np.array([jump.position for jump in jumps])
    return np.searchsorted(positions, np.atleast_1d(x), side="right")


def stencil_weights(
    jumps: Sequence[Jump], nodes: np.ndarray, centre: float
) -> np.ndarray:
    """Weights approximating ``D u`` at ``centre`` from ``u`` at ``nodes``.

    The moment conditions ``sum_i w_i phi_j(x_i) = (D phi_j)(centre)`` on the
    translated basis (EABE p. 16, dissertation p. 74): the stencil reproduces
    the operator on every piecewise polynomial that satisfies the interface
    conditions, which is what Fig. 4-3 / EABE Fig. 4 plot.
    """
    nodes = np.asarray(nodes, dtype=float)
    degree = jumps[0].degree
    if nodes.size != degree + 1:
        raise ValueError(f"{degree + 1} nodes are needed for degree {degree}")
    anchor = int(region_index(jumps, centre)[0])
    regions = translated_basis(jumps, anchor)
    which = region_index(jumps, nodes)
    phi = np.empty((nodes.size, degree + 1))
    for r, region in enumerate(regions):
        mask = which == r
        if mask.any():
            phi[mask] = region.evaluate(jumps, nodes[mask])
    d = regions[anchor].operator_values(jumps, centre)[0]
    return np.linalg.solve(phi.T, d)
