"""Reference solutions for the 1-D problems (plan D4).

Equilibrium: ``(alpha u')' = 0`` with ``u(-1) = u_L`` and ``u(1) = u_R`` has
constant flux ``alpha u' = B``, so ``u(x) = u_L + B F(x)`` with
``F(x) = ∫_{-1}^{x} dξ / alpha(ξ)`` and ``B = (u_R - u_L) / F(1)``. ``F`` is
integrated by Gauss–Legendre on panels split at the interfaces, so a jump
costs nothing and each smooth piece converges spectrally in the node count
and at order ``2 n_gauss`` in the panel width. This is the 6400-node
reference of dissertation Fig. 4-6 replaced by something exact to rounding.

Parabolic: Chebyshev collocation, one Gauss–Lobatto grid per material piece
with ``u`` and ``alpha u_x`` matched at every interface, integrated in time by
Radau at a tight tolerance after the constrained unknowns (the piece ends)
are eliminated. The same assembly solves ``(alpha v')' = c v``, the profile
of the separable solution ``e^{ct} v(x)`` that plays the part of dissertation
eq. 86 in one dimension.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .domain import X_MAX, X_MIN, Medium1D, PiecewiseAlpha


def inverse_alpha_integral(
    medium: Medium1D, x: np.ndarray, n_gauss: int = 24, n_panels: int = 1
) -> np.ndarray:
    """``F(x) = ∫_{-1}^{x} dξ / alpha(ξ)`` at every point of ``x`` (any shape).

    [-1, 1] is cut at the interfaces and at the requested points; every cut
    interval is split into ``n_panels`` equal panels carrying ``n_gauss``
    Gauss–Legendre nodes. Nothing is snapped: a panel between a point and a
    nearby interface, however thin, is integrated with the piece it lies in,
    and interface values of alpha are never sampled.
    """
    x = np.asarray(x, dtype=float)
    if x.size and (x.min() < X_MIN or x.max() > X_MAX):
        raise ValueError("evaluation points must lie in [-1, 1]")
    cuts = np.unique(np.concatenate([[X_MIN, X_MAX], medium.interfaces, x.ravel()]))
    a, b = cuts[:-1], cuts[1:]
    fractions = np.linspace(0.0, 1.0, n_panels + 1)
    edges = a[:, None] + (b - a)[:, None] * fractions[None, :]
    lo, hi = edges[:, :-1].ravel(), edges[:, 1:].ravel()
    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    samples = mid[:, None] + half[:, None] * nodes[None, :]
    integrand = 1.0 / medium.alpha(samples.ravel()).reshape(samples.shape)
    per_panel = half * (integrand @ weights)
    at_edges = np.concatenate([[0.0], np.cumsum(per_panel)])
    at_cuts = at_edges[::n_panels]
    return at_cuts[np.searchsorted(cuts, x.ravel())].reshape(x.shape)


def equilibrium_flux(
    medium: Medium1D, u_left: float, u_right: float, n_gauss: int = 24
) -> float:
    """The constant ``B = alpha u'`` of the equilibrium solution."""
    total = float(inverse_alpha_integral(medium, np.array(X_MAX), n_gauss))
    return (u_right - u_left) / total


def equilibrium_exact(
    medium: Medium1D,
    u_left: float,
    u_right: float,
    x: np.ndarray,
    n_gauss: int = 24,
    n_panels: int = 1,
) -> np.ndarray:
    """``u(x) = u_L + B F(x)`` at the points ``x``."""
    x = np.asarray(x, dtype=float)
    f = inverse_alpha_integral(
        medium, np.concatenate([x.ravel(), [X_MAX]]), n_gauss, n_panels
    )
    b = (u_right - u_left) / f[-1]
    return (u_left + b * f[:-1]).reshape(x.shape)


def chebyshev_lobatto(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n + 1`` Gauss–Lobatto nodes on [-1, 1], ascending, ``D``, barycentric weights.

    Trefethen, *Spectral Methods in MATLAB*, program ``cheb``, with the nodes
    reversed; the sign pattern of the weights is a common factor that the
    ratios in ``D`` and in the barycentric formula do not see.
    """
    if n < 2:
        raise ValueError("a piece needs at least three nodes")
    k = np.arange(n + 1)
    x = -np.cos(np.pi * k / n)
    c = np.where((k == 0) | (k == n), 2.0, 1.0) * (-1.0) ** k
    dx = x[:, None] - x[None, :] + np.eye(n + 1)
    d = np.outer(c, 1.0 / c) / dx
    d -= np.diag(d.sum(axis=1))
    return x, d, 1.0 / c


def _barycentric(
    nodes: np.ndarray, weights: np.ndarray, values: np.ndarray, x: np.ndarray
) -> np.ndarray:
    d = x[:, None] - nodes[None, :]
    hit = d == 0.0
    d[hit] = 1.0
    terms = weights[None, :] / d
    out = (terms @ values) / terms.sum(axis=1)
    rows = hit.any(axis=1)
    out[rows] = values[np.argmax(hit[rows], axis=1)]
    return out


@dataclass(frozen=True)
class ChebyshevPieces:
    """Piecewise Chebyshev collocation of ``(alpha u')'`` on a ``PiecewiseAlpha``.

    ``x`` holds every piece's nodes in turn (an interface appears twice, once
    as the end of a piece and once as the start of the next); ``operator`` is
    the block-diagonal ``alpha D² + alpha' D`` with each piece's own alpha;
    ``constraints`` are the ``2 p`` rows ``M u = g``: the Dirichlet ends and,
    per interface, ``u_L - u_R = 0`` and ``alpha_L u_L' - alpha_R u_R' = 0``.
    ``constrained`` indexes the piece ends, one unknown per row.
    """

    medium: PiecewiseAlpha
    n_cheb: int
    x: np.ndarray
    operator: np.ndarray
    constraints: np.ndarray
    constrained: np.ndarray
    interior: np.ndarray

    @classmethod
    def build(cls, medium: PiecewiseAlpha, n_cheb: int = 48) -> ChebyshevPieces:
        edges = np.concatenate([[X_MIN], medium.interfaces, [X_MAX]])
        xi, d, _ = chebyshev_lobatto(n_cheb)
        width = n_cheb + 1
        pieces = len(medium.pieces)
        size = pieces * width
        x = np.empty(size)
        operator = np.zeros((size, size))
        derivative = np.zeros((size, size))
        for k, piece in enumerate(medium.pieces):
            a, b = edges[k], edges[k + 1]
            rows = slice(k * width, (k + 1) * width)
            xk = (a + b) / 2 + (b - a) / 2 * xi
            dk = 2.0 / (b - a) * d
            x[rows] = xk
            derivative[rows, rows] = dk
            operator[rows, rows] = (
                piece.alpha(xk)[:, None] * (dk @ dk) + piece.alpha_x(xk)[:, None] * dk
            )
        starts = np.arange(pieces) * width
        ends = starts + n_cheb
        constrained = np.sort(np.concatenate([starts, ends]))
        constraints = np.zeros((2 * pieces, size))
        constraints[0, starts[0]] = 1.0
        for k, xk in enumerate(medium.interfaces):
            left, right = medium.pieces[k], medium.pieces[k + 1]
            xk = np.asarray(xk, dtype=float)
            constraints[2 * k + 1, ends[k]] = 1.0
            constraints[2 * k + 1, starts[k + 1]] = -1.0
            constraints[2 * k + 2] = (
                float(left.alpha(xk)) * derivative[ends[k]]
                - float(right.alpha(xk)) * derivative[starts[k + 1]]
            )
        constraints[-1, ends[-1]] = 1.0
        interior = np.setdiff1d(np.arange(size), constrained)
        return cls(medium, n_cheb, x, operator, constraints, constrained, interior)

    def constraint_values(self, u_left: float, u_right: float) -> np.ndarray:
        g = np.zeros(self.constraints.shape[0])
        g[0], g[-1] = u_left, u_right
        return g

    def evaluate(self, values: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Interpolate nodal ``values`` to ``x`` (any shape), piece by piece."""
        x = np.asarray(x, dtype=float)
        flat = x.ravel()
        _, _, weights = chebyshev_lobatto(self.n_cheb)
        width = self.n_cheb + 1
        idx = self.medium.piece_index(flat)
        out = np.empty_like(flat)
        for k in range(len(self.medium.pieces)):
            mask = idx == k
            if mask.any():
                rows = slice(k * width, (k + 1) * width)
                out[mask] = _barycentric(
                    self.x[rows], weights, values[rows], flat[mask]
                )
        return out.reshape(x.shape)


def chebyshev_equilibrium(
    medium: PiecewiseAlpha,
    u_left: float,
    u_right: float,
    x: np.ndarray,
    n_cheb: int = 48,
    shift: float = 0.0,
) -> np.ndarray:
    """``(alpha v')' = shift v`` with ``v(-1) = u_left``, ``v(1) = u_right``, at ``x``.

    ``shift = 0`` is the equilibrium problem (a cross-check of the
    quadrature reference); ``shift = c > 0`` gives the profile of the
    separable solution ``u = e^{ct} v(x)`` of the heat equation with
    ``u(-1, t) = u_left e^{ct}``, the 1-D twin of dissertation eq. 85–86.
    """
    cp = ChebyshevPieces.build(medium, n_cheb)
    size = cp.x.size
    matrix = cp.operator - shift * np.eye(size)
    rhs = np.zeros(size)
    matrix[cp.constrained] = cp.constraints
    rhs[cp.constrained] = cp.constraint_values(u_left, u_right)
    return cp.evaluate(np.linalg.solve(matrix, rhs), x)


def chebyshev_parabolic(
    medium: PiecewiseAlpha,
    initial: Callable[[np.ndarray], np.ndarray],
    boundary: Callable[[float], tuple[float, float]],
    t_end: float,
    x: np.ndarray,
    n_cheb: int = 48,
    rtol: float = 1e-12,
    atol: float = 1e-13,
) -> np.ndarray:
    """``u_t = (alpha u_x)_x`` from ``initial(x)`` to ``t_end``, evaluated at ``x``.

    The piece ends are eliminated through the constraints, ``u_C = M_C⁻¹ (g(t)
    - M_I u_I)``, leaving a dense linear ODE ``u_I' = A u_I + B g(t)`` that
    Radau integrates with its constant Jacobian. An initial condition that
    disagrees with ``g(0)`` at the ends is overridden there.
    """
    cp = ChebyshevPieces.build(medium, n_cheb)
    ii, cc = cp.interior, cp.constrained
    m_c_inv = np.linalg.inv(cp.constraints[:, cc])
    m_i = cp.constraints[:, ii]
    a = cp.operator[np.ix_(ii, ii)] - cp.operator[np.ix_(ii, cc)] @ m_c_inv @ m_i
    b = cp.operator[np.ix_(ii, cc)] @ m_c_inv

    def rate(t: float, u_i: np.ndarray) -> np.ndarray:
        return a @ u_i + b @ cp.constraint_values(*boundary(t))

    u_i0 = np.asarray(initial(cp.x[ii]), dtype=float)
    sol = solve_ivp(
        rate, (0.0, t_end), u_i0, method="Radau", jac=a, rtol=rtol, atol=atol
    )
    if not sol.success:
        raise RuntimeError(f"Radau failed: {sol.message}")
    u = np.empty(cp.x.size)
    u[ii] = sol.y[:, -1]
    u[cc] = m_c_inv @ (cp.constraint_values(*boundary(t_end)) - m_i @ u[ii])
    return cp.evaluate(u, x)
