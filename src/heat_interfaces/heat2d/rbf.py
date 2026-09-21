"""Gaussian RBF-FD weights with polynomial augmentation, batched over stencils.

Dissertation §3.2 / §5.2 and EABE §2.1, eq. 2: the weights ``w`` of a linear
operator ``L`` at a stencil's centre solve the saddle-point system

    [ A   P ] [ w ]   [ L φ_j(x_c) ]
    [ P^T 0 ] [ λ ] = [ L p_k(x_c) ]

with ``A_ij = φ(|x_i - x_j|)`` the Gaussian ``exp(-(ε r)^2)`` centred at each
node and ``P_ik = p_k(x_i)`` the bivariate monomials through a given degree.
The shape parameter is ``ε = 0.4 / d`` with ``d`` the distance from the centre
to its nearest neighbour (EABE eq. 31, dissertation eq. 45), the MATLAB's
``exp(-shp² r² / dmin²)``. Every stencil is solved in local coordinates
scaled by its radius, as the MATLAB scaled its polynomial block by the
distance to the farthest node (``normfactor``), so the systems are as well
conditioned at 20,000 nodes as at 1250; the weights are scaled back by the
radius to the power of the derivative order.

The Laplacian of a Gaussian is written in closed form rather than through
the Laguerre recursion the MATLAB used for its hyperviscosity powers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GA_SHAPE = 0.4
"""Numerator of ``ε = GA_SHAPE / d`` (EABE eq. 31)."""

OPERATORS = ("dx", "dy", "dxx", "dxy", "dyy", "lap")
"""The operators a stencil can approximate; ``lap`` is ``dxx + dyy``."""

ORDER = {"dx": 1, "dy": 1, "dxx": 2, "dxy": 2, "dyy": 2, "lap": 2}
"""Derivative order of each operator, the power of the radius in the scaling."""

BATCH = 4096
"""Stencils solved per ``np.linalg.solve`` call, to bound the working memory."""

COINCIDENCE = 1e-8
"""Two nodes closer than this fraction of the stencil radius count as one.

Below it the Gaussian matrix has two rows equal to rounding and the solve
returns a large, meaningless weight row without raising; the repulsion node
sets keep pairs at 0.8 spacings or more, so only a malformed stencil trips it.
"""


@dataclass(frozen=True)
class StencilSpec:
    """``size`` nodes (the centre included) with polynomials through ``degree``.

    EABE §3: 42 / 5 away from interfaces and boundaries, 30 / 4 across
    interfaces and near boundaries, 19 / 3 in the iterative-solver study.
    """

    size: int
    degree: int

    def __post_init__(self) -> None:
        if self.size < polynomial_count(self.degree):
            raise ValueError(
                f"{self.size} nodes cannot carry the {polynomial_count(self.degree)} "
                f"monomials of degree {self.degree}"
            )


def polynomial_count(degree: int) -> int:
    """``(d + 1)(d + 2) / 2`` bivariate monomials through degree ``d``."""
    return (degree + 1) * (degree + 2) // 2


INTERIOR = StencilSpec(42, 5)
BOUNDARY = StencilSpec(30, 4)
ITERATIVE = StencilSpec(19, 3)


def polynomial_exponents(degree: int) -> np.ndarray:
    """``(i, j)`` of ``x^i y^j`` for ``i + j <= degree``, graded, ``x`` powers first.

    ``1, x, y, x², xy, y², …``: the MATLAB's ``augLwithpolys1`` order.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    pairs = [(t - j, j) for t in range(degree + 1) for j in range(t + 1)]
    return np.array(pairs, dtype=int).reshape(-1, 2)


def polynomial_block(xi: np.ndarray, eta: np.ndarray, degree: int) -> np.ndarray:
    """``P[..., k, p] = xi_k^i eta_k^j``, the monomials of ``polynomial_exponents``."""
    e = polynomial_exponents(degree)
    return xi[..., None] ** e[:, 0] * eta[..., None] ** e[:, 1]


def polynomial_rhs(degree: int, ops: tuple[str, ...]) -> np.ndarray:
    """``L p_k`` at the origin for every monomial, one column per operator.

    Only the monomial that ``L`` maps to a constant survives: ``x`` for
    ``dx``, ``x²`` (giving 2) for ``dxx``, both ``x²`` and ``y²`` for ``lap``.
    """
    e = polynomial_exponents(degree)
    out = np.zeros((len(e), len(ops)))
    for c, op in enumerate(ops):
        _check_operator(op)
        for (i, j), value in _POLY_AT_ORIGIN[op]:
            hit = np.flatnonzero((e[:, 0] == i) & (e[:, 1] == j))
            out[hit, c] = value
    return out


_POLY_AT_ORIGIN = {
    "dx": (((1, 0), 1.0),),
    "dy": (((0, 1), 1.0),),
    "dxx": (((2, 0), 2.0),),
    "dxy": (((1, 1), 1.0),),
    "dyy": (((0, 2), 2.0),),
    "lap": (((2, 0), 2.0), ((0, 2), 2.0)),
}


def _check_operator(op: str) -> None:
    if op not in OPERATORS:
        raise ValueError(f"unknown operator {op!r}; the operators are {OPERATORS}")


def gaussian(dx: np.ndarray, dy: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """``exp(-ε² (dx² + dy²))`` with ``eps`` broadcast over the leading axes."""
    return np.exp(-(eps**2) * (dx**2 + dy**2))


def gaussian_derivative(
    dx: np.ndarray, dy: np.ndarray, eps: np.ndarray, op: str
) -> np.ndarray:
    """``L φ_j`` at the centre for a Gaussian centred at offset ``(dx, dy)``.

    ``φ_j(x) = exp(-ε² |x - x_j|²)`` evaluated at ``x = x_c``, with
    ``(dx, dy) = x_j - x_c``; the odd derivatives change sign with the offset
    convention, which is why they read ``+2ε² dx φ`` here.
    """
    _check_operator(op)
    phi = gaussian(dx, dy, eps)
    e2 = eps**2
    if op == "dx":
        return 2.0 * e2 * dx * phi
    if op == "dy":
        return 2.0 * e2 * dy * phi
    if op == "dxx":
        return (4.0 * e2**2 * dx**2 - 2.0 * e2) * phi
    if op == "dyy":
        return (4.0 * e2**2 * dy**2 - 2.0 * e2) * phi
    if op == "dxy":
        return 4.0 * e2**2 * dx * dy * phi
    return (4.0 * e2**2 * (dx**2 + dy**2) - 4.0 * e2) * phi


def augmented_solve(
    a: np.ndarray, p: np.ndarray, b_rbf: np.ndarray, b_poly: np.ndarray
) -> np.ndarray:
    """Weights of the saddle-point system for a batch of stencils.

    ``a`` is ``(m, k, k)``, ``p`` is ``(m, k, q)``, ``b_rbf`` ``(m, k, c)`` and
    ``b_poly`` ``(m, q, c)`` (or ``(q, c)``, shared); the result is the
    ``(m, k, c)`` block of node weights, the multipliers discarded. E2.3
    passes its translated polynomial block through here.
    """
    m, k, q = p.shape
    c = b_rbf.shape[-1]
    lhs = np.zeros((m, k + q, k + q))
    lhs[:, :k, :k] = a
    lhs[:, :k, k:] = p
    lhs[:, k:, :k] = np.swapaxes(p, 1, 2)
    rhs = np.empty((m, k + q, c))
    rhs[:, :k, :] = b_rbf
    rhs[:, k:, :] = b_poly
    return np.linalg.solve(lhs, rhs)[:, :k, :]


def check_coincidence(
    xi: np.ndarray, eta: np.ndarray, limit: float = COINCIDENCE
) -> None:
    """Raise if any two nodes of a stencil are closer than ``limit`` (scaled units).

    ``xi, eta`` are ``(m, k)`` or ``(k,)`` node coordinates in units of the
    stencil radius. Shared by ``rbf_fd_weights`` and the interface stencils
    of E2.3, which build their own Gaussian block.
    """
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    eta = np.atleast_2d(np.asarray(eta, dtype=float))
    k = xi.shape[-1]
    dxi = xi[:, :, None] - xi[:, None, :]
    deta = eta[:, :, None] - eta[:, None, :]
    closest = (np.hypot(dxi, deta) + np.eye(k)).min(axis=(1, 2))
    if np.any(closest < limit):
        raise ValueError(
            "two nodes of a stencil coincide or nearly so: the closest pair is "
            f"{closest.min():.1e} stencil radii apart"
        )


def rbf_fd_weights(
    dx: np.ndarray,
    dy: np.ndarray,
    ops: tuple[str, ...],
    degree: int,
    shape: float = GA_SHAPE,
) -> np.ndarray:
    """Weights ``(len(ops), m, k)`` for ``m`` stencils given as offsets ``(m, k)``.

    ``dx, dy`` are neighbour minus centre (``neighbors.offsets``), the centre
    itself among them at offset 0. Each stencil is scaled by its radius
    ``R = max r``, its shape parameter is ``shape / d`` with ``d`` the smallest
    nonzero offset, and the weights of an order-``n`` operator come back
    divided by ``R^n``. Applied to samples of any polynomial through
    ``degree`` they return its exact derivative at the centre.
    """
    dx = np.atleast_2d(np.asarray(dx, dtype=float))
    dy = np.atleast_2d(np.asarray(dy, dtype=float))
    if dx.shape != dy.shape:
        raise ValueError("dx and dy must have the same shape")
    m, k = dx.shape
    if k < polynomial_count(degree):
        raise ValueError(
            f"{k} nodes cannot carry the {polynomial_count(degree)} monomials "
            f"of degree {degree}"
        )
    for op in ops:
        _check_operator(op)
    r = np.hypot(dx, dy)
    radius = r.max(axis=1)
    nearest = np.where(r > 0.0, r, np.inf).min(axis=1)
    if not np.all(np.isfinite(nearest)) or np.any(radius <= 0.0):
        raise ValueError("a stencil has no node away from its centre")
    out = np.empty((len(ops), m, k))
    b_poly = polynomial_rhs(degree, ops)
    for lo in range(0, m, BATCH):
        sl = slice(lo, min(lo + BATCH, m))
        xi = dx[sl] / radius[sl, None]
        eta = dy[sl] / radius[sl, None]
        eps = (shape * radius[sl] / nearest[sl])[:, None]
        check_coincidence(xi, eta)
        dxi = xi[:, :, None] - xi[:, None, :]
        deta = eta[:, :, None] - eta[:, None, :]
        a = gaussian(dxi, deta, eps[..., None])
        p = polynomial_block(xi, eta, degree)
        b_rbf = np.stack([gaussian_derivative(xi, eta, eps, op) for op in ops], axis=-1)
        w = augmented_solve(a, p, b_rbf, b_poly)
        for c, op in enumerate(ops):
            out[c, sl] = w[:, :, c] / radius[sl, None] ** ORDER[op]
    return out
