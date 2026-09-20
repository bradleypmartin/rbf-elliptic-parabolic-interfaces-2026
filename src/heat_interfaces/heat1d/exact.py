"""Quadrature reference for the 1-D equilibrium problem (plan D4).

``(alpha u')' = 0`` with ``u(-1) = u_L`` and ``u(1) = u_R`` has constant flux
``alpha u' = B``, so ``u(x) = u_L + B F(x)`` with
``F(x) = ∫_{-1}^{x} dξ / alpha(ξ)`` and ``B = (u_R - u_L) / F(1)``. ``F`` is
integrated by Gauss–Legendre on panels split at the interfaces, so a jump
costs nothing and each smooth piece converges spectrally in the node count
and at order ``2 n_gauss`` in the panel width. This is the 6400-node
reference of dissertation Fig. 4-6 replaced by something exact to rounding.
"""

from __future__ import annotations

import numpy as np

from .domain import X_MAX, X_MIN, Medium1D


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
