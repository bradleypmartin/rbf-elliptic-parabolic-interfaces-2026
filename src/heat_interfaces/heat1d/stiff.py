"""Seed stencils for ``(alpha u_x)_x`` through a sub-grid smooth edge (E3.4, #29).

``docs/stiff-diffusion.md`` §1.2–1.4. A stencil whose window sees an edge of
width δ below the node spacing replaces the monomials by *seeds*, the t = 0
profiles of the solutions polynomial in time, anchored at the evaluation
point ``x_e`` with ``alpha_e = alpha(x_e)``:

    phi_0 = 1;   alpha phi_1' = alpha_e;
    L phi_k = k (k - 1) alpha_e phi_{k-2},   phi_k(x_e) = phi_k'(x_e) = 0.

Constant alpha gives the monomials ``(x - x_e)^k`` and a jump the translated
basis of dissertation eq. 74 (``interface.translated_basis``), so the seeds
are one construction with two implementations: E1.2's algebra for a jump and
the ODE march here for a smooth edge, which handles the jump too (§1.4). The
weights reproduce the operator on the seeds, ``sum_i w_i phi_k(x_i) =
(L phi_k)(x_e) = 2 alpha_e delta_{k2}``: the moment conditions of
``interface.stencil_weights`` on a different basis.

The chain is marched as one first-order system in ``(phi_k, psi_k = alpha
phi_k')`` (§1.3) so alpha is only ever evaluated, never differentiated, in
the stencil coordinate ``xi = (x - x_e) / h_s`` in which the chain is
invariant and ``phi_k ~ xi^k``: the interpolation matrix ``A[k, i] =
phi_k(xi_i)`` is a perturbed Vandermonde matrix (condition number 23.5 for
the centred five-point window at constant alpha) and the weights are
``w = A^{-1} (2 alpha_e e_2) / h_s^2``. DOP853 at ``SEED_RTOL`` / ``SEED_ATOL``
with a fresh segment at every edge centre and at ``EDGE_STOP`` δ on either
side of it, so the adaptive step never has to find an edge inside a long
step over flat material; on each segment alpha comes from the piece of the
medium's element that contains it (``Medium1D.elements``), which is what
makes the evaluation one-sided at a jump. Every row of an operator whose
window shares the pattern ``xi_i`` marches in one batch: the rows' stops
union to a handful per side, so an operator costs a few solves whatever
the count of seeded rows (§2.3 has the timings).

Which rows are seeded: those whose window span overlaps ``(x_c - reach δ,
x_c + reach δ)`` about an edge centre, ``reach = TANH_REACH`` by default,
beyond which ``SmoothEdges.alpha`` is the piece bit for bit; at δ = 0 that
is exactly ``operators.straddling_windows``. Both edges of a thin layer
inside one window are stops of the same march (the double-cross, §1.3).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from .domain import TANH_REACH, Grid1D, Medium1D
from .operators import (
    STENCIL_WIDTH,
    direct_operator,
    jump_aware_operator,
    naive_operator,
)

EDGE_STOP = 10.0
"""The march restarts at ``x_c ± EDGE_STOP δ`` about every edge centre, and at it.

Ten widths out the transition is ``e^{-20}`` from the piece: the segment from
the flank to the centre holds the whole steep part, and the segments beyond
are flat material the step control crosses in a few steps (the companion's
choice; ``TANH_REACH`` = 20 is where alpha is the piece to the bit, and is
the reach of the seeded rows, not of the stops).
"""

SEED_RTOL, SEED_ATOL = 1e-13, 1e-15
"""DOP853's tolerances for the chain in the stencil coordinate.

The seeds are O(1) there, so the absolute tolerance is a relative one, and
the chain is not stiff in the ODE sense (a nilpotent chain of integrations:
on constant alpha DOP853 reproduces the monomials to rounding). Tightening
``rtol`` to SciPy's floor of 100 eps gains nothing measurable: the row
residual on the true equilibrium is rounding (3e-16 in units of h⁻²) for
δ ≳ h/10 and reaches 1e-12 at δ = h/200, where the march crosses the edge in
many small steps, either way (§2.3).
"""

MERGE_TOL = 1e-12
"""Stops closer than this, in stencil units, to a node or to each other are one stop."""

Mode = Literal["naive", "direct", "jump-aware", "seeds"]


def edge_width(medium: Medium1D) -> float:
    """The edge width δ of a ``SmoothEdges``; 0 for any other medium (a jump)."""
    return float(getattr(medium, "delta", 0.0))


def seeded_windows(
    grid: Grid1D,
    medium: Medium1D,
    width: int = STENCIL_WIDTH,
    reach: float = TANH_REACH,
) -> list[tuple[int, int, list[int]]]:
    """``(row, lo, centres)`` for every row whose window sees an edge.

    A window sees edge ``k`` when its span ``[x_lo, x_hi]`` overlaps the open
    interval ``(x_c - reach δ, x_c + reach δ)``; with δ = 0 that is a window
    with nodes strictly on both sides of ``x_c``, ``straddling_windows``.
    """
    x = grid.snapped(medium.interfaces)
    n = grid.n
    if n < width:
        raise ValueError("fewer nodes than the stencil width")
    lo = np.clip(np.arange(n) - width // 2, 0, n - width)
    span_lo, span_hi = x[lo][:, None], x[lo + width - 1][:, None]
    xc = np.asarray(medium.interfaces, dtype=float)[None, :]
    half = reach * edge_width(medium)
    seen = (span_lo < xc + half) & (xc - half < span_hi)
    return [
        (int(i), int(lo[i]), [int(k) for k in np.flatnonzero(seen[i])])
        for i in np.flatnonzero(seen.any(axis=1))
    ]


def _stops(x_e: np.ndarray, h_s: np.ndarray, medium: Medium1D) -> np.ndarray:
    """Every row's segment boundaries in its own stencil coordinate, flattened."""
    delta = edge_width(medium)
    centres = np.asarray(medium.interfaces, dtype=float)
    flanks = np.array([-EDGE_STOP, 0.0, EDGE_STOP]) * delta if delta > 0 else [0.0]
    stops = (centres[:, None] + np.asarray(flanks)[None, :]).ravel()
    return ((stops[None, :] - x_e[:, None]) / h_s[:, None]).ravel()


def march_targets(ahead: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """The node positions ``ahead`` (exact) and the stops among them, increasing.

    A stop within ``MERGE_TOL`` of a node, or of a stop already kept, is
    dropped: a restart point may move by that much, a node may not. The 2-D
    march (``heat2d.seeds``) merges its node and edge targets the same way.
    """
    kept = list(np.unique(ahead))
    for s in np.sort(stops):
        if all(abs(s - k) > MERGE_TOL for k in kept):
            kept.append(float(s))
    return np.array(sorted(kept))


def _piece_groups(
    medium: Medium1D, x_mid: np.ndarray
) -> list[tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]]:
    """``(alpha, rows)`` per distinct piece among the elements holding ``x_mid``."""
    edges, pieces = medium.elements()
    idx = np.searchsorted(edges[1:-1], x_mid, side="right")
    groups: dict[int, tuple[Callable, list[int]]] = {}
    for k in np.unique(idx):
        groups.setdefault(id(pieces[k]), (pieces[k].alpha, []))[1].append(int(k))
    return [(alpha, np.isin(idx, ks)) for alpha, ks in groups.values()]


def _chain(
    x_e: np.ndarray,
    h_s: np.ndarray,
    alpha_e: np.ndarray,
    coef: np.ndarray,
    groups: list[tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]],
) -> Callable[[float, np.ndarray], np.ndarray]:
    """The first-order chain of §1.3 on one segment, for every row of a batch.

    ``y`` is ``(rows, 2, count)`` flattened: ``phi_k' = psi_k / alpha`` and
    ``psi_k' = k (k - 1) alpha_e phi_{k-2}`` with alpha read from each row's
    piece for this segment (``groups``).
    """
    r, count = x_e.size, coef.size + 2

    def rate(t: float, y: np.ndarray) -> np.ndarray:
        y = y.reshape(r, 2, count)
        x = x_e + h_s * t
        a = np.empty(r)
        for alpha, rows in groups:
            a[rows] = alpha(x[rows])
        d = np.empty_like(y)
        d[:, 0] = y[:, 1] / a[:, None]
        d[:, 1, :2] = 0.0
        d[:, 1, 2:] = coef * alpha_e[:, None] * y[:, 0, :-2]
        return d.reshape(-1)

    return rate


def seed_profiles(
    x_e: np.ndarray,
    h_s: np.ndarray,
    xi: np.ndarray,
    medium: Medium1D,
    count: int = STENCIL_WIDTH,
    rtol: float = SEED_RTOL,
    atol: float = SEED_ATOL,
) -> np.ndarray:
    """``phi[r, k, i] = phi_k(xi_i)`` for the stencil at ``x_e[r]``, scale ``h_s[r]``.

    The batched march of §1.3: ``count`` seeds per row, all rows sharing the
    node pattern ``xi`` (positions in stencil units, 0 the anchor if it is a
    node), every row's stops folded into one segment list per side.
    """
    x_e = np.atleast_1d(np.asarray(x_e, dtype=float))
    h_s = np.broadcast_to(np.asarray(h_s, dtype=float), x_e.shape)
    xi = np.asarray(xi, dtype=float)
    r = x_e.size
    alpha_e = np.atleast_1d(np.asarray(medium.alpha(x_e), dtype=float))
    k = np.arange(count)
    coef = (k * (k - 1)).astype(float)[2:]
    y0 = np.zeros((r, 2, count))
    y0[:, 0, 0] = 1.0
    y0[:, 1, 1] = alpha_e
    stops = _stops(x_e, h_s, medium)
    out = np.empty((r, count, xi.size))
    out[:, :, xi == 0.0] = y0[:, 0][:, :, None]
    for side in (-1.0, 1.0):
        ahead = side * xi > 0.0
        if not ahead.any():
            continue
        far = np.max(side * xi[ahead])
        inside = stops[(side * stops > 0.0) & (side * stops < far)]
        y, t0 = y0.reshape(-1), 0.0
        for t in march_targets(side * xi[ahead], side * inside):
            t1 = side * t
            groups = _piece_groups(medium, x_e + h_s * (0.5 * (t0 + t1)))
            rate = _chain(x_e, h_s, alpha_e, coef, groups)
            sol = solve_ivp(rate, (t0, t1), y, method="DOP853", rtol=rtol, atol=atol)
            if not sol.success:
                raise RuntimeError(f"the seed march failed at xi = {t1}: {sol.message}")
            y, t0 = sol.y[:, -1], t1
            hit = np.isclose(xi, t1, rtol=0.0, atol=MERGE_TOL)
            out[:, :, hit] = y.reshape(r, 2, count)[:, 0][:, :, None]
    return out


def seed_basis(
    nodes: np.ndarray,
    centre: float,
    medium: Medium1D,
    rtol: float = SEED_RTOL,
    atol: float = SEED_ATOL,
) -> tuple[np.ndarray, float]:
    """One stencil's ``A[k, i] = phi_k(xi_i)`` in stencil units, and its ``h_s``.

    ``h_s = max |x_i - centre|`` (the companion's half-width normalisation),
    so ``xi_i ∈ [-1, 1]``; ``centre`` need not be a node.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.size < 3:
        raise ValueError("a second derivative needs at least three nodes")
    h_s = float(np.max(np.abs(nodes - centre)))
    xi = (nodes - centre) / h_s
    phi = seed_profiles([centre], [h_s], xi, medium, nodes.size, rtol, atol)
    return phi[0], h_s


def seed_weights(
    nodes: np.ndarray,
    centre: float,
    medium: Medium1D,
    rtol: float = SEED_RTOL,
    atol: float = SEED_ATOL,
) -> np.ndarray:
    """Weights approximating ``(alpha u_x)_x`` at ``centre`` from ``u`` at ``nodes``.

    The same contract as ``interface.stencil_weights``; ``A w̃ = 2 alpha_e e_2``
    and ``w = w̃ / h_s²`` (§1.3).
    """
    a, h_s = seed_basis(nodes, centre, medium, rtol, atol)
    b = np.zeros(a.shape[0])
    b[2] = 2.0 * float(np.atleast_1d(medium.alpha(np.array([centre])))[0])
    return np.linalg.solve(a, b) / h_s**2


def seed_operator(
    grid: Grid1D,
    medium: Medium1D,
    degree: int = 4,
    reach: float = TANH_REACH,
    rtol: float = SEED_RTOL,
    atol: float = SEED_ATOL,
) -> sp.csr_array:
    """The seed operator: direct rows, seed rows on the windows that see an edge.

    The same shape as ``jump_aware_operator`` (``degree + 1`` nodes per row)
    with ``seeded_windows(reach)`` in place of the straddling ones; at δ = 0
    the two agree to rounding on constant pieces and differ at relative
    O(h alpha'/alpha) on smoothly varying ones (§1.4). Rows are batched by
    their node pattern, one march per pattern.
    """
    width = degree + 1
    x = grid.snapped(medium.interfaces)
    op = direct_operator(grid, medium, width).tolil()
    windows = seeded_windows(grid, medium, width, reach)
    if not windows:
        return op.tocsr()
    rows = np.array([i for i, _, _ in windows])
    cols = np.array([lo for _, lo, _ in windows])[:, None] + np.arange(width)[None, :]
    centre = x[rows]
    nodes = x[cols]
    h_s = np.max(np.abs(nodes - centre[:, None]), axis=1)
    xi = (nodes - centre[:, None]) / h_s[:, None]
    alpha_e = np.asarray(medium.alpha(centre), dtype=float)
    patterns = {}
    for r, row in enumerate(np.round(xi, 12)):
        patterns.setdefault(tuple(row), []).append(r)
    for members in patterns.values():
        sel = np.array(members)
        phi = seed_profiles(
            centre[sel], h_s[sel], xi[sel[0]], medium, width, rtol, atol
        )
        b = np.zeros((sel.size, width, 1))
        b[:, 2, 0] = 2.0 * alpha_e[sel]
        w = np.linalg.solve(phi, b)[:, :, 0] / h_s[sel, None] ** 2
        for r, weights in zip(sel, w, strict=True):
            op[rows[r], :] = 0.0
            op[rows[r], cols[r]] = weights
    return op.tocsr()


OPERATOR_MODES: dict[str, Callable[[Grid1D, Medium1D], sp.csr_array]] = {
    "naive": naive_operator,
    "direct": direct_operator,
    "jump-aware": jump_aware_operator,
    "seeds": seed_operator,
}
"""The 1-D operators by name: eq. 76, the blind stencil, §4.1's rows, the seeds."""


def build_operator(
    grid: Grid1D, medium: Medium1D, mode: Mode = "seeds"
) -> sp.csr_array:
    """``OPERATOR_MODES[mode](grid, medium)``: the dispatch the plan's E3.4 names."""
    try:
        build = OPERATOR_MODES[mode]
    except KeyError:
        raise ValueError(
            f"unknown operator {mode!r}; one of {sorted(OPERATOR_MODES)}"
        ) from None
    return build(grid, medium)


__all__: Sequence[str] = (
    "EDGE_STOP",
    "MERGE_TOL",
    "OPERATOR_MODES",
    "SEED_ATOL",
    "SEED_RTOL",
    "build_operator",
    "edge_width",
    "march_targets",
    "seed_basis",
    "seed_operator",
    "seed_profiles",
    "seed_weights",
    "seeded_windows",
)
