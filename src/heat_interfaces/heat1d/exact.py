"""Reference solutions for the 1-D problems (plan D4).

Equilibrium: ``(alpha u')' = 0`` with ``u(-1) = u_L`` and ``u(1) = u_R`` has
constant flux ``alpha u' = B``, so ``u(x) = u_L + B F(x)`` with
``F(x) = ∫_{-1}^{x} dξ / alpha(ξ)`` and ``B = (u_R - u_L) / F(1)``. ``F`` is
integrated by Gauss–Legendre on panels split at the interfaces, so a jump
costs nothing and each smooth piece converges spectrally in the node count
and at order ``2 n_gauss`` in the panel width. This is the 6400-node
reference of dissertation Fig. 4-6 replaced by something exact to rounding.

Parabolic: Chebyshev collocation, one Gauss–Lobatto grid per element with
``u`` and ``alpha u_x`` matched at every element edge, integrated in time by
Radau at a tight tolerance after the constrained unknowns (the element ends)
are eliminated. The same assembly solves ``(alpha v')' = c v``, the profile
of the separable solution ``e^{ct} v(x)`` that plays the part of dissertation
eq. 86 in one dimension.

Both references cut [-1, 1] at the medium's ``elements`` (``Medium1D``): the
interfaces of a ``PiecewiseAlpha``, where the matching is the jump's, and for
a ``SmoothEdges`` also ``x_c ± EDGE_CUTS δ`` about every edge centre, where
the matching is plain C¹ continuity and the elements are what resolves a
transition of width δ ≪ 1 (E3.2, plan D4). A ``ParabolicReference`` keeps
the nodal solution with its metadata and is cached under ``outputs/`` by the
drivers through ``parabolic_reference``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .domain import X_MAX, X_MIN, Medium1D

RADAU_RTOL, RADAU_ATOL = 1e-9, 1e-11
"""Radau's tolerances for the parabolic reference.

Radau's Newton iteration on the linear system converges in one step, and
the second step's update is the round-off of the solve, about ``eps · h
‖A‖``; the iteration is declared failed when that exceeds the tolerance,
the step is cut, and on a sub-grid edge (``‖A‖`` above 1e10 with 32
Chebyshev nodes on elements of width δ = 0.0025) the march dies with
"required step size is less than spacing between numbers" at rtol 1e-12
and takes 10–100× the steps at 1e-10. At 1e-9 every δ of the E3 study runs
in about 400 steps, and the result agrees with the 1e-12 run to 1e-13
where that one completes (δ = 0): Radau's error on these smooth-in-time
solutions is far below its tolerance (E3.2, `docs/stiff-diffusion.md` §2.1).
"""


def inverse_alpha_integral(
    medium: Medium1D, x: np.ndarray, n_gauss: int = 24, n_panels: int = 1
) -> np.ndarray:
    """``F(x) = ∫_{-1}^{x} dξ / alpha(ξ)`` at every point of ``x`` (any shape).

    [-1, 1] is cut at the medium's element edges (the interfaces, and the
    ``EDGE_CUTS`` of a smooth edge) and at the requested points; every cut
    interval is split into ``n_panels`` equal panels carrying ``n_gauss``
    Gauss–Legendre nodes. Nothing is snapped: a panel between a point and a
    nearby interface, however thin, is integrated with the piece it lies in,
    and interface values of alpha are never sampled.
    """
    x = np.asarray(x, dtype=float)
    if x.size and (x.min() < X_MIN or x.max() > X_MAX):
        raise ValueError("evaluation points must lie in [-1, 1]")
    edges, _ = medium.elements()
    cuts = np.unique(np.concatenate([edges, x.ravel()]))
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


def _element_nodes(edges: np.ndarray, n_cheb: int) -> np.ndarray:
    """Every element's Gauss–Lobatto nodes in turn (a shared edge appears twice)."""
    xi, _, _ = chebyshev_lobatto(n_cheb)
    a, b = edges[:-1], edges[1:]
    return ((a + b)[:, None] / 2 + (b - a)[:, None] / 2 * xi[None, :]).ravel()


def _evaluate_elements(
    edges: np.ndarray, n_cheb: int, nodes: np.ndarray, values: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """Barycentric interpolation of nodal ``values`` to ``x``, element by element.

    A point on a shared edge is read from the element to its right; the
    matching makes the two readings equal to solver precision.
    """
    x = np.asarray(x, dtype=float)
    flat = x.ravel()
    _, _, weights = chebyshev_lobatto(n_cheb)
    width = n_cheb + 1
    idx = np.searchsorted(edges[1:-1], flat, side="right")
    out = np.empty_like(flat)
    for k in range(len(edges) - 1):
        mask = idx == k
        if mask.any():
            rows = slice(k * width, (k + 1) * width)
            out[mask] = _barycentric(nodes[rows], weights, values[rows], flat[mask])
    return out.reshape(x.shape)


def split_elements(
    edges: np.ndarray, pieces: tuple, max_width: float | None
) -> tuple[np.ndarray, tuple]:
    """``edges`` with every element wider than ``max_width`` cut into equal parts."""
    edges = np.asarray(edges, dtype=float)
    if max_width is None:
        return edges, tuple(pieces)
    out, out_pieces = [edges[0]], []
    for a, b, piece in zip(edges[:-1], edges[1:], pieces, strict=True):
        parts = max(1, int(np.ceil((b - a) / max_width - 1e-12)))
        # The original edges are kept to the bit; only the new cuts are computed.
        out.extend(a + (b - a) * np.arange(1, parts) / parts)
        out.append(b)
        out_pieces.extend([piece] * parts)
    return np.array(out), tuple(out_pieces)


@dataclass(frozen=True)
class ChebyshevPieces:
    """Chebyshev collocation of ``(alpha u')'`` on the elements of a ``Medium1D``.

    ``x`` holds every element's nodes in turn (an edge appears twice, once as
    the end of an element and once as the start of the next); ``operator`` is
    the block-diagonal ``alpha D² + alpha' D`` with each element's own piece;
    ``constraints`` are the ``2 m`` rows ``M u = g``: the Dirichlet ends and,
    per internal edge, ``u_L - u_R = 0`` and ``alpha_L u_L' - alpha_R u_R' =
    0`` with the one-sided values of the pieces (equal on a smooth edge's
    cuts, so the matching is C¹ there). ``constrained`` indexes the element
    ends, one unknown per row.
    """

    medium: Medium1D
    n_cheb: int
    edges: np.ndarray
    x: np.ndarray
    operator: np.ndarray
    constraints: np.ndarray
    constrained: np.ndarray
    interior: np.ndarray

    @classmethod
    def build(
        cls, medium: Medium1D, n_cheb: int = 48, max_width: float | None = None
    ) -> ChebyshevPieces:
        """The elements of ``medium``, any wider than ``max_width`` split equally.

        Splitting is a resolution choice, not a property of the material:
        32 nodes resolve eq. 75's half-unit sinusoid layer to 1e-8 but a
        quarter-unit piece of it to 1e-12, at a smaller cost than 48 nodes
        on the whole (E3.2).
        """
        edges, pieces = split_elements(*medium.elements(), max_width)
        _, d, _ = chebyshev_lobatto(n_cheb)
        width = n_cheb + 1
        count = len(pieces)
        size = count * width
        x = _element_nodes(edges, n_cheb)
        operator = np.zeros((size, size))
        derivative = np.zeros((size, size))
        for k, piece in enumerate(pieces):
            a, b = edges[k], edges[k + 1]
            rows = slice(k * width, (k + 1) * width)
            xk = x[rows]
            dk = 2.0 / (b - a) * d
            derivative[rows, rows] = dk
            operator[rows, rows] = (
                piece.alpha(xk)[:, None] * (dk @ dk) + piece.alpha_x(xk)[:, None] * dk
            )
        starts = np.arange(count) * width
        ends = starts + n_cheb
        constrained = np.sort(np.concatenate([starts, ends]))
        constraints = np.zeros((2 * count, size))
        constraints[0, starts[0]] = 1.0
        for k, xk in enumerate(edges[1:-1]):
            left, right = pieces[k], pieces[k + 1]
            xk = np.asarray(xk, dtype=float)
            constraints[2 * k + 1, ends[k]] = 1.0
            constraints[2 * k + 1, starts[k + 1]] = -1.0
            constraints[2 * k + 2] = (
                float(left.alpha(xk)) * derivative[ends[k]]
                - float(right.alpha(xk)) * derivative[starts[k + 1]]
            )
        constraints[-1, ends[-1]] = 1.0
        interior = np.setdiff1d(np.arange(size), constrained)
        return cls(
            medium, n_cheb, edges, x, operator, constraints, constrained, interior
        )

    def constraint_values(self, u_left: float, u_right: float) -> np.ndarray:
        g = np.zeros(self.constraints.shape[0])
        g[0], g[-1] = u_left, u_right
        return g

    def evaluate(self, values: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Interpolate nodal ``values`` to ``x`` (any shape), element by element."""
        return _evaluate_elements(self.edges, self.n_cheb, self.x, values, x)


def chebyshev_equilibrium(
    medium: Medium1D,
    u_left: float,
    u_right: float,
    x: np.ndarray,
    n_cheb: int = 48,
    shift: float = 0.0,
    max_width: float | None = None,
) -> np.ndarray:
    """``(alpha v')' = shift v`` with ``v(-1) = u_left``, ``v(1) = u_right``, at ``x``.

    ``shift = 0`` is the equilibrium problem (a cross-check of the
    quadrature reference); ``shift = c > 0`` gives the profile of the
    separable solution ``u = e^{ct} v(x)`` of the heat equation with
    ``u(-1, t) = u_left e^{ct}``, the 1-D twin of dissertation eq. 85–86.
    ``max_width`` splits wide elements (``ChebyshevPieces.build``).
    """
    cp = ChebyshevPieces.build(medium, n_cheb, max_width)
    size = cp.x.size
    matrix = cp.operator - shift * np.eye(size)
    rhs = np.zeros(size)
    matrix[cp.constrained] = cp.constraints
    rhs[cp.constrained] = cp.constraint_values(u_left, u_right)
    return cp.evaluate(np.linalg.solve(matrix, rhs), x)


@dataclass(frozen=True)
class ParabolicReference:
    """A Chebyshev-element solution of ``u_t = (alpha u_x)_x`` at one time.

    ``u`` holds the nodal values on the elements' Gauss–Lobatto nodes
    (``edges``, ``n_cheb``); ``evaluate`` interpolates them to any points, so
    one reference serves every grid of a sweep. ``meta`` records how it was
    made: the medium (its ``repr``), the caller's ``problem`` label for the
    initial and boundary data, ``t_end``, ``n_cheb``, the tolerances, the
    Radau step count and the run time. ``save`` writes the arrays as ``.npz``
    with the metadata as JSON beside them, and ``matches`` is what
    ``parabolic_reference`` checks before reusing a cache; ``reused`` says
    it did (not saved, so a loaded reference can report it).
    """

    edges: np.ndarray
    n_cheb: int
    u: np.ndarray
    meta: dict
    reused: bool = field(default=False, compare=False)

    KEYS = ("medium", "problem", "t_end", "n_cheb", "max_width", "rtol", "atol")
    """The metadata a cached reference must share with a request to be reused."""

    @property
    def x(self) -> np.ndarray:
        return _element_nodes(self.edges, self.n_cheb)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """The reference at ``x`` (any shape), interpolated element by element."""
        return _evaluate_elements(self.edges, self.n_cheb, self.x, self.u, x)

    def matches(self, meta: dict) -> bool:
        return all(self.meta.get(k) == meta.get(k) for k in self.KEYS)

    @staticmethod
    def files(stem: Path) -> tuple[Path, Path]:
        """``<stem>.npz`` and ``<stem>.json``; a dot in the stem (``d0.0025``) stays."""
        stem = Path(stem)
        return stem.with_name(stem.name + ".npz"), stem.with_name(stem.name + ".json")

    def save(self, stem: Path) -> None:
        arrays, meta = self.files(stem)
        with arrays.open("wb") as f:
            np.savez_compressed(f, edges=self.edges, u=self.u)
        meta.write_text(json.dumps(self.meta, indent=1) + "\n")

    @classmethod
    def load(cls, stem: Path) -> ParabolicReference:
        arrays, meta_path = cls.files(stem)
        meta = json.loads(meta_path.read_text())
        with np.load(arrays) as loaded:
            edges, u = loaded["edges"], loaded["u"]
        return cls(edges, int(meta["n_cheb"]), u, meta)


def parabolic_reference(
    medium: Medium1D,
    initial: Callable[[np.ndarray], np.ndarray],
    boundary: Callable[[float], tuple[float, float]],
    t_end: float,
    n_cheb: int = 48,
    rtol: float = RADAU_RTOL,
    atol: float = RADAU_ATOL,
    problem: str = "",
    cache: Path | None = None,
    max_width: float | None = None,
) -> ParabolicReference:
    """``u_t = (alpha u_x)_x`` from ``initial(x)`` to ``t_end`` on the elements.

    The element ends are eliminated through the constraints, ``u_C = M_C⁻¹
    (g(t) - M_I u_I)``, leaving a dense linear ODE ``u_I' = A u_I + B g(t)``
    that Radau integrates with its constant Jacobian. An initial condition
    that disagrees with ``g(0)`` at the ends is overridden there. The
    default tolerances are ``RADAU_RTOL`` / ``RADAU_ATOL``; see them for why
    tighter ones do not pay. ``max_width`` splits wide elements
    (``ChebyshevPieces.build``).

    With ``cache`` (the files' stem; ``.npz`` and ``.json`` are appended) the
    reference is loaded when the JSON matches the request in every
    ``ParabolicReference.KEYS`` entry and is solved and saved otherwise. The
    medium is identified by its ``repr``, stable for ``Constant`` and
    ``Sinusoid`` pieces (a ``Smooth`` piece's callables make it unique per
    process and defeat the cache);
    ``problem`` is the caller's name for ``initial`` and ``boundary``, which
    cannot be compared, so it must change when they do.
    """
    meta = {
        "medium": repr(medium),
        "problem": problem,
        "t_end": float(t_end),
        "n_cheb": int(n_cheb),
        "max_width": None if max_width is None else float(max_width),
        "rtol": float(rtol),
        "atol": float(atol),
    }
    if cache is not None and all(f.exists() for f in ParabolicReference.files(cache)):
        ref = ParabolicReference.load(cache)
        if ref.matches(meta):
            return replace(ref, reused=True)
    t0 = time.perf_counter()
    cp = ChebyshevPieces.build(medium, n_cheb, max_width)
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
    meta.update(
        {
            "elements": int(len(cp.edges) - 1),
            "unknowns": int(ii.size),
            "steps": int(sol.t.size - 1),
            "seconds": time.perf_counter() - t0,
        }
    )
    ref = ParabolicReference(cp.edges, n_cheb, u, meta)
    if cache is not None:
        ref.save(cache)
    return ref


def chebyshev_parabolic(
    medium: Medium1D,
    initial: Callable[[np.ndarray], np.ndarray],
    boundary: Callable[[float], tuple[float, float]],
    t_end: float,
    x: np.ndarray,
    n_cheb: int = 48,
    rtol: float = RADAU_RTOL,
    atol: float = RADAU_ATOL,
) -> np.ndarray:
    """``parabolic_reference`` evaluated at ``x``, uncached (the E1.3 entry point)."""
    ref = parabolic_reference(medium, initial, boundary, t_end, n_cheb, rtol, atol)
    return ref.evaluate(x)
