"""Grids and materials for the 1-D heat operator ``d/dx (alpha d/dx)`` on [-1, 1].

Dissertation §4.1–4.2 (eq. 51, 64, 75) and Martin & Fornberg 2017 §2.2.2. The
grid is equispaced with both ends on nodes so the Dirichlet conditions are
plain rows. Where an interface then falls relative to the nodes is fixed by
the node count, not by a switch: with ``h = 2 / (n - 1)`` an interface at 0
sits on a node for odd ``n`` and halfway between two nodes for even ``n``,
and the dissertation's pair (0, 0.5) can never both sit mid-cell. The knee
studies (E3) select the placement through ``node_counts`` / ``grid_for``.

``SmoothEdges`` is the stiff-edge medium of ``docs/stiff-diffusion.md`` §1.1:
the same pieces, each jump replaced by a tanh transition of width ``delta``,
the jump itself at ``delta = 0``. Every material reports its ``elements``, the
intervals on which alpha is smooth up to their ends, and the references in
``exact.py`` cut on them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import exp, factorial
from typing import Literal, Protocol

import numpy as np

X_MIN, X_MAX = -1.0, 1.0

PLACEMENT_TOL = 1e-9
"""How close to a node or a cell midpoint counts as "on" it, as a fraction of h.

The one tolerance for interface placement: ``Grid1D.placement``,
``node_counts`` and ``Grid1D.snapped`` all use it. Materials never snap;
``PiecewiseAlpha`` decides ownership by exact equality, so the quadrature
reference samples the piece a point is really in.
"""

TANH_REACH = 20.0
"""Beyond ``TANH_REACH * delta`` from an edge centre ``SmoothEdges.alpha`` is the piece.

Bit for bit, for the study's media: the far piece's weight ``1 / (1 + e^{2z})``
is 4.2e-18 at ``z = 20``, below half an ulp of the near piece's value once the
values are within a factor ten of each other. At 19δ (the figure
``docs/stiff-diffusion.md`` §1.1 quotes) the blend is within two ulps.
"""

EDGE_CUTS = (1.0, 3.0, 9.0, 27.0)
"""Where the references cut a smooth edge: at ``x_c ± m delta`` for these ``m``.

Each element then spans at most a factor three in ``z = (x - x_c) / delta``
away from the centre element ``[-δ, δ]``, which keeps the transition's pole
at ``z = iπ/2`` and, for contrasts of order ten, the complex zero of alpha at
Bernstein-ellipse parameter ``ρ ≳ 3.4`` from every element, so 24 Gauss
points or 48 Chebyshev nodes per element resolve it to rounding. Past the
last cut the tails are below 1e-23 (and past ``TANH_REACH`` exactly the
piece), so the outermost elements run to the domain ends, or to the next
edge's cuts.
"""

Side = Literal["left", "right"]
Placement = Literal["node", "cell"]


@dataclass(frozen=True)
class Grid1D:
    """``n`` equispaced nodes ``x`` on [-1, 1], spacing ``h``; both ends are nodes."""

    n: int
    x: np.ndarray
    h: float

    def offset(self, xi: float) -> float:
        """Where ``xi`` falls in its cell, in units of ``h`` and in [0, 1).

        0 means ``xi`` is on a node, 0.5 that it is halfway between two.
        """
        return _offset(xi, self.n)

    def placement(self, xi: float, tol: float = PLACEMENT_TOL) -> Placement | None:
        """``"node"``, ``"cell"`` (mid-cell) or ``None`` for anything else."""
        return _placement(xi, self.n, tol)

    def snapped(
        self, interfaces: Sequence[float], tol: float = PLACEMENT_TOL
    ) -> np.ndarray:
        """The nodes, with any within ``tol * h`` of an interface moved onto it.

        ``linspace`` lands a node on an interface only up to rounding; the
        operators evaluate alpha here so such a node gets the owner's value.
        """
        x = self.x.copy()
        for xi in interfaces:
            near = np.abs(x - xi) <= tol * self.h
            x[near] = xi
        return x


def _offset(xi: float, n: int) -> float:
    t = (xi - X_MIN) * (n - 1) / (X_MAX - X_MIN)
    return float(t - np.floor(t))


def _placement(xi: float, n: int, tol: float) -> Placement | None:
    d = _offset(xi, n)
    if min(d, 1.0 - d) < tol:
        return "node"
    if abs(d - 0.5) < tol:
        return "cell"
    return None


def equispaced_grid(n: int) -> Grid1D:
    """``n`` nodes from -1 to 1 inclusive (the dissertation's 101-node runs)."""
    if n < 5:
        raise ValueError("five-point stencils need at least five nodes")
    x = np.linspace(X_MIN, X_MAX, n)
    return Grid1D(n=n, x=x, h=float(x[1] - x[0]))


def node_counts(
    interfaces: Sequence[float], placement: Placement, n_min: int, n_max: int
) -> list[int]:
    """Node counts in [``n_min``, ``n_max``] giving every interface the placement.

    ``"node"``: each interface coincides with a node; ``"cell"``: each sits
    halfway between two nodes. Empty when the interfaces are incompatible
    (the dissertation's 0 and 0.5 have no common mid-cell grid).
    """
    return [
        n
        for n in range(max(n_min, 5), n_max + 1)
        if all(_placement(xi, n, PLACEMENT_TOL) == placement for xi in interfaces)
    ]


def grid_for(
    interfaces: Sequence[float], placement: Placement, n_target: int
) -> Grid1D:
    """The grid with the admissible node count nearest ``n_target`` (ties go up)."""
    lo, hi = max(5, n_target // 2), 2 * n_target + 4
    counts = node_counts(interfaces, placement, lo, hi)
    if not counts:
        raise ValueError(
            f"no node count in [{lo}, {hi}] puts every interface {interfaces} "
            f"at a {placement!r} position"
        )
    return equispaced_grid(min(counts, key=lambda n: (abs(n - n_target), -n)))


class Medium1D(Protocol):
    """What the operators, the reference and the interface stencils need of a material.

    ``taylor`` serves E1.2's continuity matrices with the pieces' expansions
    about ``interfaces[i]``, a smooth edge included (``SmoothEdges.taylor``).
    ``elements`` serves the references: the edges ``-1 = e_0 < … < e_m = 1``
    and, per element, a ``Piece`` smooth on it up to its ends, so that alpha
    can be evaluated one-sidedly at a jump and a smooth edge is cut where its
    transition needs resolving.
    """

    interfaces: tuple[float, ...]

    def alpha(self, x: np.ndarray) -> np.ndarray: ...

    def alpha_x(self, x: np.ndarray) -> np.ndarray: ...

    def taylor(self, i: int, side: Side, degree: int) -> np.ndarray: ...

    def elements(self) -> tuple[np.ndarray, tuple[Piece, ...]]: ...


class Piece(Protocol):
    """A smooth diffusivity on one side of an interface."""

    def alpha(self, x: np.ndarray) -> np.ndarray: ...

    def alpha_x(self, x: np.ndarray) -> np.ndarray: ...

    def taylor(self, x0: float, degree: int) -> np.ndarray:
        """Coefficients ``a_k`` of ``sum a_k (x - x0)^k``, ``k = 0..degree``."""
        ...


@dataclass(frozen=True)
class Constant:
    value: float

    def alpha(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), self.value)

    def alpha_x(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(x, dtype=float))

    def taylor(self, x0: float, degree: int) -> np.ndarray:
        a = np.zeros(degree + 1)
        a[0] = self.value
        return a


@dataclass(frozen=True)
class Sinusoid:
    """``offset + amplitude * sin(wavenumber * x + phase)``.

    The layers of dissertation eq. 64 and 75 (``0.5 + 0.1 sin 2πx``,
    ``0.1 + 0.4 sin 2πx``); its Taylor coefficients about x = 0 are the entries
    of the multiplication matrix in eq. 66 (EABE eq. 18).
    """

    offset: float
    amplitude: float
    wavenumber: float
    phase: float = 0.0

    def alpha(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.offset + self.amplitude * np.sin(self.wavenumber * x + self.phase)

    def alpha_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        k = self.wavenumber
        return self.amplitude * k * np.cos(k * x + self.phase)

    def taylor(self, x0: float, degree: int) -> np.ndarray:
        # d^m/dx^m sin(k x + phi) = k^m sin(k x + phi + m pi / 2).
        m = np.arange(degree + 1)
        theta = self.wavenumber * x0 + self.phase + m * np.pi / 2
        a = self.amplitude * self.wavenumber**m * np.sin(theta)
        a /= np.array([factorial(int(i)) for i in m], dtype=float)
        a[0] += self.offset
        return a


@dataclass(frozen=True)
class Smooth:
    """Any smooth piece given as a callable with as many derivatives as needed."""

    f: Callable[[np.ndarray], np.ndarray]
    derivatives: tuple[Callable[[np.ndarray], np.ndarray], ...] = ()

    def alpha(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.f(np.asarray(x, dtype=float)), dtype=float)

    def alpha_x(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._deriv(1)(np.asarray(x, dtype=float)), dtype=float)

    def taylor(self, x0: float, degree: int) -> np.ndarray:
        a = np.empty(degree + 1)
        a[0] = float(self.f(np.asarray(x0, dtype=float)))
        for k in range(1, degree + 1):
            a[k] = float(self._deriv(k)(np.asarray(x0, dtype=float))) / factorial(k)
        return a

    def _deriv(self, k: int) -> Callable[[np.ndarray], np.ndarray]:
        if k > len(self.derivatives):
            raise ValueError(
                f"Smooth piece has {len(self.derivatives)} derivatives, {k} needed"
            )
        return self.derivatives[k - 1]


@dataclass(frozen=True)
class PiecewiseAlpha:
    """Diffusivity on [-1, 1]: ``pieces`` separated by increasing ``interfaces``.

    ``at_interface[i]`` names the piece whose value alpha takes *at*
    ``x = interfaces[i]`` (only a node sitting on an interface ever asks).
    Eq. 75's layer ``[0, 0.5]`` is closed at both ends, so its owners are
    ``("right", "left")``; eq. 64's ``[-1, 0), [0, 1]`` gives ``("right",)``,
    the default.
    """

    interfaces: tuple[float, ...]
    pieces: tuple[Piece, ...]
    at_interface: tuple[Side, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.pieces) != len(self.interfaces) + 1:
            raise ValueError("one more piece than interfaces is needed")
        xs = np.asarray(self.interfaces, dtype=float)
        if xs.size and not (
            np.all(np.diff(xs) > 0) and xs[0] > X_MIN and xs[-1] < X_MAX
        ):
            raise ValueError("interfaces must increase strictly inside (-1, 1)")
        if self.at_interface is None:
            object.__setattr__(self, "at_interface", ("right",) * len(self.interfaces))
        elif len(self.at_interface) != len(self.interfaces):
            raise ValueError("one owner per interface is needed")

    def piece_index(self, x: np.ndarray) -> np.ndarray:
        """Index of the piece that supplies alpha at each ``x``.

        Ownership is decided by exact equality: only a point *at* an
        interface asks. Grids snap their nodes first (``Grid1D.snapped``);
        quadrature samples never sit on an interface.
        """
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(np.asarray(self.interfaces), x, side="right")
        for i, (xi, side) in enumerate(
            zip(self.interfaces, self.at_interface, strict=True)
        ):
            if side == "left":
                idx = np.where(x == xi, i, idx)
        return idx

    def alpha(self, x: np.ndarray) -> np.ndarray:
        return self._piecewise("alpha", x)

    def alpha_x(self, x: np.ndarray) -> np.ndarray:
        """Derivative of the owning piece; it ignores the jump, by design."""
        return self._piecewise("alpha_x", x)

    def taylor(self, i: int, side: Side, degree: int) -> np.ndarray:
        """Taylor coefficients about ``interfaces[i]`` of the piece on ``side``."""
        piece = self.pieces[i] if side == "left" else self.pieces[i + 1]
        return piece.taylor(self.interfaces[i], degree)

    def elements(self) -> tuple[np.ndarray, tuple[Piece, ...]]:
        """The pieces between the interfaces: the elements of a jump are its pieces."""
        edges = np.array([X_MIN, *self.interfaces, X_MAX], dtype=float)
        return edges, self.pieces

    def _piecewise(self, method: str, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        flat = x.ravel()
        idx = self.piece_index(flat)
        out = np.empty_like(flat)
        for i, piece in enumerate(self.pieces):
            mask = idx == i
            if mask.any():
                out[mask] = getattr(piece, method)(flat[mask])
        return out.reshape(x.shape)


def _logistic_pair(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(s, 1 - s)`` for ``s = ½ (1 + tanh z) = 1 / (1 + e^{-2z})``.

    Each is computed as ``e / (1 + e)`` or ``1 / (1 + e)`` with ``e =
    e^{-2|z|}`` in (0, 1], so the small one is accurate to rounding in the
    tails rather than the round-off of ``1 - s``.
    """
    e = np.exp(-2.0 * np.abs(z))
    small, large = e / (1.0 + e), 1.0 / (1.0 + e)
    positive = z >= 0.0
    return np.where(positive, large, small), np.where(positive, small, large)


def edge_blend(
    a: np.ndarray, b: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(1 − s) a + s b`` and ``s′(z) = 2 s (1 − s)``, ``s = ½ (1 + tanh z)``.

    One tanh edge of a ``SmoothEdges`` (and of the 2-D ``heat2d.domain.
    SmoothBand``, whose ``z`` is the signed distance to a curve over δ). The
    blend is formed from the near value on each side, so the far one's share
    is an addition that rounds away in the tails; the caller adds the chain
    rule's ``s′(z) z′ (b − a)`` to the blended derivatives.
    """
    s, t = _logistic_pair(z)
    near = z >= 0.0
    return np.where(near, b + t * (a - b), a + s * (b - a)), 2.0 * s * t


def edge_value(a: float, b: float, z: float) -> float:
    """``edge_blend``'s value at one point, in floats.

    The same steps (the small logistic share from the near side), with
    ``math.exp`` for NumPy's: for callers that evaluate one point at a time,
    such as the 2-D seed march (``heat2d.seeds``), which reads alpha a
    thousand times per stencil and pays NumPy's per-call cost on each.
    """
    e = exp(-2.0 * abs(z))
    small = e / (1.0 + e)
    return b + small * (a - b) if z >= 0.0 else a + small * (b - a)


def _merge_cuts(cuts: np.ndarray, gap: float) -> np.ndarray:
    """The ends and the sorted ``cuts``, dropping any within ``gap`` of a kept one."""
    kept = [X_MIN]
    for c in np.sort(cuts):
        if c - kept[-1] > gap and X_MAX - c > gap:
            kept.append(float(c))
    kept.append(X_MAX)
    return np.array(kept)


@dataclass(frozen=True)
class SmoothEdges:
    """``jump`` with each interface replaced by a tanh edge of width ``delta``.

    ``docs/stiff-diffusion.md`` §1.1: across the edge at ``x_c``,

        alpha = (1 - s) alpha⁻(x) + s alpha⁺(x),   s = ½ [1 + tanh((x - x_c) / δ)],

    with the pieces keeping their own variation on both sides. The edges are
    folded in from the left, ``alpha ← (1 - s_k) alpha + s_k alpha_{k+1}``, so
    the pieces' weights are a partition of unity even when two edges lie
    within a few δ of each other (a thin layer, §1.3's double-cross). Beyond
    ``TANH_REACH`` δ from a centre alpha is the piece bit for bit; ``delta =
    0`` is the ``jump`` itself, bit for bit, in ``alpha``, ``alpha_x``,
    ``taylor`` and ``elements``.

    ``interfaces`` are the edge centres, so ``straddling_windows`` and
    ``jump_aware_operator`` see the smooth edge as a jump there, and
    ``taylor`` delegates to the pieces: ``jump_aware_operator(grid,
    SmoothEdges(m, delta))`` is the "δ = 0 construction on a smooth edge" of
    §1.7, E3.3's baseline, with no further code. (The smooth alpha's own
    expansion about the centre, whose k-th coefficient is O(δ⁻ᵏ), would serve
    no stencil and is not offered.) The elements cut each edge at
    ``x_c ± EDGE_CUTS δ``.
    """

    jump: PiecewiseAlpha
    delta: float
    interfaces: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.delta) and self.delta >= 0.0):
            raise ValueError("the edge width delta must be finite and non-negative")
        object.__setattr__(self, "interfaces", self.jump.interfaces)

    def alpha(self, x: np.ndarray) -> np.ndarray:
        return self._blend(x)[0]

    def alpha_x(self, x: np.ndarray) -> np.ndarray:
        return self._blend(x)[1]

    def taylor(self, i: int, side: Side, degree: int) -> np.ndarray:
        """The pieces' expansions about the centre, the jump's data (see the class)."""
        return self.jump.taylor(i, side, degree)

    def elements(self) -> tuple[np.ndarray, tuple[Piece, ...]]:
        if self.delta == 0.0:
            return self.jump.elements()
        cuts = np.array(
            [
                xc + sign * m * self.delta
                for xc in self.interfaces
                for m in EDGE_CUTS
                for sign in (-1.0, 1.0)
            ]
        )
        edges = _merge_cuts(cuts, gap=0.5 * self.delta)
        return edges, (self,) * (len(edges) - 1)

    def _blend(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.delta == 0.0:
            return self.jump.alpha(x), self.jump.alpha_x(x)
        x = np.asarray(x, dtype=float)
        pieces = self.jump.pieces
        a, a_x = pieces[0].alpha(x), pieces[0].alpha_x(x)
        for xc, piece in zip(self.interfaces, pieces[1:], strict=True):
            b, b_x = piece.alpha(x), piece.alpha_x(x)
            z = (x - xc) / self.delta
            value, ds = edge_blend(a, b, z)
            slope, _ = edge_blend(a_x, b_x, z)
            a, a_x = value, slope + (ds / self.delta) * (b - a)
        return a, a_x


@dataclass(frozen=True)
class OnInterval:
    """``medium`` on ``[lo, hi] ⊂ [-1, 1]``: the same material, its elements clipped.

    For a problem posed on part of the line, such as E4.2's profile in
    ``y ∈ [0, 1]`` of the 2-D strip, which is E3.2's medium in ``y`` (stiff
    note §3.1). ``alpha``, ``alpha_x``, ``taylor`` and ``interfaces`` are the
    medium's; every interface must lie strictly inside the interval.
    ``elements`` keeps the medium's edges strictly inside ``(lo, hi)`` with
    ``lo`` and ``hi`` as the ends, each new element carrying the piece of the
    element it lies in, and drops a smooth edge's cut closer than δ/2 to an
    end, as ``SmoothEdges.elements`` drops one near ±1.
    """

    medium: Medium1D
    lo: float
    hi: float
    interfaces: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not X_MIN <= self.lo < self.hi <= X_MAX:
            raise ValueError(f"[lo, hi] must be an interval inside [{X_MIN}, {X_MAX}]")
        inside = all(self.lo < xi < self.hi for xi in self.medium.interfaces)
        if not inside:
            raise ValueError("every interface must lie strictly inside (lo, hi)")
        object.__setattr__(self, "interfaces", self.medium.interfaces)

    def alpha(self, x: np.ndarray) -> np.ndarray:
        return self.medium.alpha(x)

    def alpha_x(self, x: np.ndarray) -> np.ndarray:
        return self.medium.alpha_x(x)

    def taylor(self, i: int, side: Side, degree: int) -> np.ndarray:
        return self.medium.taylor(i, side, degree)

    def elements(self) -> tuple[np.ndarray, tuple[Piece, ...]]:
        edges, pieces = self.medium.elements()
        gap = 0.5 * float(getattr(self.medium, "delta", 0.0))
        inner = edges[1:-1]
        kept = inner[(inner > self.lo + gap) & (inner < self.hi - gap)]
        # At δ > 0 every edge is a smooth cut (the medium is every element's
        # piece); at δ = 0 the gap is zero and the interfaces, strictly
        # inside by construction, all stay.
        out = np.array([self.lo, *kept, self.hi])
        mid = 0.5 * (out[:-1] + out[1:])
        owner = np.searchsorted(inner, mid, side="right")
        return out, tuple(pieces[k] for k in owner)


DISSERTATION_BC = (1.0, 0.0)
"""``u(-1), u(1)`` of dissertation eq. 75."""


def dissertation_alpha() -> PiecewiseAlpha:
    """Eq. 75: ``0.1 + 0.4 sin 2πx`` on the closed layer [0, 0.5], 1 elsewhere.

    The equilibrium problem of Fig. 4-5–4-7 pairs it with ``DISSERTATION_BC``.
    """
    return PiecewiseAlpha(
        interfaces=(0.0, 0.5),
        pieces=(Constant(1.0), Sinusoid(0.1, 0.4, 2 * np.pi), Constant(1.0)),
        at_interface=("right", "left"),
    )


def eabe_alpha() -> PiecewiseAlpha:
    """Eq. 64, EABE §2.2.2's worked example: 1 | ``0.5 + 0.1 sin 2πx`` at 0."""
    return PiecewiseAlpha(
        interfaces=(0.0,), pieces=(Constant(1.0), Sinusoid(0.5, 0.1, 2 * np.pi))
    )


def jump_alpha(left: float, right: float, x0: float = 0.0) -> PiecewiseAlpha:
    """Two constants meeting at ``x0`` (the MATLAB problem is ``1/9 | 1`` at 0)."""
    return PiecewiseAlpha(interfaces=(x0,), pieces=(Constant(left), Constant(right)))


def matlab_alpha() -> PiecewiseAlpha:
    """``FD4heat1DAC.m``: ``k1 = 1/9`` left of 0, ``k2 = 1`` right of it.

    The port's jump is at 0 with the right piece owning it, as in eq. 64
    (``jump_alpha``'s default): a node landing on 0 (odd counts) takes
    ``alpha = 1``. The MATLAB differs in two ways that are not reproduced: it
    gives the node at 0 the left value, and it builds its four interface
    stencils about a jump half a cell to the right of that node, so its
    interface effectively sits at ``x = h / 2`` and moves with the grid.
    """
    return jump_alpha(1.0 / 9.0, 1.0)
