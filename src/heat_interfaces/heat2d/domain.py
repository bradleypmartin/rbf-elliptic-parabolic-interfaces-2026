"""The x-periodic unit strip: its interfaces, materials, boundaries and node sets.

Dissertation §5.4 and Martin & Fornberg 2017 (EABE) §3. Every 2-D test case
lives in ``[0, 1) × [0, 1]``, periodic in x, with Dirichlet rows on ``y = 0``
and ``y = 1`` and, for case 3, on the circle ``r = 0.05``. An interface is a
graph ``y = c(x)`` (flat, or case 2's ``c ± 0.02 sin 2πx``) or a circle
(case 3's ring), and the material is one smooth piece inside a closed band
between two interfaces and another outside it (eq. 84, EABE eq. 32, 35,
38). Which piece a point belongs to is decided by the exact sign of the
curve's level function, never by a tolerance (port notes §1.6).
``SmoothBand`` is the stiff-edge medium of ``docs/stiff-diffusion.md`` §3.1
(E4.2): the same band with each interface a tanh edge of width δ in the
signed normal distance, the jump's region data kept.

Node sets follow the MATLAB ``ExeprepRBFHeatLaplace4.m``: rows of fixed
nodes straddle each interface in a hexagonal layout (three rows a side, the
innermost pair half a spacing off the curve), a row of fixed Dirichlet nodes
sits on each boundary, and the remaining nodes are scattered by the
electrostatic repulsion of ``mos2dsqperiodic7`` (dissertation Fig. 5-3, EABE
Fig. 1 and 12). Case 3's rows straddle the midline of its 0.001-wide ring,
the MATLAB's ``thinFlag`` layout, so both interfaces sit between the
innermost pair (EABE Fig. 12b).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import atan, cos, factorial, hypot, sin, sqrt
from typing import Literal, Protocol

import numpy as np

from ..heat1d.domain import edge_blend, edge_value
from ..heat1d.stiff import EDGE_STOP
from .neighbors import PERIOD, knn, offsets, periodic_dx, wrap_x

Y_MIN, Y_MAX = 0.0, 1.0

TWO_PI = 2.0 * np.pi

# --- curves -----------------------------------------------------------------


class Curve(Protocol):
    """A smooth closed or x-periodic curve on the strip, parametrised by ``s ∈ [0, 1)``.

    ``level`` is a cheap signed function that is 0 on the curve, negative on
    one side and positive on the other; its exact sign is what decides which
    piece a point belongs to. ``normal`` points into ``level > 0``, and
    ``curvature`` is signed so that the curve bends towards ``+normal`` when
    it is positive. ``signed_distance`` is the true normal distance, positive
    on the ``level > 0`` side.
    """

    @property
    def length(self) -> float:
        """Arc length of one period (1 for a graph, 2πR for a circle)."""
        ...

    def level(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def point(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def tangent_angle(self, s: np.ndarray) -> np.ndarray: ...

    def normal(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def curvature(self, s: np.ndarray) -> np.ndarray: ...

    def closest(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Parameter of the point of the curve nearest ``(x, y)``."""
        ...

    def signed_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def signed_distance_at(self, x: float, y: float) -> float:
        """``signed_distance`` at one point, in floats (the seed march's, E4.4–E4.7)."""
        ...


def _as_float(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(a, dtype=float) for a in arrays)


NEWTON_STEPS = 12
"""Newton iterations of the foot-point and crossing searches (quadratic at once)."""


class _Graph:
    """``y = c(x)`` with ``c`` 1-periodic; ``s`` is ``x`` itself.

    ``level = y - c(x)`` (the MATLAB's side test), so the normal points up.
    """

    length = PERIOD

    def height(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def slope(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def second(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def height_at(self, x: float) -> float:
        """``height`` at one point in floats; subclasses override the array call."""
        return float(self.height(np.array([x]))[0])

    def slope_at(self, x: float) -> float:
        return float(self.slope(np.array([x]))[0])

    def second_at(self, x: float) -> float:
        return float(self.second(np.array([x]))[0])

    def level(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = _as_float(x, y)
        return y - self.height(x)

    def point(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        (s,) = _as_float(s)
        return wrap_x(s), self.height(s)

    def tangent_angle(self, s: np.ndarray) -> np.ndarray:
        (s,) = _as_float(s)
        return np.arctan(self.slope(s))

    def normal(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = self.tangent_angle(s)
        return -np.sin(theta), np.cos(theta)

    def curvature(self, s: np.ndarray) -> np.ndarray:
        (s,) = _as_float(s)
        return self.second(s) / (1.0 + self.slope(s) ** 2) ** 1.5

    def closest(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Newton on ``(s - x) + (c(s) - y) c'(s) = 0`` from ``s = x``.

        The iteration the MATLAB ``pointFinder1/2`` runs by projecting onto
        the tangent line, done to rounding. It converges for any point of the
        strip since ``|c - y| |c''| < 1`` there for the case-2 curves.
        """
        x, y = _as_float(x, y)
        s = np.array(x, dtype=float, copy=True)
        for _ in range(NEWTON_STEPS):
            c, cp, cpp = self.height(s), self.slope(s), self.second(s)
            g = periodic_dx(s - x) + (c - y) * cp
            gp = 1.0 + cp**2 + (c - y) * cpp
            step = g / gp
            s -= step
            if np.all(np.abs(step) < 1e-15):
                break
        return wrap_x(s)

    def signed_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = _as_float(x, y)
        s = self.closest(x, y)
        nx, ny = self.normal(s)
        return nx * periodic_dx(x - s) + ny * (y - self.height(s))

    def signed_distance_at(self, x: float, y: float) -> float:
        """``closest`` and ``signed_distance`` step for step, in floats.

        The same Newton iteration from ``s = x`` and the same projection on
        the unit normal, with ``math`` in place of NumPy, so it is
        ``signed_distance`` to rounding at a few microseconds a call instead
        of tens: the seed march reads alpha along a normal line thousands of
        times per stencil (E4.4's float path, route (a) of E4.7).
        """
        s = x
        for _ in range(NEWTON_STEPS):
            c, cp, cpp = self.height_at(s), self.slope_at(s), self.second_at(s)
            step = (_periodic_dx(s - x) + (c - y) * cp) / (1.0 + cp**2 + (c - y) * cpp)
            s -= step
            if abs(step) < 1e-15:
                break
        s %= PERIOD
        theta = atan(self.slope_at(s))
        return -sin(theta) * _periodic_dx(x - s) + cos(theta) * (y - self.height_at(s))


def _periodic_dx(dx: float) -> float:
    """``neighbors.periodic_dx`` for one float (Python's ``%`` floors as NumPy's)."""
    return (dx + PERIOD / 2) % PERIOD - PERIOD / 2


@dataclass(frozen=True)
class FlatLine(_Graph):
    """``y = c``: case 1's interfaces (eq. 84) and the rows ``y = 0``, ``y = 1``."""

    c: float

    def height(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), self.c)

    def slope(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(x, dtype=float))

    second = slope

    def height_at(self, x: float) -> float:
        return float(self.c)

    def slope_at(self, x: float) -> float:
        return 0.0

    second_at = slope_at

    def closest(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return wrap_x(x)

    def signed_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.level(x, y)

    def signed_distance_at(self, x: float, y: float) -> float:
        return y - self.c


@dataclass(frozen=True)
class SineGraph(_Graph):
    """``y = c + a sin(k x)``: MATLAB ``curvedinterface1/2``, ``a = 0.02``, ``k = 2π``.

    Case 2's pair is ``SineGraph(0.6)`` and ``SineGraph(0.8)`` (paper index,
    E0.2); the band keeps its 0.2 thickness.
    """

    c: float
    amplitude: float = 0.02
    wavenumber: float = TWO_PI

    def height(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.c + self.amplitude * np.sin(self.wavenumber * x)

    def slope(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.amplitude * self.wavenumber * np.cos(self.wavenumber * x)

    def second(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return -self.amplitude * self.wavenumber**2 * np.sin(self.wavenumber * x)

    def height_at(self, x: float) -> float:
        return self.c + self.amplitude * sin(self.wavenumber * x)

    def slope_at(self, x: float) -> float:
        return self.amplitude * self.wavenumber * cos(self.wavenumber * x)

    def second_at(self, x: float) -> float:
        return -self.amplitude * self.wavenumber**2 * sin(self.wavenumber * x)


@dataclass(frozen=True)
class Circle:
    """``r = R`` about ``(cx, cy)``; ``level = r - R``, the normal points outward.

    Case 3's ring is the pair ``Circle(0.349)``, ``Circle(0.35)`` and its
    cooling unit ``Circle(0.05)`` (EABE eq. 37–39). A circle must stay clear
    of ``x = 0`` and ``x = 1``: it is not wrapped.
    """

    radius: float
    cx: float = 0.5
    cy: float = 0.5

    @property
    def length(self) -> float:
        return TWO_PI * self.radius

    def level(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = _as_float(x, y)
        return np.hypot(x - self.cx, y - self.cy) - self.radius

    def point(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phi = TWO_PI * np.asarray(s, dtype=float)
        return self.cx + self.radius * np.cos(phi), self.cy + self.radius * np.sin(phi)

    def tangent_angle(self, s: np.ndarray) -> np.ndarray:
        phi = TWO_PI * np.asarray(s, dtype=float) + np.pi / 2
        return np.arctan2(np.sin(phi), np.cos(phi))

    def normal(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phi = TWO_PI * np.asarray(s, dtype=float)
        return np.cos(phi), np.sin(phi)

    def curvature(self, s: np.ndarray) -> np.ndarray:
        # Counter-clockwise with the outward normal: the circle bends away
        # from +normal.
        return np.full_like(np.asarray(s, dtype=float), -1.0 / self.radius)

    def closest(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = _as_float(x, y)
        return np.mod(np.arctan2(y - self.cy, x - self.cx) / TWO_PI, 1.0)

    def signed_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.level(x, y)

    def signed_distance_at(self, x: float, y: float) -> float:
        return hypot(x - self.cx, y - self.cy) - self.radius


# --- materials --------------------------------------------------------------


class Piece2D(Protocol):
    """A smooth diffusivity on one side of the interfaces."""

    def alpha(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def gradient(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def taylor(self, x0: float, y0: float, degree: int) -> np.ndarray:
        """``a[i, j]`` of ``sum a_ij (x - x0)^i (y - y0)^j`` for ``i + j <= degree``.

        A ``(degree + 1, degree + 1)`` table, zero above the anti-diagonal;
        the multiplication matrices of E2.3 rotate it into the local frame.
        """
        ...


@dataclass(frozen=True)
class Constant2D:
    value: float

    def alpha(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), self.value)

    def gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.zeros_like(np.asarray(x, dtype=float))
        return z, z.copy()

    def taylor(self, x0: float, y0: float, degree: int) -> np.ndarray:
        a = np.zeros((degree + 1, degree + 1))
        a[0, 0] = self.value
        return a

    def alpha_at(self, x: float, y: float) -> float:
        return float(self.value)


def _sine_taylor(k: float, x0: float, degree: int) -> np.ndarray:
    """Coefficients of ``sin(k x)`` about ``x0``: ``k^m sin(k x0 + mπ/2) / m!``."""
    m = np.arange(degree + 1)
    a = k**m * np.sin(k * x0 + m * np.pi / 2)
    return a / np.array([factorial(int(i)) for i in m], dtype=float)


@dataclass(frozen=True)
class SineProduct:
    """``offset + amplitude sin(kx x) sin(ky y)``: EABE eq. 35 and 38's band values."""

    offset: float
    amplitude: float
    kx: float = TWO_PI
    ky: float = TWO_PI

    def alpha(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = _as_float(x, y)
        return self.offset + self.amplitude * np.sin(self.kx * x) * np.sin(self.ky * y)

    def alpha_at(self, x: float, y: float) -> float:
        """``alpha`` at one point in floats, for the seed march (``_piece_at``)."""
        return self.offset + self.amplitude * sin(self.kx * x) * sin(self.ky * y)

    def gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = _as_float(x, y)
        sx, cx = np.sin(self.kx * x), np.cos(self.kx * x)
        sy, cy = np.sin(self.ky * y), np.cos(self.ky * y)
        return self.amplitude * self.kx * cx * sy, self.amplitude * self.ky * sx * cy

    def taylor(self, x0: float, y0: float, degree: int) -> np.ndarray:
        a = self.amplitude * np.outer(
            _sine_taylor(self.kx, x0, degree), _sine_taylor(self.ky, y0, degree)
        )
        i, j = np.indices(a.shape)
        a[i + j > degree] = 0.0
        a[0, 0] += self.offset
        return a


Side = Literal["outside", "inside"]


@dataclass(frozen=True)
class Band:
    """``inside`` where ``lower.level >= 0 >= upper.level`` (closed), else ``outside``.

    Every 2016 case is one such band: y ∈ [0.6, 0.8] (eq. 84), the sine band
    (EABE eq. 35) and the ring 0.349 ≤ r ≤ 0.35 (eq. 38). Closed at both
    ends, as the MATLAB's ``y <= c2 && y >= c1`` and the papers' brackets
    say; only a point exactly on an interface ever notices.
    """

    lower: Curve
    upper: Curve
    inside: Piece2D
    outside: Piece2D

    @property
    def interfaces(self) -> tuple[Curve, Curve]:
        return self.lower, self.upper

    @property
    def pieces(self) -> tuple[Piece2D, Piece2D]:
        """Indexed by ``piece_index``: 0 outside, 1 inside."""
        return self.outside, self.inside

    def piece_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """1 on the closed band, 0 elsewhere, by the exact signs of the levels."""
        x, y = _as_float(x, y)
        inside = (self.lower.level(x, y) >= 0.0) & (self.upper.level(x, y) <= 0.0)
        return inside.astype(int)

    def alpha(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = _as_float(x, y)
        return np.where(
            self.piece_index(x, y) == 1,
            self.inside.alpha(x, y),
            self.outside.alpha(x, y),
        )

    def gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gradient of the owning piece; it ignores the jumps, by design."""
        x, y = _as_float(x, y)
        inside = self.piece_index(x, y) == 1
        gi, go = self.inside.gradient(x, y), self.outside.gradient(x, y)
        return np.where(inside, gi[0], go[0]), np.where(inside, gi[1], go[1])

    def taylor(self, side: Side, x0: float, y0: float, degree: int) -> np.ndarray:
        piece = self.inside if side == "inside" else self.outside
        return piece.taylor(x0, y0, degree)

    def region_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """0 below the lower curve, 1 on the closed band, 2 above the upper curve.

        The 1-D picture of ``heat1d.interface``: interface ``j`` of
        ``interfaces`` separates regions ``j`` and ``j + 1``, region ``j``
        being its ``level < 0`` side (the ``-normal`` side) and ``j + 1`` its
        ``level > 0`` side. A stencil that reaches from region 0 to region 2
        is translated across both interfaces (E2.3).
        """
        x, y = _as_float(x, y)
        r = np.zeros(np.broadcast(x, y).shape, dtype=int)
        r += self.lower.level(x, y) >= 0.0
        r += self.upper.level(x, y) > 0.0
        return r

    def region_piece(self, region: int) -> Piece2D:
        """The smooth piece of a region: ``outside`` for 0 and 2, ``inside`` for 1."""
        if region not in (0, 1, 2):
            raise ValueError(f"region {region} is not one of 0, 1, 2")
        return self.inside if region == 1 else self.outside


@dataclass(frozen=True)
class SmoothBand:
    """``band`` with each interface replaced by a tanh edge of width ``delta``.

    ``docs/stiff-diffusion.md`` §3.1 (E4.2, #33): with ``d₁``, ``d₂`` the
    signed distances to the lower and upper curves (``Curve.signed_distance``,
    positive on the ``level > 0`` side) and ``s(z) = ½ (1 + tanh z)``, the
    edges are folded in from the outside piece as ``heat1d.domain.SmoothEdges``
    folds them in from the left,

        alpha ← (1 − s(d₁/δ)) outside + s(d₁/δ) inside,
        alpha ← (1 − s(d₂/δ)) alpha   + s(d₂/δ) outside,

    each step ``heat1d.domain.edge_blend``, so on case 1 ``alpha`` and
    ``gradient``'s y component are E3.2's 1-D medium in ``y`` bit for bit,
    beyond ``TANH_REACH`` δ from both curves alpha is the piece bit for bit,
    and a band thinner than its edges gets the product profile
    ``outside + (inside − outside) s(d₁/δ) (1 − s(d₂/δ))``. The pieces keep
    their own variation through the blend; ``gradient`` is the blend's, with
    ``∇d`` the unit normal at the foot point.

    Everything else is the jump's (stiff note §3.8, decision 1):
    ``region_index``, ``region_piece``, ``piece_index``, ``interfaces`` and the
    pieces' ``taylor`` tables, so the naive operator, the δ = 0 construction
    (``interface_aware_operator`` on this medium) and every 2016 driver run
    on it unchanged. ``delta = 0`` is the ``band`` itself, bit for bit.
    """

    band: Band
    delta: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.delta) and self.delta >= 0.0):
            raise ValueError("the edge width delta must be finite and non-negative")
        if isinstance(self.band, SmoothBand):
            raise TypeError("smooth a jump Band, not a SmoothBand")

    @property
    def lower(self) -> Curve:
        return self.band.lower

    @property
    def upper(self) -> Curve:
        return self.band.upper

    @property
    def inside(self) -> Piece2D:
        return self.band.inside

    @property
    def outside(self) -> Piece2D:
        return self.band.outside

    @property
    def interfaces(self) -> tuple[Curve, Curve]:
        return self.band.interfaces

    @property
    def pieces(self) -> tuple[Piece2D, Piece2D]:
        return self.band.pieces

    def piece_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.band.piece_index(x, y)

    def region_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.band.region_index(x, y)

    def region_piece(self, region: int) -> Piece2D:
        return self.band.region_piece(region)

    def taylor(self, side: Side, x0: float, y0: float, degree: int) -> np.ndarray:
        """The pieces' tables, the jump's data (the smooth alpha's are O(δ⁻ᵏ))."""
        return self.band.taylor(side, x0, y0, degree)

    def alpha(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._blend(x, y)[0]

    def alpha_at(
        self, x: float, y: float, known: tuple[int, float] | None = None
    ) -> float:
        """``alpha`` at one point, in floats: ``_blend``'s value without its gradient.

        For the seed march (E4.4), which reads alpha along a line a thousand
        times per stencil; through the array code each call costs about
        60 µs across an edge. The same steps in the same order, with
        ``edge_value`` for ``edge_blend`` and each curve's float
        ``signed_distance_at`` (a flat line's level, bit for bit), so it is
        ``alpha`` to rounding (bit for bit where ``math.exp`` and NumPy's
        ``exp`` agree). ``known = (j, d)`` supplies interface ``j``'s distance
        instead of computing it: on the normal line through a foot point of
        a curve the distance to that curve is linear in the arc along the
        line, so route (a)'s march (E4.7) pays the foot-point Newton only for
        the other curve.
        """
        if self.delta == 0.0:
            return float(self.band.alpha(np.array([x]), np.array([y]))[0])
        a = _piece_at(self.outside, x, y)
        for j, (curve, piece) in enumerate(
            ((self.lower, self.inside), (self.upper, self.outside))
        ):
            if known is not None and known[0] == j:
                d = known[1]
            else:
                d = curve.signed_distance_at(x, y)
            a = edge_value(a, _piece_at(piece, x, y), d / self.delta)
        return a

    def gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, gx, gy = self._blend(x, y)
        return gx, gy

    def _blend(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.delta == 0.0:
            return self.band.alpha(x, y), *self.band.gradient(x, y)
        x, y = np.broadcast_arrays(*_as_float(x, y))
        a = self.outside.alpha(x, y)
        ax, ay = self.outside.gradient(x, y)
        for curve, piece in ((self.lower, self.inside), (self.upper, self.outside)):
            b = piece.alpha(x, y)
            bx, by = piece.gradient(x, y)
            z = curve.signed_distance(x, y) / self.delta
            value, ds = edge_blend(a, b, z)
            gx, _ = edge_blend(ax, bx, z)
            gy, _ = edge_blend(ay, by, z)
            nx, ny = curve.normal(curve.closest(x, y))
            k = (ds / self.delta) * (b - a)
            a, ax, ay = value, gx + k * nx, gy + k * ny
        return a, ax, ay

    def normal_profile(self, j: int, x: float, y: float, scale: float) -> NormalProfile:
        """The material along interface ``j``'s normal through the anchor ``(x, y)``.

        Stiff note §3.2–3.3: the line through the anchor along the unit
        normal at interface ``j``'s foot point nearest it (``frame_at``'s
        ``y'``, into ``level > 0``), in units of the stencil radius
        ``scale``; the stops are both curves' crossings of the line and, at
        δ > 0, their ``± EDGE_STOP δ`` flanks. A flat interface's are
        formed as ``heat1d.stiff``'s ``_stops`` forms them, so case 1 is the
        1-D march's stops bit for bit. A sine graph's crossing is route
        (a)'s Newton iteration on its level along the line (E4.7, #38,
        ``_line_crossing``), and its flanks are ``EDGE_STOP δ`` in its own
        normal distance to first order, which is all a restart point needs;
        on a curved foot curve the profile also carries the anchor's
        distance to it (``NormalProfile.foot``), exact along the line. The
        ring's circles are E4.8's (#39), marched as widths.
        """
        curve = self.interfaces[j]
        s = curve.closest(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        nx, ny = (float(c) for c in curve.normal(s))
        flanks = (-EDGE_STOP * self.delta, 0.0, EDGE_STOP * self.delta)
        stops = []
        for other in self.interfaces:
            if isinstance(other, FlatLine):
                for f in flanks if self.delta > 0.0 else (0.0,):
                    stops.append(((other.c + f) - float(y)) / (scale * ny))
                continue
            if not isinstance(other, _Graph):
                raise NotImplementedError(
                    "normal profiles cross graphs only; the ring's circles are "
                    "E4.8 (#39), marched as widths from the outer radius"
                )
            centre = _line_crossing(other, float(x), float(y), nx, ny, scale)
            px = float(x) + scale * centre * nx
            ox, oy = (float(c) for c in other.normal(np.array(px % PERIOD)))
            cosine = abs(nx * ox + ny * oy)
            for f in flanks if self.delta > 0.0 else (0.0,):
                stops.append(centre + f / (scale * cosine))
        stops = np.unique(stops)
        foot = None
        if not isinstance(curve, FlatLine):
            foot = (j, curve.signed_distance_at(float(x), float(y)))
        line = NormalProfile(float(x), float(y), nx, ny, float(scale), stops, (), foot)
        if self.delta > 0.0:
            pieces = (self,) * (stops.size + 1)
        else:
            probes = np.concatenate(
                [[stops[0] - 1.0], 0.5 * (stops[:-1] + stops[1:]), [stops[-1] + 1.0]]
            )
            regions = self.region_index(*line.point(probes))
            pieces = tuple(self.region_piece(int(r)) for r in regions)
        return replace(line, pieces=pieces)


def _line_crossing(
    curve: _Graph, x0: float, y0: float, nx: float, ny: float, scale: float
) -> float:
    """Where the line ``(x0, y0) + scale η (nx, ny)`` meets the graph, in η.

    Newton on the level ``y − c(x)`` along the line from the crossing of the
    graph's value at ``x0``, the level's derivative along the line being
    ``scale (ny − c′ nx)``. A normal line of a case-2 curve is within 7.2°
    of vertical, so it meets each sine graph once and the iteration is
    quadratic from the first step.
    """
    eta = (curve.height_at(x0) - y0) / (scale * ny)
    for _ in range(NEWTON_STEPS):
        px, py = x0 + scale * eta * nx, y0 + scale * eta * ny
        step = (py - curve.height_at(px)) / (scale * (ny - curve.slope_at(px) * nx))
        eta -= step
        if abs(step) < 1e-15:
            break
    return eta


@dataclass(frozen=True)
class NormalProfile:
    """alpha on the normal line of one stencil, in its stencil coordinate η.

    Stiff note §3.2–3.3, what E4.4's march samples: the line is ``(x0, y0) +
    scale η (nx, ny)``, ``stops`` (increasing) are where the march restarts,
    and ``pieces[k]`` supplies alpha on segment ``k``, between ``stops[k −
    1]`` and ``stops[k]`` (unbounded at either end): at δ = 0 the piece of
    the region the segment lies in, so alpha is one-sided at a jump as
    ``Medium1D.elements`` makes it in 1-D, and at δ > 0 the smooth medium on
    every segment. Made by ``SmoothBand.normal_profile``.
    """

    x0: float
    y0: float
    nx: float
    ny: float
    scale: float
    stops: np.ndarray
    pieces: tuple[Piece2D | SmoothBand, ...]
    foot: tuple[int, float] | None = None

    def point(self, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        step = self.scale * np.asarray(eta, dtype=float)
        return self.x0 + step * self.nx, self.y0 + step * self.ny

    def segment(self, eta: np.ndarray) -> np.ndarray:
        """The segment of each η; a point on a stop is in the segment above it."""
        return np.searchsorted(self.stops, np.asarray(eta, dtype=float), side="right")

    def alpha(self, eta: np.ndarray, segment: int | None = None) -> np.ndarray:
        """alpha at ``eta``, from ``pieces[segment]`` or each point's own segment."""
        eta = np.asarray(eta, dtype=float)
        if segment is not None:
            return self.pieces[segment].alpha(*self.point(eta))
        flat = eta.ravel()
        seg = self.segment(flat)
        out = np.empty_like(flat)
        for k in np.unique(seg):
            mask = seg == k
            out[mask] = self.pieces[k].alpha(*self.point(flat[mask]))
        return out.reshape(eta.shape)

    @property
    def alpha_e(self) -> float:
        """alpha at the anchor, ``α_e`` of the chain."""
        return float(self.alpha(np.zeros(1))[0])

    def alpha_function(self, segment: int) -> Callable[[float], float]:
        """``η ↦ alpha(η, segment)`` in floats, the march's right-hand side (E4.4).

        A constant piece is its value; the smooth band goes through
        ``SmoothBand.alpha_at`` (with the foot curve's distance from ``foot``
        when the line has one) and any other piece through ``_piece_at``, at
        ``point``'s arithmetic, so the function is ``alpha(η, segment)`` to
        rounding at a few microseconds a call.
        """
        piece = self.pieces[segment]
        if isinstance(piece, Constant2D):
            value = float(piece.value)
            return lambda eta: value
        x0, y0, nx, ny, scale = self.x0, self.y0, self.nx, self.ny, self.scale
        if isinstance(piece, SmoothBand) and self.foot is not None:
            j, d0 = self.foot

            def along(eta: float) -> float:
                step = scale * eta
                return piece.alpha_at(x0 + step * nx, y0 + step * ny, (j, d0 + step))

            return along
        at = piece.alpha_at if isinstance(piece, SmoothBand) else _point_alpha(piece)

        def alpha(eta: float) -> float:
            step = scale * eta
            return at(x0 + step * nx, y0 + step * ny)

        return alpha


def _piece_at(piece: Piece2D, x: float, y: float) -> float:
    """One piece's alpha at one point: its float ``alpha_at`` when it has one."""
    if isinstance(piece, Constant2D):
        return float(piece.value)
    at = getattr(piece, "alpha_at", None)
    if at is not None:
        return at(x, y)
    return float(piece.alpha(np.array([x]), np.array([y]))[0])


def _point_alpha(piece: Piece2D) -> Callable[[float, float], float]:
    return lambda x, y: _piece_at(piece, x, y)


# --- domains ----------------------------------------------------------------


@dataclass(frozen=True)
class Domain:
    """A case's geometry: what to straddle, where the Dirichlet rows go, what is cut.

    ``straddle`` are the curves that receive straddling rows (the interfaces,
    or a thin band's midline); ``dirichlet`` the curves carrying a row of
    boundary nodes, in the order E2.2's boundary values are given; ``holes``
    the curves whose ``level < 0`` side is outside the domain.
    """

    material: Band | SmoothBand
    straddle: tuple[Curve, ...]
    dirichlet: tuple[Curve, ...]
    holes: tuple[Curve, ...] = ()

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Strictly inside the strip and outside every hole."""
        x, y = _as_float(x, y)
        ok = (y > Y_MIN) & (y < Y_MAX)
        for hole in self.holes:
            ok &= hole.level(x, y) > 0.0
        return ok


STRIP = (FlatLine(Y_MIN), FlatLine(Y_MAX))
"""The Dirichlet rows every case has, bottom first."""


def case1() -> Domain:
    """Eq. 84 / EABE eq. 32: ``α = 0.2`` on ``y ∈ [0.6, 0.8]``, 1 elsewhere."""
    band = Band(FlatLine(0.6), FlatLine(0.8), Constant2D(0.2), Constant2D(1.0))
    return Domain(band, band.interfaces, STRIP)


def case2() -> Domain:
    """EABE eq. 35: ``0.2 + 0.1 sin 2πx sin 2πy`` between ``0.6, 0.8 + 0.02 sin 2πx``.

    ``Constant2D(1.0)`` elsewhere.
    """
    band = Band(SineGraph(0.6), SineGraph(0.8), SineProduct(0.2, 0.1), Constant2D(1.0))
    return Domain(band, band.interfaces, STRIP)


RING = (0.349, 0.35)
COOLING_RADIUS = 0.05

CASE3_S = 1000.0
"""EABE eq. 40's extremizing parameter at which eq. 37–39, case 3, is recovered."""


def ring_radii(s: float = CASE3_S) -> tuple[float, float]:
    """The insulating ring ``0.35 − 1/s ≤ r ≤ 0.35`` of EABE eq. 40 (§3.3.3).

    ``s = 1000`` gives ``RING`` bit for bit (``0.35 − 1/1000 == 0.349`` in
    double precision, and the midline ``0.35 − 0.5/1000 == 0.3495``). At
    ``s = 10¹¹`` the ring is 1.8e5 ulps of its radius wide and the stored
    width is off by 8e-8 relative, the floor of everything derived from the
    two radii (port notes §2.9).
    """
    if not s > 0.0:
        raise ValueError("the extremizing parameter s must be positive")
    inner = 0.35 - 1.0 / s
    if not inner > COOLING_RADIUS:
        raise ValueError(
            "the ring 0.35 - 1/s must lie outside the cooling circle: s > 1/0.3"
        )
    return (inner, 0.35)


def case3(s: float = CASE3_S) -> Domain:
    """EABE eq. 37–39, or eq. 40 at any ``s``: the insulating ring and cooling circle.

    ``α = 1/(1.5 s) + (1/(3 s)) sin 2πx sin 2πy`` on the ring
    ``0.35 − 1/s ≤ r ≤ 0.35`` (``ring_radii``), 1 elsewhere; eq. 38's
    ``1/1500 + (1/3000) sin 2πx sin 2πy`` on ``0.349 ≤ r ≤ 0.35`` at the
    default ``s = 1000``. The rows straddle the ring's midline, since at
    every count of Fig. 14 the ring is thinner than the spacing; the circle
    ``r = 0.05`` is a Dirichlet row and its inside is cut out.
    """
    inner, outer = ring_radii(s)
    band = Band(
        Circle(inner),
        Circle(outer),
        SineProduct(1.0 / (1.5 * s), 1.0 / (3.0 * s)),
        Constant2D(1.0),
    )
    cooling = Circle(COOLING_RADIUS)
    return Domain(band, (Circle(outer - 0.5 / s),), (*STRIP, cooling), (cooling,))


CASES = {1: case1, 2: case2, 3: case3}


def with_smooth_edges(domain: Domain, delta: float) -> Domain:
    """``domain`` with its band's interfaces tanh edges of width ``delta``.

    The material becomes ``SmoothBand`` over the domain's jump band (its
    own band if it is smooth already, so δ is replaced, not compounded); the
    geometry is untouched, so the straddling rows, Dirichlet rows and holes
    stay where the jump put them and ``build_node_set`` makes the same nodes.
    """
    material = domain.material
    band = material.band if isinstance(material, SmoothBand) else material
    return replace(domain, material=SmoothBand(band, delta))


# --- node sets --------------------------------------------------------------

FREE, ROW, DIRICHLET = 0, 1, 2
"""``NodeSet.kind`` values: scattered, straddling row, Dirichlet row."""

ROW_OFFSETS = (0.5, 0.5 + sqrt(3.0) / 2.0, 0.5 + sqrt(3.0))
"""Normal offsets of the straddling rows in units of the row spacing.

The MATLAB's hexagonal layout: the innermost pair half a spacing off the
curve, the next pair a further √3/2 out and staggered by half a spacing
along it, the third another √3/2 out and in line with the first.
"""

ROW_STAGGER = (0.0, 0.5, 0.0)

STRADDLE_FRACTION = 0.95
"""``numIntNodes = round(0.95 √N)`` nodes per unit length of row."""

STEP_SCALE = 0.05
"""The MATLAB's repulsion step, ``0.05 √(2500/N) / iteration``."""


def _round_half_up(x: float) -> int:
    """MATLAB's ``round``: halves go up (900 nodes give 29 per row, not 28)."""
    return int(np.floor(x + 0.5))


def straddle_count(n: int) -> int:
    """Nodes per unit length of a fixed row in an ``n``-node set (``numIntNodes``)."""
    return _round_half_up(STRADDLE_FRACTION * sqrt(n))


def row_count(curve: Curve, h: float) -> int:
    """How many nodes a row along ``curve`` holds at spacing about ``h``."""
    return max(1, _round_half_up(curve.length / h))


def step_delta(n: int, iteration: int) -> float:
    """Distance every free node moves at the 1-based ``iteration``."""
    return STEP_SCALE * sqrt(2500.0 / n) / iteration


@dataclass(frozen=True)
class Row:
    """One row of fixed nodes: its curve, its offset (in spacings) and its indices."""

    curve: int
    offset: float
    h: float
    index: np.ndarray


@dataclass(frozen=True)
class NodeSet:
    """``n`` nodes on the strip with ``0 <= x < 1``, labelled by ``kind``.

    ``straddle_rows`` are ordered by curve, then offset (innermost first),
    then side (``-`` before ``+``); paired rows share their parameters, so
    ``index[i]`` of the ``-0.5`` row and of the ``+0.5`` row face each other
    across the curve. ``boundary_index`` names the Dirichlet curve of a node
    in a Dirichlet row and is ``-1`` elsewhere.
    """

    x: np.ndarray
    y: np.ndarray
    kind: np.ndarray
    h: float
    straddle_rows: tuple[Row, ...] = ()
    dirichlet_rows: tuple[Row, ...] = ()
    boundary_index: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def xy(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])

    @property
    def dirichlet(self) -> np.ndarray:
        return self.kind == DIRICHLET

    @property
    def free(self) -> np.ndarray:
        return self.kind == FREE

    def rows_of(self, curve: int, offset: float) -> tuple[Row, Row]:
        """The ``-offset`` and ``+offset`` rows straddling ``straddle[curve]``."""
        found = [r for r in self.straddle_rows if r.curve == curve]
        lo = [r for r in found if r.offset == -offset]
        hi = [r for r in found if r.offset == offset]
        if len(lo) != 1 or len(hi) != 1:
            raise KeyError(f"no row pair at offset {offset} on straddle curve {curve}")
        return lo[0], hi[0]


def build_node_set(
    domain: Domain,
    n: int,
    seed: int = 0,
    rows_per_side: int = 3,
    iterations: int = 100,
    neighbors: int = 10,
) -> NodeSet:
    """``n`` nodes on ``domain`` the way ``ExeprepRBFHeatLaplace4.m`` makes them.

    Fixed rows first: ``2 * rows_per_side`` rows straddling every curve of
    ``domain.straddle`` with ``round(0.95 √n)`` nodes per unit length, then a
    row on each Dirichlet curve; the rest are drawn uniformly, kept out of
    the straddled bands and the holes, and moved ``iterations`` times by
    ``step_delta`` along the unit resultant of ``1/r^4`` repulsions from
    their ``neighbors`` nearest nodes, periodic in x. A free node that
    leaves the domain or enters a band is redrawn (the MATLAB redraws its
    ``y``). Deterministic in ``seed``.
    """
    if not 1 <= rows_per_side <= len(ROW_OFFSETS):
        raise ValueError(f"rows_per_side must be 1..{len(ROW_OFFSETS)}")
    count = straddle_count(n)
    h = 1.0 / count
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    kinds: list[np.ndarray] = []
    bounds: list[np.ndarray] = []
    straddle_rows: list[Row] = []
    dirichlet_rows: list[Row] = []
    total = 0

    def add(x: np.ndarray, y: np.ndarray, kind: int, boundary: int) -> np.ndarray:
        nonlocal total
        index = np.arange(total, total + len(x))
        total += len(x)
        xs.append(wrap_x(x))
        ys.append(np.asarray(y, dtype=float))
        kinds.append(np.full(len(x), kind, dtype=np.int8))
        bounds.append(np.full(len(x), boundary, dtype=int))
        return index

    clearances: list[tuple[Curve, float]] = []
    for ci, curve in enumerate(domain.straddle):
        m = row_count(curve, h)
        h_row = curve.length / m
        clearances.append((curve, ROW_OFFSETS[rows_per_side - 1] * h_row))
        for k in range(rows_per_side):
            s = (np.arange(m) + ROW_STAGGER[k]) / m
            px, py = curve.point(s)
            nx, ny = curve.normal(s)
            for sign in (-1.0, 1.0):
                off = sign * ROW_OFFSETS[k] * h_row
                index = add(px + off * nx, py + off * ny, ROW, -1)
                straddle_rows.append(Row(ci, sign * ROW_OFFSETS[k], h_row, index))
    for ci, curve in enumerate(domain.dirichlet):
        m = row_count(curve, h)
        px, py = curve.point(np.arange(m) / m)
        index = add(px, py, DIRICHLET, ci)
        dirichlet_rows.append(Row(ci, 0.0, curve.length / m, index))

    n_free = n - total
    if n_free < 1:
        raise ValueError(
            f"n = {n} leaves no free nodes after {total} fixed ones; "
            f"the rows alone need more than {total} nodes"
        )
    fixed = np.column_stack([np.concatenate(xs), np.concatenate(ys)])

    def admissible(xy: np.ndarray) -> np.ndarray:
        ok = domain.contains(xy[:, 0], xy[:, 1])
        for curve, clearance in clearances:
            ok &= np.abs(curve.level(xy[:, 0], xy[:, 1])) > clearance
        return ok

    rng = np.random.default_rng(seed)
    free = _draw(rng, n_free, admissible)
    k = min(neighbors + 1, n)
    for it in range(1, iterations + 1):
        allxy = np.vstack([fixed, free])
        idx, _ = knn(allxy, k, query=free)
        dx, dy = offsets(allxy, idx, centre=free)
        r = np.hypot(dx, dy)
        with np.errstate(divide="ignore"):
            w = np.where(r > 0.0, r**-5.0, 0.0)
        fx, fy = -(dx * w).sum(axis=1), -(dy * w).sum(axis=1)
        norm = np.hypot(fx, fy)
        with np.errstate(invalid="ignore", divide="ignore"):
            ux = np.where(norm > 0.0, fx / norm, 0.0)
            uy = np.where(norm > 0.0, fy / norm, 0.0)
        delta = step_delta(n, it)
        free = free + delta * np.column_stack([ux, uy])
        free[:, 0] = wrap_x(free[:, 0])
        bad = ~admissible(free)
        if bad.any():
            free[bad] = _draw(rng, int(bad.sum()), admissible)

    add(free[:, 0], free[:, 1], FREE, -1)
    return NodeSet(
        x=np.concatenate(xs),
        y=np.concatenate(ys),
        kind=np.concatenate(kinds),
        h=h,
        straddle_rows=tuple(straddle_rows),
        dirichlet_rows=tuple(dirichlet_rows),
        boundary_index=np.concatenate(bounds),
    )


def _draw(rng: np.random.Generator, m: int, admissible) -> np.ndarray:
    """``m`` points uniform on the strip, rejecting inadmissible ones."""
    out = np.empty((0, 2))
    while len(out) < m:
        cand = rng.uniform(0.0, 1.0, size=(2 * (m - len(out)) + 8, 2))
        out = np.vstack([out, cand[admissible(cand)]])
    return out[:m]
