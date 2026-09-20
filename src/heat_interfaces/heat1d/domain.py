"""Grids and materials for the 1-D heat operator ``d/dx (alpha d/dx)`` on [-1, 1].

Dissertation §4.1–4.2 (eq. 51, 64, 75) and Martin & Fornberg 2017 §2.2.2. The
grid is equispaced with both ends on nodes so the Dirichlet conditions are
plain rows. Where an interface then falls relative to the nodes is fixed by
the node count, not by a switch: with ``h = 2 / (n - 1)`` an interface at 0
sits on a node for odd ``n`` and halfway between two nodes for even ``n``,
and the dissertation's pair (0, 0.5) can never both sit mid-cell. The knee
studies (E3) select the placement through ``node_counts`` / ``grid_for``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import factorial
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

    ``taylor`` serves E1.2's continuity matrices; a smooth edge (E3) has the
    same expansion from either side and may ignore ``side``.
    """

    interfaces: tuple[float, ...]

    def alpha(self, x: np.ndarray) -> np.ndarray: ...

    def alpha_x(self, x: np.ndarray) -> np.ndarray: ...

    def taylor(self, i: int, side: Side, degree: int) -> np.ndarray: ...


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
