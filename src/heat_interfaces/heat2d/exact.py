"""Separable solutions on the strip, and the radial equilibrium through case 3's ring.

Dissertation eq. 86 and EABE eq. 34: with ``α`` piecewise constant in ``y``
and ``u = e^{c t} sin(κ x) v(y)``, ``u_t = ∇·(α ∇u)`` reduces to
``v'' = (κ² + c/α) v`` on each layer, so ``v`` is a pair of exponentials per
layer; continuity of ``v`` and of ``α v'`` at each interface, ``v(0) = 0``
and ``v(1) = 1`` fix the ``2m`` constants (the ``c₁ … c₆`` of the papers and of
MATLAB ``laplaceSetup.m``). The control problem (``α ≡ 1``, ``c = 0``) is the
one-layer case ``v = sinh(κ y) / sinh κ``.

``RingMode`` is the other separable solution this repo needs: the harmonic
mode ``u = R(r) cos(mθ)`` through concentric rings of constant α,
``R = a_k r^m + b_k r^-m`` on each ring and ``r^m`` at the centre, with ``R``
and the flux ``α R'`` continuous at every radius. It is not a solution of any
case (case 3's ring α varies, and the strip is no annulus) but it is smooth
away from the ring, regular at the centre, and satisfies every interface
condition the translated basis enforces at the ring's 1500 : 1 contrast, so
it is what E2.7 reads through the fine stencils to measure the resampling.
(The radial equilibrium ``a + b ln r`` was tried first; its fifth derivative
at the cooling circle is 1e7 and the reading measured that, not the ring.)

``SeparableReference`` is ``LayeredExact`` for a flat band of any edge width
(E4.2, ``docs/stiff-diffusion.md`` §3.1 and §4.1): with α a function of ``y``
alone the same separation holds, and ``v`` is E3.2's Chebyshev-element
collocation in ``y`` on the band's 1-D medium (``profile_medium``), cut at
the ``EDGE_CUTS`` of each tanh edge.

``ProductGridReference`` is plan D4's Fourier × Chebyshev product grid for
the bands that do not separate (E4.7, stiff note §4.6): case 2's sine pair,
at any edge width including the jump, with either piece. A shear ``ShearMap``
makes both curves coordinate lines, so the elements in the new coordinate
are E3.2's again and the reference is spectral in both directions.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import spsolve

from ..heat1d.domain import (
    Constant,
    Medium1D,
    OnInterval,
    PiecewiseAlpha,
    SmoothEdges,
)
from ..heat1d.exact import (
    ChebyshevPieces,
    chebyshev_lobatto,
    chebyshev_profile,
    split_elements,
)
from .domain import (
    CASE3_S,
    TWO_PI,
    Y_MAX,
    Y_MIN,
    Band,
    Circle,
    Constant2D,
    FlatLine,
    SineGraph,
    SmoothBand,
    case1,
    ring_gap,
    ring_radii,
)


@dataclass(frozen=True)
class LayeredExact:
    """``u(x, y, t) = e^{c t} sin(κ x) v(y)`` for ``α = alphas[k]`` on layer ``k``.

    Layer ``k`` is ``breaks[k-1] <= y < breaks[k]`` (bottom to top, so a
    point on a break belongs to the layer above it; ``u`` is continuous, so
    only ``v_y`` notices). ``growth`` is ``c``: 0 for equilibrium,
    ``c_t = 1`` in the dissertation's parabolic case 1.
    """

    alphas: tuple[float, ...]
    breaks: tuple[float, ...]
    growth: float = 0.0
    wavenumber: float = TWO_PI
    _coefficients: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.breaks) != len(self.alphas) - 1:
            raise ValueError("one break fewer than layers is needed")
        if any(a <= 0.0 for a in self.alphas):
            raise ValueError("every layer needs a positive α")
        if not all(0.0 < b < 1.0 for b in self.breaks) or list(self.breaks) != sorted(
            self.breaks
        ):
            raise ValueError("breaks must increase inside (0, 1)")
        object.__setattr__(self, "_coefficients", self._solve())

    @property
    def kappas(self) -> np.ndarray:
        """``sqrt(κ² + c/α)`` per layer.

        Eq. 86's ``sqrt(4π² + c_t)`` outside the band and ``sqrt(4π² + 5c_t)`` in it.
        """
        a = np.asarray(self.alphas, dtype=float)
        return np.sqrt(self.wavenumber**2 + self.growth / a)

    @property
    def edges(self) -> np.ndarray:
        return np.array([0.0, *self.breaks, 1.0])

    def _solve(self) -> np.ndarray:
        # Unknowns (a_k, b_k) per layer, v_k(y) = a_k e^{κ_k s} + b_k e^{-κ_k s} with
        # s = y - (the layer's lower edge), so no exponential exceeds e^{κ}.
        m = len(self.alphas)
        kap = self.kappas
        edges = self.edges
        rows = np.zeros((2 * m, 2 * m))
        rhs = np.zeros(2 * m)
        rows[0, 0:2] = [1.0, 1.0]  # v(0) = 0
        r = 1
        for k, yb in enumerate(self.breaks):
            below = kap[k] * (yb - edges[k])
            e_b = np.array([np.exp(below), np.exp(-below)])
            e_a = np.array([1.0, 1.0])  # layer k+1 starts at yb
            rows[r, 2 * k : 2 * k + 2] = e_b
            rows[r, 2 * k + 2 : 2 * k + 4] = -e_a
            rows[r + 1, 2 * k : 2 * k + 2] = self.alphas[k] * kap[k] * e_b * [1.0, -1.0]
            rows[r + 1, 2 * k + 2 : 2 * k + 4] = (
                -self.alphas[k + 1] * kap[k + 1] * e_a * [1.0, -1.0]
            )
            r += 2
        top = kap[-1] * (1.0 - edges[-2])
        rows[r, 2 * m - 2 : 2 * m] = [np.exp(top), np.exp(-top)]
        rhs[r] = 1.0  # v(1) = 1
        return np.linalg.solve(rows, rhs)

    def layer(self, y: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.asarray(self.breaks, dtype=float), y, side="right")

    def _pieces(self, y: np.ndarray, derivative: bool) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        k = self.layer(y)
        kap = self.kappas[k]
        z = kap * (y - self.edges[k])
        a = self._coefficients[2 * k]
        b = self._coefficients[2 * k + 1]
        if derivative:
            return kap * (a * np.exp(z) - b * np.exp(-z))
        return a * np.exp(z) + b * np.exp(-z)

    def v(self, y: np.ndarray) -> np.ndarray:
        return self._pieces(y, derivative=False)

    def v_y(self, y: np.ndarray) -> np.ndarray:
        return self._pieces(y, derivative=True)

    def __call__(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.exp(self.growth * t) * np.sin(self.wavenumber * x) * self.v(y)

    def flux_y(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """``α u_y``, continuous across the breaks."""
        x = np.asarray(x, dtype=float)
        a = np.asarray(self.alphas, dtype=float)[self.layer(y)]
        return np.exp(self.growth * t) * np.sin(self.wavenumber * x) * a * self.v_y(y)


def control_exact(growth: float = 0.0) -> LayeredExact:
    """``α ≡ 1``: ``u = e^{c t} sin(2πx) sinh(κ y) / sinh κ``, the control problem."""
    return LayeredExact((1.0,), (), growth)


def case1_exact(growth: float = 0.0) -> LayeredExact:
    """Eq. 34 / dissertation eq. 86: ``α = 0.2`` on ``[0.6, 0.8]``, 1 elsewhere."""
    return LayeredExact((1.0, 0.2, 1.0), (0.6, 0.8), growth)


REFERENCE_N_CHEB, REFERENCE_MAX_WIDTH = 20, 0.1
"""The separable reference's resolution: Chebyshev nodes per element, widest element.

E3.2's recipe for the same medium in 1-D (stiff note §2.1, ``scripts/
heat1d_stiff.py``'s ``N_CHEB``, ``MAX_WIDTH``): twenty nodes resolve a tanh
element to 1e-11 and keep the collocation's round-off floor, which grows
with the node count on the δ-wide elements, near 1e-12. On case 1 the
reference agrees with 24 nodes on 0.05-wide elements to 1e-12 (δ = 0.04),
1e-11 (δ = 0.0025) and 2e-11 (δ = 5e-4), and with ``case1_exact`` at δ = 0
to 4e-13 (stiff note §4.1).
"""


def profile_medium(material: Band | SmoothBand) -> OnInterval:
    """The band's α along ``y`` as a 1-D medium on ``[0, 1]``: flat, constant pieces.

    Stiff note §3.1: ``SmoothEdges`` over ``outside | inside | outside`` at the
    two lines, of the band's δ (0 for a ``Band``), the closed band owning
    both lines as ``Band.piece_index`` does. On such a band
    ``SmoothBand.alpha(x, y)`` is this medium's ``alpha(y)`` bit for bit, and
    ``gradient``'s y component its ``alpha_x``.
    """
    band = material.band if isinstance(material, SmoothBand) else material
    delta = material.delta if isinstance(material, SmoothBand) else 0.0
    lines = (band.lower, band.upper)
    pieces = (band.outside, band.inside)
    if not all(isinstance(c, FlatLine) for c in lines):
        raise ValueError("a separable reference needs flat interfaces")
    if not all(isinstance(p, Constant2D) for p in pieces):
        raise ValueError("a separable reference needs constant pieces")
    out, inside = (Constant(p.value) for p in pieces)
    jump = PiecewiseAlpha(
        (band.lower.c, band.upper.c), (out, inside, out), ("right", "left")
    )
    return OnInterval(SmoothEdges(jump, delta), Y_MIN, Y_MAX)


@dataclass(frozen=True)
class SeparableReference:
    """``u(x, y, t) = e^{c t} sin(κ x) v(y)`` through a flat band of edge width δ.

    E4.2 (#33), plan D4: with α a function of ``y`` alone,
    ``u_t = ∇·(α ∇u)`` separates as in eq. 86, and ``v`` solves

        (α v′)′ − (κ² α + c) v = 0,   v(0) = 0,   v(1) = 1,

    the elliptic problem at ``c = 0`` and, at ``c > 0``, the 1-D parabolic
    problem in ``y``: ``e^{ct} v(y)`` solves ``w_t = (α w_y)_y − κ² α w``
    with ``w(0, t) = 0``, ``w(1, t) = e^{ct}``, exactly in ``t``. So no time
    integrator enters (none of Radau's tolerance floor, stiff note §2.1),
    and the reference is callable at any ``t``, BD4's analytic history at
    ``t < 0`` included. ``v`` is ``heat1d.exact.chebyshev_profile`` on
    ``medium``'s elements (``profile_medium``: E3.2's ``EDGE_CUTS`` about
    each edge, wider elements split at ``max_width``), ``v`` and ``α v′``
    matched at every cut. At δ = 0 it is ``LayeredExact`` to 4e-13; at
    δ > 0 its accuracy is its agreement with a finer resolution (stiff note
    §4.1). The methods mirror ``LayeredExact``'s; a point on an element edge
    reads the element above it, as ``LayeredExact`` reads the layer above a
    break.
    """

    medium: Medium1D
    growth: float = 0.0
    wavenumber: float = TWO_PI
    n_cheb: int = REFERENCE_N_CHEB
    max_width: float | None = REFERENCE_MAX_WIDTH
    pieces: ChebyshevPieces = field(init=False, repr=False, compare=False)
    nodal: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        cp, v = chebyshev_profile(
            self.medium,
            0.0,
            1.0,
            self.n_cheb,
            self.growth,
            self.max_width,
            self.wavenumber,
        )
        v_y = cp.derivative @ v
        object.__setattr__(self, "pieces", cp)
        # v, v', and the flux α v' from each element's own α (one-sided at a
        # jump), per node.
        object.__setattr__(self, "nodal", np.stack([v, v_y, cp.alpha * v_y]))

    @property
    def elements(self) -> int:
        return len(self.pieces.edges) - 1

    @property
    def unknowns(self) -> int:
        """The collocation nodes that are not element ends (E3.2's count)."""
        return int(self.pieces.interior.size)

    def v(self, y: np.ndarray) -> np.ndarray:
        return self.pieces.evaluate(self.nodal[0], y)

    def v_y(self, y: np.ndarray) -> np.ndarray:
        return self.pieces.evaluate(self.nodal[1], y)

    def _factor(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.exp(self.growth * t) * np.sin(self.wavenumber * x)

    def __call__(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        return self._factor(x, t) * self.v(y)

    def flux_y(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """``α u_y``, continuous across the band."""
        return self._factor(x, t) * self.pieces.evaluate(self.nodal[2], y)

    def top(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """The Dirichlet row at ``y = 1``: ``e^{ct} sin κx``, exactly (``v(1) = 1``)."""
        return self._factor(x, t)

    def boundary_values(self) -> tuple[float, Callable[..., np.ndarray]]:
        """``(0, top)`` for the Dirichlet curves of ``STRIP`` (y = 0, then y = 1).

        What ``solve_equilibrium`` and ``march_parabolic`` take as ``values``;
        ``top`` takes ``t`` with a default, so one tuple serves both.
        """
        return 0.0, self.top


def separable_reference(
    material: Band | SmoothBand,
    growth: float = 0.0,
    n_cheb: int = REFERENCE_N_CHEB,
    max_width: float | None = REFERENCE_MAX_WIDTH,
) -> SeparableReference:
    """The ``SeparableReference`` of a flat band with constant pieces, ``κ = 2π``."""
    return SeparableReference(
        profile_medium(material), growth, TWO_PI, n_cheb, max_width
    )


def case1_reference(
    delta: float = 0.0,
    growth: float = 0.0,
    n_cheb: int = REFERENCE_N_CHEB,
    max_width: float | None = REFERENCE_MAX_WIDTH,
) -> SeparableReference:
    """Case 1 with tanh edges of width ``delta``; ``case1_exact`` at δ = 0 to 4e-13."""
    band = SmoothBand(case1().material, delta)
    return separable_reference(band, growth, n_cheb, max_width)


PRODUCT_N_X = 49
"""Fourier points in ``x`` of the product-grid reference; odd, so no Nyquist mode.

On case 2 at δ ∈ {0, 0.0025, 0.005, 0.01} the 33-point grid is 8e-12 from
the 49- and 65-point ones and those two agree to below that (stiff note §4.6),
so 49 carries a margin; the material's x-variation is a few harmonics of the
sine and the edge's width moves by 5 % along it.
"""


SHEAR_NEWTON_STEPS = 30
"""Newton iterations of ``ShearMap.eta`` (quadratic; five or six are used)."""


@dataclass(frozen=True)
class ShearMap:
    """``y = η + a sin(k x) β(η)``: the strip with two sine graphs made straight.

    ``β(η) = η (1 − η)(A + B η)`` is the cubic with ``β(0) = β(1) = 0`` and
    ``β(c₁) = β(c₂) = 1``, so the rows ``y = 0`` and ``y = 1`` are ``η = 0``
    and ``η = 1`` and the graphs ``c_k + a sin kx`` of case 2's band are the
    lines ``η = c_k``, whatever ``x``. A signed distance to either graph
    therefore vanishes on ``η = c_k`` exactly, and a tanh edge in it is a
    tanh in ``η − c_k`` whose width moves with ``x`` only through the metric
    (by ±5 % at ``a = 0.02``): E3.2's elements cut at ``c_k ± EDGE_CUTS δ``
    resolve it. The map is monotone in ``η`` while ``|a β′| < 1`` (0.17 at
    ``a = 0.02``), which ``__post_init__`` checks. ``a = 0`` is the identity.
    """

    amplitude: float
    wavenumber: float
    lower: float
    upper: float
    coefficients: tuple[float, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        c = np.array([self.lower, self.upper])
        m = np.column_stack([c * (1.0 - c), c**2 * (1.0 - c)])
        a, b = np.linalg.solve(m, np.ones(2))
        object.__setattr__(self, "coefficients", (float(a), float(b)))
        eta = np.linspace(0.0, 1.0, 1001)
        if abs(self.amplitude) * np.abs(self.beta_prime(eta)).max() >= 1.0:
            raise ValueError("the shear folds the strip: |a β′| must stay below 1")

    def beta(self, eta: np.ndarray) -> np.ndarray:
        a, b = self.coefficients
        return eta * (1.0 - eta) * (a + b * eta)

    def beta_prime(self, eta: np.ndarray) -> np.ndarray:
        a, b = self.coefficients
        return a + 2.0 * (b - a) * eta - 3.0 * b * eta**2

    def y(self, x: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return eta + self.amplitude * np.sin(self.wavenumber * x) * self.beta(eta)

    def y_x(self, x: np.ndarray, eta: np.ndarray) -> np.ndarray:
        k = self.wavenumber
        return self.amplitude * k * np.cos(k * x) * self.beta(eta)

    def y_eta(self, x: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return 1.0 + self.amplitude * np.sin(self.wavenumber * x) * self.beta_prime(eta)

    def eta(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The inverse at fixed ``x``: Newton on the cubic from ``η = y``.

        Monotone in ``η`` (``__post_init__``), so the iteration converges from
        any point of the strip; ``SHEAR_NEWTON_STEPS`` without settling to
        1e-12 is refused rather than returned.
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        eta = np.array(y, dtype=float, copy=True)
        step = np.full_like(eta, np.inf)
        for _ in range(SHEAR_NEWTON_STEPS):
            step = (self.y(x, eta) - y) / self.y_eta(x, eta)
            eta -= step
            if np.all(np.abs(step) < 1e-15):
                break
        if np.abs(step).max(initial=0.0) > 1e-12:
            raise RuntimeError(
                f"the shear's inverse did not converge in {SHEAR_NEWTON_STEPS} "
                "Newton steps"
            )
        return eta


def fourier_derivative(n: int) -> np.ndarray:
    """``d/dx`` on ``n`` (odd) equispaced points of the period ``[0, 1)``.

    Trefethen, *Spectral Methods in MATLAB*, eq. (3.10) for odd ``n``:
    ``½ (−1)^{i−j} csc(π (i − j)/n)`` off the diagonal, times ``2π`` for the
    unit period; exact on the trigonometric polynomials of degree ``(n−1)/2``.
    """
    if n < 3 or n % 2 == 0:
        raise ValueError("the Fourier grid needs an odd number of points, 3 or more")
    k = np.arange(n)
    diff = k[:, None] - k[None, :]
    off = diff != 0
    d = np.zeros((n, n))
    d[off] = 0.5 * (-1.0) ** diff[off] / np.sin(np.pi * diff[off] / n)
    return 2.0 * np.pi * d


def shear_of(material: Band | SmoothBand) -> ShearMap:
    """The ``ShearMap`` that straightens ``material``'s two graphs.

    Both interfaces must be graphs of one shape: two ``FlatLine``s (the
    identity) or two ``SineGraph``s of equal amplitude and wavenumber, as
    case 2's are; anything else has no such shear and is refused.
    """
    lower, upper = material.interfaces
    if isinstance(lower, FlatLine) and isinstance(upper, FlatLine):
        return ShearMap(0.0, TWO_PI, lower.c, upper.c)
    if (
        isinstance(lower, SineGraph)
        and isinstance(upper, SineGraph)
        and lower.amplitude == upper.amplitude
        and lower.wavenumber == upper.wavenumber
    ):
        return ShearMap(lower.amplitude, lower.wavenumber, lower.c, upper.c)
    raise ValueError(
        "a product-grid reference needs two flat lines or two parallel sine graphs"
    )


@dataclass(frozen=True)
class ProductGridReference:
    """``u = e^{ct} U(x, y)`` through a band of sine graphs, by a product grid.

    Plan D4 and stiff note §4.6 (E4.7, #38). ``U`` solves
    ``∇·(α ∇U) − c U = 0`` on the strip with ``U = 0`` on ``y = 0`` and
    ``U = sin κx`` on ``y = 1``, so ``e^{ct} U`` solves ``u_t = ∇·(α ∇u)``
    with ``e^{ct} sin κx`` on the top row, exactly in ``t``: the elliptic
    problem at ``c = 0`` and E2.5's parabolic one at ``c = 1``, with BD4's
    analytic history at ``t < 0`` for free, as ``SeparableReference`` has it
    on a flat band. In the ``ShearMap`` coordinates ``(x, η)``, with
    ``J = y_η``,

        ∇·(α ∇U) = (1/J) [∂_x F_x + ∂_η F_η],
        F_x = J α U_x − α y_x U_η,   F_η = −α y_x U_x + α (1 + y_x²)/J U_η,

    collocated on ``n_x`` Fourier points in ``x`` (``fourier_derivative``)
    times E3.2's Chebyshev elements in ``η`` (``n_cheb`` nodes per element,
    the separable reference's cuts about ``η = c_k`` and ``max_width``
    splits), the flux form differentiated as it stands. At the elements'
    shared ends ``U`` and ``F_η`` — the flux through ``η = const``, which is
    the normal flux through the curve there — are matched, with each
    element's own α: one-sided at a jump, so δ = 0 is the same solver with
    the band's pieces on their elements. The rows are equilibrated before
    SuperLU's ``NATURAL`` ordering factors the banded matrix: unscaled, the
    rows span seven decades (Dirichlet, continuity, flux, interior) and
    partial pivoting leaves 1e-7 of spurious modes at δ = 0.0025 where the
    exact answer has none. The solution is read at any point spectrally:
    ``η`` from ``ShearMap.eta``, barycentric Lagrange in its element, the
    trigonometric interpolant in ``x``. At ``a = 0`` with constant pieces
    it is ``SeparableReference`` to 2e-12 (stiff note §4.6).
    """

    material: Band | SmoothBand
    growth: float = 0.0
    n_x: int = PRODUCT_N_X
    n_cheb: int = REFERENCE_N_CHEB
    max_width: float | None = REFERENCE_MAX_WIDTH
    wavenumber: float = TWO_PI
    shear: ShearMap = field(init=False, repr=False, compare=False)
    edges: np.ndarray = field(init=False, repr=False, compare=False)
    eta: np.ndarray = field(init=False, repr=False, compare=False)
    values: np.ndarray = field(init=False, repr=False, compare=False)
    seconds: float = field(init=False, repr=False, compare=False)
    _modes: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        t0 = time.perf_counter()
        shear = shear_of(self.material)
        band = (
            self.material.band
            if isinstance(self.material, SmoothBand)
            else (self.material)
        )
        delta = self.material.delta if isinstance(self.material, SmoothBand) else 0.0
        stand_in = Band(
            FlatLine(shear.lower),
            FlatLine(shear.upper),
            Constant2D(1.0),
            Constant2D(1.0),
        )
        edges, _ = split_elements(
            *profile_medium(SmoothBand(stand_in, delta)).elements(), self.max_width
        )
        cheb, d, _ = chebyshev_lobatto(self.n_cheb)
        width = self.n_cheb + 1
        count = len(edges) - 1
        lo, hi = edges[:-1, None], edges[1:, None]
        eta = ((lo + hi) / 2 + (hi - lo) / 2 * cheb[None, :]).ravel()
        n_x = self.n_x
        x = np.arange(n_x) / n_x
        xx, ee = np.meshgrid(x, eta)
        yy = shear.y(xx, ee)
        y_x, jac = shear.y_x(xx, ee), shear.y_eta(xx, ee)
        if delta > 0.0:
            alpha = self.material.alpha(xx, yy)
        else:
            mid = np.repeat(0.5 * (edges[:-1] + edges[1:]), width)
            region = (mid > shear.lower).astype(int) + (mid > shear.upper)
            alpha = np.empty_like(xx)
            for r in (0, 1, 2):
                rows = region == r
                alpha[rows] = band.region_piece(r).alpha(xx[rows], yy[rows])
        size = eta.size * n_x
        d_eta = sp.block_diag(
            [sp.csr_array(2.0 / (edges[k + 1] - edges[k]) * d) for k in range(count)],
            format="csr",
        )
        u_x = sp.kron(sp.identity(eta.size), sp.csr_array(fourier_derivative(n_x)))
        u_eta = sp.kron(d_eta, sp.identity(n_x))

        def diag(a: np.ndarray) -> sp.dia_array:
            return sp.diags_array(a.ravel())

        f_x = diag(jac * alpha) @ u_x - diag(alpha * y_x) @ u_eta
        f_eta = (
            -diag(alpha * y_x) @ u_x + diag(alpha * (1.0 + y_x**2) / jac) @ u_eta
        ).tocsr()
        operator = (u_x @ f_x + u_eta @ f_eta - self.growth * diag(jac)).tocsr()

        def rows_at(p: int) -> np.ndarray:
            return p * n_x + np.arange(n_x)

        starts = np.arange(count) * width
        ends = starts + self.n_cheb
        eye = sp.identity(size, format="csr")
        placed = [(rows_at(starts[0]), eye[rows_at(starts[0])])]
        for k in range(count - 1):
            left, right = rows_at(ends[k]), rows_at(starts[k + 1])
            placed.append((left, eye[left] - eye[right]))
            placed.append((right, f_eta[left] - f_eta[right]))
        placed.append((rows_at(ends[-1]), eye[rows_at(ends[-1])]))
        target = np.concatenate([t for t, _ in placed])
        keep = np.ones(size)
        keep[target] = 0.0
        stacked = sp.vstack([m for _, m in placed], format="csr")
        put = sp.csr_array(
            (np.ones(target.size), (target, np.arange(target.size))),
            shape=(size, target.size),
        )
        matrix = (sp.diags_array(keep) @ operator + put @ stacked).tocsr()
        rhs = np.zeros(size)
        rhs[rows_at(ends[-1])] = np.sin(self.wavenumber * x)
        scale = 1.0 / np.abs(matrix).max(axis=1).toarray().ravel()
        matrix = sp.diags_array(scale) @ matrix
        u = spsolve(sp.csc_array(matrix), scale * rhs, permc_spec="NATURAL")
        values = np.asarray(u).reshape(eta.size, n_x)
        object.__setattr__(self, "shear", shear)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "_modes", np.fft.rfft(values, axis=1) / n_x)
        object.__setattr__(self, "seconds", time.perf_counter() - t0)

    @property
    def elements(self) -> int:
        return len(self.edges) - 1

    @property
    def unknowns(self) -> int:
        return int(self.values.size)

    def steady(self, x: np.ndarray, y: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """``U(x, y)``: ``η`` by the shear's inverse, then the spectral interpolant."""
        x, y = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (x, y)))
        flat_x, flat_y = x.ravel(), y.ravel()
        eta = self.shear.eta(flat_x, flat_y)
        cheb, _, weights = chebyshev_lobatto(self.n_cheb)
        width = self.n_cheb + 1
        element = np.clip(
            np.searchsorted(self.edges[1:-1], eta, side="right"), 0, self.elements - 1
        )
        lo, hi = self.edges[element], self.edges[element + 1]
        t = (2.0 * eta - (lo + hi)) / (hi - lo)
        m = np.arange(self._modes.shape[1])
        factor = np.where(m == 0, 1.0, 2.0)
        out = np.empty(flat_x.size)
        for start in range(0, flat_x.size, chunk):
            part = slice(start, start + chunk)
            gap = t[part, None] - cheb[None, :]
            hit = gap == 0.0
            gap[hit] = 1.0
            w = weights[None, :] / gap
            w /= w.sum(axis=1, keepdims=True)
            on = hit.any(axis=1)
            w[on] = hit[on]
            rows = element[part, None] * width + np.arange(width)[None, :]
            modes = np.einsum("pj,pjm->pm", w, self._modes[rows])
            waves = np.exp(2j * np.pi * m[None, :] * flat_x[part, None])
            out[part] = np.einsum("pm,pm->p", factor * modes, waves).real
        return out.reshape(x.shape)

    def __call__(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        return np.exp(self.growth * t) * self.steady(x, y)

    def top(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """The Dirichlet row at ``y = 1``: ``e^{ct} sin κx``, as the grid imposes it."""
        return np.exp(self.growth * t) * np.sin(self.wavenumber * np.asarray(x))

    def boundary_values(self) -> tuple[float, Callable[..., np.ndarray]]:
        """``(0, top)`` for ``STRIP``'s Dirichlet curves, as the separable one's."""
        return 0.0, self.top


@dataclass(frozen=True)
class RingMode:
    """``u = R(r) cos(mθ)``, harmonic through concentric rings of constant α.

    Ring ``k`` is ``radii[k-1] <= r < radii[k]`` (the innermost reaches the
    centre, the outermost infinity; on a radius, the ring above it, as with
    ``LayeredExact``). ``R_k = a_k r^m + b_k r^-m`` with ``b_0 = 0``, so ``u``
    is the harmonic polynomial ``Re (x + iy)^m`` scaled at the centre;
    ``R`` and ``α R'`` are continuous at every radius, and the whole is
    scaled so that ``u = 1`` at ``(r, θ) = (scale_radius, 0)``. Across an
    insulating ring ``R`` climbs by about ``m α_out / α_ring`` times the
    ring's width times ``r^(m-1)``, relative to ``r^m`` inside: at case 3's
    ring for ``m = 2`` that is 1.05 against 0.12, which the scaling turns
    into 0.665 against 0.078 in ``radial``.
    """

    radii: tuple[float, ...]
    alphas: tuple[float, ...]
    mode: int = 2
    scale_radius: float = 0.5
    cx: float = 0.5
    cy: float = 0.5
    widths: tuple[float, ...] | None = None
    _coefficients: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.radii) != len(self.alphas) - 1:
            raise ValueError("one radius fewer than rings is needed")
        if self.widths is not None and len(self.widths) != len(self.radii) - 1:
            raise ValueError("one width between each pair of radii is needed")
        if any(a <= 0.0 for a in self.alphas):
            raise ValueError("every ring needs a positive α")
        if self.mode < 1:
            raise ValueError("the mode must be 1 or more")
        if not self.radii or self.radii[0] <= 0.0:
            raise ValueError("at least one positive radius is needed")
        if list(self.radii) != sorted(set(self.radii)):
            raise ValueError("radii must increase")
        object.__setattr__(self, "_coefficients", self._solve())

    def _solve(self) -> np.ndarray:
        # Walk outward: R and α (a r^m − b r^-m), the flux times r/m, are
        # continuous, and the next ring's pair follows from them. Both are
        # carried across each ring as increments of r^m and r^-m formed with
        # expm1 / log1p: on a ring 1/s thick at α ~ 1/s (EABE eq. 40) the
        # pair (a, b) is O(s) and evaluating a r^m + b r^-m at the outer
        # radius would cancel two O(s) terms to an O(1) climb, losing
        # log10(s) digits; the increments are O(1) each. With ``widths`` the
        # rings' widths are those, not the stored radii's differences (a
        # ``Band.gap``, E4.8).
        m = self.mode
        coef = np.zeros((len(self.alphas), 2))
        coef[0] = [1.0, 0.0]
        first = self.radii[0]
        value, scaled_flux = first**m, self.alphas[0] * first**m
        for k, r in enumerate(self.radii):
            a, b = coef[k]
            if k > 0:
                below = self.radii[k - 1]
                width = r - below if self.widths is None else self.widths[k - 1]
                step = m * np.log1p(width / below)
                rise = a * below**m * np.expm1(step)
                fall = b * below**-m * np.expm1(-step)
                value = value + rise + fall
                scaled_flux = scaled_flux + self.alphas[k] * (rise - fall)
            up = scaled_flux / self.alphas[k + 1]
            coef[k + 1] = [(value + up) / (2 * r**m), (value - up) / (2 * r**-m)]
        last = coef[-1]
        norm = last[0] * self.scale_radius**m + last[1] * self.scale_radius**-m
        return coef / norm

    def ring(self, r: np.ndarray) -> np.ndarray:
        """Index of the ring holding ``r``; on a radius, the ring above it."""
        r = np.asarray(r, dtype=float)
        return np.searchsorted(np.asarray(self.radii, dtype=float), r, side="right")

    def radial(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        a, b = self._coefficients[self.ring(r)].T
        return a * r**self.mode + b * r**-self.mode

    def flux(self, r: np.ndarray) -> np.ndarray:
        """``α R'(r)``, continuous across the radii."""
        r = np.asarray(r, dtype=float)
        k = self.ring(r)
        a, b = self._coefficients[k].T
        alpha = np.asarray(self.alphas, dtype=float)[k]
        m = self.mode
        return alpha * m * (a * r ** (m - 1) - b * r ** (-m - 1))

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        dx, dy = x - self.cx, y - self.cy
        return self.radial(np.hypot(dx, dy)) * np.cos(self.mode * np.arctan2(dy, dx))


def ring_exact(mode: int = 2, s: float = CASE3_S, gap: bool = False) -> RingMode:
    """Case 3's ring at its constant part: ``1/(1.5 s)`` on ``0.35 − 1/s ≤ r ≤ 0.35``.

    ``1/1500`` on ``0.349 ≤ r ≤ 0.35`` at the default ``s``; EABE eq. 40's
    ring at any other. ``u = r^m cos mθ`` scaled inside, 1 at ``(0.5, θ = 0)``;
    nearly all of the change is across the ring, as in case 3 itself, and
    the climb tends to ``1.5 × α R'`` as ``s`` grows (the ring's resistance
    ``(1/s) / (1/(1.5 s))`` is 1.5 at every ``s``). ``gap=True`` climbs across
    the exact width ``1/s`` (``Band.gap``, the seeds' ring, E4.8) rather than
    the stored radii's, which is the jump-aware stencils' (E2.9).
    """
    widths = (ring_gap(s),) if gap else None
    return RingMode(ring_radii(s), (1.0, 1.0 / (1.5 * s), 1.0), mode, widths=widths)


RING_REACH = 40.0
"""Edge widths from either circle beyond which a resistance-composed ring is its
pieces to rounding (``SmoothRingMode``, ``matched_radial``).

The resistivity blend's tail carries the ring's contrast (``SmoothBand``'s
``composition="resistance"``): beyond a circle it is ``(ρ_in − ρ_out)
(1 − e^{−2w/δ}) e^{−2z}``, at most ``1.5/δ · e^{−2z}`` on EABE eq. 40's ring,
which ``TANH_REACH``'s 20 leaves at 1e-12 relative for δ = 1e-5 and 40 at
1e-30.
"""

RING_CUTS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
"""Element cuts at ``± RING_CUTS δ`` about each circle of a smooth ring."""


def _radial_medium(
    medium: Band | SmoothBand,
) -> tuple[float, float, float, Callable, Callable]:
    """``(inner, outer, gap, α(r), α′(r))`` along a radius of a concentric ring.

    The ring must be radial (constant pieces about one centre); alpha and its
    radial derivative are the medium's own on the ray ``θ = 0``.
    """
    band = medium.band if isinstance(medium, SmoothBand) else medium
    if not all(isinstance(c, Circle) for c in band.interfaces) or not all(
        isinstance(p, Constant2D) for p in band.pieces
    ):
        raise ValueError("a radial ring needs concentric circles and constant pieces")
    inner, outer = band.lower, band.upper
    if (inner.cx, inner.cy) != (outer.cx, outer.cy):
        raise ValueError("the ring's circles must share a centre")
    gap = band.gap if band.gap is not None else outer.radius - inner.radius
    cx, cy = outer.cx, outer.cy

    def alpha(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return medium.alpha(cx + r, np.full_like(r, cy))

    def slope(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return medium.gradient(cx + r, np.full_like(r, cy))[0]

    return outer.radius - gap, outer.radius, gap, alpha, slope


def radial_elements(medium: SmoothBand) -> np.ndarray:
    """Element edges across a smooth ring: ``± RING_CUTS δ`` about both circles.

    Within ``RING_REACH δ`` of the circles; cuts closer than δ/8 are one.
    """
    inner, outer, _, _, _ = _radial_medium(medium)
    delta = medium.delta
    lo, hi = inner - RING_REACH * delta, outer + RING_REACH * delta
    cuts = [lo, hi]
    for centre in (inner, outer):
        for c in RING_CUTS:
            cuts.extend((centre - c * delta, centre + c * delta))
    kept: list[float] = []
    for c in sorted(v for v in cuts if lo <= v <= hi):
        if not kept or c - kept[-1] > delta / 8.0:
            kept.append(c)
    kept[-1] = hi
    return np.array(kept)


RADIAL_RTOL, RADIAL_ATOL = 1e-13, 1e-16
"""DOP853's tolerances for ``SmoothRingMode``'s radial march (R and F are O(1)).

A Chebyshev collocation of the same equation on ``radial_elements`` sat on a
rounding floor of 1e-10 to 1e-8 whatever its order (the δ-wide elements put
``p D²`` at 1e17, equilibrated or not; 2026-09-23), which the probe's weights,
summing to ~1e4, would read as truncation. The march in flux form at
``rtol`` 1e-12 agrees with this one to 3e-12 or better on EABE eq. 40's ring
at s = 10³ and 10¹¹, δ from 1e-5 to 1e-3 (``tests/heat2d/test_exact.py``).
"""


@dataclass(frozen=True)
class SmoothRingMode:
    """``u = R(r) cos(mθ)`` through a smooth concentric ring (E4.8, #39).

    ``RingMode``'s twin for a ``SmoothBand`` ring of constant pieces at any
    δ > 0 (either composition): ``(r α R′)′ = m² α R / r`` in flux form,
    ``R′ = F/(r α)``, ``F′ = m² α R / r``, marched by DOP853 at
    ``RADIAL_RTOL`` from ``R = r^m`` below the ring, where alpha is its piece,
    restarting at every cut of ``radial_elements``, to ``a r^m + b r^{−m}``
    above it; scaled so that ``u = 1`` at ``(scale_radius, 0)``. It solves
    ``∇·(α∇u) = 0`` through the smooth ring with nothing shared with the
    seeds' chain but the medium's alpha, so each row applied to it is that
    row's truncation error (the probe, stiff note §4.8).
    """

    medium: SmoothBand
    mode: int = 2
    scale_radius: float = 0.5
    rtol: float = RADIAL_RTOL
    _solution: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (isinstance(self.medium, SmoothBand) and self.medium.delta > 0.0):
            raise ValueError("a smooth ring needs a SmoothBand with delta > 0")
        _, _, _, alpha, _ = _radial_medium(self.medium)
        edges = radial_elements(self.medium)
        hi = float(edges[-1])
        (top,) = self._march(edges, alpha, np.array([hi]))
        rise = hi * top[1] / (hi * float(alpha(np.array([hi]))[0]) * self.mode)
        m = self.mode
        outside = ((top[0] + rise) / (2 * hi**m), (top[0] - rise) * hi**m / 2)
        a, b = outside
        norm = a * self.scale_radius**m + b * self.scale_radius**-m
        if self.scale_radius < hi:
            raise ValueError("scale_radius must lie beyond the smooth ring")
        object.__setattr__(self, "_solution", (edges, alpha, outside, norm))

    def _march(
        self, edges: np.ndarray, alpha: Callable, targets: np.ndarray
    ) -> np.ndarray:
        """``(R, F)`` at the sorted ``targets`` in ``[edges[0], edges[-1]]``."""
        m = self.mode

        def rate(r: float, y: np.ndarray) -> np.ndarray:
            a = float(alpha(np.array([r]))[0])
            return np.array([y[1] / (r * a), m * m * a * y[0] / r])

        lo = float(edges[0])
        y = np.array([lo**m, m * float(alpha(np.array([lo]))[0]) * lo**m])
        out = np.empty((targets.size, 2))
        out[targets == lo] = y
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            inside = (targets > a) & (targets <= b)
            here = targets[inside]
            stops = here if here.size and here[-1] == b else np.append(here, b)
            sol = solve_ivp(
                rate,
                (a, b),
                y,
                method="DOP853",
                t_eval=stops,
                rtol=self.rtol,
                atol=RADIAL_ATOL,
            )
            if not sol.success:
                raise RuntimeError(f"the radial march failed: {sol.message}")
            out[inside] = sol.y[:, : here.size].T
            y = sol.y[:, -1]
        return out

    def radial(self, r: np.ndarray) -> np.ndarray:
        edges, alpha, (a, b), norm = self._solution
        r = np.asarray(r, dtype=float)
        flat = r.ravel()
        m = self.mode
        out = np.where(flat < edges[0], flat**m, a * flat**m + b * flat**-m)
        inside = (flat >= edges[0]) & (flat <= edges[-1])
        if inside.any():
            targets, where = np.unique(flat[inside], return_inverse=True)
            out[inside] = self._march(edges, alpha, targets)[where, 0]
        return (out / norm).reshape(r.shape)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        band = self.medium.band
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        dx, dy = x - band.upper.cx, y - band.upper.cy
        return self.radial(np.hypot(dx, dy)) * np.cos(self.mode * np.arctan2(dy, dx))


def matched_radial(
    medium: Band | SmoothBand, r: np.ndarray, n_gauss: int = 32
) -> np.ndarray:
    """The matched radial profile through a ring: ``∇·(α∇u) = 4``, ``u = r²`` inside.

    Stiff note §3.6: regular at the centre, ``u′ = 2r/α(r)`` and
    ``u = ∫ 2r/α dr``. At δ = 0 it is E2.9's ``r²``, ``r²/α + b₁``, ``r² + b₂``
    with the climb formed from the ring's ``gap`` when it has one (the stored
    radii's otherwise, E2.9's); at δ > 0 the integral is Gauss–Legendre on
    ``radial_elements`` split at every ``r``, below which alpha is its piece.
    """
    r = np.asarray(r, dtype=float)
    inner, outer, gap, alpha, _ = _radial_medium(medium)
    smooth = isinstance(medium, SmoothBand) and medium.delta > 0.0
    if not smooth:
        band = medium.band if isinstance(medium, SmoothBand) else medium
        ring, out = band.inside.value, band.outside.value
        climb = gap * (inner + outer) * (1.0 / ring - 1.0 / out)
        on_ring = r * r / ring - inner * inner * (1.0 / ring - 1.0 / out)
        return np.where(
            r < inner,
            r * r / out,
            np.where(r <= outer, on_ring, r * r / out + climb),
        )
    edges = radial_elements(medium)
    lo, hi = edges[0], edges[-1]
    flat = r.ravel()
    cuts = np.unique(np.concatenate([edges, np.clip(flat, lo, hi)]))
    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
    mid, half = (cuts[:-1] + cuts[1:]) / 2, (cuts[1:] - cuts[:-1]) / 2
    samples = mid[:, None] + half[:, None] * nodes[None, :]
    integrand = 2.0 * samples / alpha(samples.ravel()).reshape(samples.shape)
    at_cuts = lo * lo / float(alpha(np.array([lo]))[0]) + np.concatenate(
        [[0.0], np.cumsum(half * (integrand @ weights))]
    )
    inside = at_cuts[np.searchsorted(cuts, np.clip(flat, lo, hi))]
    out_piece = float(alpha(np.array([hi]))[0])
    value = np.where(
        flat < lo,
        flat * flat / float(alpha(np.array([lo]))[0]),
        np.where(flat > hi, inside + (flat * flat - hi * hi) / out_piece, inside),
    )
    return value.reshape(r.shape)
