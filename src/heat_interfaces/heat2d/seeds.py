"""Scalar seeds for ``div(alpha grad)`` across a straight smooth feature (E4.4, #35).

``docs/stiff-diffusion.md`` §3.2–3.4. A stencil that sees an edge of width δ
below the node spacing replaces its 15 monomials ``ξᵃ ηᵇ`` (degree ≤ 4) by
*seeds* anchored at the evaluation node, in E2.3's frame orientation (``ξ``
along the tangent of the nearest interface, ``η`` along its normal into
``level > 0``) and in units of the stencil radius ``h_s``. Where alpha
depends on the normal coordinate alone the seed of ``ξᵃ ηᵇ`` is

    φ_ab(ξ, η) = Σ_j g_j(η) ξʲ,    j = a, a − 2, …, a mod 2,

and the 2-D chain ``L φ_ab = α_e [a (a − 1) φ_{a−2,b} + b (b − 1) φ_{a,b−2}]``
becomes, per level, a first-order pair in the flux variable ``ψ_j = α g_j′``
that never differentiates alpha:

    g_j′ = ψ_j / α_n(η),
    ψ_j′ = α_e [a (a − 1) g_j^{(a−2,b)} + b (b − 1) g_j^{(a,b−2)}]
           − (j + 2)(j + 1) α_n(η) g_{j+2}^{(a,b)},

a missing level being zero, from the anchor data ``g_a(0) = [b = 0]``,
``ψ_a(0) = α_e [b = 1]`` and zero for every other state. For degree 4 that
is 22 levels and 44 states, marched together (one linear system, two
constant 22 × 22 matrices) by DOP853 at ``heat1d.stiff``'s tolerances in
both directions from the anchor, restarted at every node's ``η`` so node
values are integrated and never interpolated, and at the stops of the
material's ``NormalProfile`` (each edge centre and its ``± EDGE_STOP δ``
flanks); on each segment alpha is the profile's, one-sided at a jump. So at
δ = 0 on constant pieces the seeds span E2.3's translated basis, and with
the same Gaussian block the weights are E2.3's (§3.4, H2).

``seed_basis`` builds one stencil: its frame, the block ``S[i, e] =
φ_e(ξ_i, η_i)`` that takes the monomials' place in the saddle-point system
(``rbf.augmented_solve``), the warp coordinate ``φ₀₁`` at the nodes (§3.4),
and the moment conditions' right-hand side, the true operator on each seed
at the anchor in stencil units: ``2 α_e`` on the seeds of ``ξ²`` and
``η²``, ``h_s α_ξ`` on the seed of ``ξ`` (the tangential derivative the
frozen profile leaves out), zero on the rest. Flat interfaces only, as
``SmoothBand.normal_profile``; curved ones are route (a), E4.7 (#38).

``seed_weights`` (E4.5, #36) closes the saddle-point system: the Gaussians
in ``(ξ, φ₀₁(η))``, the warp the march brings with it, and their right-hand
side the chain rule for ``L G(ξ, η̃(η))`` with the cancellation
``α η̃′ ≡ α_e`` checked, not assumed. ``operators.seed_operator`` puts those
rows into the global matrix on the stencils that see an edge.

``tangential=True`` (E4.11, #81, §3.10) builds the same stencil in the foot
curve's own coordinates, ``x = γ(σ) + d n(σ)``, where the edge is a
coordinate line at every ξ: every seed keeps its levels to ``ξ⁴``, and
alpha's and the metric's variation along the curve, expanded in ξ, couples
them (``coupled_chain``, 75 levels, 150 states). That is what route (a)'s
frozen profile cannot carry on a curved or tangentially varying edge
(§4.6); on a flat edge with alpha a function of the normal alone the extra
levels stay zero and the seeds are the ones above.

On EABE eq. 40's ring (a ``Band`` with a ``gap``, E4.8, #39) the march runs
in the offset from the outer circle, so the ring it crosses is the gap's
width to rounding at any s (stiff note §3.6's widths rule), and the series
along its segments come from a checked piecewise Chebyshev interpolant in η
(``RING_POINTS``), which keeps the contrast's rounding out of the step control.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache
from math import acos, atan2, copysign, exp, factorial, inf, pi

import numpy as np
from scipy.integrate import solve_ivp

from ..fd_weights import fornberg_weights
from ..heat1d.domain import TANH_REACH
from ..heat1d.stiff import MERGE_TOL, SEED_ATOL, SEED_RTOL, march_targets
from .domain import (
    Band,
    Circle,
    Constant2D,
    Curve,
    FlatLine,
    NormalProfile,
    Piece2D,
    SmoothBand,
)
from .interface import Frame
from .neighbors import periodic_dx
from .rbf import (
    GA_SHAPE,
    augmented_solve,
    check_coincidence,
    gaussian,
    gaussian_derivative,
    polynomial_count,
    polynomial_exponents,
)

SEED_DEGREE = 4
"""The seeds' degree: 15 of them on 30 nodes, the interface group's spec (§3.4)."""

WARP_TOL = 1e-9
"""How far ``ψ₀₁ = α η̃′`` may stray from ``α_e`` before the warp is refused.

§3.4's cancellation: the warp ``η̃ = φ₀₁`` is the constant-flux seed, so
``α η̃′ ≡ α_e`` along the whole line and the ``G_η̃`` term of
``L G(ξ, η̃(η))`` vanishes identically. That state's right-hand side is
identically zero, so the march returns ``α_e`` to the bit (§4.3);
``seed_coordinates`` checks it rather than assuming it, and a failure means
the march, not the geometry.
"""


SERIES_DEGREE = 4
"""The tangential chain's series in ξ (§3.10): ``α/m̂``, ``α m̂`` and ``m̂`` to ``ξ⁴``.

Level ``j`` of seed ``(a, b)`` is ``O(h^{j−a})`` above its top and the
series' terms ``O(h^k)``; ``A_5`` would enter only levels that the order
count already drops, as ``O(h⁶)`` in u.
"""

SAMPLE_HALF, SAMPLE_STEP = 5, 0.2
"""The tangential series' samples: ``2 · 5 + 1`` points 0.2 stencil radii apart
along each coordinate line, ``ξ ∈ [−1, 1]``, for Fornberg's weights at ξ = 0
(§3.10: 15 points at 0.12 moved case 2's weights by 3.6e-10)."""


@dataclass(frozen=True)
class Chain:
    """The level bookkeeping of §3.2 for one degree.

    ``levels[l] = (a, b, j)``: state ``l`` is level ``j`` of the seed of
    ``ξᵃ ηᵇ``, seeds in ``rbf.polynomial_exponents`` order and each seed's
    levels from its top ``j = a`` down. ``source[l, m]`` is the coefficient
    of ``g_m`` in ``ψ_l′`` per unit ``α_e`` (the lower seeds at the same
    level), ``lower[l, m]`` that per unit ``α_n`` (the seed's own level
    ``j + 2``, entered with a minus sign); ``seed_of[e, l]`` is 1 where state
    ``l`` belongs to seed ``e``.
    """

    degree: int
    levels: tuple[tuple[int, int, int], ...]
    source: np.ndarray
    lower: np.ndarray
    seed_of: np.ndarray
    power: np.ndarray

    @property
    def size(self) -> int:
        return len(self.levels)

    def index(self, a: int, b: int, j: int) -> int:
        """The state of level ``j`` of the seed of ``ξᵃ ηᵇ``."""
        return self.levels.index((a, b, j))

    def initial(self, alpha_e: float) -> np.ndarray:
        """``(g, ψ)`` at the anchor: the monomials' top-level jets in η."""
        y = np.zeros(2 * self.size)
        for m, (a, b, j) in enumerate(self.levels):
            if j == a and b == 0:
                y[m] = 1.0
            if j == a and b == 1:
                y[self.size + m] = alpha_e
        return y

    def rate(
        self, alpha_e: float, alpha: Callable[[float], float]
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """The first-order system on one segment, alpha from ``alpha(η)``."""
        n = self.size
        source = alpha_e * self.source
        lower = self.lower

        def rate(eta: float, y: np.ndarray) -> np.ndarray:
            a = alpha(eta)
            g = y[:n]
            out = np.empty_like(y)
            out[:n] = y[n:] / a
            out[n:] = source @ g - a * (lower @ g)
            return out

        return rate


@cache
def chain(degree: int = SEED_DEGREE) -> Chain:
    """The ``Chain`` of the seeds of degree ≤ ``degree`` (22 levels at degree 4)."""
    exponents = [tuple(int(v) for v in e) for e in polynomial_exponents(degree)]
    levels = tuple((a, b, j) for a, b in exponents for j in range(a, -1, -2))
    where = {state: m for m, state in enumerate(levels)}
    n = len(levels)
    source, lower = np.zeros((n, n)), np.zeros((n, n))
    seed_of = np.zeros((len(exponents), n))
    for m, (a, b, j) in enumerate(levels):
        seed_of[exponents.index((a, b)), m] = 1.0
        for (a2, b2), c in (((a - 2, b), a * (a - 1)), ((a, b - 2), b * (b - 1))):
            if c and (a2, b2, j) in where:
                source[m, where[(a2, b2, j)]] += c
        if (a, b, j + 2) in where:
            lower[m, where[(a, b, j + 2)]] = (j + 2) * (j + 1)
    power = np.array([j for _, _, j in levels])
    return Chain(degree, levels, source, lower, seed_of, power)


@dataclass(frozen=True)
class CoupledChain:
    """The level bookkeeping of §3.10: every level of every seed, coupled in ξ.

    ``levels[l] = (a, b, j)`` for ``j = degree`` down to 0 for every seed, in
    ``polynomial_exponents`` order: 75 levels at degree 4. Every seed keeps
    the stencil's degree in ξ, whatever b: across a jump the far side of the
    seed of ``ξᵃ ηᵇ`` carries the flux ratio to the power ``⌈b/2⌉``, so the
    η-seeds' tangential levels are not small in practice although they are
    of high order in h (§3.10: cutting at ``degree − b`` left case 2's error
    at 40,000 nodes 5× what every level gives, set by the rows above its
    upper curve where the band's piece is smallest). With the series in
    ξ of ``m̂``, ``α/m̂`` and ``1/(α m̂)`` (``M_k``, ``A_k``, ``(1/B)_k``) the
    level fluxes ``ψ_i = Σ_k B_k g′_{i−k}`` and the values ``g`` march as

        g′ = Σ_k (1/B)_k shift_k ψ,    ψ′ = α_e Σ_k M_k source_k g − Σ_k A_k along_k g,

    with ``shift_k`` taking each seed's level ``i − k`` to its level ``i``,
    ``source_k`` the lower seeds of the chain (``Chain.source``'s) shifted by
    k, and ``along_k`` the ξ-part ``∂_ξ (A ∂_ξ ·)``, level ``i + 2 − k`` into
    level ``i`` with ``(i + 2 − k)(i + 1)``; each is stacked over
    ``k = 0 … series`` as one ``((series + 1) · size, size)`` matrix.
    ``seed_of`` and ``power`` are ``Chain``'s, so ``SeedProfiles`` evaluates
    either chain.
    """

    degree: int
    series: int
    levels: tuple[tuple[int, int, int], ...]
    shift: np.ndarray
    source: np.ndarray
    along: np.ndarray
    seed_of: np.ndarray
    power: np.ndarray

    @property
    def size(self) -> int:
        return len(self.levels)

    def index(self, a: int, b: int, j: int) -> int:
        """The state of level ``j`` of the seed of ``ξᵃ ηᵇ``."""
        return self.levels.index((a, b, j))

    def initial(self, alpha_e: float) -> np.ndarray:
        """``(g, ψ)`` at the anchor: §3.2's data, every other level zero (``m̂ = 1``)."""
        y = np.zeros(2 * self.size)
        for m, (a, b, j) in enumerate(self.levels):
            if j == a and b == 0:
                y[m] = 1.0
            if j == a and b == 1:
                y[self.size + m] = alpha_e
        return y

    def rate(
        self,
        alpha_e: float,
        coefficients: Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """The first-order system on one segment; ``coefficients(η) = (M, A, 1/B)``."""
        n, q = self.size, self.series + 1
        shift, source, along = self.shift, self.source, self.along

        def rate(eta: float, y: np.ndarray) -> np.ndarray:
            m, a, inverse = coefficients(eta)
            g = y[:n]
            out = np.empty_like(y)
            out[:n] = inverse @ (shift @ y[n:]).reshape(q, n)
            out[n:] = alpha_e * (m @ (source @ g).reshape(q, n)) - a @ (
                along @ g
            ).reshape(q, n)
            return out

        return rate


@cache
def coupled_chain(
    degree: int = SEED_DEGREE, series: int = SERIES_DEGREE
) -> CoupledChain:
    """The ``CoupledChain`` of degree ≤ ``degree`` (75 levels at degree 4)."""
    exponents = [tuple(int(v) for v in e) for e in polynomial_exponents(degree)]
    levels = tuple((a, b, j) for a, b in exponents for j in range(degree, -1, -1))
    where = {state: m for m, state in enumerate(levels)}
    n, q = len(levels), series + 1
    shift, along = np.zeros((q, n, n)), np.zeros((q, n, n))
    lower = np.zeros((n, n))
    seed_of = np.zeros((len(exponents), n))
    for m, (a, b, i) in enumerate(levels):
        seed_of[exponents.index((a, b)), m] = 1.0
        for k in range(q):
            if (a, b, i - k) in where:
                shift[k, m, where[(a, b, i - k)]] = 1.0
            s = i + 2 - k
            if s >= 1 and (a, b, s) in where:
                along[k, m, where[(a, b, s)]] = s * (i + 1)
        for (a2, b2), c in (((a - 2, b), a * (a - 1)), ((a, b - 2), b * (b - 1))):
            if c and (a2, b2, i) in where:
                lower[m, where[(a2, b2, i)]] += c
    source = np.einsum("kij,jl->kil", shift, lower)
    power = np.array([j for _, _, j in levels])
    return CoupledChain(
        degree,
        series,
        levels,
        shift.reshape(q * n, n),
        source.reshape(q * n, n),
        along.reshape(q * n, n),
        seed_of,
        power,
    )


def _series_product(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``p q`` truncated at ``p``'s degree: coefficients in ξ."""
    return np.convolve(p, q)[: p.size]


def _series_inverse(p: np.ndarray) -> np.ndarray:
    """``1 / p`` truncated at ``p``'s degree (``p₀ ≠ 0``), in floats (the rate's)."""
    c = p.tolist()
    inverse = 1.0 / c[0]
    r = [inverse]
    for k in range(1, len(c)):
        r.append(-sum(c[i] * r[k - i] for i in range(1, k + 1)) * inverse)
    return np.array(r)


@dataclass(frozen=True)
class SeedProfiles:
    """The chain's state at points ``eta`` of one normal line (§3.2).

    ``g[l, i]`` and ``psi[l, i]`` are state ``l`` of ``chain.levels`` at
    ``eta[i]``, in stencil units. ``psi`` is ``α g′``, continuous through a
    jump; E4.8's residual probes read it. For a ``CoupledChain`` (§3.10) it is
    each level's share of the normal flux, ``Σ_k B_k g′_{i−k}``.
    """

    eta: np.ndarray
    g: np.ndarray
    psi: np.ndarray
    alpha_e: float
    chain: Chain | CoupledChain

    def values(self, xi: np.ndarray) -> np.ndarray:
        """``φ_e(ξ_i, η_i)`` as ``(..., len(eta), q)``; ``xi`` broadcasts to ``eta``."""
        xi = np.broadcast_to(
            np.asarray(xi, dtype=float), (*np.shape(xi)[:-1], self.eta.size)
        )
        terms = self.g * xi[..., None, :] ** self.chain.power[:, None]
        return np.swapaxes(self.chain.seed_of @ terms, -1, -2)


def seed_profiles(
    profile: NormalProfile,
    eta: np.ndarray,
    degree: int = SEED_DEGREE,
    alpha_e: float | None = None,
    rtol: float = SEED_RTOL,
    atol: float = SEED_ATOL,
) -> SeedProfiles:
    """March the chain along ``profile`` to the points ``eta`` (§3.3).

    Both directions from the anchor ``η = 0``, one ``solve_ivp`` per segment
    between consecutive targets: the points themselves, exact, and the
    profile's stops within reach, dropped when within ``MERGE_TOL`` of a
    point (``heat1d.stiff.march_targets``). Alpha on a segment is the
    profile's piece for it. ``alpha_e`` defaults to the profile's.
    """
    eta = np.asarray(eta, dtype=float)
    if eta.ndim != 1:
        raise ValueError("eta must be a 1-D array of points on the line")
    ch = chain(degree)
    a_e = profile.alpha_e if alpha_e is None else float(alpha_e)

    def rate_of(segment: int) -> Callable[[float, np.ndarray], np.ndarray]:
        return ch.rate(a_e, profile.alpha_function(segment))

    out = _march(ch.initial(a_e), eta, profile, rate_of, rtol, atol)
    return SeedProfiles(eta, out[: ch.size], out[ch.size :], a_e, ch)


def _march(
    y0: np.ndarray,
    eta: np.ndarray,
    profile: NormalProfile,
    rate_of: Callable[[int], Callable[[float, np.ndarray], np.ndarray]],
    rtol: float,
    atol: float,
) -> np.ndarray:
    """The state at every ``eta``: both directions from ``y0`` at the anchor (§3.3).

    One ``solve_ivp`` per segment between consecutive targets (the points,
    exact, and the profile's stops within reach), ``rate_of(segment)`` the
    system on each; either chain's march. On a profile with ``offsets`` (a
    ring with a ``gap``, E4.8) the march runs in the offset ``τ = η −
    origin`` from the outer crossing, so that the segment across the ring is
    the gap to rounding (``SmoothBand._gap_line``); elsewhere ``τ`` is ``η``.
    """
    origin = profile.origin
    if profile.offsets is None:
        tau, offsets, anchor = eta, profile.stops, 0.0
    else:
        tau, offsets, anchor = eta - origin, profile.offsets, -origin
    out = np.empty((y0.size, eta.size))
    out[:, eta == 0.0] = y0[:, None]
    for side in (-1.0, 1.0):
        ahead = side * eta > 0.0
        if not ahead.any():
            continue
        far = np.max(side * tau[ahead])
        stops = side * offsets
        inside = stops[(stops > side * anchor + MERGE_TOL) & (stops < far)]
        y, t0 = y0, anchor
        for t in march_targets(side * tau[ahead], inside):
            t1 = side * t
            segment = int(profile.segment(0.5 * (t0 + t1) + origin))
            rate = rate_of(segment)
            if profile.offsets is not None:
                rate = _shifted(rate, origin)
            sol = solve_ivp(rate, (t0, t1), y, method="DOP853", rtol=rtol, atol=atol)
            if not sol.success:
                raise RuntimeError(
                    f"the seed march failed at eta = {t1 + origin}: {sol.message}"
                )
            y, t0 = sol.y[:, -1], t1
            hit = ahead & np.isclose(tau, t1, rtol=0.0, atol=MERGE_TOL)
            out[:, hit] = y[:, None]
    return out


def _shifted(
    rate: Callable[[float, np.ndarray], np.ndarray], origin: float
) -> Callable[[float, np.ndarray], np.ndarray]:
    """``rate`` in the offset ``τ = η − origin``: the medium only sees ``η``."""
    return lambda tau, y: rate(tau + origin, y)


@dataclass(frozen=True)
class FootCoordinates:
    """One stencil in its foot curve's normal coordinates (§3.10).

    A point near ``curve`` (interface ``interface``) is ``γ(σ) + d n(σ)``;
    about the anchor's foot point ``s0`` and distance ``d_e``, in units of
    ``scale``, ``ξ = mu (σ − s0) / scale`` and ``η = (d − d_e) / scale``, with
    ``mu`` the metric ``|γ′| (1 − κ d)`` at the anchor, so that ``m̂ = 1``
    there and ξ is the arc length of the anchor's parallel curve. ``px, py``
    and ``nx, ny`` are the foot points and unit normals of the sample lines
    ``ξ_k = SAMPLE_STEP · (−SAMPLE_HALF … SAMPLE_HALF)``, ``weights``
    Fornberg's at ξ = 0 on them over ``k!`` (Taylor coefficients), and
    ``metric`` the pair ``(G, K)`` of series with ``m̂ = G − d K``.
    """

    curve: Curve
    interface: int
    s0: float
    d_e: float
    mu: float
    scale: float
    px: np.ndarray
    py: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    weights: np.ndarray
    metric: tuple[np.ndarray, np.ndarray]

    def points(self, eta: float) -> tuple[np.ndarray, np.ndarray]:
        """The samples at ``η``: one point on each line, at distance ``d_e + h_s η``."""
        d = self.d_e + self.scale * eta
        return self.px + d * self.nx, self.py + d * self.ny

    def series(self, samples: np.ndarray) -> np.ndarray:
        """Taylor coefficients in ξ at ξ = 0 of a function sampled on the lines.

        A constant is its own series exactly, so that a flat line with alpha
        a function of the normal alone couples nothing (§3.10).
        """
        if np.all(samples == samples[0]):
            out = np.zeros(self.weights.shape[0])
            out[0] = samples[0]
            return out
        return self.weights @ samples

    def metric_at(self, eta: float) -> np.ndarray:
        """``m̂``'s series in ξ on the line ``η``."""
        g, k = self.metric
        return g - (self.d_e + self.scale * eta) * k


@cache
def _sample_weights(series: int) -> np.ndarray:
    xi = SAMPLE_STEP * np.arange(-SAMPLE_HALF, SAMPLE_HALF + 1)
    w = fornberg_weights(0.0, xi, series)
    return w / np.array([factorial(k) for k in range(series + 1)])[:, None]


def foot_coordinates(
    curve: Curve,
    interface: int,
    x: np.ndarray,
    y: np.ndarray,
    scale: float,
    series: int = SERIES_DEGREE,
) -> tuple[FootCoordinates, np.ndarray, np.ndarray]:
    """The ``FootCoordinates`` of the nodes ``(x, y)`` (anchor first), and their ξ, η.

    Each node's foot point and distance by ``closest`` and
    ``signed_distance``, one Newton each, inside the focal distance that
    ``normal_profile``'s ``FOOT_CURVATURE`` guards; ξ periodic in the curve's
    parameter (x on a graph, the turn on a circle).
    """
    s, d = curve.closest(x, y), curve.signed_distance(x, y)
    s0, d_e = float(s[0]), float(d[0])
    mu = float(curve.speed(s[:1])[0]) * (1.0 - float(curve.curvature(s[:1])[0]) * d_e)
    sk = s0 + scale * SAMPLE_STEP * np.arange(-SAMPLE_HALF, SAMPLE_HALF + 1) / mu
    px, py = (np.asarray(v, dtype=float) for v in curve.point(sk))
    nx, ny = (np.asarray(v, dtype=float) for v in curve.normal(sk))
    coords = FootCoordinates(
        curve,
        interface,
        s0,
        d_e,
        mu,
        scale,
        px,
        py,
        nx,
        ny,
        _sample_weights(series),
        (np.zeros(series + 1), np.zeros(series + 1)),
    )
    speed = curve.speed(sk) / mu
    metric = (coords.series(speed), coords.series(speed * curve.curvature(sk)))
    xi = mu * periodic_dx(s - s0) / scale
    return replace(coords, metric=metric), xi, (d - d_e) / scale


FAR_POINTS = 17
"""Chebyshev points per sample line for the other curve's distance (§3.10).

Along a straight sample line the signed distance to a smooth curve is analytic
out to the curve's focal set, at least ``1/κ_max − 0.2 ≈ 1`` from case 2's
lines, against a march of at most ``2.2 h_s ≤ 0.2``: 17 points put the
interpolant at rounding, where a float Newton per sample and per rate call
cost half a smooth row (65 ms at δ = 0.01 on 2500 nodes, 2026-09-22)."""


def _other_edge(
    medium: SmoothBand, coords: FootCoordinates, eta: np.ndarray
) -> float | Callable[[float], np.ndarray]:
    """The other curve's signed distance at the samples, as the series need it.

    ``±inf`` if its edge is saturated on every sample: the samples on the
    march's lines lie within ``reach`` of the anchor (the largest distance to
    the lines' ends; each line is straight in η), and a signed distance moves
    by at most the distance moved, so beyond ``TANH_REACH δ + reach`` the other
    edge's blend is its piece to ``e^{−40}`` times the contrast at every
    sample (the fold only). Otherwise ``η ↦`` the distance at the 11 samples,
    interpolated on ``FAR_POINTS`` Chebyshev points of each line over the
    march's range (§3.10); on a ring with a ``gap``, the foot distance shifted
    by the gap, exact.
    """
    other = medium.interfaces[1 - coords.interface]
    if medium.gap is not None:
        # Concentric circles: the other is the foot circle's coordinate line
        # ``d ∓ gap``, exactly (§3.6's widths from the outer radius).
        shift = -medium.gap if coords.interface == 0 else medium.gap
        count = coords.px.size

        def parallel(eta: float) -> np.ndarray:
            return np.full(count, coords.d_e + coords.scale * eta + shift)

        return parallel
    ax, ay = coords.points(0.0)
    ax, ay = float(ax[SAMPLE_HALF]), float(ay[SAMPLE_HALF])
    lo, hi = min(0.0, float(eta.min())), max(0.0, float(eta.max()))
    reach = 0.0
    for t in (lo, hi):
        px, py = coords.points(t)
        reach = max(reach, float(np.hypot(periodic_dx(px - ax), py - ay).max()))
    d0 = other.signed_distance_at(ax, ay)
    # The fold's far edge is its piece beyond TANH_REACH δ; the resistance
    # composition's tail carries the band's contrast and is always sampled.
    saturated = abs(d0) - reach >= TANH_REACH * medium.delta
    if saturated and medium.composition == "fold":
        return copysign(inf, d0)
    theta = np.pi * (np.arange(FAR_POINTS) + 0.5) / FAR_POINTS
    t = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(theta)
    d = coords.d_e + coords.scale * t
    x = coords.px[None, :] + d[:, None] * coords.nx[None, :]
    y = coords.py[None, :] + d[:, None] * coords.ny[None, :]
    samples = other.signed_distance(x.ravel(), y.ravel()).reshape(x.shape)
    k = np.arange(FAR_POINTS)
    series = (2.0 / FAR_POINTS) * np.cos(np.outer(k, theta)) @ samples
    series[0] *= 0.5
    transposed = np.ascontiguousarray(series.T)

    def far(eta: float) -> np.ndarray:
        # T_k(t) = cos(k arccos t) in one call: NumPy's ``chebval`` loops over
        # the degrees, a quarter of a smooth row's rate evaluation.
        t = min(1.0, max(-1.0, (2.0 * eta - lo - hi) / (hi - lo)))
        return transposed @ np.cos(k * acos(t))

    return far


def _alpha_series(
    coords: FootCoordinates,
    piece: Piece2D | SmoothBand,
    far: float | Callable[[float], np.ndarray] | None,
) -> Callable[[float], np.ndarray]:
    """``η ↦`` alpha's series in ξ on the coordinate line ``η`` (§3.10).

    A constant piece is its own series. The smooth band's foot edge is a
    function of ``d`` alone, exact; its other edge is ``far``, from
    ``_other_edge``: a saturated value, or the distance at the samples
    interpolated along each line (``None`` at δ = 0, where the pieces are
    sampled directly). With the other edge saturated the blend is
    linear in the two pieces with one weight per line, so their series are
    blended instead of their samples (``_blend``'s steps, to rounding), which
    is most of what a smooth row costs. Any other piece is sampled as it
    stands: at δ = 0 the segment's own piece on both sides of the curve, its
    smooth extension, as E2.3's Taylor tables are.
    """
    if isinstance(piece, Constant2D):
        constant = np.zeros(coords.weights.shape[0])
        constant[0] = piece.value
        return lambda eta: constant
    if isinstance(piece, SmoothBand):
        j = coords.interface
        if not callable(far):
            saturated = float(far)
            inside = _alpha_series(coords, piece.inside, None)
            outside = _alpha_series(coords, piece.outside, None)

            def linear(eta: float) -> np.ndarray:
                z = (coords.d_e + coords.scale * eta) / piece.delta
                out = outside(eta)
                zs = (z, saturated) if j == 0 else (saturated, z)
                a = _edge_series(out, inside(eta), zs[0])
                return _edge_series(a, out, zs[1])

            return linear

        def blended(eta: float) -> np.ndarray:
            x, y = coords.points(eta)
            d, distance = coords.d_e + coords.scale * eta, far(eta)
            return coords.series(
                piece.alpha_given(x, y, (d, distance) if j == 0 else (distance, d))
            )

        return blended
    weights = coords.weights
    return lambda eta: weights @ piece.alpha(*coords.points(eta))


def _edge_series(a: np.ndarray, b: np.ndarray, z: float) -> np.ndarray:
    """``edge_value`` on two series with one weight: ``(1 − s) a + s b``, ``s(z)``."""
    e = exp(-2.0 * abs(z))
    small = e / (1.0 + e)
    return b + small * (a - b) if z >= 0.0 else a + small * (b - a)


def _coefficients(
    coords: FootCoordinates, alpha: Callable[[float], np.ndarray]
) -> Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``η ↦ (M, A, 1/B)``: the series of ``m̂``, ``α/m̂`` and ``1/(α m̂)``."""

    def coefficients(eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, m = alpha(eta), coords.metric_at(eta)
        return (
            m,
            _series_product(a, _series_inverse(m)),
            _series_inverse(_series_product(a, m)),
        )

    return coefficients


def tangential_profiles(
    coords: FootCoordinates,
    profile: NormalProfile,
    eta: np.ndarray,
    medium: SmoothBand,
    alpha_e: float,
    degree: int = SEED_DEGREE,
    rtol: float = SEED_RTOL,
    atol: float = SEED_ATOL,
) -> SeedProfiles:
    """March the coupled chain along ``profile``'s line to the points ``eta`` (§3.10).

    ``seed_profiles``' march (``_march``) with ``CoupledChain``'s rate and the
    tangential series of each segment's piece; the line ``ξ = 0`` of
    ``coords`` is ``profile``'s.
    """
    eta = np.asarray(eta, dtype=float)
    if eta.ndim != 1:
        raise ValueError("eta must be a 1-D array of points on the line")
    far = _other_edge(medium, coords, eta) if medium.delta > 0.0 else None
    return _tangential_march(coords, profile, eta, far, alpha_e, degree, rtol, atol)


RING_POINTS, RING_TOL, RING_NOISE, RING_DEPTH = 9, 1e-13, 1e-11, 14
"""On a ring with a ``gap`` (E4.8) the series along each segment of the march are
read from a piecewise Chebyshev interpolant in η, ``RING_POINTS`` points a piece,
each piece halved until it matches the sampled series at two points off its nodes
to ``RING_TOL`` of their largest coefficient, or until halving no longer helps
(``_ring_coefficients``).

The series sampled at every stage carry the Fornberg weights' rounding, a few
hundred ulps of the samples, and on the insulating ring ``1/(α m̂)`` multiplies
it by the contrast: 1.5 s across the jump ring's gap (δ = 0) and ~1500 on the
smooth ring's plateau and tails. The small tangential levels of the η-seeds,
driven by those coefficients, then carried more noise than DOP853's absolute
tolerance, and it took up to 150,000 rate evaluations a row (170–240 ms a row
at every s ≥ 10⁵ at δ = 0, a median 90 ms and 5 s at worst at δ = 0.001,
2026-09-23). The interpolant is one polynomial for each piece, so the stages
see a smooth function, and its error is checked; elsewhere (no gap) the series
are sampled at every stage as §3.10 decided.
"""


def _chebyshev_piece(
    coefficients: Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]],
    lo: float,
    hi: float,
) -> tuple[list[np.ndarray], float, float]:
    """``coefficients``' Chebyshev tables on ``[lo, hi]``, with its middle and half."""
    middle, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    theta = np.pi * (np.arange(RING_POINTS) + 0.5) / RING_POINTS
    samples = [coefficients(middle + half * t) for t in np.cos(theta)]
    basis = (2.0 / RING_POINTS) * np.cos(np.outer(np.arange(RING_POINTS), theta))
    basis[0] *= 0.5
    return (
        [basis @ np.stack(parts) for parts in zip(*samples, strict=True)],
        middle,
        half,
    )


def _chebyshev_value(
    tables: list[np.ndarray], middle: float, half: float, eta: float
) -> tuple[np.ndarray, ...]:
    t = min(1.0, max(-1.0, (eta - middle) / half))
    chebyshev = np.cos(np.arange(RING_POINTS) * acos(t))
    return tuple(chebyshev @ table for table in tables)


def _ring_coefficients(
    coefficients: Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]],
    lo: float,
    hi: float,
) -> Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``coefficients`` on ``[lo, hi]`` as ``RING_POINTS``-point Chebyshev pieces.

    A piece is halved while its check misses ``RING_TOL``, unless it is below
    ``RING_NOISE`` and the last halving gained less than 4×: a nine-point
    interpolant of a smooth function gains ~500× a halving once resolved, and
    one that stalls there is at the sampled series' own rounding, which no
    halving removes (5e-14 on the smooth ring). A piece still above
    ``RING_NOISE`` after ``RING_DEPTH`` halvings is sampled directly.
    """
    pieces: list[tuple[float, float, tuple | None]] = []
    stack: list[tuple[float, float, int, float]] = [(lo, hi, 0, inf)]
    while stack:
        a, b, depth, parent = stack.pop()
        piece = _chebyshev_piece(coefficients, a, b)
        error = 0.0
        for offset in (-0.3, 0.3):
            eta = piece[1] + offset * piece[2]
            got, want = _chebyshev_value(*piece, eta), coefficients(eta)
            for g, w in zip(got, want, strict=True):
                size = float(np.abs(w).max())
                error = max(error, float(np.abs(g - w).max()) / size if size else 0.0)
        helps = error > RING_NOISE or error < 0.25 * parent
        if error > RING_TOL and depth < RING_DEPTH and helps:
            middle = 0.5 * (a + b)
            stack.extend([(middle, b, depth + 1, error), (a, middle, depth + 1, error)])
            continue
        pieces.append((a, b, piece if error <= RING_NOISE else None))
    pieces.sort(key=lambda p: p[0])
    starts = np.array([p[0] for p in pieces[1:]])

    def model(eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, _, piece = pieces[int(np.searchsorted(starts, eta, side="right"))]
        if piece is None:
            return coefficients(eta)
        return _chebyshev_value(*piece, eta)

    return model


def _tangential_march(
    coords: FootCoordinates,
    profile: NormalProfile,
    eta: np.ndarray,
    far: float | Callable[[float], np.ndarray] | None,
    alpha_e: float,
    degree: int,
    rtol: float,
    atol: float,
) -> SeedProfiles:
    """``tangential_profiles`` with the far edge already built (``_other_edge``)."""
    ch = coupled_chain(degree)
    ring = profile.offsets is not None
    lo, hi = min(0.0, float(eta.min())), max(0.0, float(eta.max()))
    models: dict[int, Callable] = {}

    def rate_of(segment: int) -> Callable[[float, np.ndarray], np.ndarray]:
        if segment not in models:
            alpha = _alpha_series(coords, profile.pieces[segment], far)
            coefficients = _coefficients(coords, alpha)
            if ring:
                # The segment's extent within the march (RING_POINTS).
                below = profile.stops[segment - 1] if segment > 0 else -inf
                above = profile.stops[segment] if segment < profile.stops.size else inf
                a, b = max(lo, float(below)), min(hi, float(above))
                if b > a:
                    coefficients = _ring_coefficients(coefficients, a, b)
            models[segment] = coefficients
        return ch.rate(alpha_e, models[segment])

    out = _march(ch.initial(alpha_e), eta, profile, rate_of, rtol, atol)
    return SeedProfiles(eta, out[: ch.size], out[ch.size :], alpha_e, ch)


@dataclass(frozen=True)
class SeedBasis:
    """One seed stencil: its frame, seeds at the nodes and moment right-hand side.

    ``xy`` are the nodes, the anchor first; ``frame`` has its origin at the
    anchor and E2.3's orientation at the foot point of interface
    ``interface`` (the nearest), in units of ``scale``, the stencil radius;
    ``xi, eta`` are the nodes in it and ``profile`` the material along its
    normal. ``block`` is ``S[i, e] = φ_e(ξ_i, η_i)``, the ``P`` of EABE eq. 2
    for the seeds; ``rhs[e]`` is ``(L φ_e)(anchor)`` in stencil units (the
    weights are ``w̃ / scale²``); ``gradient`` is ``∇α`` at the anchor rotated
    into the frame and scaled, ``(h_s α_ξ, h_s α_η)``, whose first component
    is ``rhs``'s only entry that is not a moment condition and whose second
    the warp cancels (§3.4, ``seed_weights``).
    """

    xy: np.ndarray
    medium: SmoothBand
    interface: int
    frame: Frame
    profile: NormalProfile
    xi: np.ndarray
    eta: np.ndarray
    profiles: SeedProfiles
    block: np.ndarray
    rhs: np.ndarray
    gradient: tuple[float, float]

    @property
    def scale(self) -> float:
        return self.frame.scale

    @property
    def alpha_e(self) -> float:
        return self.profiles.alpha_e

    @property
    def warp(self) -> np.ndarray:
        """``η̃ = φ₀₁(η)`` at the nodes: the Gaussians' normal coordinate (§3.4)."""
        return self.block[:, _column(self.profiles.chain.degree, 0, 1)]


@dataclass(frozen=True)
class TangentialBasis:
    """One tangential seed stencil (§3.10): ``SeedBasis`` in the foot curve's frame.

    ``coordinates`` takes the frame's place and ``xi, eta`` are the nodes'
    normal coordinates in it; ``profile`` is the normal line ``ξ = 0``
    (route (a)'s, unchanged) and ``profiles`` the 150 states of
    ``coupled_chain`` on it. ``block`` and ``rhs`` are ``SeedBasis``'s, the
    right-hand side ``2 α_e`` on the two quadratics and nothing else, since
    the chain holds on the whole line ``ξ = 0``. ``gradient`` is the
    Gaussians' first-order coefficients at the anchor, ``(A_1(0), B_η(0))``
    = ``(∂_ξ (α/m̂), ∂_η (α m̂))``, which are ``SeedBasis.gradient`` on a flat
    line; ``anchor_flux`` is ``(ψ, ψ′)`` of φ₀₁'s level 0 at the anchor, the
    warp's ``α η̃′`` and its derivative (``α_e`` and 0 by the chain's
    structure, read from the march and the rate).
    """

    xy: np.ndarray
    medium: SmoothBand
    interface: int
    coordinates: FootCoordinates
    profile: NormalProfile
    xi: np.ndarray
    eta: np.ndarray
    profiles: SeedProfiles
    block: np.ndarray
    rhs: np.ndarray
    gradient: tuple[float, float]
    anchor_flux: tuple[float, float]

    @property
    def scale(self) -> float:
        return self.coordinates.scale

    @property
    def alpha_e(self) -> float:
        return self.profiles.alpha_e

    @property
    def warp(self) -> np.ndarray:
        """``η̃ = φ₀₁(ξ, η)`` at the nodes, all its levels (§3.10)."""
        return self.block[:, _column(self.profiles.chain.degree, 0, 1)]


def _column(degree: int, a: int, b: int) -> int:
    e = polynomial_exponents(degree)
    return int(np.flatnonzero((e[:, 0] == a) & (e[:, 1] == b))[0])


def nearest_interface(medium: Band | SmoothBand, x: float, y: float) -> int:
    """The interface whose curve is nearest ``(x, y)`` in signed normal distance."""
    d = [
        abs(float(c.signed_distance(np.array([x]), np.array([y]))[0]))
        for c in medium.interfaces
    ]
    return int(np.argmin(d))


def seed_basis(
    xy: np.ndarray,
    medium: Band | SmoothBand,
    degree: int = SEED_DEGREE,
    rtol: float = SEED_RTOL,
    atol: float = SEED_ATOL,
    tangential: bool = False,
) -> SeedBasis | TangentialBasis:
    """The ``SeedBasis`` of the nodes ``xy``, anchored at ``xy[0]``.

    ``h_s`` is the largest distance from the anchor (periodic in x), E2.3's
    scale. ``α_e`` is the medium's own value at the anchor, the owner's at a
    jump as in E2.3 and the 1-D march. A jump ``Band`` is marched as its
    ``SmoothBand`` at δ = 0. ``tangential=True`` is §3.10's
    ``TangentialBasis`` on the same normal line.
    """
    xy = np.asarray(xy, dtype=float)
    q = polynomial_count(degree)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < q:
        raise ValueError(f"xy must be (k, 2) with k at least the {q} seeds")
    if not isinstance(medium, SmoothBand):
        medium = SmoothBand(medium, 0.0)
    x, y = xy[:, 0], xy[:, 1]
    dx, dy = periodic_dx(x - x[0]), y - y[0]
    scale = float(np.hypot(dx, dy).max())
    check_coincidence(dx / scale, dy / scale)
    x0, y0 = float(x[0]), float(y[0])
    j = nearest_interface(medium, x0, y0)
    profile = medium.normal_profile(j, x0, y0, scale)
    if tangential:
        return _tangential_basis(xy, medium, j, profile, scale, degree, rtol, atol)
    frame = Frame(x0, y0, atan2(profile.ny, profile.nx) - pi / 2, scale)
    xi, eta = frame.local(x, y)
    alpha_e = float(medium.alpha(x[:1], y[:1])[0])
    profiles = seed_profiles(profile, eta, degree, alpha_e, rtol, atol)
    gx, gy = (float(g[0]) for g in medium.gradient(x[:1], y[:1]))
    gradient = tuple(scale * c for c in frame.rotate_in(gx, gy))
    rhs = np.zeros(q)
    rhs[_column(degree, 2, 0)] = rhs[_column(degree, 0, 2)] = 2.0 * alpha_e
    rhs[_column(degree, 1, 0)] = gradient[0]
    return SeedBasis(
        xy,
        medium,
        j,
        frame,
        profile,
        xi,
        eta,
        profiles,
        profiles.values(xi),
        rhs,
        gradient,
    )


def _coordinate_lines(a: Curve, b: Curve) -> bool:
    """Whether ``b`` is a line ``d = const`` of ``a``'s normal coordinates."""
    if isinstance(a, FlatLine) and isinstance(b, FlatLine):
        return True
    return (
        isinstance(a, Circle) and isinstance(b, Circle) and (a.cx, a.cy) == (b.cx, b.cy)
    )


def _tangential_basis(
    xy: np.ndarray,
    medium: SmoothBand,
    j: int,
    profile: NormalProfile,
    scale: float,
    degree: int,
    rtol: float,
    atol: float,
) -> TangentialBasis:
    """``seed_basis(tangential=True)`` past the checks the two bases share."""
    x, y = xy[:, 0], xy[:, 1]
    curve, other = medium.interfaces[j], medium.interfaces[1 - j]
    if medium.delta == 0.0 and not _coordinate_lines(curve, other):
        side = other.level(x, y)
        if (side > 0.0).any() and (side < 0.0).any():
            raise NotImplementedError(
                "the stencil crosses both curves, and the other is not a "
                "coordinate line of the foot curve's frame (§3.10)"
            )
    coords, xi, eta = foot_coordinates(curve, j, x, y, scale, SERIES_DEGREE)
    alpha_e = float(medium.alpha(x[:1], y[:1])[0])
    # One far edge for the march and the anchor's coefficients: its
    # interpolant is a vectorized Newton on 187 points when it is not saturated.
    far = _other_edge(medium, coords, eta) if medium.delta > 0.0 else None
    profiles = _tangential_march(coords, profile, eta, far, alpha_e, degree, rtol, atol)
    q = polynomial_count(degree)
    rhs = np.zeros(q)
    rhs[_column(degree, 2, 0)] = rhs[_column(degree, 0, 2)] = 2.0 * alpha_e
    # The Gaussians' first-order coefficients at the anchor: ∂_ξ(α/m̂) from the
    # series the march uses, and ∂_η(α m̂) = h_s (∂_n α − α_e K_0) from the
    # medium's gradient and the metric (m̂ = G − d K is 1 there).
    ch = profiles.chain
    segment = int(profile.segment(np.zeros(1))[0])
    coefficients = _coefficients(
        coords, _alpha_series(coords, profile.pieces[segment], far)
    )
    _, a, _ = coefficients(0.0)
    gx, gy = (float(g[0]) for g in medium.gradient(x[:1], y[:1]))
    normal = gx * profile.nx + gy * profile.ny
    gradient = (float(a[1]), scale * (normal - alpha_e * float(coords.metric[1][0])))
    rate = ch.rate(alpha_e, coefficients)
    level = ch.index(0, 1, 0)
    anchor_flux = (
        float(profiles.psi[level, 0]),
        float(rate(0.0, ch.initial(alpha_e))[ch.size + level]),
    )
    return TangentialBasis(
        xy,
        medium,
        j,
        coords,
        profile,
        xi,
        eta,
        profiles,
        profiles.values(xi),
        rhs,
        gradient,
        anchor_flux,
    )


def seed_coordinates(
    sb: SeedBasis | TangentialBasis, warp: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """The nodes in the Gaussian block's coordinates: ``(ξ, φ₀₁(η))``, or ``(ξ, η)``.

    §3.4's warp. ``φ₀₁`` is the seed of ``η``, the constant-flux solution
    ``∫ α_e / α_n``: continuous with its flux through the edge, E2.4's
    piecewise-linear stretch at δ = 0 (E4.4 measured 1.8e-15 over 576
    stencils) and the smooth stretch at δ > 0, read off the same march as the
    block. The flux ``ψ₀₁`` must be ``α_e`` along the whole line for the
    ``G_η̃`` term of ``L G`` to cancel, so it is checked here against
    ``WARP_TOL``. ``warp=False`` is the plain-Gaussian ablation (H7). A
    ``TangentialBasis``' warp is ``φ₀₁(ξ, η)`` with all its levels, whose
    level-0 flux is not constant along the line on a curved or tangentially
    varying edge: only its anchor value enters, as ``anchor_flux`` (§3.10).
    """
    if not warp:
        return sb.xi, sb.eta
    if isinstance(sb, TangentialBasis):
        return sb.xi, sb.warp
    flux = sb.profiles.psi[sb.profiles.chain.index(0, 1, 0)]
    drift = float(np.abs(flux - sb.alpha_e).max() / abs(sb.alpha_e))
    if drift > WARP_TOL:
        raise RuntimeError(
            f"the warp's flux alpha eta-tilde' left alpha_e by {drift:.1e} "
            f"relative, above {WARP_TOL:g}: the march, not the geometry"
        )
    return sb.xi, sb.warp


def seed_weights(
    xy: np.ndarray,
    medium: Band | SmoothBand,
    degree: int = SEED_DEGREE,
    shape: float = GA_SHAPE,
    warp: bool = True,
    rtol: float = SEED_RTOL,
    atol: float = SEED_ATOL,
    tangential: bool = False,
) -> np.ndarray:
    """Weights of ``div(alpha grad u)`` at ``xy[0]`` from ``u`` at the nodes ``xy``.

    ``interface.stencil_weights``'s contract with the seeds in place of the
    translated basis (§3.4): EABE eq. 2 with ``seed_basis``'s block for ``P``
    and its moment conditions for the polynomial right-hand side, Gaussians
    ``ε = shape / d`` in the coordinates of ``seed_coordinates``, and for
    their right-hand side ``L`` applied to ``G(ξ, η̃(η))`` at the anchor by
    the chain rule,

        L G = α G_ξξ + α_ξ G_ξ + (α η̃′)′ G_η̃ + α η̃′² G_η̃η̃,

    ``α η̃′`` read from the marched ``ψ₀₁`` and ``(α η̃′)′`` from the chain's
    own rate for that state. Both collapse: ``ψ₀₁`` is ``α_e`` from the
    anchor's initial condition, and its rate is *structurally* zero — the
    state ``(0, 1, 0)`` has no lower seed and no level above it, so the
    chain's ``source`` and ``lower`` rows are empty at every η and for every
    profile. So the term the 2016 rows carry as ``α_η G_η`` is cancelled by
    the warp's curvature and ``L G`` is ``α_e (G_ξξ + G_η̃η̃) + α_ξ G_ξ``. The
    two coefficients are assembled rather than written as 1 and 0 so that a
    chain which ever acquires a source there (a curved feature's anchor
    correction, E4.7) carries it instead of silently losing it, and
    ``seed_coordinates``'s check on the whole line is what actually guards
    the cancellation. With ``warp=False`` the coordinates are the frame's own
    and the plain Gaussian's right-hand side carries the true ``α_η G_η``.
    The weights come back in physical units, ``w̃ / h_s²``. ``tangential=True``
    is the same system on §3.10's ``TangentialBasis``, the Gaussians in
    ``(ξ, φ₀₁(ξ, η))`` of the foot curve's coordinates with
    ``α_e Δ + A_1(0) ∂_ξ`` for their right-hand side (``+ B_η(0) ∂_η``
    plain).
    """
    sb = seed_basis(xy, medium, degree, rtol, atol, tangential)
    return weights_of(sb, shape, warp)


def weights_of(
    sb: SeedBasis | TangentialBasis, shape: float = GA_SHAPE, warp: bool = True
) -> np.ndarray:
    """``seed_weights`` on a basis already marched (the ablations' entry)."""
    xi, eta, eps = _gaussian_coordinates(sb, shape, warp)
    a_e = sb.alpha_e
    g_xi, g_eta = sb.gradient
    ch = sb.profiles.chain
    if warp:
        # (α η̃′) at the anchor and its derivative, from the march and from
        # the chain: 1 and 0 for this chain, whatever the profile (see
        # ``seed_weights``), and read rather than written so that they are
        # the chain's own numbers.
        if isinstance(sb, TangentialBasis):
            flux, d_flux = sb.anchor_flux
        else:
            segment = int(sb.profile.segment(np.zeros(1))[0])
            rate = ch.rate(a_e, sb.profile.alpha_function(segment))
            flux = float(sb.profiles.psi[ch.index(0, 1, 0), 0])
            d_flux = float(rate(0.0, ch.initial(a_e))[ch.size + ch.index(0, 1, 0)])
        slope = flux / a_e
        b_rbf = (
            a_e * gaussian_derivative(xi, eta, eps, "dxx")
            + g_xi * gaussian_derivative(xi, eta, eps, "dx")
            + d_flux * gaussian_derivative(xi, eta, eps, "dy")
            + a_e * slope**2 * gaussian_derivative(xi, eta, eps, "dyy")
        )
    else:
        b_rbf = (
            a_e * gaussian_derivative(xi, eta, eps, "lap")
            + g_xi * gaussian_derivative(xi, eta, eps, "dx")
            + g_eta * gaussian_derivative(xi, eta, eps, "dy")
        )
    block = gaussian(xi[:, None] - xi[None, :], eta[:, None] - eta[None, :], eps)
    w = augmented_solve(
        block[None], sb.block[None], b_rbf[None, :, None], sb.rhs[None, :, None]
    )
    return w[0, :, 0] / sb.scale**2


def _gaussian_coordinates(
    sb: SeedBasis | TangentialBasis, shape: float, warp: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    """The Gaussian block's coordinates (``seed_coordinates``) and its ``ε``."""
    xi, eta = seed_coordinates(sb, warp)
    x, y = sb.xy[:, 0], sb.xy[:, 1]
    r = np.hypot(periodic_dx(x - x[0]), y - y[0])
    eps = shape * sb.scale / float(np.where(r > 0.0, r, np.inf).min())
    return xi, eta, eps


def saddle_system(
    sb: SeedBasis | TangentialBasis, shape: float = GA_SHAPE, warp: bool = True
) -> np.ndarray:
    """EABE eq. 2's matrix ``[[A, S], [Sᵀ, 0]]`` of ``weights_of``'s solve.

    The Gaussians in ``seed_coordinates``' coordinates and the seed block;
    for the conditioning tables (E4.8), the solve itself is ``weights_of``'s.
    """
    xi, eta, eps = _gaussian_coordinates(sb, shape, warp)
    block = gaussian(xi[:, None] - xi[None, :], eta[:, None] - eta[None, :], eps)
    q = sb.block.shape[1]
    return np.block([[block, sb.block], [sb.block.T, np.zeros((q, q))]])


def block_condition(block: np.ndarray) -> tuple[float, float]:
    """``cond`` of a ``(k, q)`` basis block, raw and with its columns scaled to 1.

    The raw number carries the seeds' ``α_e`` normalisation and the contrast;
    the scaled one is what the moment conditions see (§3.3, H3).
    """
    scaled = block / np.abs(block).max(axis=0)
    return float(np.linalg.cond(block)), float(np.linalg.cond(scaled))
