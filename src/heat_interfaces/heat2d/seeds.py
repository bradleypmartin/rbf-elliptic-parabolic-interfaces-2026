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
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from math import atan2, pi

import numpy as np
from scipy.integrate import solve_ivp

from ..heat1d.stiff import MERGE_TOL, SEED_ATOL, SEED_RTOL, march_targets
from .domain import Band, NormalProfile, SmoothBand
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
class SeedProfiles:
    """The chain's state at points ``eta`` of one normal line (§3.2).

    ``g[l, i]`` and ``psi[l, i]`` are state ``l`` of ``chain.levels`` at
    ``eta[i]``, in stencil units. ``psi`` is ``α g′``, continuous through a
    jump; E4.8's residual probes read it.
    """

    eta: np.ndarray
    g: np.ndarray
    psi: np.ndarray
    alpha_e: float
    chain: Chain

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
    y0 = ch.initial(a_e)
    out = np.empty((2 * ch.size, eta.size))
    out[:, eta == 0.0] = y0[:, None]
    for side in (-1.0, 1.0):
        ahead = side * eta > 0.0
        if not ahead.any():
            continue
        far = np.max(side * eta[ahead])
        stops = side * profile.stops
        inside = stops[(stops > MERGE_TOL) & (stops < far)]
        y, t0 = y0, 0.0
        for t in march_targets(side * eta[ahead], inside):
            t1 = side * t
            segment = int(profile.segment(0.5 * (t0 + t1)))
            rate = ch.rate(a_e, profile.alpha_function(segment))
            sol = solve_ivp(rate, (t0, t1), y, method="DOP853", rtol=rtol, atol=atol)
            if not sol.success:
                raise RuntimeError(
                    f"the seed march failed at eta = {t1}: {sol.message}"
                )
            y, t0 = sol.y[:, -1], t1
            hit = ahead & np.isclose(eta, t1, rtol=0.0, atol=MERGE_TOL)
            out[:, hit] = y[:, None]
    return SeedProfiles(eta, out[: ch.size], out[ch.size :], a_e, ch)


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
) -> SeedBasis:
    """The ``SeedBasis`` of the nodes ``xy``, anchored at ``xy[0]``.

    ``h_s`` is the largest distance from the anchor (periodic in x), E2.3's
    scale. ``α_e`` is the medium's own value at the anchor, the owner's at a
    jump as in E2.3 and the 1-D march. A jump ``Band`` is marched as its
    ``SmoothBand`` at δ = 0.
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


def seed_coordinates(sb: SeedBasis, warp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """The nodes in the Gaussian block's coordinates: ``(ξ, φ₀₁(η))``, or ``(ξ, η)``.

    §3.4's warp. ``φ₀₁`` is the seed of ``η``, the constant-flux solution
    ``∫ α_e / α_n``: continuous with its flux through the edge, E2.4's
    piecewise-linear stretch at δ = 0 (E4.4 measured 1.8e-15 over 576
    stencils) and the smooth stretch at δ > 0, read off the same march as the
    block. The flux ``ψ₀₁`` must be ``α_e`` along the whole line for the
    ``G_η̃`` term of ``L G`` to cancel, so it is checked here against
    ``WARP_TOL``. ``warp=False`` is the plain-Gaussian ablation (H7).
    """
    if not warp:
        return sb.xi, sb.eta
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
    The weights come back in physical units, ``w̃ / h_s²``.
    """
    return weights_of(seed_basis(xy, medium, degree, rtol, atol), shape, warp)


def weights_of(sb: SeedBasis, shape: float = GA_SHAPE, warp: bool = True) -> np.ndarray:
    """``seed_weights`` on a ``SeedBasis`` already marched (the ablations' entry)."""
    xi, eta = seed_coordinates(sb, warp)
    x, y = sb.xy[:, 0], sb.xy[:, 1]
    r = np.hypot(periodic_dx(x - x[0]), y - y[0])
    eps = shape * sb.scale / float(np.where(r > 0.0, r, np.inf).min())
    a_e = sb.alpha_e
    g_xi, g_eta = sb.gradient
    ch = sb.profiles.chain
    if warp:
        # (α η̃′) at the anchor and its derivative, from the march and from
        # the chain: 1 and 0 for this chain, whatever the profile (see
        # ``seed_weights``), and read rather than written so that they are
        # the chain's own numbers.
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


def block_condition(block: np.ndarray) -> tuple[float, float]:
    """``cond`` of a ``(k, q)`` basis block, raw and with its columns scaled to 1.

    The raw number carries the seeds' ``α_e`` normalisation and the contrast;
    the scaled one is what the moment conditions see (§3.3, H3).
    """
    scaled = block / np.abs(block).max(axis=0)
    return float(np.linalg.cond(block)), float(np.linalg.cond(scaled))
