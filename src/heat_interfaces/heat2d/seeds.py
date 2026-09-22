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
from .rbf import check_coincidence, polynomial_count, polynomial_exponents

SEED_DEGREE = 4
"""The seeds' degree: 15 of them on 30 nodes, the interface group's spec (§3.4)."""


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
    weights are ``w̃ / scale²``).
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
    rhs = np.zeros(q)
    rhs[_column(degree, 2, 0)] = rhs[_column(degree, 0, 2)] = 2.0 * alpha_e
    rhs[_column(degree, 1, 0)] = scale * frame.rotate_in(gx, gy)[0]
    return SeedBasis(
        xy, medium, j, frame, profile, xi, eta, profiles, profiles.values(xi), rhs
    )


def block_condition(block: np.ndarray) -> tuple[float, float]:
    """``cond`` of a ``(k, q)`` basis block, raw and with its columns scaled to 1.

    The raw number carries the seeds' ``α_e`` normalisation and the contrast;
    the scaled one is what the moment conditions see (§3.3, H3).
    """
    scaled = block / np.abs(block).max(axis=0)
    return float(np.linalg.cond(block)), float(np.linalg.cond(scaled))
