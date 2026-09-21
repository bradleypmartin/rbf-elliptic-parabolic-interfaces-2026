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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .domain import RING, TWO_PI


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
    ring's width times ``r^(m-1)``: 1.05 at case 3's ring for ``m = 2``,
    against 0.12 inside.
    """

    radii: tuple[float, ...]
    alphas: tuple[float, ...]
    mode: int = 2
    scale_radius: float = 0.5
    cx: float = 0.5
    cy: float = 0.5
    _coefficients: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.radii) != len(self.alphas) - 1:
            raise ValueError("one radius fewer than rings is needed")
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
        # Walk outward: V = R and F = α (a r^m − b r^-m) (the flux times r/m)
        # are continuous, and the next ring's pair follows from them.
        m = self.mode
        coef = np.zeros((len(self.alphas), 2))
        coef[0] = [1.0, 0.0]
        for k, r in enumerate(self.radii):
            a, b = coef[k]
            value = a * r**m + b * r**-m
            flux = self.alphas[k] * (a * r**m - b * r**-m)
            up = flux / self.alphas[k + 1]
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


def ring_exact(mode: int = 2) -> RingMode:
    """Case 3's ring at its constant part: ``α = 1/1500`` on ``0.349 <= r <= 0.35``.

    ``u = r^m cos mθ`` scaled inside, 1 at ``(0.5, θ = 0)``; nearly all of the
    change is across the ring, as in case 3 itself.
    """
    return RingMode(RING, (1.0, 1.0 / 1500.0, 1.0), mode)
