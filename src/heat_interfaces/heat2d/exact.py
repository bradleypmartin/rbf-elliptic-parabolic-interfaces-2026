"""Separable solutions on the strip: ``u = e^{c t} sin(κx) v(y)``, α constant per layer.

Dissertation eq. 86 and EABE eq. 34: with ``α`` piecewise constant in ``y``
and ``u = e^{c t} sin(κ x) v(y)``, ``u_t = ∇·(α ∇u)`` reduces to
``v'' = (κ² + c/α) v`` on each layer, so ``v`` is a pair of exponentials per
layer; continuity of ``v`` and of ``α v'`` at each interface, ``v(0) = 0``
and ``v(1) = 1`` fix the ``2m`` constants (the ``c₁ … c₆`` of the papers and of
MATLAB ``laplaceSetup.m``). The control problem (``α ≡ 1``, ``c = 0``) is the
one-layer case ``v = sinh(κ y) / sinh κ``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .domain import TWO_PI


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
