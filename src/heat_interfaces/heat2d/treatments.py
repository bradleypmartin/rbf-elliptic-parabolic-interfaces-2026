"""Coefficient treatments for a sub-grid edge on scattered nodes (E4.9, #40).

Plan §3.4 and ``docs/stiff-diffusion.md`` §3.7 H12, the 2-D twin of E3.5's
``heat1d/treatments.py``: change the *medium* the naive operator samples,
never the operator. A nodal treatment is a ``NodalAlpha2D``, a table of
alpha at the nodes of a ``NodeSet``, which ``operators.naive_operator``
reads through ``alpha_matrix`` exactly as it reads any other material; the
sweep then sets ``Dx A Dx + Dy A Dy`` on the treated table against the
true-δ reference.

- **T1**, ``harmonic_discs``: the harmonic mean ``area / ∫ dA/alpha`` over a
  disc of radius ``r`` about each node, the scattered-node form of the
  finite-volume conductance rule (Patankar's interface conductivity in 1-D).
- **T2**, ``arithmetic_discs``: the arithmetic mean ``∫ alpha dA / area``
  over the same disc.
- **T0**, ``widened_edge``: the true band with δ replaced by ``max(δ, m h)``,
  a ``SmoothBand`` again, so the references stay those of the same class.

The plan's radii are ``h/2`` and ``h`` (1-D's one and two cells), with ``h``
the node set's nominal spacing. The discs are clipped at the strip's
Dirichlet rows ``y = 0`` and ``y = 1`` and periodic in ``x``, as 1-D's
windows are clipped at the domain ends. There is no twin of 1-D's T1-FV:
a conservative scheme with exact face conductances needs faces, and a
scattered node set has none (stiff note §2.4, H12). T3, the band-limited
alpha, is dropped, as in 1-D (``LITERATURE.md`` §1d). No source averages
alpha over a disc about each node; the manuscript calls T1 here the natural
scattered-node analogue of the harmonic-mean conductance (§1a K10).

The disc integrals are exact to rounding for the materials of the sweeps
(``disc_integrals``): in the product-grid reference's ``ShearMap``
coordinates ``(x, η)`` both curves of a flat or case-2 band are the lines
``η = c_k`` and the Dirichlet rows are ``η = 0, 1``, so a line of constant
``η`` never crosses an edge, and the only non-smooth points of the outer
integral are the edges themselves, where E3.2's ``EDGE_CUTS`` cut it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..heat1d.domain import EDGE_CUTS
from .domain import Band, NodeSet, SmoothBand
from .exact import ShearMap, shear_of
from .neighbors import wrap_x

DISC_GAUSS = 24
"""Gauss–Legendre points per outer element in ``η`` and along each inner chord.

On case 2 at δ from 0 to 0.04 and radii 0.0053 and 0.0294 the three integrals
agree with 36 points to 1e-14, and with a brute-force polar rule that shears
nothing (at δ = 0 cut at the ray–curve crossings and where the curve meets the
circle, at δ > 0 in 40–120 panels a direction through the tanh) to 5e-15;
``tests/heat2d/test_treatments.py`` pins both at 1e-13.
"""

DISC_CHUNK = 2000
"""Discs per vectorised batch; a batch holds about 3 million quadrature points."""

CHORD_STEPS = 100
"""Newton steps allowed for a chord's end (monotone from outside; 5–15 are used)."""

EXTREME_STEPS = 50
"""Fixed-point steps for the top and bottom of a disc in ``η`` (contracting by κ r)."""


@dataclass(frozen=True, repr=False)
class NodalAlpha2D:
    """A diffusivity given by its ``values`` at the nodes of ``nodes``, nowhere else.

    The material of a nodal treatment. ``alpha`` returns the table when asked
    at the node set's own coordinates, which is the one question the naive
    operator asks (``alpha_matrix``), and refuses any other point: a table on
    scattered nodes has no interpolant worth trusting, and a reference or a
    diagnostic that reached for one would read the treatment where it should
    read the true medium. The ``repr`` carries a hash of the whole table
    (E3.5's lesson: numpy's summary elides the middle of a long array, where
    the edge is, so two treatments could otherwise print alike).
    """

    nodes: NodeSet
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float)
        if values.shape != (self.nodes.n,):
            raise ValueError("one value per node is needed")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("a diffusivity must be positive and finite")
        object.__setattr__(self, "values", values)

    def __repr__(self) -> str:
        digest = hashlib.sha1(self.values.tobytes()).hexdigest()
        return f"NodalAlpha2D(n={self.nodes.n}, h={self.nodes.h!r}, sha1={digest})"

    def alpha(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if not (np.array_equal(x, self.nodes.x) and np.array_equal(y, self.nodes.y)):
            raise ValueError("a nodal treatment is defined at its own nodes only")
        return self.values.copy()


def _extremes(
    shear: ShearMap, x0: np.ndarray, y0: np.ndarray, r: float, top: bool
) -> tuple[np.ndarray, np.ndarray]:
    """``(η, x)`` at the top (or bottom) of each disc's preimage in ``(x, η)``.

    On the upper half circle ``(x₀ + r sin φ, y₀ + r cos φ)`` the derivative
    of ``η`` in ``φ`` is ``−(r / y_η)(y_x cos φ + sin φ)``, so the top is at
    ``tan φ = −y_x``; the bottom, on the lower half, at ``tan φ = y_x``. The
    fixed point contracts by about ``r |y_xx|`` a step (1e-2 here), and it is
    ``φ = 0`` on flat lines.
    """
    sign = 1.0 if top else -1.0
    phi = np.zeros_like(x0)
    for _ in range(EXTREME_STEPS):
        x = x0 + r * np.sin(phi)
        eta = shear.eta(x, y0 + sign * r * np.cos(phi))
        new = -sign * np.arctan(shear.y_x(x, eta))
        settled = np.all(np.abs(new - phi) < 1e-15)
        phi = new
        if settled:
            break
    else:
        raise RuntimeError("the disc's extreme in η did not settle")
    x = x0 + r * np.sin(phi)
    return shear.eta(x, y0 + sign * r * np.cos(phi)), x


def _chord(
    shear: ShearMap, x0: np.ndarray, y0: np.ndarray, r: float, eta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The ends of ``{x : (x − x₀)² + (y(x, η) − y₀)² ≤ r²}`` at each ``η``.

    ``q(x) = (x − x₀)² + (y(x, η) − y₀)² − r²`` is convex while
    ``|y − y₀| |y_xx| < 1`` (``r`` times 0.8 on case 2), so Newton started
    outside the disc, at ``x₀ ± 1.01 r`` where ``q > 0``, walks monotonically
    onto the nearer root and never overshoots, even where the two roots
    nearly meet at the top or bottom of the disc. An end is converged when
    the step is a few ulps of ``x`` itself (the coordinates are O(1), so an
    ulp is 1e-14 of a small radius) or ``q`` is at its rounding, about
    ``2 ε r (|x| + |y|)``: near the top the end is ill-conditioned, the
    slope being twice the half chord, but that chord and its Gauss weight
    are both small and the integral does not see it (the brute-force check
    agrees to 5e-15).
    """
    eps = np.finfo(float).eps
    floor = 4.0 * eps * r * (np.abs(x0) + np.abs(y0) + r)
    ends = []
    for sign in (-1.0, 1.0):
        x = x0 + sign * 1.01 * r
        for _ in range(CHORD_STEPS):
            y = shear.y(x, eta)
            q = (x - x0) ** 2 + (y - y0) ** 2 - r**2
            step = q / (2.0 * (x - x0) + 2.0 * (y - y0) * shear.y_x(x, eta))
            x = x - step
            small = np.abs(step) <= 4.0 * np.spacing(np.abs(x))
            if np.all(small | (np.abs(q) <= floor)):
                break
        else:
            raise RuntimeError("a disc's chord end did not converge")
        ends.append(x)
    return ends[0], ends[1]


def _outer_elements(
    shear: ShearMap,
    delta: float,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(disc, a, b)``: each disc's elements in ``η``, cut at the edges.

    The cuts are the two curves' levels ``c_k`` and, at δ > 0, ``c_k ± m δ``
    for ``m`` in ``EDGE_CUTS`` (E3.2's, whose elements keep the tanh's pole
    well away), wherever they fall strictly inside the disc's ``[lo, hi]``.
    """
    levels = np.array([shear.lower, shear.upper])
    offsets = np.array([0.0, *(s * m for m in EDGE_CUTS for s in (-1.0, 1.0))])
    cuts = np.unique((levels[:, None] + offsets[None, :] * delta).ravel())
    inside = (cuts[None, :] > lo[:, None]) & (cuts[None, :] < hi[:, None])
    count = inside.sum(axis=1) + 1
    points = np.column_stack(
        [lo, np.sort(np.where(inside, cuts[None, :], np.inf), axis=1), hi]
    )
    points[np.arange(lo.size), count] = hi
    keep = np.arange(points.shape[1] - 1)[None, :] < count[:, None]
    disc, _ = np.nonzero(keep)
    return disc, points[:, :-1][keep], points[:, 1:][keep]


def disc_integrals(
    material: Band | SmoothBand,
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    n_gauss: int = DISC_GAUSS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(area, ∫ alpha dA, ∫ dA / alpha)`` over the disc of ``radius`` about each point
    ``(x, y)``.

    Each disc is clipped at ``y = 0`` and ``y = 1`` and wraps in ``x``. The
    integral is taken in the ``ShearMap`` coordinates of ``material``'s two
    graphs (``exact.shear_of``: two flat lines, where the map is the
    identity, or two parallel sine graphs), with ``dA = y_η dx dη``:

    - *outer*, in ``η``, from the disc's bottom to its top (``_extremes``),
      clipped to ``[0, 1]``: cut at the edges (``_outer_elements``) and
      substituted ``η = m + w sin θ`` over the whole disc, so the chord's
      square-root behaviour at the top and bottom is analytic in ``θ``;
    - *inner*, in ``x``, over the chord at each ``η`` (``_chord``), which
      crosses no edge, so one Gauss–Legendre panel resolves it.

    A jump is therefore integrated piece by piece, exactly, and a tanh edge
    on E3.2's elements; ``material.alpha`` is sampled only at interior
    points of the elements, never on an interface. Other curves (the ring's
    circles) have no such shear and are refused by ``shear_of``.
    """
    if not (np.isfinite(radius) and radius > 0.0):
        raise ValueError("the disc radius must be positive")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    shear = shear_of(material)
    delta = float(getattr(material, "delta", 0.0))
    t_out, w_out = np.polynomial.legendre.leggauss(n_gauss)
    t_in, w_in = np.polynomial.legendre.leggauss(n_gauss)
    out = np.zeros((3, x.size))
    for s in range(0, x.size, DISC_CHUNK):
        x0, y0 = x[s : s + DISC_CHUNK], y[s : s + DISC_CHUNK]
        top, _ = _extremes(shear, x0, y0, radius, top=True)
        bottom, _ = _extremes(shear, x0, y0, radius, top=False)
        mid, half = 0.5 * (top + bottom), 0.5 * (top - bottom)
        lo, hi = np.maximum(bottom, 0.0), np.minimum(top, 1.0)
        disc, a, b = _outer_elements(shear, delta, lo, hi)
        theta_a = np.arcsin(np.clip((a - mid[disc]) / half[disc], -1.0, 1.0))
        theta_b = np.arcsin(np.clip((b - mid[disc]) / half[disc], -1.0, 1.0))
        centre, width = 0.5 * (theta_a + theta_b), 0.5 * (theta_b - theta_a)
        theta = (centre[:, None] + width[:, None] * t_out[None, :]).ravel()
        owner = np.repeat(disc, n_gauss)
        eta = mid[owner] + half[owner] * np.sin(theta)
        weight = (width[:, None] * w_out[None, :]).ravel()
        weight *= half[owner] * np.cos(theta)
        left, right = _chord(shear, x0[owner], y0[owner], radius, eta)
        centre_x, half_x = 0.5 * (left + right), 0.5 * (right - left)
        px = centre_x[:, None] + half_x[:, None] * t_in[None, :]
        pe = np.broadcast_to(eta[:, None], px.shape)
        py = shear.y(px, pe)
        w = (weight * half_x)[:, None] * w_in[None, :] * shear.y_eta(px, pe)
        values = material.alpha(wrap_x(px.ravel()), py.ravel()).reshape(px.shape)
        for k, f in enumerate((w, w * values, w / values)):
            out[k, s : s + DISC_CHUNK] = np.bincount(
                owner, weights=f.sum(axis=1), minlength=x0.size
            )
    return out[0], out[1], out[2]


def disc_means(
    nodes: NodeSet,
    material: Band | SmoothBand,
    radius: float,
    n_gauss: int = DISC_GAUSS,
) -> tuple[NodalAlpha2D, NodalAlpha2D]:
    """``(T1, T2)``: the harmonic and arithmetic disc means from one quadrature."""
    area, integral, resistance = disc_integrals(
        material, nodes.x, nodes.y, radius, n_gauss
    )
    return NodalAlpha2D(nodes, area / resistance), NodalAlpha2D(nodes, integral / area)


def harmonic_discs(
    nodes: NodeSet,
    material: Band | SmoothBand,
    radius: float,
    n_gauss: int = DISC_GAUSS,
) -> NodalAlpha2D:
    """T1: ``area / ∫ dA/alpha`` over each node's disc of ``radius``.

    Across a flat jump it is the area-weighted harmonic mean of the two
    pieces, with the circular segments' areas as the weights; on a smooth
    alpha it is ``alpha + (r²/8)(Δalpha − 2|∇alpha|²/alpha) + O(r⁴)``.
    """
    return disc_means(nodes, material, radius, n_gauss)[0]


def arithmetic_discs(
    nodes: NodeSet,
    material: Band | SmoothBand,
    radius: float,
    n_gauss: int = DISC_GAUSS,
) -> NodalAlpha2D:
    """T2: ``∫ alpha dA / area``, ``alpha + (r²/8) Δalpha + O(r⁴)`` where smooth."""
    return disc_means(nodes, material, radius, n_gauss)[1]


def widened_edge(nodes: NodeSet, material: Band | SmoothBand, m: float) -> SmoothBand:
    """T0: the band's edges widened to ``max(δ, m h)``, ``h`` the node set's spacing.

    A ``SmoothBand`` keeps its band and composition; a ``Band`` is the jump
    (δ = 0). Below ``δ = m h`` the treatment does not depend on δ at all,
    and its own error floor is the widened reference's distance from the
    true one (1-D's ``c (m h − δ)``, stiff note §2.4).
    """
    if not (np.isfinite(m) and m > 0.0):
        raise ValueError("the widening factor m must be positive")
    if isinstance(material, SmoothBand):
        width = max(material.delta, m * nodes.h)
        return SmoothBand(material.band, width, material.composition)
    return SmoothBand(material, m * nodes.h)


__all__ = (
    "DISC_GAUSS",
    "NodalAlpha2D",
    "arithmetic_discs",
    "disc_integrals",
    "disc_means",
    "harmonic_discs",
    "widened_edge",
)
