"""Piecewise-polynomial bases across 2-D interfaces (dissertation §5.3, EABE §2.2.3).

The 2-D twin of ``heat1d.interface``. Everything acts on coefficient vectors
``c`` of bivariate polynomials ``sum c_k x'^i y'^j`` through degree ``p`` in
the graded order of ``rbf.polynomial_exponents``, written in a local frame
at a point of the interface: ``x'`` along the tangent, ``y'`` along the
normal into the ``level > 0`` side, both in units of the stencil radius
(the MATLAB's ``normfactor``; epic #4, item 5). Differentiation and
multiplication by a smooth ``alpha`` through its Taylor table (EABE eq. 15
in 2-D), the operator ``D = Dx M Dx + Dy M Dy`` (eq. 1 / 77), and the
continuity matrices of eq. 28 / 80 that equate, along the interface
``y' = f(x')`` (eq. 29 / 82), the first ``p + 1 - 2k`` coefficients of
``D^k u`` and the first ``p - 2k`` of the normal flux
``alpha (d/dy' - f' d/dx') D^k u`` (eq. 27, 30 / 79, 83). Those counts are
the MATLAB ``continuityCreator``'s (the text's ``p - 2k`` and ``p - 2k - 1``
count from one), and they make ``C`` square. The translation
``C_to^{-1} C_from`` carries the standard monomials from the stencil
centre's side across each interface the stencil reaches (eq. 81), with an
exact change of frame between the interfaces of a stencil that crosses two
(§5.4.3, case 3's ring), and ``stencil_weights`` puts the translated basis
in place of the monomials of the RBF-FD system (eq. 2).

Curvature enters twice, as in the papers: the interface expansion ``f`` is
inserted for ``y'`` before the coefficients along the interface are read
off, and the normal of eq. 30 turns ``d/dy'`` into ``d/dy' - f'(x') d/dx'``.
The papers multiply by the expansions of ``cos θ'`` and ``sin θ'`` instead;
the common factor ``cos θ' = (1 + f'^2)^{-1/2}`` multiplies both sides of
every flux row and is invertible as a series, so the rows span the same
space and it is dropped here. ``f`` is found numerically, as the papers'
"standard FD method" does: Fornberg weights on ``2p + 1`` samples of the
curve about the foot point. The flat variant (EABE Fig. 10's "linear
interface") sets ``f = 0`` and keeps everything else; on a flat interface
the two are identical to rounding.

The Gaussians of the RBF-FD system are "warped" across the interfaces
(EABE §2.2.4, dissertation Fig. 5-1): in the anchor frame the normal
coordinate of every node across an interface is stretched by
``alpha_centre / alpha_across``, so that a Gaussian of the stretched
coordinates has continuous value and continuous ``alpha ∂_n`` at the
interface, and the RBF part of the stencil upholds the interface conditions
to first order where the translated polynomials uphold them to order ``p``.
``Warp`` is the piecewise-linear stretch of one stencil, ``build_warp``
walks it outward from the anchor region across each interface, and
``stencil_weights(..., warp=False)`` is the plain-Gaussian ablation of EABE
Fig. 11. ``interface_stencil`` holds one crossing stencil's translated basis
and Gaussian coordinates so that ``stencil_weights`` (the operator at the
centre) and ``interpolation_weights`` (the identity at points off the nodes,
E2.6's interface-aware resampling) solve the same system.

Frames are built from each curve's ``normal`` alone, since ``Circle`` and
the graphs orient their tangents differently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import atan2, cos, factorial, pi, sin

import numpy as np
from scipy.signal import convolve2d

from ..fd_weights import fornberg_weights
from .domain import Band, Curve, Piece2D
from .neighbors import periodic_dx
from .rbf import (
    GA_SHAPE,
    augmented_solve,
    check_coincidence,
    gaussian,
    gaussian_derivative,
    polynomial_block,
    polynomial_count,
    polynomial_exponents,
)

# --- coefficient algebra ------------------------------------------------------


def _positions(degree: int) -> np.ndarray:
    """``table[i, j]``: the position of ``x^i y^j`` in the graded order, -1 above it."""
    e = polynomial_exponents(degree)
    table = -np.ones((degree + 1, degree + 1), dtype=int)
    table[e[:, 0], e[:, 1]] = np.arange(len(e))
    return table


def table_to_vector(table: np.ndarray) -> np.ndarray:
    """A ``(p + 1, p + 1)`` table ``a[i, j]`` as a graded coefficient vector."""
    table = np.asarray(table, dtype=float)
    e = polynomial_exponents(table.shape[0] - 1)
    return table[e[:, 0], e[:, 1]]


def vector_to_table(vector: np.ndarray, degree: int) -> np.ndarray:
    e = polynomial_exponents(degree)
    table = np.zeros((degree + 1, degree + 1))
    table[e[:, 0], e[:, 1]] = np.asarray(vector, dtype=float)
    return table


def coefficient_dx(degree: int) -> np.ndarray:
    """``d/dx`` on coefficient vectors: ``x^i y^j -> i x^(i-1) y^j`` (eq. 13 in 2-D)."""
    e, pos = polynomial_exponents(degree), _positions(degree)
    d = np.zeros((len(e), len(e)))
    for k, (i, j) in enumerate(e):
        if i > 0:
            d[pos[i - 1, j], k] = i
    return d


def coefficient_dy(degree: int) -> np.ndarray:
    e, pos = polynomial_exponents(degree), _positions(degree)
    d = np.zeros((len(e), len(e)))
    for k, (i, j) in enumerate(e):
        if j > 0:
            d[pos[i, j - 1], k] = j
    return d


def multiplication_matrix(table: np.ndarray) -> np.ndarray:
    """Multiplication by ``sum a_ij x^i y^j``, truncated to the table's degree.

    Column ``k`` is ``a`` shifted by the exponents of monomial ``k``; the
    product's terms above degree ``p`` are dropped, as in eq. 15 / 66.
    """
    table = np.asarray(table, dtype=float)
    degree = table.shape[0] - 1
    e = polynomial_exponents(degree)
    m = np.zeros((len(e), len(e)))
    ii, jj = np.indices(table.shape)
    for k, (i, j) in enumerate(e):
        shifted = np.zeros_like(table)
        shifted[i:, j:] = table[: degree + 1 - i, : degree + 1 - j]
        shifted[ii + jj > degree] = 0.0
        m[:, k] = table_to_vector(shifted)
    return m


def coefficient_operator(table: np.ndarray) -> np.ndarray:
    """``D = Dx M Dx + Dy M Dy`` on coefficient vectors (eq. 1 / 77).

    Exact below degree ``p``: the degree-``p`` part of ``D c`` would need the
    Taylor terms of ``alpha`` above ``p`` and is truncated with them, which
    is why the continuity rows of order ``k`` stop at degree ``p - 2k``.
    """
    degree = np.asarray(table).shape[0] - 1
    dx, dy = coefficient_dx(degree), coefficient_dy(degree)
    m = multiplication_matrix(table)
    return dx @ m @ dx + dy @ m @ dy


def _product(a: np.ndarray, b: np.ndarray, degree: int) -> np.ndarray:
    return convolve2d(a, b)[: degree + 1, : degree + 1]


def substitution_matrix(
    degree: int, angle: float, shift: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """``S`` with ``c' = S c`` when ``(x, y) = R(angle) (x', y') + shift``.

    ``sum c_k x^i y^j`` and ``sum c'_k x'^i y'^j`` are the same polynomial;
    a rotation with a shift keeps the degree, so nothing is truncated.
    Column ``k`` is the expansion of ``x^i y^j`` in the primed variables.
    """
    c, s = cos(angle), sin(angle)
    x = np.zeros((degree + 1, degree + 1))
    y = np.zeros((degree + 1, degree + 1))
    x[0, 0], x[1, 0], x[0, 1] = shift[0], c, -s
    y[0, 0], y[1, 0], y[0, 1] = shift[1], s, c
    unit = np.zeros((degree + 1, degree + 1))
    unit[0, 0] = 1.0
    xp, yp = [unit], [unit]
    for _ in range(degree):
        xp.append(_product(xp[-1], x, degree))
        yp.append(_product(yp[-1], y, degree))
    e = polynomial_exponents(degree)
    out = np.empty((len(e), len(e)))
    for k, (i, j) in enumerate(e):
        out[:, k] = table_to_vector(_product(xp[i], yp[j], degree))
    return out


def restriction_matrix(expansion: np.ndarray, degree: int) -> np.ndarray:
    """``(p + 1, q)``: coefficients in ``x'`` of ``c(x', f(x'))`` through degree ``p``.

    Eq. 29 / 82 with ``f`` the interface expansion ``sum f_m x'^m``: column
    ``k`` is ``x'^i f(x')^j`` truncated. With ``f = 0`` it picks out the pure
    ``x'`` powers, the MATLAB's ``colCounter`` rows.
    """
    f = np.asarray(expansion, dtype=float)
    if f.shape != (degree + 1,):
        raise ValueError(f"the expansion needs {degree + 1} coefficients")
    powers = [np.eye(1, degree + 1, 0)[0]]
    for _ in range(degree):
        powers.append(np.convolve(powers[-1], f)[: degree + 1])
    e = polynomial_exponents(degree)
    r = np.zeros((degree + 1, len(e)))
    for k, (i, j) in enumerate(e):
        r[i:, k] = powers[j][: degree + 1 - i]
    return r


# --- continuity ----------------------------------------------------------------


def continuity_conditions(degree: int) -> list[tuple[str, int, int]]:
    """``(kind, k, count)`` in row order: ``(u, 0, p+1), (flux, 0, p), (u, 1, p-1) …``

    Temperature after ``k`` applications of ``D`` contributes ``p + 1 - 2k``
    rows, the flux ``p - 2k``; the total is ``(p + 1)(p + 2) / 2``, one row
    per monomial, so the continuity matrices are square. Degree 4 gives
    5 + 4 + 3 + 2 + 1 = 15.
    """
    out: list[tuple[str, int, int]] = []
    k = 0
    while degree + 1 - 2 * k > 0:
        out.append(("u", k, degree + 1 - 2 * k))
        if degree - 2 * k > 0:
            out.append(("flux", k, degree - 2 * k))
        k += 1
    return out


def continuity_matrix(table: np.ndarray, expansion: np.ndarray) -> np.ndarray:
    """One side's ``C`` of ``C_L u_L = C_R u_R`` (eq. 28 / 80).

    ``table`` is ``alpha``'s Taylor table on that side in the frame's units,
    ``expansion`` the interface ``f``. Row ``("u", k, m)`` is the coefficient
    of ``x'^m`` in ``(D^k c)(x', f(x'))``; row ``("flux", k, m)`` that of
    ``alpha (d/dy' - f' d/dx') D^k c`` restricted the same way (eq. 27, 30).
    """
    table = np.asarray(table, dtype=float)
    degree = table.shape[0] - 1
    f = np.asarray(expansion, dtype=float)
    dx, dy = coefficient_dx(degree), coefficient_dy(degree)
    d = coefficient_operator(table)
    fprime = np.zeros((degree + 1, degree + 1))
    fprime[:degree, 0] = f[1:] * np.arange(1, degree + 1)
    flux = multiplication_matrix(table) @ (dy - multiplication_matrix(fprime) @ dx)
    restrict = restriction_matrix(f, degree)
    rows = []
    for kind, k, count in continuity_conditions(degree):
        d_k = np.linalg.matrix_power(d, k)
        rows.append((restrict @ (d_k if kind == "u" else flux @ d_k))[:count])
    return np.vstack(rows)


def translation_matrix(
    table_from: np.ndarray, table_to: np.ndarray, expansion: np.ndarray
) -> np.ndarray:
    """``C_to^{-1} C_from``: coefficients on the ``to`` side from the ``from`` side.

    Each row of ``[C_from | C_to]`` is scaled by its largest entry first, the
    MATLAB's equilibration; it leaves the product unchanged and keeps the
    solve well conditioned for the extreme contrasts of EABE §3.3.3.
    """
    c_from = continuity_matrix(table_from, expansion)
    c_to = continuity_matrix(table_to, expansion)
    scale = np.abs(np.hstack([c_from, c_to])).max(axis=1)[:, None]
    return np.linalg.solve(c_to / scale, c_from / scale)


# --- frames ---------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """Local coordinates at ``(x0, y0)``: ``x'`` along ``angle``, ``y'`` to its left.

    In units of ``scale``, the stencil radius. Built by ``frame_at`` so that
    ``y'`` points along the curve's normal into ``level > 0``.
    """

    x0: float
    y0: float
    angle: float
    scale: float

    def local(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(x', y')`` of global points, x measured to the nearest image."""
        dx = periodic_dx(np.asarray(x, dtype=float) - self.x0)
        dy = np.asarray(y, dtype=float) - self.y0
        c, s = cos(self.angle), sin(self.angle)
        return (c * dx + s * dy) / self.scale, (-s * dx + c * dy) / self.scale

    def rotate_in(self, gx: float, gy: float) -> tuple[float, float]:
        """Components of a global vector along ``x'`` and ``y'`` (no scaling)."""
        c, s = cos(self.angle), sin(self.angle)
        return c * gx + s * gy, -s * gx + c * gy


def frame_at(curve: Curve, s: float, scale: float) -> Frame:
    """The frame at ``curve.point(s)`` with ``y'`` along ``curve.normal(s)``."""
    x0, y0 = curve.point(np.asarray(s, dtype=float))
    nx, ny = curve.normal(np.asarray(s, dtype=float))
    return Frame(float(x0), float(y0), atan2(float(ny), float(nx)) - pi / 2, scale)


def frame_change(a: Frame, b: Frame, degree: int) -> np.ndarray:
    """``S`` with ``c_b = S c_a``: one polynomial in two frames of one stencil."""
    # Exact equality on purpose: one stencil threads one Python float through
    # every frame it builds, so a mismatch is a wiring error, not rounding.
    if a.scale != b.scale:
        raise ValueError("frames of one stencil share their scale")
    xi, eta = a.local(b.x0, b.y0)
    return substitution_matrix(degree, b.angle - a.angle, (float(xi), float(eta)))


def local_taylor(piece: Piece2D, frame: Frame, degree: int) -> np.ndarray:
    """``alpha``'s Taylor table about the frame's origin in the frame's coordinates.

    ``piece.taylor`` expands in global offsets; the coefficients are scaled
    by ``scale^(i + j)`` and rotated by ``substitution_matrix``.
    """
    table = piece.taylor(frame.x0, frame.y0, degree)
    i, j = np.indices(table.shape)
    scaled = table * frame.scale ** (i + j)
    rotated = substitution_matrix(degree, frame.angle) @ table_to_vector(scaled)
    return vector_to_table(rotated, degree)


def interface_expansion(
    curve: Curve, s: float, frame: Frame, degree: int
) -> np.ndarray:
    """``f_0 … f_p`` of ``y' = f(x')`` in the frame's units, from samples of the curve.

    Fornberg weights at ``x' = 0`` on ``2p + 1`` samples at arc spacing
    ``scale / p`` about ``curve.point(s)``: the papers' numerical route to
    the local interface expansion. ``f_0`` and ``f_1`` are rounding, since
    the frame is tangent at its origin; ``f_2`` is half the signed curvature
    times ``scale``.
    """
    ds = frame.scale / degree / curve.length
    samples = float(s) + ds * np.arange(-degree, degree + 1)
    px, py = curve.point(samples)
    xi, eta = frame.local(px, py)
    weights = fornberg_weights(0.0, xi, degree)
    return (weights @ eta) / np.array([factorial(k) for k in range(degree + 1)])


@dataclass(frozen=True)
class LocalInterface:
    """One interface as one stencil sees it: its frame, expansion and ``alpha`` tables.

    ``minus`` is ``alpha`` on the ``level < 0`` side (region ``j`` of the
    band), ``plus`` on the ``level > 0`` side (region ``j + 1``), both in
    the frame's units.
    """

    frame: Frame
    expansion: np.ndarray
    minus: np.ndarray
    plus: np.ndarray

    @property
    def degree(self) -> int:
        return self.minus.shape[0] - 1


def local_interface(
    band: Band,
    j: int,
    x: float,
    y: float,
    scale: float,
    degree: int,
    curvature: bool = True,
) -> LocalInterface:
    """Interface ``j`` of ``band`` framed at its point nearest ``(x, y)``.

    ``curvature=False`` is the flat variant: the expansion is zero and the
    interface is its tangent line at the foot point.
    """
    curve = band.interfaces[j]
    s = float(curve.closest(x, y))
    frame = frame_at(curve, s, scale)
    f = (
        interface_expansion(curve, s, frame, degree)
        if curvature
        else np.zeros(degree + 1)
    )
    return LocalInterface(
        frame,
        f,
        local_taylor(band.region_piece(j), frame, degree),
        local_taylor(band.region_piece(j + 1), frame, degree),
    )


# --- the translated basis --------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """The basis on one region of a stencil, in the frame of interface ``frame``.

    Column ``j`` of ``coefficients`` is basis function ``j``.
    """

    frame: int
    coefficients: np.ndarray


def translated_basis(
    interfaces: Mapping[int, LocalInterface], anchor: int, lowest: int, highest: int
) -> dict[int, Region]:
    """Bases on regions ``lowest … highest`` continuous in ``u`` and normal flux.

    Region ``anchor`` carries the standard monomials (the ``U_R`` of eq. 81)
    in the frame of interface ``anchor`` (or ``anchor - 1`` at the top);
    every other region's basis comes from translating across the interfaces
    in between, changing frame at each. ``interfaces`` holds interface ``j``
    for ``lowest <= j < highest``. Which region is anchored only changes the
    basis of the span, not the span itself.
    """
    if not lowest <= anchor <= highest or lowest >= highest:
        raise ValueError("anchor must lie in lowest..highest, with lowest < highest")
    frame = anchor if anchor < highest else anchor - 1
    degree = interfaces[frame].degree
    q = polynomial_count(degree)
    regions = {anchor: Region(frame, np.eye(q))}

    def carried_to(region: Region, k: int) -> np.ndarray:
        if region.frame == k:
            return region.coefficients
        a, b = interfaces[region.frame].frame, interfaces[k].frame
        return frame_change(a, b, degree) @ region.coefficients

    for r in range(anchor, highest):
        it = interfaces[r]
        across = translation_matrix(it.minus, it.plus, it.expansion)
        regions[r + 1] = Region(r, across @ carried_to(regions[r], r))
    for r in range(anchor, lowest, -1):
        it = interfaces[r - 1]
        across = translation_matrix(it.plus, it.minus, it.expansion)
        regions[r - 1] = Region(r - 1, across @ carried_to(regions[r], r - 1))
    return regions


# --- warped RBFs ----------------------------------------------------------------


@dataclass(frozen=True)
class Warp:
    """The coordinate stretch of EABE §2.2.4 for one stencil, in its anchor frame.

    A node in region ``r`` has its normal coordinate replaced by
    ``eta_tilde = slope[r - lowest] * eta + intercept[r - lowest]``: the
    identity on the anchor's region, continuous across each interface at
    that interface's ``eta`` in the frame, and with
    ``alpha_minus * slope_minus == alpha_plus * slope_plus`` at each, so that
    a smooth function of ``(xi, eta_tilde)`` has continuous ``alpha ∂_eta``
    there. ``alpha`` is each side's value at the interface's foot point (the
    MATLAB's ``rhoEval / rhoAcross`` used the pieces' constant parts).
    """

    frame: int
    lowest: int
    slope: np.ndarray
    intercept: np.ndarray

    def apply(self, eta: np.ndarray, region: np.ndarray) -> np.ndarray:
        """``eta_tilde`` of nodes at ``eta`` (frame units) in ``region``."""
        r = np.asarray(region, dtype=int) - self.lowest
        return self.slope[r] * np.asarray(eta, dtype=float) + self.intercept[r]


def build_warp(
    interfaces: Mapping[int, LocalInterface], anchor: int, lowest: int, highest: int
) -> Warp:
    """The ``Warp`` of a stencil centred in region ``anchor`` of ``lowest … highest``.

    Walked outward from the anchor region one interface at a time, as
    ``translated_basis`` walks the translation: across interface ``r`` the
    slope is multiplied by ``alpha_minus / alpha_plus`` going up and by the
    inverse going down, and the intercept keeps ``eta_tilde`` continuous at
    the interface's ``eta`` in the anchor frame. That ``eta`` is zero for the
    anchor's own interface; the other interface's is its foot point read in
    the anchor frame, exact for parallel lines and concentric circles and
    the MATLAB's ``zoneWidth`` approximation for the sine pair of case 2.
    """
    if not lowest <= anchor <= highest or lowest >= highest:
        raise ValueError("anchor must lie in lowest..highest, with lowest < highest")
    frame = anchor if anchor < highest else anchor - 1
    base = interfaces[frame].frame
    m = highest - lowest + 1
    slope, intercept = np.ones(m), np.zeros(m)

    def eta_of(j: int) -> float:
        if j == frame:
            return 0.0
        other = interfaces[j].frame
        return float(base.local(other.x0, other.y0)[1])

    def step(r: int, j: int, ratio: float, to: int) -> None:
        b = eta_of(j)
        s = slope[r - lowest] * ratio
        intercept[to - lowest] = slope[r - lowest] * b + intercept[r - lowest] - s * b
        slope[to - lowest] = s

    for r in range(anchor, highest):
        it = interfaces[r]
        step(r, r, it.minus[0, 0] / it.plus[0, 0], r + 1)
    for r in range(anchor, lowest, -1):
        it = interfaces[r - 1]
        step(r, r - 1, it.plus[0, 0] / it.minus[0, 0], r - 1)
    return Warp(frame, lowest, slope, intercept)


def stencil_weights(
    xy: np.ndarray,
    band: Band,
    degree: int,
    shape: float = GA_SHAPE,
    curvature: bool = True,
    warp: bool = True,
) -> np.ndarray:
    """Weights of ``div(alpha grad u)`` at ``xy[0]`` from ``u`` at the nodes ``xy``.

    The RBF-FD system of eq. 2 with the translated basis in place of the
    monomials (EABE p. 24): Gaussians ``ε = shape / d`` in offsets from the
    centre scaled by the stencil radius, and for each node the basis of its
    region evaluated in that region's frame. The right-hand side is the
    operator on the centre's own side, ``alpha ∇² + ∇alpha · ∇``, applied to
    each function at the centre. With ``warp`` the Gaussians are those of
    EABE §2.2.4: their offsets are taken in the anchor frame with the normal
    coordinate stretched by ``build_warp``, on both sides of the Gaussian
    block and in its right-hand side; the centre's own region is unstretched,
    so its derivatives at the centre are the plain Gaussian's at the warped
    offset, and ``∇alpha`` is rotated into the frame as the polynomial
    right-hand side already is. ``warp=False`` keeps the plain Gaussians of
    the global frame. The stencil must reach more than one region of
    ``band``; the weights come back in physical units.
    """
    st = interface_stencil(xy, band, degree, shape, curvature, warp)
    x0, y0 = st.xy[0]
    piece = band.region_piece(st.anchor)
    a0 = float(piece.alpha(st.xy[:1, 0], st.xy[:1, 1])[0])
    gx, gy = (
        float(g[0]) * st.scale for g in piece.gradient(st.xy[:1, 0], st.xy[:1, 1])
    )
    frame = st.frame
    xi0, eta0 = frame.local(x0, y0)
    v = polynomial_block(xi0, eta0, degree)
    ddx, ddy = coefficient_dx(degree), coefficient_dy(degree)
    c = st.regions[st.anchor].coefficients
    g_xi, g_eta = frame.rotate_in(gx, gy)
    b_poly = a0 * (v @ (ddx @ ddx + ddy @ ddy) @ c) + g_xi * (v @ ddx @ c)
    b_poly = b_poly + g_eta * (v @ ddy @ c)

    if st.warp is not None:
        gx, gy = g_xi, g_eta
    b_rbf = (
        a0 * gaussian_derivative(st.xi, st.eta, st.eps, "lap")
        + gx * gaussian_derivative(st.xi, st.eta, st.eps, "dx")
        + gy * gaussian_derivative(st.xi, st.eta, st.eps, "dy")
    )
    w = augmented_solve(
        st.gaussian_block()[None],
        st.polynomial_block()[None],
        b_rbf[None, :, None],
        b_poly[None, :, None],
    )
    return w[0, :, 0] / st.scale**2


@dataclass(frozen=True)
class InterfaceStencil:
    """One crossing stencil: its regions, translated basis and Gaussian coordinates.

    Built by ``interface_stencil`` for the ``k`` nodes ``xy`` (the centre
    first) of a stencil that reaches more than one region of ``band``.
    ``regions`` is the translated basis of ``translated_basis`` anchored on
    the centre's region and ``interfaces`` the local interfaces it was built
    from; ``xi, eta`` are the coordinates the Gaussian block is written in,
    offsets from the centre in units of ``scale``, the stencil radius: the
    anchor frame's with the normal coordinate stretched by ``warp`` (EABE
    §2.2.4), or the global ones when ``warp`` is ``None``; ``eps`` is the
    shape parameter in those units. ``stencil_weights`` puts the operator on
    the right-hand side of the system, ``interpolation_weights`` the identity
    at points off the nodes (E2.6's resampling).
    """

    xy: np.ndarray
    band: Band
    degree: int
    region: np.ndarray
    scale: float
    eps: float
    interfaces: Mapping[int, LocalInterface]
    regions: Mapping[int, Region]
    warp: Warp | None
    xi: np.ndarray
    eta: np.ndarray

    @property
    def anchor(self) -> int:
        return int(self.region[0])

    @property
    def frame(self) -> Frame:
        """The anchor frame: that of the interface next to the centre's region."""
        return self.interfaces[self.regions[self.anchor].frame].frame

    def reach(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The region of each point, refused where the stencil has no basis.

        Both ``basis`` and ``gaussian_coordinates`` go through here, so an
        out-of-reach point is a ``ValueError`` whichever is called first; the
        warp's slope table would otherwise be indexed past its end (or, with
        the warp off, the point read as if the stencil covered it).
        """
        region = self.band.region_index(x, y)
        if region.size == 0:
            raise ValueError("no points to read")
        if region.min() < min(self.regions) or region.max() > max(self.regions):
            raise ValueError("a point lies in a region the stencil does not reach")
        return region

    def basis(
        self,
        x: np.ndarray,
        y: np.ndarray,
        regions: Mapping[int, Region] | None = None,
    ) -> np.ndarray:
        """``(len(x), q)``: the translated basis at points, in each point's frame.

        ``regions`` is another basis of the same span on the same interfaces,
        such as ``translated_basis(self.interfaces, 1, ...)`` anchored on the
        band the way the MATLAB anchored (E2.9's conditioning comparison);
        the stencil's own by default.
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        region = self.reach(x, y)
        regions = self.regions if regions is None else regions
        out = np.empty((x.size, polynomial_count(self.degree)))
        for index, reg in regions.items():
            mask = region == index
            if mask.any():
                xi, eta = self.interfaces[reg.frame].frame.local(x[mask], y[mask])
                out[mask] = polynomial_block(xi, eta, self.degree) @ reg.coefficients
        return out

    def gaussian_coordinates(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(xi, eta)`` of points in the Gaussian block's coordinates (see ``xi``)."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        region = self.reach(x, y)
        return _gaussian_coordinates(
            x, y, region, self.xy[0], self.anchor, self.scale, self.frame, self.warp
        )

    def polynomial_block(
        self, regions: Mapping[int, Region] | None = None
    ) -> np.ndarray:
        """``(k, q)``: the translated basis at the nodes, the ``P`` of eq. 2."""
        return self.basis(self.xy[:, 0], self.xy[:, 1], regions)

    def gaussian_block(self) -> np.ndarray:
        """``(k, k)``: ``φ(|x̃_i − x̃_j|)`` in the (warped) coordinates."""
        return gaussian(
            self.xi[:, None] - self.xi[None, :],
            self.eta[:, None] - self.eta[None, :],
            self.eps,
        )


def _gaussian_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    region: np.ndarray,
    centre: np.ndarray,
    anchor: int,
    scale: float,
    frame: Frame,
    warp: Warp | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Offsets from ``centre`` in the Gaussian block's coordinates."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if warp is None:
        return periodic_dx(x - centre[0]) / scale, (y - centre[1]) / scale
    xi, eta = frame.local(x, y)
    eta = warp.apply(eta, region)
    xi0, eta0 = frame.local(centre[0], centre[1])
    eta0 = warp.apply(eta0, anchor)
    return xi - xi0, eta - eta0


def interface_stencil(
    xy: np.ndarray,
    band: Band,
    degree: int,
    shape: float = GA_SHAPE,
    curvature: bool = True,
    warp: bool = True,
) -> InterfaceStencil:
    """The ``InterfaceStencil`` of the nodes ``xy``, centred on ``xy[0]``.

    Frames, expansions and α tables at each interface the stencil reaches
    (``local_interface``; ``curvature=False`` is the flat variant), the
    translated basis anchored on the centre's region, and the Gaussian
    coordinates, warped by ``build_warp`` unless ``warp`` is off. The stencil
    must reach more than one region of ``band``.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < polynomial_count(degree):
        raise ValueError("xy must be (k, 2) with k at least the monomial count")
    x, y = xy[:, 0], xy[:, 1]
    region = band.region_index(x, y)
    lowest, highest = int(region.min()), int(region.max())
    if lowest == highest:
        raise ValueError("the stencil lies in one region; use the direct weights")
    dx, dy = periodic_dx(x - x[0]), y - y[0]
    r = np.hypot(dx, dy)
    scale = float(r.max())
    nearest = float(r[1:].min())
    check_coincidence(dx / scale, dy / scale)

    anchor = int(region[0])
    locals_ = {
        j: local_interface(band, j, float(x[0]), float(y[0]), scale, degree, curvature)
        for j in range(lowest, highest)
    }
    regions = translated_basis(locals_, anchor, lowest, highest)
    stretch = build_warp(locals_, anchor, lowest, highest) if warp else None
    frame = locals_[regions[anchor].frame].frame
    xi, eta = _gaussian_coordinates(x, y, region, xy[0], anchor, scale, frame, stretch)
    return InterfaceStencil(
        xy,
        band,
        degree,
        region,
        scale,
        shape * scale / nearest,
        locals_,
        regions,
        stretch,
        xi,
        eta,
    )


def interpolation_weights(
    xy: np.ndarray,
    band: Band,
    degree: int,
    points: tuple[np.ndarray, np.ndarray],
    shape: float = GA_SHAPE,
    curvature: bool = True,
    warp: bool = True,
) -> np.ndarray:
    """``(len(points), k)`` weights reading ``u`` at ``points`` from a crossing stencil.

    The system of ``stencil_weights`` with the identity for the operator: the
    right-hand side is each (warped) Gaussian and each translated basis
    function at the point, in the region the point lies in, so the
    interpolant upholds the interface conditions to the order of the basis
    (E2.6, ``heat2d.resample``). ``points`` is ``(x, y)`` of arrays; a point in
    a region the stencil does not reach is refused.
    """
    st = interface_stencil(xy, band, degree, shape, curvature, warp)
    px, py = (np.atleast_1d(np.asarray(p, dtype=float)) for p in points)
    if px.shape != py.shape or px.ndim != 1:
        raise ValueError("points must be a pair of matching 1-D arrays")
    b_poly = st.basis(px, py).T
    xe, ye = st.gaussian_coordinates(px, py)
    b_rbf = gaussian(
        xe[:, None] - st.xi[None, :], ye[:, None] - st.eta[None, :], st.eps
    )
    w = augmented_solve(
        st.gaussian_block()[None],
        st.polynomial_block()[None],
        b_rbf.T[None],
        b_poly[None],
    )
    return w[0].T
