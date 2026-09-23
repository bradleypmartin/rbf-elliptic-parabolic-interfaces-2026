"""E4.8 (#39): EABE eq. 40's ring with seeds, at δ = 0 and with smooth edges.

The ring's exact width (``Band.gap``) and the seeds' march from the outer circle
(stiff note §3.6's widths rule); the resistance composition of the smooth ring;
the ring's piecewise Chebyshev series (``seeds.RING_POINTS``); the radial
references the probe and the matched residual read (``RingMode`` with a width,
``SmoothRingMode``, ``matched_radial``).
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from heat_interfaces.heat1d.stiff import EDGE_STOP
from heat_interfaces.heat2d import seeds
from heat_interfaces.heat2d.domain import (
    GAP_TOL,
    Band,
    Circle,
    Constant2D,
    FlatLine,
    SineProduct,
    SmoothBand,
    build_node_set,
    case3,
    layer_share,
    layer_share_at,
    ring_gap,
    ring_radii,
    with_smooth_edges,
)
from heat_interfaces.heat2d.exact import (
    RING_REACH,
    RingMode,
    SmoothRingMode,
    matched_radial,
    radial_elements,
    ring_exact,
)
from heat_interfaces.heat2d.interface import stencil_weights
from heat_interfaces.heat2d.neighbors import knn
from heat_interfaces.heat2d.operators import interface_crossings
from heat_interfaces.heat2d.seeds import (
    flux_exponents,
    saddle_system,
    seed_basis,
    weights_of,
)


def constant_ring(s: float, delta: float = 0.0, gap: bool = True):
    """Eq. 40's ring at its constant part ``1/(1.5 s)``, 1 outside."""
    inner, outer = ring_radii(s)
    band = Band(
        Circle(inner),
        Circle(outer),
        Constant2D(1.0 / (1.5 * s)),
        Constant2D(1.0),
        gap=ring_gap(s) if gap else None,
    )
    return SmoothBand(band, delta, "resistance") if delta else band


def ring_stencils(s: float, n: int = 1250, count: int = 24):
    """Every ``len // count``-th 30-node stencil of eq. 40's set crossing the ring."""
    domain = case3(s)
    nodes = build_node_set(domain, n, seed=0, iterations=20)
    index, _ = knn(nodes.xy, 30)
    rows = index[interface_crossings(nodes, domain.material, index)]
    return nodes, rows[:: max(1, len(rows) // count)]


# --- the gap --------------------------------------------------------------------------


def test_case3_carries_the_rings_exact_width():
    for s in (1e3, 1e8, 1e11):
        band = case3(s).material
        assert band.gap == ring_gap(s) == 1.0 / s
        stored = band.upper.radius - band.lower.radius
        assert abs(stored - band.gap) <= 1e-7 * band.gap
    # s = 10³ stores it to rounding; 10¹¹ to 8e-8 (port notes §2.9).
    assert abs(ring_radii(1e11)[1] - ring_radii(1e11)[0] - 1e-11) > 1e-19


def test_a_gap_belongs_to_two_concentric_circles_at_their_stored_width():
    pieces = (Constant2D(0.2), Constant2D(1.0))
    Band(Circle(0.3), Circle(0.35), *pieces, gap=0.05 * (1 + 0.5 * GAP_TOL))
    for curves, gap in (
        ((Circle(0.3), Circle(0.35)), 0.051),
        ((Circle(0.35), Circle(0.3)), 0.05),
        ((Circle(0.3), Circle(0.35, cx=0.51)), 0.05),
        ((FlatLine(0.6), FlatLine(0.8)), 0.2),
    ):
        with pytest.raises(ValueError):
            Band(*curves, *pieces, gap=gap)


@pytest.mark.parametrize("delta", (0.0, 0.0004))
def test_the_rings_stops_are_offsets_from_the_outer_circle(delta):
    # §3.6's widths rule: the inner crossing is -gap/scale exactly in the
    # march's coordinate, whatever the stored radii say.
    s = 1e11
    band = SmoothBand(case3(s).material, delta, "resistance")
    x, y, scale = 0.5 + 0.345 * np.cos(0.4), 0.5 + 0.345 * np.sin(0.4), 0.05
    for j in (0, 1):
        p = band.normal_profile(j, x, y, scale)
        width = (1.0 / s) / scale
        assert 0.0 in p.offsets and -width in p.offsets
        np.testing.assert_array_equal(p.stops, p.origin + p.offsets)
        np.testing.assert_allclose(p.origin, (0.35 - 0.345) / scale, rtol=1e-13)
        flanks = [-EDGE_STOP * delta / scale, EDGE_STOP * delta / scale]
        assert p.offsets.size == (6 if delta else 2)
        if delta:
            np.testing.assert_allclose(sorted(p.offsets)[-1], flanks[1], rtol=1e-15)
        else:
            gap = int(np.flatnonzero(p.offsets == 0.0)[0])
            assert p.pieces[gap] is band.inside
            assert p.pieces[0] is p.pieces[-1] is band.outside


# --- the resistance composition -------------------------------------------------------


def logistic(z):
    return 0.5 * (1.0 + np.tanh(z))


def test_layer_share_is_the_difference_of_two_edges_without_cancellation():
    a = np.linspace(-6.0, 6.0, 49)
    for width in (0.3, 2.0, 7.0):
        share, s1, s2 = layer_share(a, a - width, width)
        np.testing.assert_allclose(share, logistic(a) - logistic(a - width), atol=3e-16)
        np.testing.assert_allclose(s1, 0.5 / np.cosh(a) ** 2, rtol=1e-14)
        np.testing.assert_allclose(s2, 0.5 / np.cosh(a - width) ** 2, rtol=1e-14)
        floats = [layer_share_at(float(v), float(v) - width, width) for v in a]
        np.testing.assert_allclose(floats, share, rtol=1e-15, atol=1e-300)
    # EABE eq. 40 at s = 10¹¹ and δ = 1e-3: a width of 1e-8 edge widths, where
    # the tanh difference keeps no digit; the share is width · s′ to first order.
    width = 1e-8
    share, s1, _ = layer_share(a, a - width, width)
    np.testing.assert_allclose(share, width * s1, rtol=1e-7)


@pytest.mark.parametrize("s", (1e3, 1e11))
@pytest.mark.parametrize("delta", (1e-3, 2.5e-4))
def test_the_resistance_composition_keeps_the_rings_contact_resistance(s, delta):
    # The decision on #39: ∫ (1/α − 1) dr across the smooth ring is the jump's
    # w (1/α_in − 1) at every (s, δ); the fold loses it once δ ≳ w/10.
    band = constant_ring(s, delta)
    outer = band.upper.radius

    def excess(r):
        return 1.0 / band.alpha(np.array([0.5 + r]), np.array([0.5]))[0] - 1.0

    reach = RING_REACH * delta
    points = sorted({outer - 1.0 / s, outer})
    got = quad(excess, outer - reach, outer + reach, points=points, limit=400)[0]
    want = (1.0 / s) * (1.5 * s - 1.0)
    assert got == pytest.approx(want, rel=1e-9)
    fold = SmoothBand(band.band, delta)

    def folded(r):
        return 1.0 / fold.alpha(np.array([0.5 + r]), np.array([0.5]))[0] - 1.0

    lost = quad(folded, outer - reach, outer + reach, points=points, limit=400)[0]
    assert lost < 0.05 * want


@pytest.mark.parametrize("gap", (True, False))
def test_the_resistance_blend_agrees_on_every_path(gap):
    # ``alpha``, ``alpha_at`` (with and without a known distance) and
    # ``alpha_given`` are one function; ``gradient`` is its derivative.
    inside = SineProduct(1.0 / 1500.0, 1.0 / 3000.0)
    band = replace(case3().material, inside=inside)
    if not gap:
        band = replace(band, gap=None)
    m = SmoothBand(band, 4e-4, "resistance")
    theta = np.linspace(0.1, 6.0, 9)
    r = 0.3495 + np.linspace(-0.004, 0.004, 9)
    x, y = 0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)
    alpha = m.alpha(x, y)
    d1, d2 = band.lower.signed_distance(x, y), band.upper.signed_distance(x, y)
    if gap:
        d1 = d2 + band.gap
    np.testing.assert_allclose(m.alpha_given(x, y, (d1, d2)), alpha, rtol=1e-15)
    for k in range(x.size):
        xk, yk = float(x[k]), float(y[k])
        assert m.alpha_at(xk, yk) == pytest.approx(alpha[k], rel=1e-14)
        for j, d in ((0, d1[k]), (1, d2[k])):
            assert m.alpha_at(xk, yk, (j, float(d))) == pytest.approx(
                alpha[k], rel=1e-12
            )
    gx, gy = m.gradient(x, y)
    step = 1e-8
    fx = (m.alpha(x + step, y) - m.alpha(x - step, y)) / (2 * step)
    fy = (m.alpha(x, y + step) - m.alpha(x, y - step)) / (2 * step)
    scale = np.abs(np.concatenate([fx, fy])).max()
    np.testing.assert_allclose(gx, fx, atol=1e-6 * scale)
    np.testing.assert_allclose(gy, fy, atol=1e-6 * scale)


def test_composition_is_checked_and_kept_by_with_smooth_edges():
    with pytest.raises(ValueError, match="composition"):
        SmoothBand(case3().material, 1e-3, "harmonic")
    domain = with_smooth_edges(case3(), 1e-3, "resistance")
    assert domain.material.composition == "resistance"
    assert domain.material.gap == 1e-3
    assert with_smooth_edges(domain, 2e-3).material.composition == "fold"


# --- the seeds across the ring --------------------------------------------------------


@pytest.mark.parametrize("s", (1e3, 1e11))
def test_the_seeds_are_exact_on_the_matched_profile_at_every_s(s):
    # H11: with the gap the weights reproduce ∇·(α∇u) = 4 through the ring to
    # rounding at every s; on the stored radii the stops' rounding in stencil
    # units grows like s (2500 nodes: 8e-10 at 10¹¹), and E2.3 (E2.9) like s.
    nodes, rows = ring_stencils(s)
    r = np.hypot(nodes.x - 0.5, nodes.y - 0.5)
    band, stored = constant_ring(s), constant_ring(s, gap=False)
    u, u_stored = matched_radial(band, r), matched_radial(stored, r)
    worst = {"gap": 0.0, "stored": 0.0, "E2.3": 0.0}
    for idx in rows:
        xy = nodes.xy[idx]
        for name, medium, profile in (("gap", band, u), ("stored", stored, u_stored)):
            w = weights_of(seed_basis(xy, medium, tangential=True))
            rel = abs(w @ profile[idx] - 4.0) / (np.abs(w) @ np.abs(profile[idx]))
            worst[name] = max(worst[name], rel)
        w = stencil_weights(xy, stored, 4)
        rel = abs(w @ u_stored[idx] - 4.0) / (np.abs(w) @ np.abs(u_stored[idx]))
        worst["E2.3"] = max(worst["E2.3"], rel)
    assert worst["gap"] < 1e-14
    if s == 1e11:
        assert worst["stored"] > 1e-11 and worst["E2.3"] > 1e-8


def test_the_march_from_the_outer_circle_is_the_absolute_one_at_case_3():
    # At s = 10³ the stored radii carry the width to rounding, so marching in
    # the offset from the outer crossing changes the seeds at rounding only.
    nodes, rows = ring_stencils(1e3, count=6)
    band = case3().material
    for idx in rows:
        xy = nodes.xy[idx]
        a = seed_basis(xy, band, tangential=True).block
        b = seed_basis(xy, replace(band, gap=None), tangential=True, flux=True).block
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-11 * np.abs(b).max())


def test_the_seed_block_does_not_condition_like_s():
    # §3.6 corrected: s w = 1 on eq. 40, so φ₀₁'s climb across the ring is
    # 1.5 α_e / h_s at every s and the seed block's condition number is flat.
    nodes, rows = ring_stencils(1e3, count=6)
    conds = {}
    for s in (1e3, 1e11):
        band = constant_ring(s)
        conds[s] = np.array(
            [
                np.linalg.cond(seed_basis(nodes.xy[i], band, tangential=True).block)
                for i in rows
            ]
        )
    ratio = conds[1e11] / conds[1e3]
    assert np.all((ratio > 0.3) & (ratio < 3.0))


def test_saddle_system_is_the_matrix_the_weights_solve():
    # Its blocks are the Gaussians' (unit diagonal, symmetric) and the seeds',
    # and the weights meet the moment conditions on the same seed block.
    nodes, rows = ring_stencils(1e3, count=2)
    xy = nodes.xy[rows[0]]
    sb = seed_basis(xy, case3().material, tangential=True)
    k, q = sb.block.shape
    for warp in (True, False):
        m = saddle_system(sb, warp=warp)
        assert m.shape == (k + q, k + q)
        np.testing.assert_array_equal(m[:k, k:], sb.block)
        np.testing.assert_array_equal(m[k:, :k], sb.block.T)
        np.testing.assert_array_equal(m[k:, k:], 0.0)
        np.testing.assert_allclose(m[:k, :k], m[:k, :k].T, rtol=0, atol=0)
        np.testing.assert_array_equal(np.diag(m[:k, :k]), 1.0)
        w = weights_of(sb, warp=warp) * sb.scale**2
        np.testing.assert_allclose(sb.block.T @ w, sb.rhs, atol=1e-9 * np.abs(w).sum())


def test_the_rings_series_are_read_from_their_interpolant_and_cost_little(monkeypatch):
    # RING_POINTS: across the gap at s = 10¹¹ (δ = 0) and through the smooth
    # ring (δ = 0.001), a row marches in a few thousand rate evaluations, where
    # the sampled series took up to 150,000; the interpolant matches the
    # sampled series to RING_TOL.
    calls = {"n": 0}
    real = seeds.solve_ivp

    def counted(*args, **kwargs):
        sol = real(*args, **kwargs)
        calls["n"] += sol.nfev
        return sol

    monkeypatch.setattr(seeds, "solve_ivp", counted)
    for s, delta, n in ((1e11, 0.0, 1250), (1e3, 1e-3, 2500)):
        domain = case3(s)
        material = (
            SmoothBand(domain.material, delta, "resistance")
            if delta
            else domain.material
        )
        nodes = build_node_set(domain, n, seed=0, iterations=20)
        index, _ = knn(nodes.xy, 30)
        rows = index[interface_crossings(nodes, domain.material, index)][::40]
        for idx in rows:
            calls["n"] = 0
            seed_basis(nodes.xy[idx], material, tangential=True)
            assert calls["n"] < 5000


def test_the_ring_interpolant_matches_the_series():
    def coefficients(eta):
        v = np.array([np.exp(eta), np.sin(3 * eta), 1e-12 * np.cos(eta)])
        return v, 2 * v, 1500.0 * v

    model = seeds._ring_coefficients(coefficients, -0.4, 0.3)
    for eta in np.linspace(-0.4, 0.3, 23):
        for got, want in zip(model(eta), coefficients(eta), strict=True):
            np.testing.assert_allclose(
                got, want, rtol=0, atol=1e-12 * np.abs(want).max()
            )


def test_the_far_edge_of_a_ring_is_its_foot_distance_shifted_by_the_gap():
    band = SmoothBand(case3(1e11).material, 1e-3, "resistance")
    nodes, rows = ring_stencils(1e11, count=2)
    xy = nodes.xy[rows[0]]
    sb = seed_basis(xy, band, tangential=True)
    coords = sb.coordinates
    far = seeds._other_edge(band, coords, sb.eta)
    shift = -1e-11 if coords.interface == 0 else 1e-11
    for eta in (-0.7, 0.0, 0.4):
        want = coords.d_e + coords.scale * eta + shift
        np.testing.assert_array_equal(far(eta), np.full(coords.px.size, want))


# --- the radial references ------------------------------------------------------------


def test_ring_mode_climbs_across_the_exact_width():
    # E2.9's RingMode climbs across the stored radii's width, 8e-8 off 1/s at
    # s = 10¹¹; with ``widths`` it climbs across the gap: 1.5 times the flux to
    # O(w), the ring's resistance.
    s = 1e11
    inner, outer = ring_radii(s)
    for mode, bound in ((ring_exact(s=s, gap=True), 1e-9), (ring_exact(s=s), None)):
        below = mode.radial(np.array([np.nextafter(inner, 0.0)]))[0]
        climb = mode.radial(np.array([outer]))[0] - below
        off = abs(climb / (1.5 * mode.flux(np.array([outer]))[0]) - 1.0)
        if bound:
            assert off < bound
        else:
            assert off > 1e-8
    case3_mode = ring_exact(s=1e3, gap=True)
    np.testing.assert_allclose(
        case3_mode.radial(np.array([0.5])),
        ring_exact().radial(np.array([0.5])),
        rtol=1e-14,
    )
    with pytest.raises(ValueError, match="width"):
        RingMode((0.3, 0.35), (1.0, 0.2, 1.0), widths=(0.05, 0.01))


@pytest.mark.parametrize("s", (1e3, 1e11))
@pytest.mark.parametrize("delta", (1e-3, 1e-5))
def test_the_smooth_ring_mode_solves_the_radial_equation(s, delta):
    band = constant_ring(s, delta)
    mode = SmoothRingMode(band)
    # Against itself at a looser tolerance, across the ring.
    r = np.linspace(0.3, 0.4, 1001)
    np.testing.assert_allclose(
        SmoothRingMode(band, rtol=1e-12).radial(r), mode.radial(r), atol=5e-12
    )
    # Far from it, the resistance is the jump's to first order in δ and the
    # mode the jump's to O(δ).
    far = np.array([0.2, 0.5, 0.6])
    jump = ring_exact(s=s, gap=True)
    np.testing.assert_allclose(mode.radial(far), jump.radial(far), atol=20 * delta)
    assert mode(np.array([0.5]), np.array([0.5 + 0.5]))[0] == pytest.approx(
        -mode.radial(np.array([0.5]))[0]
    )


def test_the_smooth_ring_mode_without_a_ring_is_r_to_the_m():
    band = Band(Circle(0.3), Circle(0.35), Constant2D(1.0), Constant2D(1.0), gap=0.05)
    mode = SmoothRingMode(SmoothBand(band, 1e-3, "resistance"))
    r = np.linspace(0.05, 0.6, 111)
    np.testing.assert_allclose(mode.radial(r), (r / 0.5) ** 2, rtol=1e-14)


@pytest.mark.parametrize("s", (1e3, 1e11))
def test_the_matched_profile_is_the_quadrature_of_2r_over_alpha(s):
    for delta in (0.0, 1e-3, 2.5e-4):
        band = constant_ring(s, delta)
        r = np.linspace(0.1, 0.6, 501)
        u = matched_radial(band, r)
        np.testing.assert_allclose(u[r < 0.3], r[r < 0.3] ** 2, rtol=1e-15)
        # Resistance-preserving: beyond the ring it is the jump's at every δ.
        jump = matched_radial(constant_ring(s), r)
        np.testing.assert_allclose(u[r > 0.4], jump[r > 0.4], rtol=1e-13)
        if delta:
            mid = 0.5 * (r[1:] + r[:-1])
            alpha = band.alpha(0.5 + mid, np.full_like(mid, 0.5))
            slope = np.diff(u) / np.diff(r)
            smooth = np.abs(mid - 0.35) > 40 * delta
            np.testing.assert_allclose(
                slope[smooth], 2 * mid[smooth] / alpha[smooth], rtol=1e-5
            )


def test_radial_elements_cut_both_circles_within_reach():
    band = constant_ring(1e3, 2.5e-4)
    edges = radial_elements(band)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == pytest.approx(0.349 - RING_REACH * 2.5e-4)
    assert edges[-1] == pytest.approx(0.35 + RING_REACH * 2.5e-4)
    for centre in (0.349, 0.35):
        assert np.abs(edges - centre).min() < 1e-12


# --- the flux seeds (§3.11) ---------------------------------------------------------


def test_the_flux_seeds_are_the_degree_five_seeds_that_carry_a_flux():
    exponents = flux_exponents(4)
    assert len(exponents) == 20
    assert exponents[:15] == [(a, b) for a, b in exponents if a + b <= 4]
    assert exponents[15:] == [(4, 1), (3, 2), (2, 3), (1, 4), (0, 5)]


def test_the_flux_seeds_are_on_by_default_on_a_ring_only():
    nodes, rows = ring_stencils(1e3, count=2)
    xy = nodes.xy[rows[0]]
    band = case3().material
    assert seed_basis(xy, band, tangential=True).block.shape == (30, 20)
    assert seed_basis(xy, band, tangential=True, flux=False).block.shape == (30, 15)
    no_gap = replace(band, gap=None)
    assert seed_basis(xy, no_gap, tangential=True).block.shape == (30, 15)
    with pytest.raises(NotImplementedError, match="tangential"):
        seed_basis(xy, band, flux=True)
    # The block's first 15 columns are the degree-5 chain's seeds of degree 4,
    # and the moment conditions are 2 α_e on the two quadratics alone.
    sb = seed_basis(xy, band, tangential=True)
    full = seed_basis(xy, band, degree=5, tangential=True, flux=False)
    np.testing.assert_array_equal(sb.block[:, :15], full.block[:, :15])
    want = np.zeros(20)
    want[[3, 5]] = 2.0 * sb.alpha_e
    np.testing.assert_array_equal(sb.rhs, want)


def test_the_flux_seeds_fit_the_mode_across_the_ring_one_order_better():
    # §3.11: on the rows with most of their nodes across the ring the far side
    # carries R·F(σ), the flux at the ring times its contact resistance, with
    # no factor of the distance; the degree-4 seeds fit the mode there at
    # O(h⁴) and with the flux seeds at O(h⁵) (per halving of h: 4 against 5.7
    # per doubling of the count).
    domain = case3()
    band = replace(domain.material, inside=Constant2D(1.0 / 1500.0))
    mode = ring_exact(gap=True)
    fits = {}
    for n in (10000, 20000):
        nodes = build_node_set(domain, n, seed=0, iterations=20)
        index, _ = knn(nodes.xy, 30)
        rows = index[interface_crossings(nodes, band, index)]
        worst = {15: [], 20: []}
        for idx in rows[::6]:
            xy = nodes.xy[idx]
            r = np.hypot(xy[:, 0] - 0.5, xy[:, 1] - 0.5)
            far = (r > 0.3495) != (r[0] > 0.3495)
            if far.sum() < 10:
                continue
            u = mode(xy[:, 0], xy[:, 1])
            for q, flux in ((15, False), (20, True)):
                block = seed_basis(xy, band, tangential=True, flux=flux).block
                c, *_ = np.linalg.lstsq(block, u, rcond=None)
                worst[q].append(np.sqrt(np.mean((u - block @ c)[far] ** 2)))
        fits[n] = {q: np.sqrt(np.mean(np.square(v))) for q, v in worst.items()}
    assert fits[10000][15] / fits[20000][15] < 4.6
    assert fits[10000][20] / fits[20000][20] > 5.0
    assert fits[20000][20] < 0.1 * fits[20000][15]


def test_on_a_ring_the_gaussians_are_shaped_on_the_warped_spacing():
    # §3.11: with the anchor inside the smooth ring the warp compresses its
    # neighbours by α_e/α, so ε comes from the nearest node in (ξ, φ₀₁); off a
    # ring, and plain, it is E2.2's physical rule.
    domain = case3()
    band = SmoothBand(domain.material, 2.5e-3, "resistance")
    nodes = build_node_set(domain, 5000, seed=0, iterations=20)
    index, _ = knn(nodes.xy, 30)
    r = np.hypot(nodes.x - 0.5, nodes.y - 0.5)
    inside = np.argmin(np.abs(r - 0.3495))
    sb = seed_basis(nodes.xy[index[inside]], band, tangential=True)
    assert sb.alpha_e < 0.5
    xi, eta, eps = seeds._gaussian_coordinates(sb, 0.4, True)
    d = np.hypot(xi[1:] - xi[0], eta[1:] - eta[0])
    assert eps == pytest.approx(0.4 / d.min())
    xy = nodes.xy[index[inside]]
    physical = np.hypot(xy[1:, 0] - xy[0, 0], xy[1:, 1] - xy[0, 1]).min()
    _, _, plain = seeds._gaussian_coordinates(sb, 0.4, False)
    assert plain == pytest.approx(0.4 * sb.scale / physical)
    assert eps > 2.0 * plain
