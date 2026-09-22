import numpy as np
import pytest

from heat_interfaces.heat1d.domain import (
    TANH_REACH,
    Constant,
    PiecewiseAlpha,
    SmoothEdges,
)
from heat_interfaces.heat1d.stiff import EDGE_STOP, _stops
from heat_interfaces.heat2d.domain import (
    DIRICHLET,
    FREE,
    ROW,
    ROW_OFFSETS,
    Band,
    Circle,
    Constant2D,
    FlatLine,
    SineGraph,
    SineProduct,
    SmoothBand,
    build_node_set,
    case1,
    case2,
    case3,
    ring_radii,
    row_count,
    step_delta,
    straddle_count,
    with_smooth_edges,
)
from heat_interfaces.heat2d.neighbors import nearest_spacing, periodic_dx

CURVES = [FlatLine(0.6), SineGraph(0.6), SineGraph(0.8), Circle(0.3495), Circle(0.05)]
S = np.linspace(0.0, 1.0, 13, endpoint=False) + 0.013


# --- curves -----------------------------------------------------------------


def test_flat_line_geometry():
    c = FlatLine(0.6)
    x, y = np.array([0.1, 1.3, -0.2]), np.array([0.7, 0.6, 0.5])
    np.testing.assert_allclose(c.level(x, y), [0.1, 0.0, -0.1])
    np.testing.assert_allclose(c.signed_distance(x, y), [0.1, 0.0, -0.1])
    np.testing.assert_allclose(c.closest(x, y), [0.1, 0.3, 0.8])
    px, py = c.point([0.25, 1.25])
    np.testing.assert_allclose(px, [0.25, 0.25])
    np.testing.assert_allclose(py, 0.6)
    nx, ny = c.normal(S)
    np.testing.assert_allclose(nx, 0.0)
    np.testing.assert_allclose(ny, 1.0)
    np.testing.assert_allclose(c.curvature(S), 0.0)
    assert c.length == 1.0


def test_circle_geometry():
    c = Circle(0.3495)
    assert c.length == pytest.approx(2 * np.pi * 0.3495)
    np.testing.assert_allclose(c.level(0.5 + 0.3495, 0.5), 0.0, atol=1e-16)
    assert c.level(0.5, 0.9) == pytest.approx(0.0505)
    assert c.signed_distance(0.5, 0.2) == pytest.approx(-0.0495)
    assert c.closest(0.5, 0.9) == pytest.approx(0.25)
    assert c.closest(0.2, 0.5) == pytest.approx(0.5)
    assert c.closest(0.5, 0.2) == pytest.approx(0.75)
    px, py = c.point(c.closest(0.9, 0.9))
    assert np.hypot(px - 0.5, py - 0.5) == pytest.approx(0.3495)
    assert (px - 0.5) == pytest.approx(py - 0.5)
    np.testing.assert_allclose(c.curvature(S), -1 / 0.3495)
    nx, ny = c.normal(0.125)
    assert nx == pytest.approx(np.cos(np.pi / 4)) and ny == pytest.approx(nx)
    assert c.tangent_angle(0.0) == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("curve", CURVES)
def test_normal_is_perpendicular_to_the_tangent_and_unit(curve):
    nx, ny = curve.normal(S)
    theta = curve.tangent_angle(S)
    np.testing.assert_allclose(nx * np.cos(theta) + ny * np.sin(theta), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.hypot(nx, ny), 1.0)


@pytest.mark.parametrize("curve", CURVES)
def test_curvature_sign_and_size_from_the_curve_itself(curve):
    # To leading order n(s) · (p(s + ds) - p(s)) = (kappa / 2) |p(s + ds) - p(s)|^2.
    ds = 1e-4
    px, py = curve.point(S)
    qx, qy = curve.point(S + ds)
    dx, dy = periodic_dx(qx - px), qy - py
    nx, ny = curve.normal(S)
    measured = 2 * (nx * dx + ny * dy) / (dx**2 + dy**2)
    np.testing.assert_allclose(measured, curve.curvature(S), atol=2e-3)


def test_sine_graph_derivatives_and_curvature():
    g = SineGraph(0.6)
    x = np.array([0.0, 0.125, 0.25, 0.75])
    np.testing.assert_allclose(g.height(x), 0.6 + 0.02 * np.sin(2 * np.pi * x))
    np.testing.assert_allclose(g.slope(x), 0.04 * np.pi * np.cos(2 * np.pi * x))
    eps = 1e-6
    np.testing.assert_allclose(
        g.second(x), (g.slope(x + eps) - g.slope(x - eps)) / (2 * eps), rtol=1e-6
    )
    assert g.curvature(0.25) == pytest.approx(-0.02 * 4 * np.pi**2)
    assert g.curvature(0.0) == pytest.approx(0.0, abs=1e-15)
    assert g.tangent_angle(0.0) == pytest.approx(np.arctan(0.04 * np.pi))


def test_sine_graph_closest_point_minimises_the_distance_and_wraps():
    g = SineGraph(0.8)
    x = np.array([0.02, 0.3, 0.98, 0.51, 0.77])
    y = np.array([0.95, 0.6, 0.1, 0.81, 0.79])
    s = g.closest(x, y)
    assert np.all((s >= 0) & (s < 1))

    def dist(t):
        px, py = g.point(t)
        return np.hypot(periodic_dx(x - px), y - py)

    d0 = dist(s)
    for ds in (1e-3, -1e-3, 1e-5, -1e-5):
        assert np.all(dist(s + ds) >= d0 - 1e-14)
    sd = g.signed_distance(x, y)
    np.testing.assert_allclose(np.abs(sd), d0, rtol=1e-12)
    np.testing.assert_array_equal(np.sign(sd), np.sign(g.level(x, y)))
    assert np.all(np.abs(sd) <= np.abs(g.level(x, y)) + 1e-15)
    # A point displaced along the normal at s = 0.003 lands across the seam
    # (x < 0 wrapped to 0.99...), and its foot is found back across it.
    px, py = g.point(0.003)
    nx, ny = g.normal(0.003)
    qx, qy = px + 0.05 * nx, py + 0.05 * ny
    assert qx < 0.0
    qx = qx % 1.0
    assert g.closest(qx, qy) == pytest.approx(0.003, abs=1e-12)
    assert g.signed_distance(qx, qy) == pytest.approx(0.05, rel=1e-12)


# --- materials --------------------------------------------------------------


def test_sine_product_value_gradient_and_taylor_table():
    p = SineProduct(0.2, 0.1)
    x0, y0 = 0.31, 0.72
    assert p.alpha(x0, y0) == pytest.approx(
        0.2 + 0.1 * np.sin(2 * np.pi * x0) * np.sin(2 * np.pi * y0)
    )
    eps = 1e-6
    gx, gy = p.gradient(x0, y0)
    fd_x = (p.alpha(x0 + eps, y0) - p.alpha(x0 - eps, y0)) / (2 * eps)
    fd_y = (p.alpha(x0, y0 + eps) - p.alpha(x0, y0 - eps)) / (2 * eps)
    assert gx == pytest.approx(fd_x, rel=1e-6) and gy == pytest.approx(fd_y, rel=1e-6)
    a = p.taylor(x0, y0, 5)
    assert a.shape == (6, 6)
    i, j = np.indices(a.shape)
    assert np.all(a[i + j > 5] == 0.0)
    assert a[0, 0] == pytest.approx(p.alpha(x0, y0))
    assert a[1, 0] == pytest.approx(gx) and a[0, 1] == pytest.approx(gy)
    dx, dy = 0.01, -0.012
    series = sum(a[i, j] * dx**i * dy**j for i in range(6) for j in range(6))
    assert series == pytest.approx(p.alpha(x0 + dx, y0 + dy), abs=1e-9)


def test_constant_piece():
    c = Constant2D(0.2)
    np.testing.assert_allclose(c.alpha(np.zeros(3), np.ones(3)), 0.2)
    gx, gy = c.gradient(np.zeros(3), np.ones(3))
    assert not gx.any() and not gy.any()
    a = c.taylor(0.3, 0.4, 4)
    assert a[0, 0] == 0.2 and np.count_nonzero(a) == 1


def test_case1_band_is_closed_at_both_ends_by_exact_comparison():
    m = case1().material
    x = np.full(6, 0.37)
    y = np.array([0.6, 0.8, np.nextafter(0.6, 0.0), np.nextafter(0.8, 1.0), 0.7, 0.1])
    np.testing.assert_array_equal(m.piece_index(x, y), [1, 1, 0, 0, 1, 0])
    np.testing.assert_allclose(m.alpha(x, y), [0.2, 0.2, 1.0, 1.0, 0.2, 1.0])
    assert m.pieces == (m.outside, m.inside) and m.interfaces == (m.lower, m.upper)
    a = m.taylor("inside", 0.0, 0.6, 3)
    assert a[0, 0] == 0.2


def test_case2_band_follows_the_sine_interfaces():
    m = case2().material
    x = np.array([0.25, 0.25, 0.75])
    y = np.array([0.6 + 0.02, 0.6 + 0.02 - 1e-9, 0.8 - 0.02 + 1e-9])
    np.testing.assert_array_equal(m.piece_index(x, y), [1, 0, 0])
    assert m.alpha(0.25, 0.7) == pytest.approx(0.2 + 0.1 * np.sin(1.4 * np.pi))
    gx, gy = m.gradient(np.array([0.25, 0.25]), np.array([0.7, 0.3]))
    assert gx[0] == pytest.approx(0.0, abs=1e-12) and gx[1] == 0.0
    assert gy[0] == pytest.approx(0.2 * np.pi * np.cos(1.4 * np.pi)) and gy[1] == 0.0


def test_case3_ring_owns_both_of_its_edges():
    m = case3().material
    x = np.array([0.5 + 0.349, 0.85, 0.5, 0.5, 0.5])
    y = np.array([0.5, 0.5, 0.5 + 0.3495, 0.9, 0.5 + 0.349 - 1e-12])
    np.testing.assert_array_equal(m.piece_index(x, y), [1, 1, 1, 0, 0])
    alpha = m.alpha(x, y)
    assert alpha[3] == 1.0 and alpha[4] == 1.0
    assert alpha[2] == pytest.approx(1 / 1500, abs=1e-6)
    expected = 1 / 1500 + np.sin(1.7 * np.pi) * np.sin(np.pi) / 3000
    assert alpha[1] == pytest.approx(expected)


def test_case3_domain_cuts_out_the_cooling_unit():
    d = case3()
    x = np.array([0.5, 0.5, 0.54, 0.56, 0.5, 0.5])
    y = np.array([0.5, 0.549, 0.5, 0.5, 0.0, 1.0])
    np.testing.assert_array_equal(d.contains(x, y), [0, 0, 0, 1, 0, 0])
    assert d.straddle[0].radius == pytest.approx(0.3495)
    assert d.dirichlet[2].radius == 0.05 and d.holes == (d.dirichlet[2],)
    assert case1().contains(np.array([0.5]), np.array([0.5]))[0]


def test_case3_at_s_is_eq_40_and_recovers_case_3_at_1000():
    # E2.9: EABE eq. 40's ring 0.35 − 1/s ≤ r ≤ 0.35 at
    # α = 1/(1.5 s) + (1/(3 s)) sin sin, rows on its midline; s = 1000 is case 3
    # bit for bit.
    assert ring_radii() == (0.349, 0.35) and ring_radii(1000.0) == (0.349, 0.35)
    d = case3(1000.0)
    assert d.material == case3().material and d.straddle == case3().straddle
    e = case3(1e8)
    assert e.material.lower.radius == 0.35 - 1e-8 and e.material.upper.radius == 0.35
    assert e.straddle[0].radius == 0.35 - 0.5e-8
    assert e.dirichlet == d.dirichlet and e.holes == d.holes
    m = e.material
    x = 0.5 + np.array([0.35 - 0.5e-8, 0.35 - 2e-8, 0.35 + 1e-8])
    y = np.full(3, 0.5)
    np.testing.assert_array_equal(m.region_index(x, y), [1, 0, 2])
    assert m.inside.offset == pytest.approx(1 / 1.5e8)
    assert m.inside.amplitude == pytest.approx(1 / 3e8)
    np.testing.assert_array_equal(m.alpha(x, y)[1:], 1.0)
    with pytest.raises(ValueError, match="positive"):
        ring_radii(0.0)
    with pytest.raises(ValueError, match="cooling circle"):
        ring_radii(3.0)  # 0.35 - 1/3 < 0.05
    assert ring_radii(4.0) == (0.35 - 0.25, 0.35)


def test_case3_rows_straddle_a_ring_a_billionth_wide():
    d = case3(1e9)
    ns = build_node_set(d, 1250, iterations=10)
    r = np.hypot(ns.x - 0.5, ns.y - 0.5)
    assert (d.material.region_index(ns.x, ns.y) == 1).sum() == 0
    lo, hi = ns.rows_of(0, 0.5)
    assert np.all(r[lo.index] < 0.35 - 1e-9) and np.all(r[hi.index] > 0.35)
    np.testing.assert_allclose(
        (r[lo.index] + r[hi.index]) / 2, 0.35 - 0.5e-9, rtol=1e-12
    )


# --- node sets --------------------------------------------------------------


def test_row_counts_and_step_schedule_follow_the_matlab():
    assert straddle_count(2500) == 48 and straddle_count(1250) == 34
    assert straddle_count(900) == 29  # 28.5 rounds up, as MATLAB's round does
    assert row_count(FlatLine(0.0), 1 / 48) == 48
    assert row_count(Circle(0.3495), 1 / 48) == 105
    assert row_count(Circle(0.05), 1 / 48) == 15
    assert step_delta(2500, 1) == pytest.approx(0.05)
    assert step_delta(10000, 4) == pytest.approx(0.05 * 0.5 / 4)


@pytest.fixture(scope="module")
def sets():
    return {
        1: build_node_set(case1(), 2500),
        2: build_node_set(case2(), 2500),
        3: build_node_set(case3(), 2500),
    }


def test_case1_counts_kinds_and_dirichlet_rows(sets):
    ns = sets[1]
    assert ns.n == 2500 and ns.h == pytest.approx(1 / 48)
    assert int((ns.kind == ROW).sum()) == 12 * 48
    assert int((ns.kind == DIRICHLET).sum()) == 96
    assert int((ns.kind == FREE).sum()) == 2500 - 12 * 48 - 96
    assert np.all((ns.x >= 0) & (ns.x < 1))
    assert ns.xy.shape == (2500, 2) and ns.free.sum() == (ns.kind == FREE).sum()
    bottom, top = ns.dirichlet_rows
    assert np.all(ns.y[bottom.index] == 0.0) and np.all(ns.y[top.index] == 1.0)
    np.testing.assert_array_equal(ns.x[bottom.index], np.arange(48) / 48)
    np.testing.assert_array_equal(ns.boundary_index[bottom.index], 0)
    np.testing.assert_array_equal(ns.boundary_index[top.index], 1)
    assert np.all(ns.boundary_index[~ns.dirichlet] == -1)
    assert np.all(ns.dirichlet == (ns.kind == DIRICHLET))


@pytest.mark.parametrize("case", [1, 2, 3])
def test_every_straddling_pair_sits_orthogonally_across_its_curve(sets, case):
    ns = sets[case]
    domain = {1: case1, 2: case2, 3: case3}[case]()
    for ci, curve in enumerate(domain.straddle):
        for k, offset in enumerate(ROW_OFFSETS):
            lo, hi = ns.rows_of(ci, offset)
            assert lo.h == hi.h and len(lo.index) == len(hi.index)
            dx = periodic_dx(ns.x[hi.index] - ns.x[lo.index])
            dy = ns.y[hi.index] - ns.y[lo.index]
            np.testing.assert_allclose(np.hypot(dx, dy), 2 * offset * lo.h, rtol=1e-12)
            mx, my = ns.x[lo.index] + dx / 2, ns.y[lo.index] + dy / 2
            np.testing.assert_allclose(curve.level(mx, my), 0.0, atol=1e-13)
            s = curve.closest(mx, my)
            nx, ny = curve.normal(s)
            np.testing.assert_allclose(dx * ny - dy * nx, 0.0, atol=1e-13)
            for row, sign in ((lo, -1), (hi, 1)):
                sd = curve.signed_distance(ns.x[row.index], ns.y[row.index])
                np.testing.assert_allclose(sd, sign * offset * row.h, rtol=1e-10)
            if k == 1:  # the middle pair is staggered by half a spacing along the curve
                inner = ns.rows_of(ci, ROW_OFFSETS[0])[0]
                s_inner = curve.closest(ns.x[inner.index], ns.y[inner.index])
                shift = np.mod(s - s_inner + 0.5, 1.0) - 0.5
                np.testing.assert_allclose(shift, 0.5 / len(lo.index), atol=1e-12)


def test_case1_rows_fall_on_the_expected_sides_of_the_band(sets):
    ns, m = sets[1], case1().material
    piece = m.piece_index(ns.x, ns.y)
    lo, hi = ns.rows_of(0, 0.5)
    assert np.all(piece[lo.index] == 0) and np.all(piece[hi.index] == 1)
    lo, hi = ns.rows_of(1, 0.5)
    assert np.all(piece[lo.index] == 1) and np.all(piece[hi.index] == 0)
    np.testing.assert_allclose(ns.y[ns.rows_of(0, 0.5)[0].index], 0.6 - 0.5 / 48)
    # The rows of the two interfaces do not meet: 0.2 > 2 (0.5 + sqrt 3) h.
    assert np.all(ns.y[ns.rows_of(0, ROW_OFFSETS[2])[1].index] < 0.7)


def test_case3_pair_straddles_both_ring_interfaces_and_the_cooling_row(sets):
    ns, d = sets[3], case3()
    r = np.hypot(ns.x - 0.5, ns.y - 0.5)
    lo, hi = ns.rows_of(0, 0.5)
    assert np.all(r[lo.index] < 0.349) and np.all(r[hi.index] > 0.35)
    np.testing.assert_allclose((r[lo.index] + r[hi.index]) / 2, 0.3495, rtol=1e-12)
    assert np.all(d.material.piece_index(ns.x, ns.y)[ns.kind == ROW] == 0)
    cooling = ns.dirichlet_rows[2]
    assert len(cooling.index) == 15
    np.testing.assert_allclose(r[cooling.index], 0.05, rtol=1e-12)
    np.testing.assert_array_equal(ns.boundary_index[cooling.index], 2)
    assert np.all(r[ns.free] > 0.05)
    assert np.all(d.contains(ns.x[ns.free], ns.y[ns.free]))
    assert len(ns.straddle_rows) == 6 and len(ns.dirichlet_rows) == 3


@pytest.mark.parametrize("case", [1, 2, 3])
def test_free_nodes_keep_clear_of_the_bands_and_are_quasi_uniform(sets, case):
    ns = sets[case]
    domain = {1: case1, 2: case2, 3: case3}[case]()
    fx, fy = ns.x[ns.free], ns.y[ns.free]
    assert np.all(domain.contains(fx, fy))
    for curve in domain.straddle:
        h_row = curve.length / row_count(curve, ns.h)
        assert np.all(np.abs(curve.level(fx, fy)) > ROW_OFFSETS[-1] * h_row)
    spacing = nearest_spacing(ns.xy) / ns.h
    assert spacing.min() > 0.7
    free = spacing[ns.free]
    assert 0.9 < np.median(free) < 1.05 and free.max() < 1.4


def test_node_sets_are_deterministic_in_the_seed():
    a = build_node_set(case1(), 400, seed=3, iterations=5)
    b = build_node_set(case1(), 400, seed=3, iterations=5)
    c = build_node_set(case1(), 400, seed=4, iterations=5)
    np.testing.assert_array_equal(a.xy, b.xy)
    assert not np.array_equal(a.xy[a.free], c.xy[c.free])
    np.testing.assert_array_equal(a.xy[~a.free], c.xy[~c.free])


def test_fewer_rows_per_side_and_the_refusals():
    ns = build_node_set(case2(), 900, rows_per_side=1, iterations=3)
    assert len(ns.straddle_rows) == 4 and (ns.kind == ROW).sum() == 4 * 29
    with pytest.raises(ValueError, match="rows_per_side"):
        build_node_set(case1(), 900, rows_per_side=4)
    with pytest.raises(ValueError, match="leaves no free nodes"):
        build_node_set(case1(), 100)
    with pytest.raises(KeyError):
        ns.rows_of(0, ROW_OFFSETS[1])


def test_band_rejects_nothing_but_reports_its_pieces():
    b = Band(FlatLine(0.2), FlatLine(0.4), Constant2D(3.0), Constant2D(1.0))
    assert b.alpha(0.5, 0.3) == 3.0 and b.alpha(0.5, 0.5) == 1.0


# --- the smooth band (E4.2) -------------------------------------------------

DELTAS = (0.04, 0.01, 0.0025)
"""Three of the study's edge widths (stiff note §3.1): resolved to sub-grid."""


def logistic(d, delta):
    return 0.5 * (1.0 + np.tanh(d / delta))


def case1_in_y(delta):
    """E3.2's 1-D medium over ``1 | 0.2 | 1`` at 0.6, 0.8, built by hand."""
    jump = PiecewiseAlpha(
        (0.6, 0.8), (Constant(1.0), Constant(0.2), Constant(1.0)), ("right", "left")
    )
    return SmoothEdges(jump, delta)


def sample_points(n=4001):
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, n)
    y = rng.uniform(0.0, 1.0, n)
    # Points on both of case 1's and case 2's interfaces, and just off them.
    xs = np.linspace(0.0, 1.0, 9, endpoint=False)
    on = [(xs, np.full_like(xs, c)) for c in (0.6, 0.8)]
    on += [(xs, SineGraph(c).height(xs)) for c in (0.6, 0.8)]
    on += [(xs, np.nextafter(np.full_like(xs, c), 1.0)) for c in (0.6, 0.8)]
    for px, py in on:
        x, y = np.concatenate([x, px]), np.concatenate([y, py])
    return x, y


@pytest.mark.parametrize("case", [case1, case2, case3])
def test_smooth_band_at_delta_zero_is_the_band_bit_for_bit(case):
    band = case().material
    m = SmoothBand(band, 0.0)
    x, y = sample_points()
    assert np.array_equal(m.alpha(x, y), band.alpha(x, y))
    for got, want in zip(m.gradient(x, y), band.gradient(x, y), strict=True):
        assert np.array_equal(got, want)
    assert np.array_equal(m.piece_index(x, y), band.piece_index(x, y))
    assert np.array_equal(m.region_index(x, y), band.region_index(x, y))
    assert m.interfaces == band.interfaces and m.pieces == band.pieces
    assert (m.lower, m.upper, m.inside, m.outside) == (
        band.lower,
        band.upper,
        band.inside,
        band.outside,
    )
    for r in (0, 1, 2):
        assert m.region_piece(r) is band.region_piece(r)
    for side in ("inside", "outside"):
        assert np.array_equal(
            m.taylor(side, 0.3, 0.7, 4), band.taylor(side, 0.3, 0.7, 4)
        )


@pytest.mark.parametrize("delta", DELTAS)
def test_smooth_band_on_case1_is_the_1d_medium_in_y_bit_for_bit(delta):
    # Stiff note §3.1: the same edge_blend steps in the same order, with the
    # flat lines' signed distance y − c and normal (0, 1).
    m = SmoothBand(case1().material, delta)
    x, y = sample_points()
    one_d = case1_in_y(delta)
    gx, gy = m.gradient(x, y)
    assert np.array_equal(m.alpha(x, y), one_d.alpha(y))
    assert np.array_equal(gy, one_d.alpha_x(y))
    assert np.all(gx == 0.0)
    # The jump's protocol is untouched: the band is still closed at 0.6 and 0.8.
    edge = np.array([0.6, 0.8])
    assert np.array_equal(m.piece_index(np.full(2, 0.4), edge), [1, 1])


def test_a_smooth_flat_edge_is_the_tanh_blend_to_rounding():
    delta = 0.0025
    m = SmoothBand(case1().material, delta)
    y = np.linspace(0.55, 0.65, 4001)
    x = np.full_like(y, 0.37)
    s = logistic(y - 0.6, delta)
    np.testing.assert_allclose(m.alpha(x, y), 1.0 - 0.8 * s, rtol=1e-15, atol=1e-17)
    np.testing.assert_allclose(
        m.gradient(x, y)[1],
        -0.8 / (2 * delta) / np.cosh((y - 0.6) / delta) ** 2,
        rtol=1e-13,
        atol=1e-12,
    )
    assert float(m.alpha(0.37, 0.6)) == pytest.approx(0.6)


@pytest.mark.parametrize("case", [case1, case2, case3])
@pytest.mark.parametrize("delta", DELTAS[1:])  # at 0.04 the reach covers the strip
def test_smooth_band_tails_are_the_pieces_bit_for_bit_beyond_tanh_reach(case, delta):
    band = case().material
    m = SmoothBand(band, delta)
    rng = np.random.default_rng(7)
    x, y = rng.uniform(0.0, 1.0, (2, 20000))
    far = np.ones(x.size, dtype=bool)
    for curve in band.interfaces:
        far &= np.abs(curve.signed_distance(x, y)) >= TANH_REACH * delta
    assert far.sum() > 1000
    assert np.array_equal(m.alpha(x[far], y[far]), band.alpha(x[far], y[far]))


@pytest.mark.parametrize("case", [case1, case2, case3])
def test_smooth_band_gradient_is_the_derivative_of_its_alpha(case):
    # The edge is smooth to rounding: the analytic gradient (the pieces'
    # blended, plus s′ ∇d with ∇d the normal at the foot point) against
    # central differences, near the edges where it is steep.
    delta = 0.01
    band = case().material
    m = SmoothBand(band, delta)
    rng = np.random.default_rng(11)
    x, y = rng.uniform(0.02, 0.98, (2, 4000))
    near = np.zeros(x.size, dtype=bool)
    for curve in band.interfaces:
        near |= np.abs(curve.signed_distance(x, y)) < 5 * delta
    x, y = x[near], y[near]
    assert x.size > 100
    h = 1e-6
    fx = (m.alpha(x + h, y) - m.alpha(x - h, y)) / (2 * h)
    fy = (m.alpha(x, y + h) - m.alpha(x, y - h)) / (2 * h)
    gx, gy = m.gradient(x, y)
    scale = np.abs(band.inside.alpha(x, y) - band.outside.alpha(x, y)).max() / delta
    np.testing.assert_allclose(gx, fx, atol=1e-6 * scale)
    np.testing.assert_allclose(gy, fy, atol=1e-6 * scale)


def test_a_band_thinner_than_its_edges_gets_the_product_profile():
    # Stiff note §3.1: the fold gives o + (i − o) s(d₁/δ)(1 − s(d₂/δ)), whose
    # extreme is at the midline, s(w/2δ)² of the contrast. It tends to a
    # quarter as w/δ → 0, not to zero (§4.1: a ring thinner than its edges
    # keeps a bump of width δ; the difference of the two edges, s₁ − s₂,
    # would peak at tanh(w/2δ) instead).
    for ratio in (0.0, 0.5, 2.0):
        delta = 0.001
        lower, upper = 0.6, 0.6 + ratio * delta
        band = Band(FlatLine(lower), FlatLine(upper), Constant2D(1e-3), Constant2D(1.0))
        m = SmoothBand(band, delta)
        y = np.linspace(0.59, 0.61, 4001)
        x = np.full_like(y, 0.5)
        s1, s2 = logistic(y - lower, delta), logistic(y - upper, delta)
        expected = 1.0 + (1e-3 - 1.0) * s1 * (1.0 - s2)
        np.testing.assert_allclose(m.alpha(x, y), expected, rtol=1e-13)
        mid = float(m.alpha(0.5, 0.5 * (lower + upper)))
        peak = logistic(0.5 * ratio * delta, delta) ** 2
        assert mid == pytest.approx(1.0 + (1e-3 - 1.0) * peak, rel=1e-14)
    assert 0.77 < peak < 0.78  # w = 2δ reaches 78 % of the contrast
    zero_width = SmoothBand(
        Band(FlatLine(0.6), FlatLine(0.6), Constant2D(1e-3), Constant2D(1.0)), 1e-3
    )
    assert float(zero_width.alpha(0.5, 0.6)) == pytest.approx(1.0 - 0.999 / 4)


def test_the_ring_blends_in_the_radial_distance():
    delta = 2e-4
    m = SmoothBand(case3().material, delta)
    inner, outer = ring_radii()
    theta = 0.3
    rho = np.linspace(0.345, 0.355, 2001)
    x, y = 0.5 + rho * np.cos(theta), 0.5 + rho * np.sin(theta)
    r = np.hypot(x - 0.5, y - 0.5)  # the circles' own radius, to the bit
    ring = case3().material.inside.alpha(x, y)
    s1, s2 = logistic(r - inner, delta), logistic(r - outer, delta)
    expected = 1.0 + (ring - 1.0) * s1 * (1.0 - s2)
    np.testing.assert_allclose(m.alpha(x, y), expected, rtol=1e-13)


def test_smooth_band_rejects_a_negative_or_infinite_width():
    with pytest.raises(ValueError, match="delta"):
        SmoothBand(case1().material, -1e-3)
    with pytest.raises(ValueError, match="delta"):
        SmoothBand(case1().material, np.nan)
    with pytest.raises(TypeError, match="jump"):
        SmoothBand(SmoothBand(case1().material, 0.01), 0.01)


def test_with_smooth_edges_keeps_the_geometry_and_the_node_set():
    d = with_smooth_edges(case1(), 0.01)
    assert isinstance(d.material, SmoothBand) and d.material.delta == 0.01
    assert d.material.band == case1().material
    assert (d.straddle, d.dirichlet, d.holes) == (
        case1().straddle,
        case1().dirichlet,
        case1().holes,
    )
    a, b = (
        build_node_set(d, 1250, iterations=5),
        build_node_set(case1(), 1250, iterations=5),
    )
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    # Re-smoothing replaces δ rather than compounding it.
    again = with_smooth_edges(d, 0.0025)
    assert again.material == SmoothBand(case1().material, 0.0025)


# --- the normal profile -----------------------------------------------------


@pytest.mark.parametrize("y_e", [0.5896, 0.7, 0.83])
def test_normal_profile_at_delta_zero_is_the_regions_along_the_line(y_e):
    m = SmoothBand(case1().material, 0.0)
    h_s = 0.08
    for j in (0, 1):
        p = m.normal_profile(j, 0.37, y_e, h_s)
        assert (p.nx, p.ny) == (-0.0, 1.0) and p.scale == h_s
        # The breadcrumb's stops: (0.6 − y_e)/h_s and (0.8 − y_e)/h_s.
        assert p.stops.tolist() == [(0.6 - y_e) / h_s, (0.8 - y_e) / h_s]
        assert p.pieces == (m.outside, m.inside, m.outside)
        # One-sided at the stops: each segment reads its own piece there.
        lo, hi = p.stops
        assert p.alpha(np.array([lo]), segment=0)[0] == 1.0
        assert p.alpha(np.array([lo]), segment=1)[0] == 0.2
        assert p.alpha(np.array([hi]), segment=1)[0] == 0.2
        assert p.alpha(np.array([hi]), segment=2)[0] == 1.0
        eta = np.array([lo - 0.5, 0.5 * (lo + hi), hi + 0.5])
        assert p.alpha(eta).tolist() == [1.0, 0.2, 1.0]
        assert p.alpha_e == float(m.alpha(0.37, y_e))


@pytest.mark.parametrize("delta", DELTAS)
def test_normal_profile_stops_are_the_1d_marchs_stops_bit_for_bit(delta):
    m = SmoothBand(case1().material, delta)
    y_e, h_s = 0.5896, 0.08
    p = m.normal_profile(0, 0.37, y_e, h_s)
    one_d = _stops(np.array([y_e]), np.array([h_s]), case1_in_y(delta))
    assert np.array_equal(p.stops, np.unique(one_d))
    # The flanks sit EDGE_STOP δ either side of each crossing.
    np.testing.assert_allclose(
        np.sort(p.stops),
        np.sort(
            [
                (c + f * delta - y_e) / h_s
                for c in (0.6, 0.8)
                for f in (-EDGE_STOP, 0, EDGE_STOP)
            ]
        ),
        rtol=0,
        atol=1e-14,
    )
    assert all(piece is m for piece in p.pieces)
    assert len(p.pieces) == p.stops.size + 1
    eta = np.linspace(-1.0, 4.0, 501)
    px, py = p.point(eta)
    assert np.array_equal(p.alpha(eta), m.alpha(px, py))
    assert np.array_equal(p.alpha(eta), case1_in_y(delta).alpha(y_e + h_s * eta))


def test_normal_profile_refuses_curved_interfaces_for_now():
    for case in (case2, case3):
        m = SmoothBand(case().material, 0.01)
        with pytest.raises(NotImplementedError, match="E4.7"):
            m.normal_profile(0, 0.3, 0.6, 0.08)
