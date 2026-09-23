"""E4.9 (#40): the coefficient treatments on scattered nodes, and their quadrature."""

import numpy as np
import pytest
from scipy.optimize import brentq

from heat_interfaces.heat1d.exact import edge_resistance_deficit
from heat_interfaces.heat2d.domain import (
    Band,
    Constant2D,
    FlatLine,
    SineGraph,
    SineProduct,
    SmoothBand,
    build_node_set,
    case1,
    case2,
    case3,
)
from heat_interfaces.heat2d.neighbors import wrap_x
from heat_interfaces.heat2d.operators import build_stencils, naive_operator
from heat_interfaces.heat2d.treatments import (
    NodalAlpha2D,
    arithmetic_discs,
    disc_integrals,
    disc_means,
    harmonic_discs,
    widened_edge,
)


def segment(r, height):
    """Area of the circular segment of ``height`` (0 to 2r) cut off a disc."""
    height = np.clip(height, 0.0, 2.0 * r)
    return r**2 * np.arccos((r - height) / r) - (r - height) * np.sqrt(
        2.0 * r * height - height**2
    )


def _roots(f, a, b, m):
    t = np.linspace(a, b, m + 1)
    v = f(t)
    return [
        brentq(f, t[i], t[i + 1], xtol=1e-16, rtol=1e-15)
        for i in range(m)
        if v[i] * v[i + 1] < 0.0
    ]


def polar_integrals(material, x0, y0, r, curve, panels=1, n=24):
    """``(area, ∫ α, ∫ 1/α)`` in polar coordinates about the centre: nothing sheared.

    At δ = 0 the rays are cut where they cross ``curve`` and the angles where
    the curve meets the circle (the outer integrand's kinks); at δ > 0 both
    directions are split into ``panels`` equal panels through the tanh.
    """
    t, w = np.polynomial.legendre.leggauss(n)
    jump = not isinstance(material, SmoothBand) or material.delta == 0.0

    def gap(phi, rho):
        x = x0 + rho * np.cos(phi)
        return y0 + rho * np.sin(phi) - curve.height(x)

    if jump:
        phis = np.unique(
            [0.0, 2 * np.pi, *_roots(lambda p: gap(p, r), 0, 2 * np.pi, 400)]
        )
    else:
        phis = np.linspace(0.0, 2 * np.pi, panels + 1)
    total = np.zeros(3)
    for a, b in zip(phis[:-1], phis[1:], strict=True):
        nodes, weights = 0.5 * (a + b) + 0.5 * (b - a) * t, 0.5 * (b - a) * w
        for p, wp in zip(nodes, weights, strict=True):
            if jump:
                cuts = _roots(lambda s, p=p: gap(p, s), 0.0, r, 64)
                rho_edges = np.unique([0.0, r, *cuts])
            else:
                rho_edges = np.linspace(0.0, r, panels + 1)
            c, d = rho_edges[:-1], rho_edges[1:]
            rho = (0.5 * (c + d)[:, None] + 0.5 * (d - c)[:, None] * t).ravel()
            wr = (0.5 * (d - c)[:, None] * w).ravel()
            x, y = x0 + rho * np.cos(p), y0 + rho * np.sin(p)
            alpha = material.alpha(wrap_x(x), y)
            weight = wp * wr * rho
            total += [weight.sum(), (weight * alpha).sum(), (weight / alpha).sum()]
    return total


@pytest.fixture(scope="module")
def nodes900():
    return build_node_set(case2(), 900, seed=0, iterations=20)


def test_a_constant_alpha_is_left_alone(nodes900):
    for curves in ((FlatLine(0.6), FlatLine(0.8)), (SineGraph(0.6), SineGraph(0.8))):
        band = Band(*curves, Constant2D(0.7), Constant2D(0.7))
        for material in (band, SmoothBand(band, 0.01)):
            for r in (0.5 * nodes900.h, nodes900.h):
                hm, am = disc_means(nodes900, material, r)
                assert np.abs(hm.values / 0.7 - 1).max() < 1e-13
                assert np.abs(am.values / 0.7 - 1).max() < 1e-13


def test_the_disc_area_is_exact_inside_and_clipped_at_the_rows():
    rng = np.random.default_rng(1)
    r = 0.03
    x = rng.random(200)
    y = np.concatenate(
        [rng.random(192), [0.0, 0.01, 0.029, 0.5, 0.971, 0.99, 1.0, 0.6]]
    )
    expect = np.pi * r**2 - segment(r, r - y) - segment(r, y + r - 1.0)
    for material in (case1().material, SmoothBand(case2().material, 0.0025)):
        area, _, _ = disc_integrals(material, x, y, r)
        assert np.abs(area / expect - 1).max() < 1e-13


def test_across_a_flat_jump_the_means_weight_the_pieces_by_segment_areas():
    r = 0.02
    y = 0.6 + np.linspace(-0.99, 0.99, 41) * r
    x = np.linspace(0.0, 1.0, 41, endpoint=False)
    below = segment(r, r - (y - 0.6))
    above = np.pi * r**2 - below
    area, integral, resistance = disc_integrals(case1().material, x, y, r)
    assert np.abs(integral / (below * 1.0 + above * 0.2) - 1).max() < 1e-13
    assert np.abs(resistance / (below / 1.0 + above / 0.2) - 1).max() < 1e-13


@pytest.mark.parametrize(
    ("delta", "panels"), [(0.0, 1), (0.0025, 40)], ids=["jump", "tanh"]
)
def test_case2_matches_a_polar_quadrature_that_shears_nothing(delta, panels):
    material = SmoothBand(case2().material, delta)
    curve = SineGraph(0.6)
    rng = np.random.default_rng(3)
    r = 0.01
    for _ in range(2):
        x0 = rng.random()
        y0 = curve.height_at(x0) + (rng.random() - 0.5) * 1.2 * r
        got = np.array(disc_integrals(material, np.array([x0]), np.array([y0]), r))
        ref = polar_integrals(material, x0, y0, r, curve, panels, n=24 if delta else 64)
        assert np.abs(got[:, 0] / ref - 1).max() < 1e-13


def test_more_gauss_points_change_nothing():
    rng = np.random.default_rng(4)
    x = rng.random(300)
    y = 0.6 + 0.02 * np.sin(2 * np.pi * x) + (rng.random(300) - 0.5) * 0.08
    for delta in (0.0, 0.0025, 0.04):
        material = SmoothBand(case2().material, delta)
        a = np.array(disc_integrals(material, x, y, 0.02))
        b = np.array(disc_integrals(material, x, y, 0.02, n_gauss=36))
        assert np.abs(a / b - 1).max() < 1e-13


def test_small_discs_follow_the_mean_value_expansion():
    """``AM = α + r²/8 Δα + O(r⁴)``, ``HM = α + r²/8 (Δα − 2|∇α|²/α) + O(r⁴)``."""
    piece = SineProduct(0.2, 0.1)
    band = Band(FlatLine(0.6), FlatLine(0.8), piece, piece)
    x = np.array([0.1, 0.37, 0.8])
    y = np.array([0.3, 0.55, 0.9])
    a = piece.alpha(x, y)
    gx, gy = piece.gradient(x, y)
    lap = -2.0 * (2.0 * np.pi) ** 2 * (a - 0.2)
    remainders = []
    for r in (0.02, 0.01):
        area, integral, resistance = disc_integrals(band, x, y, r)
        am, hm = integral / area, area / resistance
        remainders.append(
            (
                am - (a + r**2 / 8 * lap),
                hm - (a + r**2 / 8 * (lap - 2 * (gx**2 + gy**2) / a)),
            )
        )
    for coarse, fine in zip(*remainders, strict=True):
        assert np.abs(coarse).max() < 1e-6
        assert np.allclose(coarse / fine, 16.0, rtol=0.02)


def test_the_arithmetic_mean_is_never_below_the_harmonic(nodes900):
    material = SmoothBand(case2().material, 0.0025)
    hm, am = disc_means(nodes900, material, nodes900.h)
    assert np.all(am.values >= hm.values * (1 - 1e-14))
    assert np.array_equal(
        hm.values, harmonic_discs(nodes900, material, nodes900.h).values
    )
    assert np.array_equal(
        am.values, arithmetic_discs(nodes900, material, nodes900.h).values
    )
    across = am.values / hm.values
    assert across.max() > 1.5  # the discs that straddle an edge
    assert np.median(across) < 1.0 + 1e-3


def test_the_resistance_tends_to_the_jumps_at_first_order_and_alpha_at_second():
    """``∫ 1/α`` loses ``c δ`` per unit of the edge's chord (1-D's deficit, §2.2).

    ``∫ α`` has no first-order term: the fold's tanh is odd about the centre,
    so the arithmetic mean approaches the jump's at O(δ²) and the harmonic
    one at O(δ), with 1-D's ``edge_resistance_deficit`` as the constant.
    """
    r, x, y = 0.02, np.array([0.3]), np.array([0.61])
    chord = 2.0 * np.sqrt(r**2 - (y[0] - 0.6) ** 2)
    c = edge_resistance_deficit(1.0, 0.2)
    zero = np.array(disc_integrals(case1().material, x, y, r))[:, 0]
    gaps = []
    for d in (1e-3, 5e-4, 2.5e-4):
        smooth = np.array(disc_integrals(SmoothBand(case1().material, d), x, y, r))
        gaps.append(smooth[:, 0] - zero)
        assert abs((smooth[2, 0] - zero[2]) / (-c * d * chord) - 1) < 2 * d / r
    for coarse, fine in zip(gaps[:-1], gaps[1:], strict=True):
        assert coarse[0] == pytest.approx(0.0, abs=1e-15)
        assert coarse[1] / fine[1] == pytest.approx(4.0, rel=0.02)
        assert coarse[2] / fine[2] == pytest.approx(2.0, rel=0.02)


def test_a_ring_has_no_shear_and_is_refused():
    with pytest.raises(ValueError, match="flat lines or two parallel sine"):
        disc_integrals(case3().material, np.array([0.5]), np.array([0.5]), 0.01)
    with pytest.raises(ValueError, match="radius"):
        disc_integrals(case1().material, np.array([0.5]), np.array([0.5]), 0.0)


def test_a_nodal_table_is_read_at_its_nodes_only(nodes900):
    values = np.linspace(0.5, 1.5, nodes900.n)
    table = NodalAlpha2D(nodes900, values)
    assert np.array_equal(table.alpha(nodes900.x, nodes900.y), values)
    with pytest.raises(ValueError, match="own nodes"):
        table.alpha(nodes900.x[:10], nodes900.y[:10])
    with pytest.raises(ValueError, match="own nodes"):
        table.alpha(nodes900.x + 1e-12, nodes900.y)
    with pytest.raises(ValueError, match="one value per node"):
        NodalAlpha2D(nodes900, values[:-1])
    with pytest.raises(ValueError, match="positive"):
        NodalAlpha2D(nodes900, -values)


def test_a_nodal_table_prints_its_whole_table():
    nodes = build_node_set(case1(), 2500, seed=0, iterations=5)
    values = np.ones(nodes.n)
    other = values.copy()
    other[nodes.n // 2] = 2.0
    assert repr(NodalAlpha2D(nodes, values)) != repr(NodalAlpha2D(nodes, other))
    assert repr(NodalAlpha2D(nodes, values)) == repr(NodalAlpha2D(nodes, values.copy()))


def test_the_naive_operator_reads_a_table_as_it_reads_the_medium(nodes900):
    material = SmoothBand(case2().material, 0.01)
    stencils = build_stencils(nodes900, case2())
    table = NodalAlpha2D(nodes900, material.alpha(nodes900.x, nodes900.y))
    a = naive_operator(nodes900, material, stencils)
    b = naive_operator(nodes900, table, stencils)
    assert (a != b).nnz == 0


def test_the_widened_edge_is_the_band_at_the_wider_width(nodes900):
    h = nodes900.h
    band = case2().material
    for m in (1.0, 2.0):
        assert widened_edge(nodes900, band, m) == SmoothBand(band, m * h)
        below = [widened_edge(nodes900, SmoothBand(band, d), m) for d in (0.0, 0.1 * h)]
        assert below[0] == below[1] == SmoothBand(band, m * h)
        wide = SmoothBand(band, 3.0 * h)
        assert widened_edge(nodes900, wide, m) == wide
    ring = SmoothBand(case3().material, 0.001, "resistance")
    assert widened_edge(nodes900, ring, 1.0).composition == "resistance"
    with pytest.raises(ValueError, match="positive"):
        widened_edge(nodes900, band, 0.0)
