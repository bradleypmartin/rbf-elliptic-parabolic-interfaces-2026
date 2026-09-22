import numpy as np
import pytest

from heat_interfaces.heat1d.domain import (
    EDGE_CUTS,
    PLACEMENT_TOL,
    TANH_REACH,
    Constant,
    Grid1D,
    OnInterval,
    PiecewiseAlpha,
    Sinusoid,
    Smooth,
    SmoothEdges,
    dissertation_alpha,
    eabe_alpha,
    edge_blend,
    equispaced_grid,
    grid_for,
    jump_alpha,
    matlab_alpha,
    node_counts,
)


def test_equispaced_grid_has_both_ends_on_nodes():
    g = equispaced_grid(101)
    assert g.n == 101
    assert g.h == pytest.approx(0.02)
    assert g.x[0] == -1.0 and g.x[-1] == 1.0
    np.testing.assert_allclose(np.diff(g.x), g.h)


def test_the_dissertation_interfaces_sit_on_nodes_50_and_75_of_101():
    g = equispaced_grid(101)
    assert g.placement(0.0) == "node" and g.placement(0.5) == "node"
    assert g.x[50] == pytest.approx(0.0) and g.x[75] == pytest.approx(0.5)
    assert equispaced_grid(100).placement(0.0) == "cell"
    assert equispaced_grid(100).offset(0.0) == pytest.approx(0.5)
    assert equispaced_grid(100).placement(0.5) is None


def test_node_counts_by_placement():
    both = node_counts([0.0, 0.5], "node", 100, 1700)
    assert {101, 401, 801, 1601} <= set(both)
    assert all((n - 1) % 4 == 0 for n in both)
    assert node_counts([0.0, 0.5], "cell", 5, 5000) == []
    assert node_counts([0.0], "cell", 10, 20) == list(range(10, 21, 2))


def test_grid_for_picks_the_nearest_admissible_count():
    assert grid_for([0.0, 0.5], "node", 400).n == 401
    assert grid_for([0.0], "cell", 101).n == 102
    with pytest.raises(ValueError, match="no node count"):
        grid_for([0.0, 0.5], "cell", 100)


def test_dissertation_alpha_follows_eq_75_including_the_closed_layer():
    m = dissertation_alpha()
    x = np.array([-0.5, 0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(m.alpha(x), [1.0, 0.1, 0.5, 0.1, 1.0], atol=1e-15)
    np.testing.assert_allclose(
        m.alpha_x(x), [0.0, 0.8 * np.pi, 0.0, -0.8 * np.pi, 0.0], atol=1e-12
    )
    assert float(m.alpha(0.3)) == pytest.approx(0.1 + 0.4 * np.sin(0.6 * np.pi))


def test_eabe_alpha_gives_the_eq_66_taylor_coefficients():
    m = eabe_alpha()
    right = m.taylor(0, "right", 4)
    # Dissertation eq. 66 / EABE eq. 18, printed to two decimals.
    np.testing.assert_allclose(right, [0.5, 0.63, 0.0, -4.13, 0.0], atol=5e-3)
    np.testing.assert_allclose(m.taylor(0, "left", 4), [1, 0, 0, 0, 0])
    assert float(m.alpha(0.0)) == 0.5  # [0, 1] owns the interface


def test_dissertation_taylor_coefficients_about_both_interfaces():
    m = dissertation_alpha()
    k = 2 * np.pi
    np.testing.assert_allclose(
        m.taylor(0, "right", 4), [0.1, 0.4 * k, 0.0, -0.4 * k**3 / 6, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        m.taylor(1, "left", 4), [0.1, -0.4 * k, 0.0, 0.4 * k**3 / 6, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(m.taylor(1, "right", 3), [1, 0, 0, 0])


def test_smooth_piece_taylor_matches_the_sinusoid():
    s = Sinusoid(0.3, 0.2, 3.0, phase=0.4)
    f = Smooth(
        s.alpha,
        (
            lambda x: 0.6 * np.cos(3 * x + 0.4),
            lambda x: -1.8 * np.sin(3 * x + 0.4),
            lambda x: -5.4 * np.cos(3 * x + 0.4),
        ),
    )
    np.testing.assert_allclose(f.taylor(0.2, 3), s.taylor(0.2, 3))
    with pytest.raises(ValueError, match="derivatives"):
        f.taylor(0.2, 4)


def test_jump_alpha_and_validation():
    m = jump_alpha(1 / 9, 1.0)
    np.testing.assert_allclose(m.alpha([-0.1, 0.0, 0.1]), [1 / 9, 1.0, 1.0])
    with pytest.raises(ValueError):
        PiecewiseAlpha((0.5, 0.0), (Constant(1), Constant(2), Constant(3)))
    with pytest.raises(ValueError):
        PiecewiseAlpha((0.0,), (Constant(1),))
    with pytest.raises(ValueError):
        PiecewiseAlpha((1.0,), (Constant(1), Constant(2)))


def test_ownership_is_exact_and_grids_snap_instead():
    m = dissertation_alpha()
    # The medium never snaps: a hair off the interface is the other piece.
    np.testing.assert_allclose(m.alpha([0.5 - 1e-14, 0.5 + 1e-14]), [0.1, 1.0])
    np.testing.assert_allclose(m.alpha([-1e-14, 1e-14]), [1.0, 0.1])
    # A grid node that lands on an interface up to rounding is moved onto it.
    g = equispaced_grid(101)
    x = g.x.copy()
    x[75] += 1e-16
    nudged = Grid1D(n=g.n, x=x, h=g.h)
    snapped = nudged.snapped(m.interfaces)
    assert snapped[75] == 0.5 and snapped[50] == 0.0
    assert np.count_nonzero(snapped != x) == 1
    assert float(m.alpha(snapped[75])) == pytest.approx(0.1)
    assert float(m.alpha(x[75])) == 1.0


def test_placement_and_snapping_share_one_tolerance():
    xi = 0.37
    for n in node_counts([xi], "node", 5, 2000):
        g = equispaced_grid(n)
        j = int(np.argmin(np.abs(g.x - xi)))
        assert g.snapped([xi])[j] == xi
        assert g.placement(xi, tol=PLACEMENT_TOL / 100) in ("node", None)


# --- E3.2 (#27): the smooth-edged medium -----------------------------------


def logistic(x, xc, delta):
    return 0.5 * (1.0 + np.tanh((x - xc) / delta))


def test_delta_zero_is_the_jump_bit_for_bit():
    for jump in (matlab_alpha(), dissertation_alpha()):
        m = SmoothEdges(jump, 0.0)
        x = np.concatenate([np.linspace(-1, 1, 1001), jump.interfaces])
        assert m.interfaces == jump.interfaces
        assert np.array_equal(m.alpha(x), jump.alpha(x))
        assert np.array_equal(m.alpha_x(x), jump.alpha_x(x))
        for i in range(len(jump.interfaces)):
            for side in ("left", "right"):
                assert np.array_equal(m.taylor(i, side, 4), jump.taylor(i, side, 4))
        edges, pieces = m.elements()
        j_edges, j_pieces = jump.elements()
        assert np.array_equal(edges, j_edges) and pieces == j_pieces


def test_a_smooth_edge_between_constants_is_the_tanh_blend():
    a, b, delta = 1.0 / 9.0, 1.0, 0.0025
    m = SmoothEdges(matlab_alpha(), delta)
    x = np.linspace(-0.05, 0.05, 2001)
    s = logistic(x, 0.0, delta)
    np.testing.assert_allclose(m.alpha(x), a + (b - a) * s, rtol=1e-15, atol=1e-17)
    np.testing.assert_allclose(
        m.alpha_x(x), (b - a) / (2 * delta) / np.cosh(x / delta) ** 2, rtol=1e-13
    )
    assert float(m.alpha(0.0)) == pytest.approx((a + b) / 2)
    inner = np.abs(x) < 15 * delta  # beyond that the tails are flat to rounding
    assert np.all(np.diff(m.alpha(x[inner])) > 0)


def test_the_tails_reach_the_pieces_bit_for_bit_beyond_tanh_reach():
    for jump, delta in ((matlab_alpha(), 0.0025), (dissertation_alpha(), 0.01)):
        m = SmoothEdges(jump, delta)
        for xc in jump.interfaces:
            far = xc + np.array([-1.0, 1.0]) * TANH_REACH * delta
            assert np.array_equal(m.alpha(far), jump.alpha(far))
            nearer = xc + np.array([-1.0, 1.0]) * 19.0 * delta
            gap = np.abs(m.alpha(nearer) - jump.alpha(nearer))
            assert np.all(gap <= 2 * np.spacing(jump.alpha(nearer)))
        x = np.linspace(-1, 1, 20001)
        inside = np.any(
            [np.abs(x - xc) < TANH_REACH * delta for xc in jump.interfaces], axis=0
        )
        assert np.array_equal(m.alpha(x[~inside]), jump.alpha(x[~inside]))


def test_smoothly_varying_pieces_keep_their_variation_through_the_blend():
    # Eq. 75 with δ = 0.01 at both edges of the layer; the blend of the two
    # pieces at each point, hand-built, folding the edges in from the left.
    jump = dissertation_alpha()
    m = SmoothEdges(jump, 0.01)
    x = np.linspace(-1, 1, 4001)
    p0, p1, p2 = (p.alpha(x) for p in jump.pieces)
    s0, s1 = logistic(x, 0.0, 0.01), logistic(x, 0.5, 0.01)
    expected = (1 - s1) * ((1 - s0) * p0 + s0 * p1) + s1 * p2
    np.testing.assert_allclose(m.alpha(x), expected, rtol=1e-13, atol=1e-16)
    # The analytic derivative against a central difference of the blend.
    h = 1e-6
    fd = (m.alpha(x + h) - m.alpha(x - h)) / (2 * h)
    np.testing.assert_allclose(m.alpha_x(x), fd, rtol=1e-7, atol=1e-7)
    assert m.alpha(x).min() > 0.1


def test_a_thin_layer_blends_as_a_partition_of_unity():
    # Two edges 0.002 apart with δ = 0.001: the weights of the three pieces
    # still sum to one and alpha stays between the extreme values.
    jump = PiecewiseAlpha((0.1, 0.102), (Constant(1.0), Constant(1e-3), Constant(0.5)))
    m = SmoothEdges(jump, 1e-3)
    x = np.linspace(0.08, 0.12, 4001)
    a = m.alpha(x)
    assert np.all(a >= 1e-3) and np.all(a <= 1.0)
    assert float(m.alpha(0.101)) < 0.3  # the layer's value is seen mid-layer


def test_smooth_edges_elements_cut_at_the_edge_cuts_and_merge_collisions():
    m = SmoothEdges(matlab_alpha(), 0.0025)
    edges, pieces = m.elements()
    expected = [-1.0, *(-0.0025 * c for c in reversed(EDGE_CUTS))]
    expected += [*(0.0025 * c for c in EDGE_CUTS), 1.0]
    np.testing.assert_allclose(edges, expected, atol=1e-15)
    assert pieces == (m,) * (len(edges) - 1)
    assert edges[-2] > TANH_REACH * m.delta  # the last cut is past the reach
    # δ = 0.04 at 0 and 0.5: cuts beyond the ends are dropped, cuts of the two
    # edges that collide are merged, and every element is at least δ/2 wide.
    d = SmoothEdges(dissertation_alpha(), 0.04)
    edges, _ = d.elements()
    assert edges[0] == -1.0 and edges[-1] == 1.0
    assert np.all(np.diff(edges) > 0.02)
    for xc in d.interfaces:
        for c in EDGE_CUTS:
            for cut in (xc - c * 0.04, xc + c * 0.04):
                if -1 + 0.02 < cut < 1 - 0.02:
                    assert np.min(np.abs(edges - cut)) <= 0.02 + 1e-12


def test_smooth_edges_rejects_a_negative_or_infinite_width():
    with pytest.raises(ValueError, match="delta"):
        SmoothEdges(matlab_alpha(), -0.1)
    with pytest.raises(ValueError, match="delta"):
        SmoothEdges(matlab_alpha(), np.inf)


def test_edge_blend_is_the_logistic_blend_from_the_near_side():
    z = np.linspace(-25.0, 25.0, 2001)
    a, b = np.full_like(z, 0.2), np.full_like(z, 1.0)
    value, ds = edge_blend(a, b, z)
    s = 0.5 * (1.0 + np.tanh(z))
    np.testing.assert_allclose(value, 0.2 + 0.8 * s, rtol=1e-15)
    np.testing.assert_allclose(ds, 0.5 / np.cosh(z) ** 2, rtol=1e-13, atol=0)
    # The far side's share rounds away: exactly each value at |z| >= 20.
    assert np.all(value[z >= 20.0] == 1.0) and np.all(value[z <= -20.0] == 0.2)


def test_on_interval_clips_the_elements_and_keeps_the_material():
    # Case 1's profile in y (E4.2): 1 | 0.2 | 1 at 0.6 and 0.8, on [0, 1].
    jump = PiecewiseAlpha(
        (0.6, 0.8), (Constant(1.0), Constant(0.2), Constant(1.0)), ("right", "left")
    )
    m = OnInterval(jump, 0.0, 1.0)
    edges, pieces = m.elements()
    assert edges.tolist() == [0.0, 0.6, 0.8, 1.0] and pieces == jump.pieces
    x = np.linspace(0.0, 1.0, 101)
    assert np.array_equal(m.alpha(x), jump.alpha(x))
    assert m.interfaces == jump.interfaces
    assert np.array_equal(m.taylor(1, "left", 3), jump.taylor(1, "left", 3))
    for delta in (0.04, 0.01, 0.0025):
        smooth = OnInterval(SmoothEdges(jump, delta), 0.0, 1.0)
        edges, pieces = smooth.elements()
        full, _ = smooth.medium.elements()
        assert edges[0] == 0.0 and edges[-1] == 1.0
        assert set(edges[1:-1]) == {e for e in full if delta / 2 < e < 1 - delta / 2}
        assert pieces == (smooth.medium,) * (len(edges) - 1)
        assert np.all(np.diff(edges) > delta / 2)


def test_on_interval_drops_a_smooth_cut_within_half_a_width_of_an_end():
    # 0.2 − 27δ = 0.003 < δ/2 from 0: that cut goes; 0.2 − 9δ stays.
    delta = 0.197 / 27
    m = OnInterval(SmoothEdges(jump_alpha(1.0, 0.2, 0.2), delta), 0.0, 1.0)
    edges, _ = m.elements()
    assert edges[0] == 0.0 and edges[1] == pytest.approx(0.2 - 9 * delta)
    assert np.all(np.diff(edges) > delta / 2)


def test_on_interval_refuses_an_interface_outside_or_a_bad_interval():
    with pytest.raises(ValueError, match="strictly inside"):
        OnInterval(dissertation_alpha(), 0.0, 1.0)  # an interface at 0
    with pytest.raises(ValueError, match="interval"):
        OnInterval(matlab_alpha(), 0.5, -0.5)
    with pytest.raises(ValueError, match="interval"):
        OnInterval(matlab_alpha(), -1.5, 1.0)
