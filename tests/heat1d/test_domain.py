import numpy as np
import pytest

from heat_interfaces.heat1d.domain import (
    PLACEMENT_TOL,
    Constant,
    Grid1D,
    PiecewiseAlpha,
    Sinusoid,
    Smooth,
    dissertation_alpha,
    eabe_alpha,
    equispaced_grid,
    grid_for,
    jump_alpha,
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
