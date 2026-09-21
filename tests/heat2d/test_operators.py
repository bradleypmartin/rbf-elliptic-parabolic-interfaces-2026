"""Stencil groups, derivative matrices, the naive and direct operators, Dirichlet rows,
the equilibrium solve on the control and on case 1, and the control's spectrum."""

import numpy as np
import pytest

from heat_interfaces.heat1d.march import bd4_amplification, interior_operator
from heat_interfaces.heat2d.domain import (
    Band,
    Constant2D,
    FlatLine,
    SineProduct,
    build_node_set,
    case1,
    case3,
)
from heat_interfaces.heat2d.exact import case1_exact, control_exact
from heat_interfaces.heat2d.operators import (
    BOUNDARY_ZONE,
    alpha_matrix,
    boundary_zone,
    build_stencils,
    derivative_matrices,
    direct_operator,
    dirichlet_system,
    dirichlet_values,
    laplacian_operator,
    naive_operator,
)
from heat_interfaces.heat2d.rbf import BOUNDARY, INTERIOR, ITERATIVE
from heat_interfaces.heat2d.solve import rms_error, solve_equilibrium

DOMAIN = case1()
ONE = Constant2D(1.0)
CONTROL, CASE1 = control_exact(), case1_exact()


def top(x, y):
    return np.sin(2 * np.pi * x)


_sets = {}


def node_set(n):
    """Node sets and stencils are shared across the tests of this module."""
    if n not in _sets:
        nodes = build_node_set(DOMAIN, n)
        _sets[n] = (nodes, build_stencils(nodes, DOMAIN))
    return _sets[n]


def rate(errs):
    """Order per halving of h; the count doubles between consecutive entries."""
    e = np.asarray(errs)
    return np.log(e[:-1] / e[1:]) / np.log(np.sqrt(2.0))


# --- stencils ---------------------------------------------------------------


def test_boundary_zone_is_three_over_root_n_from_a_dirichlet_curve():
    nodes, st = node_set(1250)
    zone = boundary_zone(nodes, DOMAIN)
    width = BOUNDARY_ZONE / np.sqrt(1250)
    expect = (nodes.y < width) | (nodes.y > 1 - width)
    np.testing.assert_array_equal(zone, expect)
    assert zone[nodes.dirichlet].all()
    assert not zone[(nodes.y > 0.2) & (nodes.y < 0.8)].any()
    np.testing.assert_array_equal(st.near_boundary, zone)


def test_case3_zone_includes_the_cooling_circle():
    dom = case3()
    nodes = build_node_set(dom, 2500, iterations=5)
    zone = boundary_zone(nodes, dom)
    r = np.hypot(nodes.x - 0.5, nodes.y - 0.5)
    assert zone[nodes.boundary_index == 2].all()
    assert zone[(r > 0.05) & (r < 0.05 + 3 / 50)].all()
    assert not zone[(r > 0.2) & (nodes.y > 0.2) & (nodes.y < 0.8)].any()


def test_stencil_groups_partition_the_nodes_and_start_at_the_centre():
    nodes, st = node_set(1250)
    assert [g.spec for g in st.groups] == [INTERIOR, BOUNDARY]
    rows = np.concatenate([g.rows for g in st.groups])
    assert np.array_equal(np.sort(rows), np.arange(nodes.n))
    for g in st.groups:
        assert g.index.shape == (len(g.rows), g.spec.size)
        np.testing.assert_array_equal(g.index[:, 0], g.rows)
    assert st.spec_of(int(np.flatnonzero(nodes.dirichlet)[0])) == BOUNDARY
    assert (
        st.spec_of(int(np.flatnonzero((nodes.y > 0.4) & (nodes.y < 0.6))[0]))
        == INTERIOR
    )
    small = build_stencils(nodes, DOMAIN, ITERATIVE, ITERATIVE)
    assert len(small.groups) == 2 and all(g.spec == ITERATIVE for g in small.groups)


# --- derivative matrices ----------------------------------------------------


def test_dy_and_laplacian_are_exact_on_powers_of_y_to_each_groups_degree():
    nodes, st = node_set(1250)
    d = derivative_matrices(nodes, st, ("dy", "lap", "dyy"))
    scale = np.abs(d["lap"]).sum(axis=1)
    for g in st.groups:
        for k in range(g.spec.degree + 1):
            y = nodes.y**k
            dy = k * nodes.y ** max(k - 1, 0) if k else 0.0
            dyy = k * (k - 1) * nodes.y ** max(k - 2, 0) if k > 1 else 0.0
            np.testing.assert_allclose(
                (d["dy"] @ y)[g.rows],
                np.broadcast_to(dy, y.shape)[g.rows],
                atol=1e-12 * scale[g.rows].max(),
            )
            for op in ("lap", "dyy"):
                np.testing.assert_allclose(
                    (d[op] @ y)[g.rows],
                    np.broadcast_to(dyy, y.shape)[g.rows],
                    atol=1e-12 * scale[g.rows].max(),
                )


def test_dx_is_exact_on_powers_of_x_away_from_the_seam():
    nodes, st = node_set(1250)
    dx = derivative_matrices(nodes, st, ("dx",))["dx"]
    # A stencil that wraps sees x^k with a jump of 1 in x; keep to the middle.
    mid = (nodes.x > 0.25) & (nodes.x < 0.75)
    for k in range(5):
        got = (dx @ nodes.x**k)[mid]
        np.testing.assert_allclose(got, k * nodes.x[mid] ** max(k - 1, 0), atol=1e-9)


def test_dx_across_the_seam_is_as_accurate_as_inside():
    # Stencils that straddle x = 0 / x = 1 differentiate through the wrap:
    # cos 2πx is periodic, so a wrong image would show as an O(1/h) error there.
    errs = []
    for n in (1250, 5000):
        nodes, st = node_set(n)
        dx = derivative_matrices(nodes, st, ("dx",))["dx"]
        e = np.abs(
            dx @ np.cos(2 * np.pi * nodes.x) + 2 * np.pi * np.sin(2 * np.pi * nodes.x)
        )
        seam = np.zeros(nodes.n, dtype=bool)
        for g in st.groups:
            seam[g.rows] = np.ptp(nodes.x[g.index], axis=1) > 0.5
        inside = ~seam & ~st.near_boundary
        assert seam.sum() > 20
        assert e[seam & ~st.near_boundary].max() < 10 * e[inside].max()
        errs.append(np.sqrt(np.mean(e[seam] ** 2)))
    assert errs[0] / errs[1] > 2**3.5, errs


def test_derivative_matrices_have_one_stencil_per_row():
    nodes, st = node_set(1250)
    d = derivative_matrices(nodes, st, ("dx", "dy", "lap"))
    for m in d.values():
        assert m.shape == (nodes.n, nodes.n)
        nnz = np.diff(m.indptr)
        assert np.all((nnz == 42) | (nnz == 30)) or np.all(nnz <= 42)
    assert d["lap"].nnz == 42 * (~st.near_boundary).sum() + 30 * st.near_boundary.sum()


# --- alpha, the operators ---------------------------------------------------


def test_alpha_matrix_owns_the_closed_band_by_exact_sign():
    nodes, _ = node_set(1250)
    a = alpha_matrix(nodes, DOMAIN.material).diagonal()
    band = (nodes.y >= 0.6) & (nodes.y <= 0.8)
    np.testing.assert_array_equal(a[band], 0.2)
    np.testing.assert_array_equal(a[~band], 1.0)
    np.testing.assert_array_equal(alpha_matrix(nodes, ONE).diagonal(), 1.0)


def test_naive_operator_is_dx_a_dx_plus_dy_a_dy():
    nodes, st = node_set(1250)
    d = derivative_matrices(nodes, st, ("dx", "dy"))
    a = alpha_matrix(nodes, DOMAIN.material)
    expect = d["dx"] @ a @ d["dx"] + d["dy"] @ a @ d["dy"]
    got = naive_operator(nodes, DOMAIN.material, st)
    assert abs(got - expect).max() < 1e-12 * abs(expect).max()
    assert 100 < got.nnz / nodes.n < 200  # neighbours of neighbours


def test_direct_operator_converges_on_a_smooth_material():
    # div(alpha grad u) for alpha = 0.5 + 0.2 sin 2pi x sin 2pi y, u = sin 2pi x e^y.
    material = SineProduct(0.5, 0.2)
    errs = []
    for n in (1250, 5000):
        nodes, st = node_set(n)
        x, y = nodes.x, nodes.y
        u = np.sin(2 * np.pi * x) * np.exp(y)
        ux = 2 * np.pi * np.cos(2 * np.pi * x) * np.exp(y)
        lap = (1 - 4 * np.pi**2) * u
        ax, ay = material.gradient(x, y)
        exact = material.alpha(x, y) * lap + ax * ux + ay * u
        e = direct_operator(nodes, material, st) @ u - exact
        inside = ~st.near_boundary
        errs.append(np.sqrt(np.mean(e[inside] ** 2)))
    assert errs[0] / errs[1] > 2**3.5, errs


def test_direct_operator_on_a_jump_is_blind_to_it():
    # As in 1-D (dissertation §4.2): the equilibrium error does not fall with n.
    errs = []
    for n in (1250, 2500):
        nodes, st = node_set(n)
        u = solve_equilibrium(
            direct_operator(nodes, DOMAIN.material, st), nodes, [0.0, top]
        )
        errs.append(rms_error(u, CASE1(nodes.x, nodes.y)))
    assert errs[0] == pytest.approx(errs[1], rel=0.1)
    assert errs[1] > 1e-2


# --- Dirichlet rows ---------------------------------------------------------


def test_dirichlet_values_follow_the_curve_order():
    nodes, _ = node_set(1250)
    g = dirichlet_values(nodes, [2.0, top])
    bottom, up = nodes.boundary_index == 0, nodes.boundary_index == 1
    np.testing.assert_array_equal(g[bottom], 2.0)
    np.testing.assert_allclose(g[up], np.sin(2 * np.pi * nodes.x[up]))
    np.testing.assert_array_equal(g[~nodes.dirichlet], 0.0)
    with pytest.raises(ValueError, match="2 Dirichlet curves"):
        dirichlet_values(nodes, [1.0])


def test_dirichlet_system_replaces_the_masked_rows():
    nodes, st = node_set(1250)
    op = laplacian_operator(nodes, st)
    g = dirichlet_values(nodes, [0.0, top])
    f = np.full(nodes.n, 3.0)
    a, b = dirichlet_system(op, nodes.dirichlet, g, f)
    fixed = np.flatnonzero(nodes.dirichlet)
    rows = a[fixed].toarray()
    np.testing.assert_array_equal(rows, np.eye(nodes.n)[fixed])
    np.testing.assert_array_equal(b[fixed], g[fixed])
    np.testing.assert_array_equal(b[~nodes.dirichlet], 3.0)
    free = np.flatnonzero(~nodes.dirichlet)[:5]
    np.testing.assert_allclose(a[free].toarray(), op[free].toarray())
    with pytest.raises(ValueError, match="one entry per node"):
        dirichlet_system(op, nodes.dirichlet[:-1], g)


# --- the control problem and case 1 -----------------------------------------


def test_control_laplacian_is_at_least_fourth_order():
    errs = []
    for n in (1250, 2500, 5000):
        nodes, st = node_set(n)
        u = solve_equilibrium(laplacian_operator(nodes, st), nodes, [0.0, top])
        errs.append(rms_error(u, CONTROL(nodes.x, nodes.y)))
    r = rate(errs)
    assert np.all(r > 3.5), (errs, r)
    assert errs[0] < 5e-5 and errs[-1] < 2e-6


def test_control_naive_operator_is_at_least_fourth_order_from_2500_nodes():
    errs = []
    for n in (2500, 5000):
        nodes, st = node_set(n)
        u = solve_equilibrium(naive_operator(nodes, ONE, st), nodes, [0.0, top])
        errs.append(rms_error(u, CONTROL(nodes.x, nodes.y)))
    assert rate(errs)[0] > 3.5, errs
    assert errs[-1] < 5e-7


def test_case1_naive_operator_is_first_order():
    counts = (1250, 2500, 5000, 10000)
    errs = []
    for n in counts:
        nodes, st = node_set(n)
        u = solve_equilibrium(
            naive_operator(nodes, DOMAIN.material, st), nodes, [0.0, top]
        )
        errs.append(rms_error(u, CASE1(nodes.x, nodes.y)))
    errs = np.array(errs)
    h = 1 / np.sqrt(np.array(counts, dtype=float))
    slope = np.polyfit(np.log(h), np.log(errs), 1)[0]
    # The band is the seed scatter of a first-order line on random node sets:
    # seeds 0, 1, 2 fit at 1.1, 1.6 and 1.3 (port notes §2.2), and the same
    # naive form runs at 1.0-1.3 in 1-D. Anything near the control's 4.5 or
    # the direct operator's 0 fails; the floors catch an operator that is
    # accidentally too good.
    assert 0.7 < slope < 1.7, (errs, slope)
    assert errs[0] > 2e-3 and errs[-1] > 5e-4


# --- the control's spectrum -------------------------------------------------


def test_control_spectra_are_complex_and_bd4_damped_from_2000_nodes():
    # Scattered-node RBF-FD is not symmetric even without interfaces: most
    # eigenvalues are complex and they are not confined to the boundary zone.
    nodes, st = node_set(2000)
    for op in (laplacian_operator(nodes, st), naive_operator(nodes, ONE, st)):
        lam = np.linalg.eigvals(interior_operator(op, nodes.dirichlet).toarray())
        assert lam.real.max() == pytest.approx(-(np.pi**2), rel=1e-3)
        assert np.sum(np.abs(lam.imag) > 1e-8 * np.abs(lam).max()) > 500
        assert bd4_amplification(nodes.h * lam).max() < 1


def test_naive_control_operator_has_a_spurious_growing_mode_at_1250_nodes():
    nodes, st = node_set(1250)
    lam_naive = np.linalg.eigvals(
        interior_operator(naive_operator(nodes, ONE, st), nodes.dirichlet).toarray()
    )
    lam_lap = np.linalg.eigvals(
        interior_operator(laplacian_operator(nodes, st), nodes.dirichlet).toarray()
    )
    assert lam_naive.real.max() > 0
    assert lam_lap.real.max() == pytest.approx(-(np.pi**2), rel=1e-3)


def test_a_flat_band_with_alpha_one_is_the_control():
    nodes, st = node_set(1250)
    same = Band(FlatLine(0.6), FlatLine(0.8), Constant2D(1.0), Constant2D(1.0))
    a = naive_operator(nodes, same, st)
    b = naive_operator(nodes, ONE, st)
    assert abs(a - b).max() == 0.0
