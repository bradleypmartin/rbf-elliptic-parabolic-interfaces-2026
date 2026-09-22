"""Stencil groups, derivative matrices, the naive and direct operators, Dirichlet rows,
the equilibrium solve on the control and on case 1, the control's spectrum, the
warp-and-straddle ablation of case 2, and (E4.5, #36) the seeded-row rule, the
seed operator and the dispatch by name."""

from dataclasses import replace

import numpy as np
import pytest

from heat_interfaces.heat1d.domain import TANH_REACH
from heat_interfaces.heat1d.march import bd4_amplification, interior_operator
from heat_interfaces.heat2d.domain import (
    ROW,
    Band,
    Constant2D,
    FlatLine,
    SineProduct,
    SmoothBand,
    build_node_set,
    case1,
    case2,
    case3,
)
from heat_interfaces.heat2d.exact import case1_exact, control_exact
from heat_interfaces.heat2d.neighbors import knn
from heat_interfaces.heat2d.operators import (
    BOUNDARY_KIND,
    BOUNDARY_ZONE,
    INTERFACE_KIND,
    INTERIOR_KIND,
    OPERATOR_MODES,
    alpha_matrix,
    boundary_zone,
    build_operator,
    build_stencils,
    derivative_matrices,
    direct_operator,
    dirichlet_system,
    dirichlet_values,
    interface_aware_operator,
    interface_crossings,
    laplacian_operator,
    naive_operator,
    seed_operator,
    seeded_rows,
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


_aware = {}


def aware_set(n):
    """The same node sets with the interface group of E2.3."""
    if n not in _aware:
        nodes, _ = node_set(n)
        _aware[n] = (nodes, build_stencils(nodes, DOMAIN, interface=BOUNDARY))
    return _aware[n]


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
    assert [g.kind for g in st.groups] == [INTERIOR_KIND, BOUNDARY_KIND]
    assert not st.near_interface.any()


def test_interface_group_holds_exactly_the_nodes_whose_interior_stencil_crosses():
    nodes, st = aware_set(1250)
    assert [g.kind for g in st.groups] == [INTERIOR_KIND, BOUNDARY_KIND, INTERFACE_KIND]
    interior, boundary, interface = st.groups
    assert interface.spec == BOUNDARY and interior.spec == INTERIOR
    rows = np.concatenate([g.rows for g in st.groups])
    assert np.array_equal(np.sort(rows), np.arange(nodes.n))
    region = DOMAIN.material.region_index(nodes.x, nodes.y)
    idx42, _ = knn(nodes.xy, INTERIOR.size)
    crosses42 = np.ptp(region[idx42], axis=1) > 0
    np.testing.assert_array_equal(st.near_interface, crosses42)
    np.testing.assert_array_equal(np.sort(interface.rows), np.flatnonzero(crosses42))
    # No standard stencil sees a jump; the straddling rows all sit in the group.
    assert not interface_crossings(nodes, DOMAIN.material, interior.index).any()
    assert not interface_crossings(nodes, DOMAIN.material, boundary.index).any()
    assert np.isin(np.flatnonzero(nodes.kind == ROW), interface.rows).all()
    # Most of the group's own 30 nodes cross; a few at its edge do not.
    own = interface_crossings(nodes, DOMAIN.material, interface.index)
    assert 0.9 < own.mean() < 1.0
    assert st.group_of(int(interface.rows[0])).kind == INTERFACE_KIND
    # Without interfaces nothing crosses.
    assert not interface_crossings(nodes, ONE, interior.index).any()


def test_crossing_test_uses_the_largest_stencil_size_not_the_interior_one():
    # With a 19-node interior and 30-node boundary and interface stencils the
    # crossing test must look at 30 nodes, or a boundary stencil could see a
    # jump that the smaller interior stencil does not.
    nodes, _ = node_set(1250)
    st = build_stencils(nodes, DOMAIN, ITERATIVE, BOUNDARY, interface=BOUNDARY)
    region = DOMAIN.material.region_index(nodes.x, nodes.y)
    idx30, _ = knn(nodes.xy, BOUNDARY.size)
    np.testing.assert_array_equal(st.near_interface, np.ptp(region[idx30], axis=1) > 0)
    idx19, _ = knn(nodes.xy, ITERATIVE.size)
    assert st.near_interface.sum() > (np.ptp(region[idx19], axis=1) > 0).sum()
    for g in st.groups:
        if g.kind != INTERFACE_KIND:
            assert not interface_crossings(nodes, DOMAIN.material, g.index).any()


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


# --- the interface-aware operator (E2.3) --------------------------------------


@pytest.mark.parametrize("warp", [True, False])
def test_interface_aware_operator_is_fourth_order_on_case1(warp):
    # EABE Fig. 7: 1.0e-5, 2.6e-6, 5.5e-7 at 1250, 2500, 5000 nodes (read off
    # the rendered page); with the warped RBFs of E2.4 ours run 1.1–1.6× above
    # them, with plain Gaussians 3.7–4.2× (port notes §2.3–2.4), at the same
    # order either way.
    counts = (1250, 2500, 5000)
    errs, naive = [], []
    for n in counts:
        nodes, st = aware_set(n)
        u = solve_equilibrium(
            interface_aware_operator(nodes, DOMAIN.material, st, warp=warp),
            nodes,
            [0.0, top],
        )
        errs.append(rms_error(u, CASE1(nodes.x, nodes.y)))
        _, st0 = node_set(n)
        un = solve_equilibrium(
            naive_operator(nodes, DOMAIN.material, st0), nodes, [0.0, top]
        )
        naive.append(rms_error(un, CASE1(nodes.x, nodes.y)))
    r = rate(errs)
    assert np.all(r > 3.3), (errs, r)
    assert errs[0] < (2.5e-5 if warp else 1e-4) and errs[-1] < (1e-6 if warp else 5e-6)
    assert all(e < n / 50 for e, n in zip(errs, naive, strict=True))


def test_case2_warp_and_straddle_ablation_runs_at_three_resolutions():
    # EABE Fig. 11's four combinations on case 2, which has no analytic
    # solution: each builds and solves, the Dirichlet data bound |u| by 1, the
    # node sets without rows put nodes within a hundredth of a spacing of the
    # curves and the crossing stencils take them, and the warp moves the
    # straddled solution by an amount that falls like h^4, as the difference
    # of two fourth-order operators should. The errors against the
    # 160,000-node reference are E2.6's (#20).
    domain = case2()
    bare = replace(domain, straddle=())
    moved = []
    for n in (1250, 2500, 5000):
        for dom in (domain, bare):
            nodes = build_node_set(dom, n)
            st = build_stencils(nodes, dom, interface=BOUNDARY)
            assert st.groups[-1].kind == INTERFACE_KIND
            u = {}
            for warp in (True, False):
                op = interface_aware_operator(nodes, dom.material, st, warp=warp)
                u[warp] = solve_equilibrium(op, nodes, [0.0, top])
                assert np.isfinite(u[warp]).all()
                assert np.abs(u[warp]).max() < 1.0 + 1e-3
            if dom is domain:
                moved.append(rms_error(u[True], u[False]))
            else:
                assert len(nodes.straddle_rows) == 0
                gap = min(
                    np.abs(c.signed_distance(nodes.x, nodes.y)).min()
                    for c in dom.material.interfaces
                )
                assert gap < 0.02 * nodes.h
    moved = np.array(moved)
    assert moved[0] > 1e-5 and np.all(moved[:-1] / moved[1:] > 2.5), moved


def test_flat_and_curved_variants_coincide_on_case1_and_replace_only_crossing_rows():
    nodes, st = aware_set(1250)
    curved = interface_aware_operator(nodes, DOMAIN.material, st)
    flat = interface_aware_operator(nodes, DOMAIN.material, st, curvature=False)
    assert abs(curved - flat).max() == 0.0
    direct = direct_operator(nodes, DOMAIN.material, st)
    diff = np.asarray(abs(curved - direct).sum(axis=1)).ravel()
    interface = st.groups[-1]
    own = interface_crossings(nodes, DOMAIN.material, interface.index)
    changed = np.zeros(nodes.n, dtype=bool)
    changed[interface.rows[own]] = True
    assert (diff[changed] > 0).all() and (diff[~changed] == 0).all()
    nnz = np.diff(curved.indptr)
    assert np.all(nnz[interface.rows] == BOUNDARY.size)
    assert np.all(nnz[st.groups[0].rows] == INTERIOR.size)


def test_interface_aware_operator_without_interfaces_is_the_direct_operator():
    nodes, st = aware_set(1250)
    a = interface_aware_operator(nodes, ONE, st)
    b = direct_operator(nodes, ONE, st)
    assert abs(a - b).max() == 0.0
    u = solve_equilibrium(a, nodes, [0.0, top])
    assert rms_error(u, CONTROL(nodes.x, nodes.y)) < 5e-5


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


def test_every_operator_runs_on_the_smooth_band_and_is_the_jumps_at_delta_zero():
    # Stiff note §3.1 and §3.8 decision 1: the smooth band keeps the jump's
    # protocol, so the naive, direct and aware operators run on it unchanged
    # and δ = 0 is the jump bit for bit. At δ > 0 the aware operator is the
    # "δ = 0 construction on a smooth edge": its crossing rows read the
    # pieces' data and are the jump's to the bit, its direct rows read the
    # smooth α and ∇α. Beyond TANH_REACH δ α is the piece to the bit and ∇α
    # the blend's true derivative, 7e-16 at 20δ (E3.2's tails), so the rows
    # there move by rounding only.
    nodes, st = aware_set(1250)
    band = DOMAIN.material
    builders = {
        "naive": lambda m: naive_operator(nodes, m, st),
        "direct": lambda m: direct_operator(nodes, m, st),
        "aware": lambda m: interface_aware_operator(nodes, m, st),
    }
    jump = {name: build(band) for name, build in builders.items()}
    for name, build in builders.items():
        assert abs(build(SmoothBand(band, 0.0)) - jump[name]).max() == 0.0

    delta = 0.01
    aware = builders["aware"](SmoothBand(band, delta))
    rows = np.abs((aware - jump["aware"]).toarray()).max(axis=1)
    group = next(g for g in st.groups if g.kind == INTERFACE_KIND)
    crossing = np.zeros(nodes.n, dtype=bool)
    crossing[group.rows[interface_crossings(nodes, band, group.index)]] = True
    far = np.all(
        [np.abs(nodes.y - c) >= TANH_REACH * delta for c in (0.6, 0.8)], axis=0
    )
    assert crossing.sum() > 300 and far.sum() > 400
    assert np.all(rows[crossing] == 0.0) and rows[far].max() < 1e-13
    near = ~crossing & ~far
    assert near.sum() > 50 and rows[near].max() > 0.0
    assert abs(builders["naive"](SmoothBand(band, delta)) - jump["naive"]).max() > 1.0


# --- E4.5 (#36): the seeded-row rule, the seed operator and the dispatch ----


def test_seeded_rows_is_the_crossing_test_at_delta_zero_and_grows_with_delta():
    # §3.3's rule: the span of a stencil's signed distances meets the edge's
    # support [−20δ, 20δ]. At δ = 0 that is the straddling test, whatever the
    # spacing; at δ = 0.04 on case 1, 20δ = 0.8 covers the strip (H8).
    nodes, st = aware_set(1250)
    band = DOMAIN.material
    index, _ = knn(nodes.xy, INTERIOR.size)
    crossing = interface_crossings(nodes, band, index)
    np.testing.assert_array_equal(seeded_rows(nodes, band, index), crossing)
    np.testing.assert_array_equal(
        seeded_rows(nodes, SmoothBand(band, 0.0), index), crossing
    )
    counts = [
        seeded_rows(nodes, SmoothBand(band, r * nodes.h), index).sum()
        for r in (1 / 64, 1 / 8, 1 / 2, 1)
    ]
    assert counts[0] == crossing.sum()
    assert crossing.sum() < counts[1] < counts[2] < counts[3] == nodes.n
    assert seeded_rows(nodes, SmoothBand(band, 0.04), index).all()
    # A stencil that straddles an edge far thinner than the spacing is seeded
    # although no node of it is within 20δ.
    thin = SmoothBand(band, nodes.h / 1000)
    assert seeded_rows(nodes, thin, index).sum() == crossing.sum()
    assert not seeded_rows(nodes, ONE, index).any()


def test_a_reach_group_holds_every_stencil_that_sees_the_edge():
    # The rule one size up: with reach on, no stencil outside the interface
    # group sees the edge, so no 42 / 5 stencil ever does.
    nodes, _ = node_set(1250)
    medium = SmoothBand(DOMAIN.material, nodes.h / 8)
    domain = replace(DOMAIN, material=medium)
    st = build_stencils(nodes, domain, interface=BOUNDARY, reach=TANH_REACH)
    index, _ = knn(nodes.xy, INTERIOR.size)
    np.testing.assert_array_equal(
        st.near_interface, seeded_rows(nodes, medium, index, TANH_REACH)
    )
    for g in st.groups:
        if g.kind != INTERFACE_KIND:
            assert not seeded_rows(nodes, medium, g.index, TANH_REACH).any()
    plain = build_stencils(nodes, domain, interface=BOUNDARY)
    assert st.near_interface.sum() > plain.near_interface.sum()


def test_seed_operator_is_the_jump_aware_operator_at_delta_zero():
    # H2 at the operator level: every seeded row is E2.3's row (the seed span
    # is the translated basis's and φ₀₁ is Warp.apply), and the rows that see
    # nothing are the direct operator's in both.
    nodes, st = aware_set(1250)
    band = DOMAIN.material
    aware = interface_aware_operator(nodes, band, st)
    seeds = seed_operator(nodes, band, st)
    assert seeds.nnz == aware.nnz
    assert abs(seeds - aware).max() < 1e-11 * abs(aware).max()
    assert abs(seed_operator(nodes, SmoothBand(band, 0.0), st) - seeds).max() == 0.0
    # Without interfaces there is nothing to seed.
    assert (
        abs(seed_operator(nodes, ONE, st) - direct_operator(nodes, ONE, st)).max()
        == 0.0
    )


def test_seed_operator_replaces_exactly_the_rows_that_see_the_edge():
    nodes, _ = node_set(1250)
    medium = SmoothBand(DOMAIN.material, nodes.h / 8)
    domain = replace(DOMAIN, material=medium)
    st = build_stencils(nodes, domain, interface=BOUNDARY, reach=TANH_REACH)
    op = seed_operator(nodes, medium, st)
    direct = direct_operator(nodes, medium, st)
    moved = np.asarray(abs(op - direct).sum(axis=1)).ravel() > 0
    group = next(g for g in st.groups if g.kind == INTERFACE_KIND)
    seeded = np.zeros(nodes.n, dtype=bool)
    seeded[group.rows[seeded_rows(nodes, medium, group.index, TANH_REACH)]] = True
    np.testing.assert_array_equal(moved, seeded)
    assert 300 < seeded.sum() < nodes.n
    assert np.all(np.diff(op.indptr)[group.rows] == BOUNDARY.size)


def test_build_operator_dispatches_by_name():
    nodes, st = aware_set(1250)
    band = DOMAIN.material
    assert set(OPERATOR_MODES) == {"naive", "direct", "jump-aware", "seeds"}
    plain = node_set(1250)[1]
    assert (
        abs(
            build_operator(nodes, band, plain, "naive")
            - naive_operator(nodes, band, plain)
        ).max()
        == 0.0
    )
    assert (
        abs(
            build_operator(nodes, band, st, "jump-aware", warp=False)
            - interface_aware_operator(nodes, band, st, warp=False)
        ).max()
        == 0.0
    )
    assert OPERATOR_MODES["seeds"] is seed_operator
    assert (
        abs(build_operator(nodes, ONE, st) - direct_operator(nodes, ONE, st)).max()
        == 0.0
    )
    with pytest.raises(ValueError, match="unknown operator"):
        build_operator(nodes, band, st, "seed")
