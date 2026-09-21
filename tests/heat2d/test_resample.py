"""Interface-aware resampling of a fine solution (E2.6) and the cached ``Reference``."""

from dataclasses import replace

import numpy as np
import pytest

from heat_interfaces.heat2d.domain import (
    Band,
    Circle,
    Constant2D,
    build_node_set,
    case1,
    case3,
)
from heat_interfaces.heat2d.exact import case1_exact, ring_exact
from heat_interfaces.heat2d.fd4 import cartesian_grid
from heat_interfaces.heat2d.interface import interpolation_weights
from heat_interfaces.heat2d.neighbors import knn
from heat_interfaces.heat2d.operators import (
    build_stencils,
    interface_aware_operator,
    interface_crossings,
)
from heat_interfaces.heat2d.rbf import BOUNDARY
from heat_interfaces.heat2d.resample import Reference, reference_solution, resample
from heat_interfaces.heat2d.solve import rms_error, solve_equilibrium

DOMAIN, EXACT = case1(), case1_exact()


def top(x, y):
    return np.sin(2 * np.pi * x)


_fine = {}


def fine_set():
    """A 10,000-node case-1 set carrying the analytic solution, shared by the tests."""
    if not _fine:
        nodes = build_node_set(DOMAIN, 10000)
        _fine["nodes"] = nodes
        _fine["u"] = EXACT(nodes.x, nodes.y)
        _fine["aware"] = build_stencils(nodes, DOMAIN, interface=BOUNDARY)
        _fine["blind"] = build_stencils(nodes, DOMAIN)
    return _fine


def test_aware_resampling_reads_the_kinked_solution_and_the_blind_one_does_not():
    f = fine_set()
    coarse = build_node_set(DOMAIN, 1250, iterations=20)
    ref = EXACT(coarse.x, coarse.y)
    aware = resample(
        f["u"], f["nodes"], f["aware"], DOMAIN.material, coarse.x, coarse.y
    )
    blind = resample(
        f["u"], f["nodes"], f["blind"], DOMAIN.material, coarse.x, coarse.y
    )
    e_aware, e_blind = rms_error(aware, ref), rms_error(blind, ref)
    assert e_aware < 3e-8, e_aware
    assert e_blind > 1e-6 and e_blind > 50 * e_aware, (e_aware, e_blind)
    # The blind interpolant is wrong only where its stencil crosses the band's edges.
    far = np.abs(coarse.y - 0.6) > 0.05
    far &= np.abs(coarse.y - 0.8) > 0.05
    assert np.abs(blind - ref)[far].max() < 1e-7


def test_resampling_is_the_identity_on_the_nodes_and_covers_a_grid_and_the_seam():
    f = fine_set()
    sub = np.arange(0, f["nodes"].n, 53)
    back = resample(
        f["u"],
        f["nodes"],
        f["aware"],
        DOMAIN.material,
        f["nodes"].x[sub],
        f["nodes"].y[sub],
    )
    assert np.abs(back - f["u"][sub]).max() < 1e-12
    grid = cartesian_grid(DOMAIN, 1250)
    on_grid = resample(f["u"], f["nodes"], f["aware"], DOMAIN.material, grid.x, grid.y)
    assert rms_error(on_grid, EXACT(grid.x, grid.y)) < 3e-8
    # x = 1 is the seam: wrapped, not refused.
    seam = resample(
        f["u"],
        f["nodes"],
        f["aware"],
        DOMAIN.material,
        np.array([1.0]),
        np.array([0.5]),
    )
    assert seam[0] == pytest.approx(float(EXACT(0.0, 0.5)), abs=1e-8)


def test_points_sharing_a_crossing_centre_match_the_one_point_systems():
    # Several points nearest one fine node whose stencil crosses an interface
    # go through one translated-basis system; each must equal its own.
    f = fine_set()
    nodes, st = f["nodes"], f["aware"]
    group = st.groups[-1]
    cross = interface_crossings(nodes, DOMAIN.material, group.index)
    row = group.index[np.flatnonzero(cross)[7]]
    rng = np.random.default_rng(11)
    theta = rng.uniform(0.0, 2 * np.pi, 6)
    px = nodes.x[row[0]] + 0.2 * nodes.h * np.cos(theta)
    py = nodes.y[row[0]] + 0.2 * nodes.h * np.sin(theta)
    assert np.all(knn(nodes.xy, 1, query=np.column_stack([px, py]))[0][:, 0] == row[0])
    got = resample(f["u"], nodes, st, DOMAIN.material, px, py)
    for k in range(6):
        w = interpolation_weights(
            nodes.xy[row],
            DOMAIN.material,
            group.spec.degree,
            (px[k : k + 1], py[k : k + 1]),
        )
        assert got[k] == pytest.approx(float(w[0] @ f["u"][row]), abs=1e-12)
    assert np.abs(got - EXACT(px, py)).max() < 1e-7


def test_resample_refuses_bad_shapes():
    f = fine_set()
    with pytest.raises(ValueError, match="matching 1-D"):
        resample(
            f["u"], f["nodes"], f["aware"], DOMAIN.material, np.zeros(3), np.zeros(2)
        )
    with pytest.raises(ValueError, match="one value per node"):
        resample(f["u"][:-1], f["nodes"], f["aware"], DOMAIN.material, [0.5], [0.5])


def test_reference_solution_matches_a_direct_solve_and_round_trips(tmp_path):
    ref = reference_solution(DOMAIN, 1250, [0.0, top], iterations=20)
    nodes = build_node_set(DOMAIN, 1250, iterations=20)
    op = interface_aware_operator(
        nodes, DOMAIN.material, build_stencils(nodes, DOMAIN, interface=BOUNDARY)
    )
    np.testing.assert_array_equal(ref.u, solve_equilibrium(op, nodes, [0.0, top]))
    assert ref.meta["interface_group"] > 300 and ref.meta["iterations"] == 20
    assert set(ref.meta["seconds"]) == {"nodes", "operator", "solve"}
    path = tmp_path / "ref.npz"
    ref.save(path)
    assert path.exists() and path.with_suffix(".json").exists()
    back = Reference.load(path)
    np.testing.assert_array_equal(back.u, ref.u)
    np.testing.assert_array_equal(back.nodes.xy, ref.nodes.xy)
    np.testing.assert_array_equal(back.nodes.dirichlet, ref.nodes.dirichlet)
    assert back.nodes.h == ref.nodes.h and back.meta["n"] == 1250
    assert back.nodes.straddle_rows == () and back.nodes.dirichlet_rows == ()
    st = back.stencils(DOMAIN)
    assert st.near_interface.sum() == ref.meta["interface_group"]
    x, y = np.array([0.3, 0.7]), np.array([0.61, 0.25])
    np.testing.assert_array_equal(
        resample(back.u, back.nodes, st, DOMAIN.material, x, y),
        resample(ref.u, ref.nodes, ref.stencils(DOMAIN), DOMAIN.material, x, y),
    )


def test_aware_resampling_is_exact_through_the_ring_at_its_1500_contrast():
    # E2.6's check was flat case 1; the E2.6 note on #21 asked for the same on a
    # profile exact across the 0.001-wide ring. A matched radial quadratic
    # (div(α grad u) = 4, α = 1/1500 on the ring) lies in the translated basis,
    # so the aware read is exact to rounding; the blind read is off by the jump.
    band = Band(Circle(0.349), Circle(0.35), Constant2D(1 / 1500), Constant2D(1.0))
    domain = replace(case3(), material=band)
    fine = build_node_set(domain, 5000, iterations=20)

    def quadratic(x, y):
        r = np.hypot(x - 0.5, y - 0.5)
        a = 1.0 / 1500.0
        b1 = 0.349**2 - 0.349**2 / a
        b2 = 0.35**2 / a + b1 - 0.35**2
        return np.where(r < 0.349, r**2, np.where(r <= 0.35, r**2 / a + b1, r**2 + b2))

    u = quadratic(fine.x, fine.y)
    coarse = build_node_set(case3(), 1250, iterations=20)
    near = np.abs(np.hypot(coarse.x - 0.5, coarse.y - 0.5) - 0.3495) < 0.1
    x, y = coarse.x[near], coarse.y[near]
    ref = quadratic(x, y)
    aware = resample(
        u, fine, build_stencils(fine, domain, interface=BOUNDARY), band, x, y
    )
    blind = resample(u, fine, build_stencils(fine, domain), band, x, y)
    assert np.abs(aware - ref).max() < 1e-9, np.abs(aware - ref).max()
    assert np.abs(blind - ref).max() > 1e-2
    # A point inside the ring itself (an FD4 grid point would land there) is
    # read through the ring's own translated basis.
    s = np.linspace(0.0, 1.0, 7, endpoint=False)
    px, py = 0.5 + 0.3495 * np.cos(2 * np.pi * s), 0.5 + 0.3495 * np.sin(2 * np.pi * s)
    inside = resample(
        u, fine, build_stencils(fine, domain, interface=BOUNDARY), band, px, py
    )
    assert np.abs(inside - quadratic(px, py)).max() < 1e-9


def test_ring_mode_reads_to_the_stencils_truncation_near_the_ring():
    # The harmonic mode R(r) cos 2θ is not in the basis, so this is the read's
    # truncation at a 1500 : 1 contrast, the number the case-3 driver tabulates.
    band = Band(Circle(0.349), Circle(0.35), Constant2D(1 / 1500), Constant2D(1.0))
    domain = replace(case3(), material=band)
    exact = ring_exact()
    fine = build_node_set(domain, 10000, iterations=20)
    u = exact(fine.x, fine.y)
    coarse = build_node_set(case3(), 2500, iterations=20)
    near = np.abs(np.hypot(coarse.x - 0.5, coarse.y - 0.5) - 0.3495) < 0.1
    x, y = coarse.x[near], coarse.y[near]
    aware = resample(
        u, fine, build_stencils(fine, domain, interface=BOUNDARY), band, x, y
    )
    blind = resample(u, fine, build_stencils(fine, domain), band, x, y)
    e_aware, e_blind = rms_error(aware, exact(x, y)), rms_error(blind, exact(x, y))
    assert e_aware < 1e-6, e_aware
    assert e_blind > 100 * e_aware, (e_aware, e_blind)
