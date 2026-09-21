"""Interface-aware resampling of a fine solution (E2.6) and the cached ``Reference``."""

import numpy as np
import pytest

from heat_interfaces.heat2d.domain import build_node_set, case1
from heat_interfaces.heat2d.exact import case1_exact
from heat_interfaces.heat2d.fd4 import cartesian_grid
from heat_interfaces.heat2d.operators import (
    build_stencils,
    interface_aware_operator,
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
