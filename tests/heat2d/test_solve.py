"""The reduced interior system and the iterative solvers of §5.4.4 (E2.8)."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse.linalg import bicgstab

from heat_interfaces.heat2d.domain import (
    Band,
    Circle,
    Constant2D,
    build_node_set,
    case3,
)
from heat_interfaces.heat2d.operators import (
    build_stencils,
    dirichlet_system,
    dirichlet_values,
    laplacian_operator,
)
from heat_interfaces.heat2d.rbf import ITERATIVE
from heat_interfaces.heat2d.solve import (
    ILU_ORDERING,
    ilu_preconditioner,
    reduced_system,
    solve_equilibrium,
    solve_iterative,
)


def edge(x, y):
    return np.sin(6 * np.pi * x)


VALUES = [edge, edge, 0.0]

_control = {}


def control():
    """The §5.4.4 control on 1250 nodes: case 3's layout, ``α ≡ 1``, 19 nodes."""
    if not _control:
        band = Band(Circle(0.349), Circle(0.35), Constant2D(1.0), Constant2D(1.0))
        domain = replace(case3(), material=band)
        nodes = build_node_set(domain, 1250, iterations=30)
        stencils = build_stencils(nodes, domain, interior=ITERATIVE, boundary=ITERATIVE)
        _control["nodes"] = nodes
        _control["op"] = laplacian_operator(nodes, stencils)
    return _control["nodes"], _control["op"]


def test_reduced_system_solves_the_same_problem_as_the_identity_row_form():
    nodes, op = control()
    system = reduced_system(op, nodes, VALUES)
    assert system.n == int((~nodes.dirichlet).sum())
    u_direct = solve_equilibrium(op, nodes, VALUES)
    u = system.solve_direct()
    assert np.linalg.norm(u - u_direct) / np.linalg.norm(u_direct) < 1e-10
    # the reduced form carries the boundary values exactly, the identity rows to 1e-14
    on = nodes.dirichlet
    np.testing.assert_allclose(u[on], u_direct[on], atol=1e-12)
    # a forcing term is carried through
    f = np.ones(nodes.n)
    forced = reduced_system(op, nodes, VALUES, forcing=f).solve_direct()
    assert np.allclose(forced, solve_equilibrium(op, nodes, VALUES, forcing=f))


@pytest.mark.parametrize("method", ["gmres", "bicgstab"])
def test_iterative_solvers_reach_the_direct_solution(method):
    nodes, op = control()
    system = reduced_system(op, nodes, VALUES)
    u = system.solve_direct()
    result = solve_iterative(system, method, rtol=1e-9)
    assert result.converged and result.residual <= 1e-9
    assert result.iterations > 1
    assert np.linalg.norm(result.u - u) / np.linalg.norm(u) < 1e-7
    np.testing.assert_array_equal(result.u[nodes.dirichlet], u[nodes.dirichlet])


def test_restarted_gmres_counts_inner_iterations_and_respects_the_cap():
    nodes, op = control()
    system = reduced_system(op, nodes, VALUES)
    full = solve_iterative(system, "gmres", rtol=1e-9)
    restarted = solve_iterative(system, "gmres", rtol=1e-9, restart=20)
    assert restarted.converged and restarted.iterations > full.iterations
    capped = solve_iterative(system, "gmres", rtol=1e-12, maxiter=10)
    assert not capped.converged and capped.iterations == 10
    with pytest.raises(ValueError):
        solve_iterative(system, "cg")


def test_ilu_preconditioner_cuts_the_iteration_count():
    nodes, op = control()
    system = reduced_system(op, nodes, VALUES)
    assert ILU_ORDERING == "MMD_AT_PLUS_A"
    m, seconds = ilu_preconditioner(system)
    assert seconds >= 0.0
    # SuperLU's own ordering is still reachable (it fails only past 10,000 nodes)
    m_colamd, _ = ilu_preconditioner(system, permc_spec="COLAMD")
    assert solve_iterative(system, "bicgstab", m=m_colamd).converged
    plain = solve_iterative(system, "bicgstab")
    fast = solve_iterative(system, "bicgstab", m=m)
    assert fast.converged and fast.iterations < plain.iterations / 5
    u = system.solve_direct()
    assert np.linalg.norm(fast.u - u) / np.linalg.norm(u) < 1e-6


def test_bicgstab_breaks_down_on_the_identity_row_form():
    """The trap the module docstring names: ``r̂ = b`` on the Dirichlet rows only."""
    nodes, op = control()
    g = dirichlet_values(nodes, VALUES)
    a, b = dirichlet_system(op, nodes.dirichlet, g)
    count = [0]
    _, info = bicgstab(
        a, b, rtol=1e-8, atol=0.0, callback=lambda x: count.__setitem__(0, count[0] + 1)
    )
    assert info < 0 and count[0] <= 1
