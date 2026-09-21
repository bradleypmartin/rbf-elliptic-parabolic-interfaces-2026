"""Appendix B's diagonal-dominance preconditioner (E2.8)."""

import numpy as np
import pytest
import scipy.sparse as sp

from heat_interfaces.heat2d.domain import build_node_set, case3
from heat_interfaces.heat2d.neighbors import knn
from heat_interfaces.heat2d.operators import build_stencils, interface_aware_operator
from heat_interfaces.heat2d.precondition import (
    diagonal_dominance_ratio,
    dominance_preconditioner,
    neighbour_table,
)
from heat_interfaces.heat2d.rbf import ITERATIVE
from heat_interfaces.heat2d.solve import reduced_system


def edge(x, y):
    return np.sin(6 * np.pi * x)


VALUES = [edge, edge, 0.0]


def appendix_b_matrix() -> sp.csr_array:
    """Eq. 93's ``D_Loc``: rows ``u_{m−3} … u_{m+3}``, columns ``u_{m−5} … u_{m+5}``.

    The 1-D layer at ``α = 1/50`` sits between ``u_m`` and ``u_{m+1}``; the
    four rows the print leaves out are the standard five-point row, which no
    cancellation of the worked example reaches.
    """
    standard = [-0.08, 1.33, -2.50, 1.33, -0.08]
    printed = {
        -3: [-0.08, 1.33, -2.50, 1.33, -0.08, 0, 0, 0, 0, 0, 0],
        -2: [0, -0.07, 1.25, -2.34, 1.16, -0.00, 0, 0, 0, 0, 0],
        -1: [0, 0, -0.05, 1.09, -1.14, 0.14, -0.05, 0, 0, 0, 0],
        0: [0, 0, 0, -0.05, 0.17, -1.45, 1.41, -0.08, 0, 0, 0],
        1: [0, 0, 0, 0, -0.00, 1.17, -2.34, 1.24, -0.07, 0, 0],
        2: [0, 0, 0, 0, 0, -0.08, 1.33, -2.50, 1.33, -0.08, 0],
        3: [0, 0, 0, 0, 0, 0, -0.08, 1.33, -2.50, 1.33, -0.08],
    }
    a = np.zeros((11, 11))
    for t in range(-5, 6):
        i = 5 + t
        if t in printed:
            a[i] = printed[t]
            continue
        for s, w in zip(range(-2, 3), standard, strict=True):
            if 0 <= i + s < 11:
                a[i, i + s] = w
    return sp.csr_array(a)


def test_diagonal_dominance_ratio_by_hand():
    a = sp.csr_array(np.array([[4.0, -1.0, -1.0], [0.5, 2.0, 0.0], [0.0, 0.0, 3.0]]))
    np.testing.assert_allclose(diagonal_dominance_ratio(a), [2.0, 4.0, np.inf])
    with pytest.raises(ValueError):
        diagonal_dominance_ratio(sp.csr_array(np.ones((2, 3))))


def test_appendix_b_worked_example_is_reproduced_step_by_step():
    """Eq. 93–97: DDR 0.846 → 0.808 → 0.872 → 0.936 → 0.968, and ``P_m`` is eq. 98.

    The matrix is transcribed to two decimals, so ±0.01 on every number.
    """
    a = appendix_b_matrix()
    assert diagonal_dominance_ratio(a)[5] == pytest.approx(0.846, abs=0.01)
    neighbours = -np.ones((11, 6), dtype=int)
    neighbours[5] = [6, 4, 7, 3, 8, 2]  # m+1, m−1, m+2, m−2, m+3, m−3
    expected = {1: 0.808, 2: 0.872, 4: 0.936, 6: 0.968}
    for k, ddr in expected.items():
        p = dominance_preconditioner(a, neighbours[:, :k], rows=np.array([5]), sweeps=1)
        assert diagonal_dominance_ratio(p @ a)[5] == pytest.approx(ddr, abs=0.01)
    p_m = p[[5], :].toarray().ravel()
    np.testing.assert_allclose(
        p_m[2:9], [0.02, 0.05, 0.15, 1.0, 0.60, 0.27, 0.13], atol=0.01
    )
    assert np.all(p_m[:2] == 0.0) and np.all(p_m[9:] == 0.0)
    # every other row of P is the identity's
    identity_rows = p.toarray()
    identity_rows[5] = np.eye(11)[5]
    np.testing.assert_array_equal(identity_rows, np.eye(11))


def test_preconditioner_input_checks():
    a = appendix_b_matrix()
    with pytest.raises(ValueError):
        dominance_preconditioner(a, np.zeros((4, 2), dtype=int))
    with pytest.raises(ValueError):
        dominance_preconditioner(a, -np.ones((11, 2), dtype=int), sweeps=0)
    # a table of skips leaves the identity behind
    p = dominance_preconditioner(a, -np.ones((11, 3), dtype=int))
    np.testing.assert_array_equal(p.toarray(), np.eye(11))


_case = {}


def case3_system():
    """A 1250-node case-3 reduced system on 19-node / degree-3 stencils, shared."""
    if not _case:
        domain = case3()
        nodes = build_node_set(domain, 1250, iterations=30)
        stencils = build_stencils(
            nodes, domain, interior=ITERATIVE, boundary=ITERATIVE, interface=ITERATIVE
        )
        op = interface_aware_operator(nodes, domain.material, stencils)
        _case["nodes"], _case["stencils"] = nodes, stencils
        _case["system"] = reduced_system(op, nodes, VALUES)
    return _case["nodes"], _case["stencils"], _case["system"]


def test_neighbour_table_maps_the_nearest_nodes_to_interior_indices():
    nodes, _, system = case3_system()
    table = neighbour_table(nodes, system.interior, k=12)
    assert table.shape == (system.n, 12)
    index, _ = knn(nodes.xy, 13)
    for i in (0, system.n // 2, system.n - 1):
        node = system.interior[i]
        for t, neighbour in enumerate(index[node, 1:]):
            if nodes.dirichlet[neighbour]:
                assert table[i, t] == -1
            else:
                assert system.interior[table[i, t]] == neighbour
    with pytest.raises(ValueError):
        neighbour_table(nodes, system.interior, k=0)


def test_preconditioned_case3_system_keeps_its_solution_and_gains_dominance():
    nodes, stencils, system = case3_system()
    before = diagonal_dominance_ratio(system.a)
    group = stencils.near_interface[system.interior]
    assert before[group].min() < 0.5 < before[~group].min()
    table = neighbour_table(nodes, system.interior)
    p = dominance_preconditioner(system.a, table)
    pre = system.left_preconditioned(p)
    after = diagonal_dominance_ratio(pre.a)
    assert after.min() > 1.5 * before.min()
    assert np.median(after) > np.median(before) + 0.1
    u = system.solve_direct()
    u_pre = pre.solve_direct()
    assert np.linalg.norm(u_pre - u) / np.linalg.norm(u) < 1e-10
    # the interface group alone leaves the rest of the matrix untouched
    p_group = dominance_preconditioner(system.a, table, rows=np.flatnonzero(group))
    rows_kept = np.flatnonzero(~group)
    assert (
        p_group[rows_kept] != sp.eye_array(system.n, format="csr")[rows_kept]
    ).nnz == 0
