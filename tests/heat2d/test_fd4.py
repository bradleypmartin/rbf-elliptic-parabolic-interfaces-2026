"""The Cartesian FD4 baseline: the grid, the periodic and one-sided derivative
matrices, ``Dx A Dx + Dy A Dy``, fourth order on the control and first on case 1."""

import numpy as np
import pytest

from heat_interfaces.heat2d.domain import (
    DIRICHLET,
    FREE,
    STRIP,
    Band,
    Constant2D,
    Domain,
    FlatLine,
    build_node_set,
    case1,
    case3,
)
from heat_interfaces.heat2d.exact import case1_exact, control_exact
from heat_interfaces.heat2d.fd4 import (
    cartesian_grid,
    fd4_dx,
    fd4_dy,
    fd4_operator,
    grid_size,
)
from heat_interfaces.heat2d.solve import rms_error, solve_equilibrium

CONTROL = Domain(
    Band(FlatLine(0.6), FlatLine(0.8), Constant2D(1.0), Constant2D(1.0)), (), STRIP
)


def top(x, y):
    return np.sin(2 * np.pi * x)


def rate(errs):
    """Order per halving of h; the count quadruples between consecutive entries."""
    e = np.asarray(errs)
    return np.log(e[:-1] / e[1:]) / np.log(2.0)


def test_grid_size_and_layout():
    assert [grid_size(n) for n in (1250, 5000, 160000, 3)] == [35, 70, 400, 5]
    g = cartesian_grid(case1(), 1250)
    m = 35
    assert g.n == m * (m + 1) and g.h == 1.0 / m
    i, j = np.arange(g.n) % m, np.arange(g.n) // m
    np.testing.assert_array_equal(g.x, i / m)
    np.testing.assert_array_equal(g.y, j / m)
    assert (
        g.x.min() == 0.0 and g.x.max() < 1.0 and g.y.min() == 0.0 and g.y.max() == 1.0
    )
    assert set(g.kind[j == 0]) == set(g.kind[j == m]) == {DIRICHLET}
    assert set(g.kind[(j > 0) & (j < m)]) == {FREE}
    assert [(r.curve, len(r.index)) for r in g.dirichlet_rows] == [(0, m), (1, m)]
    np.testing.assert_array_equal(g.boundary_index[g.dirichlet_rows[1].index], 1)
    assert g.straddle_rows == ()


def test_fd4_dx_is_periodic_and_fourth_order():
    errs = []
    for m in (20, 40):
        x = np.arange(m) / m
        d = fd4_dx(m, 1.0 / m)
        assert d.shape == (m, m) and d.nnz == 5 * m
        assert d[0, m - 1] != 0.0 and d[m - 1, 0] != 0.0  # wraps
        errs.append(
            np.abs(d @ np.sin(2 * np.pi * x) - 2 * np.pi * np.cos(2 * np.pi * x)).max()
        )
    assert errs[0] / errs[1] == pytest.approx(16.0, rel=0.05)


def test_fd4_dy_is_exact_on_quartics_with_one_sided_end_rows():
    levels, h = 21, 0.05
    y = np.arange(levels) * h
    d = fd4_dy(levels, h)
    np.testing.assert_allclose(d @ y**4, 4 * y**3, atol=1e-11)
    assert set(d[[0], :].indices) == {0, 1, 2, 3, 4}
    assert set(d[[levels - 1], :].indices) == {levels - 5 + k for k in range(5)}


def test_fd4_operator_is_fourth_order_on_the_control_and_first_on_case_1():
    for domain, exact, order, floor in (
        (CONTROL, control_exact(), 3.5, 0.0),
        (case1(), case1_exact(), 0.7, 5e-4),
    ):
        errs = []
        for n in (1250, 5000, 20000):
            g = cartesian_grid(domain, n)
            op = fd4_operator(g, domain.material)
            assert op.nnz <= 17 * g.n
            u = solve_equilibrium(op, g, [0.0, top])
            errs.append(rms_error(u, exact(g.x, g.y)))
        assert np.all(rate(errs) > order), errs
        assert np.all(np.asarray(errs) >= floor), errs
        if floor:
            assert np.all(rate(errs) < 1.8), errs


def test_fd4_refuses_case_3_and_foreign_node_sets():
    with pytest.raises(NotImplementedError, match="plain strip"):
        cartesian_grid(case3(), 1250)
    nodes = build_node_set(case1(), 900, iterations=5)
    with pytest.raises(ValueError, match="cartesian_grid"):
        fd4_operator(nodes, case1().material)
