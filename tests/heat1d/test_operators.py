import numpy as np
import pytest

from heat_interfaces.heat1d.domain import (
    Constant,
    Grid1D,
    PiecewiseAlpha,
    Smooth,
    dissertation_alpha,
    equispaced_grid,
    jump_alpha,
)
from heat_interfaces.heat1d.operators import (
    alpha_matrix,
    direct_operator,
    dx_matrix,
    dxx_matrix,
    naive_operator,
)


def test_dx_is_exact_on_quartics_including_the_one_sided_rows():
    g = equispaced_grid(41)
    dx = dx_matrix(g)
    for p in range(5):
        np.testing.assert_allclose(dx @ g.x**p, p * g.x ** max(p - 1, 0), atol=1e-10)
    assert dx.nnz == 5 * g.n


def test_dx_and_dxx_are_fourth_order_on_a_smooth_function():
    errs = []
    for n in (81, 161, 321):
        g = equispaced_grid(n)
        u = np.sin(2 * g.x)
        e1 = np.max(np.abs(dx_matrix(g) @ u - 2 * np.cos(2 * g.x)))
        e2 = np.max(np.abs(dxx_matrix(g) @ u + 4 * np.sin(2 * g.x)))
        errs.append((e1, e2))
    errs = np.array(errs)
    rates = np.log2(errs[:-1] / errs[1:])
    # The one-sided second-derivative rows are third order; the max norm sees them.
    assert np.all(rates[:, 0] > 3.8)
    assert np.all(rates[:, 1] > 2.8)


def test_alpha_matrix_is_diagonal_in_alpha():
    g = equispaced_grid(101)
    m = dissertation_alpha()
    a = alpha_matrix(g, m)
    np.testing.assert_allclose(a.diagonal(), m.alpha(g.x))
    assert a.nnz == g.n


def test_constant_alpha_reduces_both_operators_to_alpha_times_a_laplacian():
    g = equispaced_grid(31)
    m = PiecewiseAlpha((), (Constant(0.7),))
    dx = dx_matrix(g)
    np.testing.assert_allclose(
        naive_operator(g, m).toarray(), 0.7 * (dx @ dx).toarray(), atol=1e-12
    )
    np.testing.assert_allclose(
        direct_operator(g, m).toarray(), 0.7 * dxx_matrix(g).toarray(), atol=1e-12
    )


def test_operators_are_fourth_order_when_alpha_is_smooth():
    # alpha = 2 + cos(pi x), u = sin(pi x): (alpha u')' is known in closed form.
    alpha = Smooth(
        lambda x: 2 + np.cos(np.pi * x), (lambda x: -np.pi * np.sin(np.pi * x),)
    )
    m = PiecewiseAlpha((), (alpha,))
    errs = {"naive": [], "direct": []}
    for n in (41, 81, 161):
        g = equispaced_grid(n)
        x = g.x
        u = np.sin(np.pi * x)
        exact = -(np.pi**2) * (2 + np.cos(np.pi * x)) * np.sin(np.pi * x) - np.pi**2 * (
            np.sin(np.pi * x) * np.cos(np.pi * x)
        )
        errs["naive"].append(np.abs(naive_operator(g, m) @ u - exact))
        errs["direct"].append(np.abs(direct_operator(g, m) @ u - exact))

    def rates(name, rows):
        e = np.array([np.max(err[rows]) for err in errs[name]])
        return np.log2(e[:-1] / e[1:])

    # The direct stencil is fourth order everywhere but the two one-sided rows.
    assert np.all(rates("direct", slice(2, -2)) > 3.8)
    # Dx A Dx at rows 2-3 differentiates the one-sided Dx rows, whose error
    # constants differ from the centred ones: those two rows lose an order.
    assert np.all(rates("naive", slice(4, -4)) > 3.8)
    assert np.all(rates("naive", slice(2, 4)) > 2.8)
    assert np.all(rates("naive", slice(2, 4)) < 3.8)


def test_naive_operator_sees_the_jump_and_the_direct_one_does_not():
    g = equispaced_grid(41)
    m = jump_alpha(1.0, 0.25)
    # The straight line is annihilated by the direct stencil but not by Dx A Dx.
    line = 0.5 * (1 - g.x)
    assert np.max(np.abs(direct_operator(g, m) @ line)) < 1e-12
    assert np.max(np.abs(naive_operator(g, m) @ line)) > 1.0


@pytest.mark.parametrize("n", [5, 6, 9])
def test_small_grids_still_assemble(n):
    g = equispaced_grid(n)
    assert dx_matrix(g).shape == (n, n)
    assert naive_operator(g, jump_alpha(1.0, 2.0)).shape == (n, n)


def test_alpha_matrix_gives_a_nudged_interface_node_the_owner_value():
    m = dissertation_alpha()
    g = equispaced_grid(101)
    x = g.x.copy()
    x[75] += 1e-16
    a = alpha_matrix(Grid1D(n=g.n, x=x, h=g.h), m)
    assert a.diagonal()[75] == float(m.alpha(0.5))  # the layer owns x = 0.5
    assert a.diagonal()[75] != float(m.alpha(x[75]))  # unsnapped it would be 1
    np.testing.assert_allclose(a.diagonal(), alpha_matrix(g, m).diagonal())
