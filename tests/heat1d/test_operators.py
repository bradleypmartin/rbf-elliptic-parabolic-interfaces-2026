import numpy as np
import pytest

from heat_interfaces.heat1d.domain import (
    DISSERTATION_BC,
    Constant,
    Grid1D,
    PiecewiseAlpha,
    Smooth,
    dissertation_alpha,
    equispaced_grid,
    jump_alpha,
)
from heat_interfaces.heat1d.exact import equilibrium_exact
from heat_interfaces.heat1d.operators import (
    alpha_matrix,
    direct_operator,
    dx_matrix,
    dxx_matrix,
    jump_aware_operator,
    naive_operator,
    straddling_windows,
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


def test_jump_aware_operator_equals_the_direct_one_off_the_straddling_rows():
    m = dissertation_alpha()
    for n, per_interface in ((101, 3), (100, 4)):
        g = equispaced_grid(n)
        windows = straddling_windows(g, m)
        assert len(windows) == 2 * per_interface
        assert all(seen in ([0], [1]) for _, _, seen in windows)
        assert all(lo == i - 2 for i, lo, _ in windows)
        diff = np.abs((jump_aware_operator(g, m) - direct_operator(g, m)).toarray())
        assert set(np.flatnonzero(diff.sum(axis=1))) == {i for i, _, _ in windows}
    # On the 101-node grid the interfaces are nodes 50 and 75.
    assert [i for i, _, _ in straddling_windows(equispaced_grid(101), m)] == [
        49,
        50,
        51,
        74,
        75,
        76,
    ]


@pytest.mark.parametrize("n", [41, 40])
def test_jump_aware_operator_is_fd4_when_the_material_does_not_change(n):
    g = equispaced_grid(n)
    m = jump_alpha(0.7, 0.7)
    assert straddling_windows(g, m)  # the rows are rebuilt, and come back the same
    np.testing.assert_allclose(
        jump_aware_operator(g, m).toarray(),
        0.7 * dxx_matrix(g).toarray(),
        rtol=1e-10,
        atol=1e-8,
    )


@pytest.mark.parametrize("n", [41, 40])
def test_jump_aware_operator_annihilates_a_piecewise_linear_equilibrium(n):
    g = equispaced_grid(n)
    m = jump_alpha(1.0, 0.25)
    u = equilibrium_exact(m, 1.0, 0.0, g.x)
    assert np.max(np.abs(jump_aware_operator(g, m) @ u)) < 1e-9
    assert np.max(np.abs(naive_operator(g, m) @ u)) > 1.0


def test_a_window_that_sees_both_interfaces_of_a_thin_layer_translates_twice():
    g = equispaced_grid(41)
    m = PiecewiseAlpha((0.0, 2 * g.h), (Constant(1.0), Constant(0.1), Constant(1.0)))
    assert [seen for _, _, seen in straddling_windows(g, m)] == [
        [0],
        [0],
        [0, 1],
        [1],
        [1],
    ]
    u = equilibrium_exact(m, 1.0, 0.0, g.x)
    assert np.max(np.abs(jump_aware_operator(g, m) @ u)) < 1e-9


def test_jump_aware_operator_is_consistent_on_the_dissertation_solution():
    # Local truncation error. The straddling rows are third order (the stencil
    # is not symmetric about the interface, so the x^5 term does not cancel)
    # and the max norm shows it; the naive residual grows like 1/h instead.
    m = dissertation_alpha()
    residuals, naive = [], []
    for n in (101, 201, 401, 801):
        g = equispaced_grid(n)
        u = equilibrium_exact(m, *DISSERTATION_BC, g.x)
        residuals.append(np.max(np.abs(jump_aware_operator(g, m) @ u)))
        naive.append(np.max(np.abs(naive_operator(g, m) @ u)))
    rates = np.log2(np.array(residuals[:-1]) / np.array(residuals[1:]))
    assert np.all(rates > 2.7), rates
    assert np.all(np.diff(naive) > 0)


@pytest.mark.parametrize("cells", [1, 2])
def test_no_nodal_alpha_makes_dx_a_dx_exact_on_a_kinked_equilibrium(cells):
    """The harmonic-mean rule is exact only as a face conductance.

    `docs/stiff-diffusion.md` §1.8: with alpha at the nodes replaced by its
    harmonic mean over ``cells`` cells, ``Dx A Dx`` still fails on the
    piecewise-linear equilibrium of the MATLAB jump ``1/9 | 1`` (mid-cell
    at even n), because ``Dx`` of the kink is already O(1) off beside it,
    while the jump-aware operator annihilates the same solution to rounding.
    The conservative three-point scheme with exact face conductances would
    give zero; E3.5 builds it as the finite-volume comparator.
    """
    g = equispaced_grid(100)
    m = jump_alpha(1.0 / 9.0, 1.0)
    u = equilibrium_exact(m, 1.0, 0.0, g.x)
    w = cells * g.h / 2
    lo, hi = g.x - w, g.x + w
    in_left = np.clip(np.minimum(hi, 0.0) - lo, 0.0, None)
    in_right = np.clip(hi - np.maximum(lo, 0.0), 0.0, None)
    treated = 2 * w / (9.0 * in_left + 1.0 * in_right)
    medium = PiecewiseAlpha(
        (), (Smooth(lambda x: np.interp(x, g.x, treated), (np.zeros_like,)),)
    )
    residual = (naive_operator(g, medium) @ u)[2:-2]
    assert np.max(np.abs(residual)) > 0.5
    assert np.max(np.abs((jump_aware_operator(g, m) @ u)[2:-2])) < 1e-9
