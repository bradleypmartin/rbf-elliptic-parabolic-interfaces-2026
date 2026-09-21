import numpy as np
import pytest

from heat_interfaces.heat1d.domain import (
    DISSERTATION_BC,
    Constant,
    PiecewiseAlpha,
    Smooth,
    dissertation_alpha,
    eabe_alpha,
    equispaced_grid,
    jump_alpha,
)
from heat_interfaces.heat1d.exact import equilibrium_exact
from heat_interfaces.heat1d.operators import (
    direct_operator,
    jump_aware_operator,
    naive_operator,
)
from heat_interfaces.heat1d.solve import (
    dirichlet_system,
    normalized_l2,
    solve_equilibrium,
)


@pytest.mark.parametrize("operator", [naive_operator, direct_operator])
def test_constant_alpha_reproduces_the_linear_solution_to_rounding(operator):
    g = equispaced_grid(101)
    m = PiecewiseAlpha((), (Constant(0.3),))
    u = solve_equilibrium(operator(g, m), 1.0, 0.0)
    np.testing.assert_allclose(u, 0.5 * (1 - g.x), atol=1e-12)


def test_dirichlet_system_keeps_interior_rows_and_pins_the_ends():
    g = equispaced_grid(21)
    op = naive_operator(g, jump_alpha(1.0, 2.0))
    a, b = dirichlet_system(op, 3.0, -1.0, forcing=np.full(g.n, 0.5))
    dense = a.toarray()
    np.testing.assert_allclose(dense[1:-1], op.toarray()[1:-1])
    assert dense[0, 0] == 1 and dense[-1, -1] == 1
    assert np.count_nonzero(dense[0]) == 1 and np.count_nonzero(dense[-1]) == 1
    assert b[0] == 3.0 and b[-1] == -1.0 and np.all(b[1:-1] == 0.5)


def test_manufactured_smooth_problem_converges_at_fourth_order():
    alpha = Smooth(
        lambda x: 2 + np.cos(np.pi * x), (lambda x: -np.pi * np.sin(np.pi * x),)
    )
    m = PiecewiseAlpha((), (alpha,))
    errs = []
    for n in (21, 41, 81, 161):
        g = equispaced_grid(n)
        x = g.x
        exact = np.sin(np.pi * x) + x
        # (alpha u')' with u = sin(pi x) + x.
        forcing = -(np.pi**2) * (2 + np.cos(np.pi * x)) * np.sin(
            np.pi * x
        ) - np.pi * np.sin(np.pi * x) * (np.pi * np.cos(np.pi * x) + 1)
        u = solve_equilibrium(naive_operator(g, m), exact[0], exact[-1], forcing)
        errs.append(normalized_l2(u, exact))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 3.7), rates


def dissertation_errors(operator, counts):
    m = dissertation_alpha()
    errs = []
    for n in counts:
        g = equispaced_grid(n)
        u = solve_equilibrium(operator(g, m), *DISSERTATION_BC)
        errs.append(normalized_l2(u, equilibrium_exact(m, *DISSERTATION_BC, g.x)))
    return np.array(errs)


def test_naive_fd4_on_the_dissertation_problem_is_first_order():
    # Fig. 4-7's top line, 100–1600 nodes with both interfaces on nodes.
    counts = (101, 201, 401, 801, 1601)
    errs = dissertation_errors(naive_operator, counts)
    slope = -np.polyfit(np.log(counts), np.log(errs), 1)[0]
    assert 0.8 < slope < 1.2, (errs, slope)
    assert np.all(np.diff(errs) < 0)


def test_direct_stencil_gives_a_straight_line_on_a_jump():
    g = equispaced_grid(101)
    m = jump_alpha(1.0, 0.1)
    u = solve_equilibrium(direct_operator(g, m), 1.0, 0.0)
    np.testing.assert_allclose(u, 0.5 * (1 - g.x), atol=1e-11)
    # The true solution has a kink: the line is a long way from it.
    assert normalized_l2(u, equilibrium_exact(m, 1.0, 0.0, g.x)) > 0.2


def test_direct_stencil_does_not_converge_on_the_dissertation_problem():
    errs = dissertation_errors(direct_operator, (101, 401, 1601))
    assert np.all(errs > 0.05)
    assert errs[-1] > 0.5 * errs[0]


def test_jump_aware_stencils_solve_the_dissertation_problem_at_fourth_order():
    # Fig. 4-7's lower line: read off the figure, 3e-3 at 100 nodes down to
    # 4.5e-8 at 1600 against a 6400-node run; ours against quadrature run
    # 4.3e-3 down to 7.4e-8 with the same slope.
    counts = (101, 201, 401, 801, 1601)
    errs = dissertation_errors(jump_aware_operator, counts)
    rates = np.log2(errs[:-1] / errs[1:])
    assert np.all(rates > 3.8), (errs, rates)
    assert errs[0] < 6e-3 and errs[-1] < 1e-7
    assert np.all(errs < dissertation_errors(naive_operator, counts))


def test_the_101_node_jump_aware_solution_has_no_oscillation():
    # Fig. 4-5: alpha u' is a negative constant, so u decreases monotonically.
    # The §4.1 solution does; the FD4 one wiggles at the interfaces.
    g = equispaced_grid(101)
    m = dissertation_alpha()
    u = solve_equilibrium(jump_aware_operator(g, m), *DISSERTATION_BC)
    assert np.all(np.diff(u) < 0)
    u_naive = solve_equilibrium(naive_operator(g, m), *DISSERTATION_BC)
    assert not np.all(np.diff(u_naive) < 0)


def test_the_eabe_problem_converges_at_fourth_order_with_the_interface_mid_cell():
    # Even node counts put x = 0 halfway between two nodes (four straddling
    # rows). The errors start near 1e-7 and reach rounding by 1600 nodes.
    m = eabe_alpha()
    errs = []
    for n in (100, 200, 400, 800):
        g = equispaced_grid(n)
        u = solve_equilibrium(jump_aware_operator(g, m), 1.0, 0.0)
        errs.append(normalized_l2(u, equilibrium_exact(m, 1.0, 0.0, g.x)))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 3.8), (errs, rates)


@pytest.mark.parametrize("n", [41, 40])
def test_a_jump_between_constants_is_solved_exactly(n):
    # The MATLAB problem's material: the solution is piecewise linear, which
    # the translated basis contains.
    m = jump_alpha(1 / 9, 1.0)
    g = equispaced_grid(n)
    u = solve_equilibrium(jump_aware_operator(g, m), 1.0, 0.0)
    assert normalized_l2(u, equilibrium_exact(m, 1.0, 0.0, g.x)) < 1e-12


@pytest.mark.parametrize("n", [41, 81])
def test_a_two_cell_layer_between_constants_is_solved_exactly(n):
    # Both interfaces fall in one window: the basis translates twice.
    g = equispaced_grid(n)
    m = PiecewiseAlpha((0.0, 2 * g.h), (Constant(1.0), Constant(0.1), Constant(1.0)))
    u = solve_equilibrium(jump_aware_operator(g, m), 1.0, 0.0)
    assert normalized_l2(u, equilibrium_exact(m, 1.0, 0.0, g.x)) < 1e-12
