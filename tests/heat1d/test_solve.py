import numpy as np
import pytest

from heat_interfaces.heat1d.domain import (
    DISSERTATION_BC,
    Constant,
    PiecewiseAlpha,
    Smooth,
    dissertation_alpha,
    equispaced_grid,
    jump_alpha,
)
from heat_interfaces.heat1d.exact import equilibrium_exact
from heat_interfaces.heat1d.operators import direct_operator, naive_operator
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
