import numpy as np
import pytest

from heat_interfaces.heat1d.domain import (
    DISSERTATION_BC,
    Constant,
    PiecewiseAlpha,
    dissertation_alpha,
    equispaced_grid,
    matlab_alpha,
)
from heat_interfaces.heat1d.exact import chebyshev_equilibrium, chebyshev_parabolic
from heat_interfaces.heat1d.march import (
    STARTUP_FRACTION,
    bd4_amplification,
    bd4_march,
    bd4_stability_boundary,
    interior_operator,
    ramp_boundary,
    rk4_dt_limit,
    rk4_march,
    smooth_step,
)
from heat_interfaces.heat1d.operators import (
    direct_operator,
    jump_aware_operator,
    naive_operator,
)
from heat_interfaces.heat1d.solve import normalized_l2

RAMP, T_END = 1.0, 2.0
"""The MATLAB study: u(-1, t) rises over [0, 1] from a zero start; errors at t = 2.

With dt = h, BD4's error on the ramp is within a factor two of the spatial
error at every count from 100 to 800; a ramp of 0.5 made it dominate at 100
nodes (3e-6 against a 6e-8 spatial error).
"""


def quartic_problem(n=41, alpha=0.7):
    """``u = p(x) g(t)`` with ``p`` quartic, so the FD4 rows are exact."""
    g = equispaced_grid(n)
    x = g.x
    p = x**4 - 2 * x**2 + x + 3
    pxx = 12 * x**2 - 4

    def gt(t):
        return np.cos(3 * t) + 0.5 * np.exp(-t)

    def gpt(t):
        return -3 * np.sin(3 * t) - 0.5 * np.exp(-t)

    operator = direct_operator(g, PiecewiseAlpha((), (Constant(alpha),)))

    def forcing(t):
        return p * gpt(t) - alpha * pxx * gt(t)

    def boundary(t):
        return p[0] * gt(t), p[-1] * gt(t)

    def exact(t):
        return p * gt(t)

    return operator, exact(0.0), boundary, forcing, exact


def test_bd4_is_fourth_order_in_time_on_a_manufactured_solution():
    operator, u0, boundary, forcing, exact = quartic_problem()
    errs = []
    for dt in (0.1, 0.05, 0.025, 0.0125):
        u = bd4_march(operator, u0, 1.0, dt, boundary, forcing)
        errs.append(np.max(np.abs(u - exact(1.0))))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all((rates > 3.9) & (rates < 4.1)), (errs, rates)


def test_rk4_is_fourth_order_in_time_below_its_limit():
    operator, u0, boundary, forcing, exact = quartic_problem()
    limit = rk4_dt_limit(operator)
    errs = []
    for dt in (limit, limit / 2):
        u = rk4_march(operator, u0, 0.5, dt, boundary, forcing)
        errs.append(np.max(np.abs(u - exact(0.5))))
    assert 3.5 < np.log2(errs[0] / errs[1]) < 4.5, errs


def test_rk4_blows_up_well_above_its_limit():
    operator, u0, boundary, forcing, _ = quartic_problem()
    with np.errstate(over="ignore", invalid="ignore"):
        u = rk4_march(operator, u0, 0.5, 3 * rk4_dt_limit(operator), boundary, forcing)
    assert not np.all(np.abs(u) < 1e6)


def test_a_run_of_three_steps_or_fewer_is_the_rk4_start_up():
    operator, u0, boundary, forcing, exact = quartic_problem()
    u = bd4_march(operator, u0, 0.03, 0.01, boundary, forcing)
    substeps = int(np.ceil(0.01 / (STARTUP_FRACTION * rk4_dt_limit(operator))))
    same = rk4_march(operator, u0, 0.03, 0.01 / substeps, boundary, forcing)
    np.testing.assert_allclose(u, same, rtol=0, atol=1e-13)
    assert np.max(np.abs(u - exact(0.03))) < 1e-8


def test_smooth_step_is_a_symmetric_monotone_ramp():
    t = np.linspace(-0.5, 1.5, 401)
    s = smooth_step(t, 1.0)
    assert np.all(s[t <= 0] == 0) and np.all(s[t >= 1] == 1)
    assert np.all(np.diff(s) >= 0)
    np.testing.assert_allclose(s + smooth_step(1 - t, 1.0), 1.0, atol=1e-15)
    assert smooth_step(0.25, 0.5) == pytest.approx(0.5)
    assert isinstance(smooth_step(0.3, 1.0), float)


@pytest.mark.parametrize("counts", [(100, 200, 400, 800), (101, 201, 401, 801)])
def test_matlab_problem_naive_first_order_and_jump_aware_fourth_order(counts):
    # Even counts put the jump mid-cell, odd counts on a node; dt = h throughout.
    m = matlab_alpha()
    boundary = ramp_boundary(RAMP)
    errs = {"naive": [], "aware": []}
    for n in counts:
        g = equispaced_grid(n)
        ref = chebyshev_parabolic(m, np.zeros_like, boundary, T_END, g.x, n_cheb=32)
        for name, op in (("naive", naive_operator), ("aware", jump_aware_operator)):
            u = bd4_march(op(g, m), np.zeros(n), T_END, g.h, boundary)
            errs[name].append(normalized_l2(u, ref))
    naive, aware = (np.array(errs[k]) for k in ("naive", "aware"))
    slope = -np.polyfit(np.log(counts), np.log(naive), 1)[0]
    assert 0.9 < slope < 1.1, (naive, slope)
    rates = np.log2(aware[:-1] / aware[1:])
    assert np.all(rates > 3.7), (aware, rates)
    assert np.all(aware < 1e-3 * naive)


def test_dissertation_problem_with_a_growing_end_value_is_fourth_order():
    # u = e^{c t} v(x), the 1-D twin of dissertation eq. 85-86 (case 1, c_t = 1):
    # v solves (alpha v')' = c v with the equilibrium end values.
    m = dissertation_alpha()
    c, t_end = 1.0, 1.0

    def boundary(t):
        return np.exp(c * t), 0.0

    counts = (101, 201, 401, 801)
    errs = {"naive": [], "aware": []}
    for n in counts:
        g = equispaced_grid(n)
        v = chebyshev_equilibrium(m, *DISSERTATION_BC, g.x, shift=c)
        for name, op in (("naive", naive_operator), ("aware", jump_aware_operator)):
            u = bd4_march(op(g, m), v, t_end, g.h, boundary)
            errs[name].append(normalized_l2(u, np.exp(c * t_end) * v))
    naive, aware = (np.array(errs[k]) for k in ("naive", "aware"))
    # The naive line's first pair (101 -> 201) runs at 1.33 before settling to
    # 1.02, 1.01: at 101 nodes the layer holds 25 nodes and both interface
    # errors still overlap. The elliptic test in test_solve.py sees the same
    # start; here the pairs from 201 on carry the first-order claim.
    naive_rates = np.log2(naive[:-1] / naive[1:])
    assert np.all((naive_rates[1:] > 0.9) & (naive_rates[1:] < 1.15)), naive_rates
    assert naive_rates[0] > 0.9
    rates = np.log2(aware[:-1] / aware[1:])
    assert np.all(rates > 3.7), (aware, rates)
    assert aware[0] < 6e-3 and aware[-1] < 2e-6


@pytest.mark.parametrize(
    ("medium", "n"),
    [
        (matlab_alpha(), 100),
        (matlab_alpha(), 101),
        (dissertation_alpha(), 101),
        (dissertation_alpha(), 401),
    ],
)
def test_jump_aware_spectrum_is_real_negative_and_damped_by_bd4(medium, n):
    g = equispaced_grid(n)
    lam = np.linalg.eigvals(interior_operator(jump_aware_operator(g, medium)).toarray())
    assert np.max(lam.real) < 0
    assert np.max(np.abs(lam.imag)) < 1e-8 * np.max(np.abs(lam))
    # The extreme is the FD4 second derivative's in the alpha = 1 material.
    assert np.min(lam.real) * g.h**2 == pytest.approx(-16 / 3, rel=0.01)
    assert np.max(bd4_amplification(g.h * lam)) < 1


def test_naive_spectrum_has_complex_pairs_yet_is_damped_by_bd4_at_dt_h():
    g = equispaced_grid(101)
    op = naive_operator(g, dissertation_alpha())
    lam = np.linalg.eigvals(interior_operator(op).toarray())
    assert np.max(lam.real) < 0
    assert np.sum(np.abs(lam.imag) > 1e-8) > 40
    assert np.max(np.abs(lam.imag)) * g.h**2 > 0.1
    assert np.max(bd4_amplification(g.h * lam)) < 1


def test_bd4_boundary_has_unit_amplification_and_the_negative_axis_is_stable():
    theta = np.linspace(0.1, 2 * np.pi - 0.1, 50)
    np.testing.assert_allclose(
        bd4_amplification(bd4_stability_boundary(theta)), 1.0, atol=1e-9
    )
    assert bd4_stability_boundary(0.0) == 0
    assert np.all(bd4_amplification(-np.logspace(-3, 3, 30)) < 1)
    # A growing mode grows: z = 0.05 is amplified by about e^0.05.
    assert bd4_amplification(0.05)[0] == pytest.approx(np.exp(0.05), rel=1e-3)


def test_dirichlet_mask_generalises_the_end_rows():
    # The same quartic problem with the grid nodes shuffled: the Dirichlet
    # nodes are named by a mask, the boundary values come in index order.
    operator, u0, boundary, forcing, exact = quartic_problem()
    n = operator.shape[0]
    rng = np.random.default_rng(11)
    perm = rng.permutation(n)  # new position -> old node
    p = np.eye(n)[perm]
    shuffled = p @ operator.toarray() @ p.T
    mask = np.zeros(n, dtype=bool)
    mask[np.flatnonzero((perm == 0) | (perm == n - 1))] = True
    ends = np.flatnonzero(mask)  # ascending index; which end is first?
    first_is_left = perm[ends[0]] == 0

    def shuffled_boundary(t):
        left, right = boundary(t)
        return (left, right) if first_is_left else (right, left)

    reference = bd4_march(operator, u0, 1.0, 0.05, boundary, forcing)
    u = bd4_march(
        shuffled,
        u0[perm],
        1.0,
        0.05,
        shuffled_boundary,
        lambda t: forcing(t)[perm],
        dirichlet=mask,
    )
    np.testing.assert_allclose(u[np.argsort(perm)], reference, atol=1e-12)
    # Index arrays and the default agree; the interior operator drops the mask.
    same = bd4_march(operator, u0, 1.0, 0.05, boundary, forcing, dirichlet=[0, n - 1])
    np.testing.assert_allclose(same, reference, atol=0)
    assert interior_operator(operator, mask[np.argsort(perm)]).shape == (n - 2, n - 2)
    lam_ref = np.sort(np.linalg.eigvals(interior_operator(operator).toarray()))
    lam = np.sort(np.linalg.eigvals(interior_operator(shuffled, mask).toarray()))
    np.testing.assert_allclose(lam, lam_ref, atol=1e-9)


def test_dirichlet_index_rejects_bad_masks():
    from heat_interfaces.heat1d.march import dirichlet_index

    np.testing.assert_array_equal(dirichlet_index(5), [0, 4])
    np.testing.assert_array_equal(dirichlet_index(5, [4, 0, 4]), [0, 4])
    with pytest.raises(ValueError, match="one entry per node"):
        dirichlet_index(5, np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="range"):
        dirichlet_index(5, [0, 5])
