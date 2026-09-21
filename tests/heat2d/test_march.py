"""BD4 on the 2-D node sets: the Dirichlet boundary in index order, the analytic
history start, case 1's parabolic convergence and the operator spectra against
BD4's region (dissertation §5.4.1, Fig. 5-5 and 5-6)."""

import numpy as np
import pytest

from heat_interfaces.heat1d.march import bd4_amplification, march_steps
from heat_interfaces.heat2d.domain import build_node_set, case1
from heat_interfaces.heat2d.exact import case1_exact
from heat_interfaces.heat2d.march import (
    analytic_history,
    dirichlet_boundary,
    interior_eigenvalues,
    march_parabolic,
)
from heat_interfaces.heat2d.operators import (
    build_stencils,
    dirichlet_values,
    interface_aware_operator,
    naive_operator,
)
from heat_interfaces.heat2d.rbf import BOUNDARY
from heat_interfaces.heat2d.solve import rms_error, solve_equilibrium

DOMAIN = case1()
T_END = 0.1


def top(x, y, t=0.0):
    return np.exp(t) * np.sin(2 * np.pi * x)


_sets = {}


def aware_set(n):
    if n not in _sets:
        nodes = build_node_set(DOMAIN, n)
        _sets[n] = (nodes, build_stencils(nodes, DOMAIN, interface=BOUNDARY))
    return _sets[n]


def rate(errs):
    e = np.asarray(errs)
    return np.log(e[:-1] / e[1:]) / np.log(np.sqrt(2.0))


def test_dirichlet_boundary_gives_the_curve_values_in_index_order():
    nodes, _ = aware_set(900)
    boundary = dirichlet_boundary(nodes, [0.0, top])
    for t in (0.0, 0.37):
        g = dirichlet_values(nodes, [0.0, lambda x, y, t=t: top(x, y, t)])
        np.testing.assert_array_equal(boundary(t), g[nodes.dirichlet])
    assert boundary(0.5).shape == (int(nodes.dirichlet.sum()),)
    assert np.all(boundary(0.5)[nodes.boundary_index[nodes.dirichlet] == 0] == 0.0)
    with pytest.raises(ValueError, match="boundary values for"):
        dirichlet_boundary(nodes, [0.0])


def test_analytic_history_is_the_solution_at_the_rounded_steps_before_zero():
    nodes, _ = aware_set(900)
    exact = case1_exact(1.0)
    steps, dt = march_steps(T_END, 0.03)
    assert steps == 3 and dt == pytest.approx(0.1 / 3)
    history = analytic_history(exact, nodes, T_END, 0.03)
    assert len(history) == 3
    for k, u in zip((3, 2, 1), history, strict=True):
        np.testing.assert_array_equal(u, exact(nodes.x, nodes.y, -k * dt))


def test_case1_parabolic_is_fourth_order_and_tracks_the_elliptic_error():
    # Dissertation Fig. 5-5: parabolic 1.8e-5, 3.8e-6, 6.3e-7, 1.7e-7 and
    # elliptic 1.0e-5, 2.6e-6, 5.5e-7, 8e-8 at 1250–10,000 nodes (read off).
    # BD4 from the analytic history with dt = h runs 3, 5, 7 and 10 steps to
    # t = 0.1. The ticket's 20,000-node point (port notes §2.5) is left to
    # the driver: its node set and operator take ten seconds to build.
    exact = case1_exact(1.0)
    errs, ell = [], []
    for n in (1250, 2500, 5000, 10000):
        nodes, st = aware_set(n)
        op = interface_aware_operator(nodes, DOMAIN.material, st)
        u0 = exact(nodes.x, nodes.y, 0.0)
        u = march_parabolic(op, nodes, u0, T_END, nodes.h, [0.0, top], solution=exact)
        errs.append(rms_error(u, exact(nodes.x, nodes.y, T_END)))
        ue = solve_equilibrium(op, nodes, [0.0, top])
        ell.append(rms_error(ue, case1_exact()(nodes.x, nodes.y)))
    r = rate(errs)
    assert np.all(r > 3.3), (errs, r)
    assert errs[0] < 3e-5 and errs[-1] < 3e-7
    ratio = np.array(errs) / np.array(ell)
    assert np.all((ratio > 0.3) & (ratio < 3.0)), ratio


def test_time_error_is_subdominant_and_the_rk4_start_agrees():
    # Halving dt moves the error by under a percent; the RK4 start-up of the
    # 1-D marcher (1300 sub-steps per step here) starts from the discrete
    # solution's own first three steps instead of the analytic ones and lands
    # within 3 % of the same error.
    exact = case1_exact(1.0)
    nodes, st = aware_set(2500)
    op = interface_aware_operator(nodes, DOMAIN.material, st)
    u0 = exact(nodes.x, nodes.y, 0.0)
    reference = exact(nodes.x, nodes.y, T_END)
    errs = {}
    for name, dt, solution in (
        ("h", nodes.h, exact),
        ("h/2", nodes.h / 2, exact),
        ("rk4", nodes.h, None),
    ):
        u = march_parabolic(op, nodes, u0, T_END, dt, [0.0, top], solution=solution)
        errs[name] = rms_error(u, reference)
    assert errs["h/2"] == pytest.approx(errs["h"], rel=0.01), errs
    assert errs["rk4"] == pytest.approx(errs["h"], rel=0.05), errs


def test_case1_spectra_are_damped_by_bd4_at_dt_h_and_at_fig_5_6s_step():
    # The done-when of #19 at 2000 nodes (the notes record 4900): every
    # eigenvalue of the interface-aware interior operator, warped or plain,
    # and of the naive one sits in BD4's stable region at dt = h and at
    # dt = 0.02, and the slowest mode is slower than the control's -π².
    nodes, st = aware_set(2000)
    plain = build_stencils(nodes, DOMAIN)
    for op in (
        interface_aware_operator(nodes, DOMAIN.material, st),
        interface_aware_operator(nodes, DOMAIN.material, st, warp=False),
        naive_operator(nodes, DOMAIN.material, plain),
    ):
        lam = interior_eigenvalues(op, nodes)
        assert lam.size == nodes.n - int(nodes.dirichlet.sum())
        assert -(np.pi**2) < lam.real.max() < 0
        assert np.sum(np.abs(lam.imag) > 1e-8 * np.abs(lam).max()) > 100
        for dt in (nodes.h, 0.02):
            assert bd4_amplification(dt * lam).max() < 1
