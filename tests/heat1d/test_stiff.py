"""The seed stencils of E3.4 (#29): `docs/stiff-diffusion.md` §1.9's P4–P9."""

import numpy as np
import pytest

from heat_interfaces.fd_weights import fornberg_weights
from heat_interfaces.heat1d.domain import (
    DISSERTATION_BC,
    Constant,
    PiecewiseAlpha,
    SmoothEdges,
    dissertation_alpha,
    equispaced_grid,
    jump_alpha,
    matlab_alpha,
)
from heat_interfaces.heat1d.exact import chebyshev_equilibrium, equilibrium_exact
from heat_interfaces.heat1d.interface import Jump, stencil_weights
from heat_interfaces.heat1d.march import (
    bd4_amplification,
    bd4_march,
    interior_operator,
)
from heat_interfaces.heat1d.operators import (
    direct_operator,
    dxx_matrix,
    jump_aware_operator,
    naive_operator,
    straddling_windows,
)
from heat_interfaces.heat1d.solve import normalized_l2, solve_equilibrium
from heat_interfaces.heat1d.stiff import (
    OPERATOR_MODES,
    build_operator,
    edge_width,
    seed_basis,
    seed_operator,
    seed_profiles,
    seed_weights,
    seeded_windows,
)

H = 0.01


def _jumps(medium: PiecewiseAlpha) -> list[Jump]:
    return [
        Jump(xi, medium.taylor(k, "left", 4), medium.taylor(k, "right", 4))
        for k, xi in enumerate(medium.interfaces)
    ]


def _matlab_window() -> tuple[np.ndarray, float, np.ndarray]:
    """§1.4's layout: the ``1/9 | 1`` jump half a cell right of the node, h = 0.01."""
    centre = -0.5 * H
    nodes = centre + H * np.arange(-2.0, 3.0)
    w_jump = stencil_weights(_jumps(matlab_alpha()), nodes, centre)
    return nodes, centre, w_jump


def _difference(w: np.ndarray, w_ref: np.ndarray) -> float:
    return float(np.max(np.abs(w - w_ref)) / np.max(np.abs(w_ref)))


@pytest.mark.parametrize("centre", [0.0, 0.01, 0.003, -0.02])
def test_constant_alpha_gives_fornberg_weights_times_alpha(centre):
    # P4: on constant alpha the seeds are the monomials and DOP853 integrates
    # the (nilpotent) chain to rounding; the centre may be off the nodes.
    m = jump_alpha(0.7, 0.7)
    nodes = H * np.arange(-2.0, 3.0)
    w = seed_weights(nodes, centre, m)
    expected = 0.7 * fornberg_weights(centre, nodes, 2)[2]
    assert _difference(w, expected) < 1e-11
    if centre == 0.0:
        assert _difference(w, expected) < 5e-15
        a, h_s = seed_basis(nodes, centre, m)
        assert h_s == 2 * H
        assert np.linalg.cond(a) == pytest.approx(23.53, abs=0.05)


@pytest.mark.parametrize("delta", [0.0, 0.05])
def test_seed_operator_is_alpha_times_dxx_when_the_material_does_not_change(delta):
    g = equispaced_grid(41)
    m = SmoothEdges(jump_alpha(0.7, 0.7), delta)
    # 41 nodes put the edge on node 20: three straddling rows; at δ = 0.05 the
    # reach (20 δ = 1) covers the domain and every row is seeded, the four
    # one-sided end rows in batches of their own.
    assert len(seeded_windows(g, m)) == (3 if delta == 0.0 else 41)
    np.testing.assert_allclose(
        seed_operator(g, m).toarray(),
        0.7 * dxx_matrix(g).toarray(),
        rtol=1e-10,
        atol=1e-8,
    )


@pytest.mark.parametrize("offset", [0.5, 1.5, 0.0])
def test_the_delta_zero_march_equals_e12s_weights_on_constant_pieces(offset):
    # §1.4: for a jump between constants the chain *is* the translated basis;
    # the march evaluates alpha one-sidedly through the elements' pieces, so a
    # jump on a node (offset 0, alpha_e the owner's) is a stop like any other.
    m = matlab_alpha()
    centre = -offset * H
    nodes = centre + H * np.arange(-2.0, 3.0)
    w_jump = stencil_weights(_jumps(m), nodes, centre)
    for medium in (m, SmoothEdges(m, 0.0)):
        assert _difference(seed_weights(nodes, centre, medium), w_jump) < 1e-12


def test_the_jump_limit_is_first_order_in_delta_over_h():
    # P4: §1.4's scratch numbers, 84 %, 47 %, 11 %, 1.1 %, 0.11 %.
    nodes, centre, w_jump = _matlab_window()
    ratios = (1.0, 0.5, 0.1, 0.01, 0.001)
    diffs = np.array(
        [
            _difference(
                seed_weights(nodes, centre, SmoothEdges(matlab_alpha(), r * H)), w_jump
            )
            for r in ratios
        ]
    )
    np.testing.assert_allclose(diffs, [0.843, 0.465, 0.113, 0.0107, 0.00107], rtol=0.02)
    assert diffs[-2] / diffs[-1] == pytest.approx(10.0, rel=0.05)
    assert diffs[-3] / diffs[-2] == pytest.approx(10.6, rel=0.05)


def test_condition_numbers_are_vandermonde_like_at_every_delta():
    # P5: cond A in the stencil coordinate stays within a factor six of the
    # constant-alpha 23.5 from δ = 1e-5 h to δ = 100 h, largest at δ ≈ h/2 and
    # equal to the translated basis' own 90 as δ → 0.
    nodes, centre, _ = _matlab_window()
    conds = {}
    for ratio in (1e-5, 1e-3, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0):
        a, _ = seed_basis(nodes, centre, SmoothEdges(matlab_alpha(), ratio * H))
        conds[ratio] = np.linalg.cond(a)
    assert all(20.0 < c < 150.0 for c in conds.values()), conds
    assert conds[100.0] == pytest.approx(23.53, rel=0.01)
    assert conds[1e-5] == pytest.approx(90.3, rel=0.01)
    assert max(conds, key=conds.get) == 0.5


def test_on_smoothly_varying_pieces_the_delta_zero_weights_differ_at_first_order_in_h():
    # §1.4: E1.2 applies the degree-4-truncated operator to plain monomials on
    # the anchor side, the chain solves (alpha phi')' exactly; at eq. 75's
    # layer edge x = 0.5 (alpha'/alpha = 25 on the sinusoid side) the two
    # five-dimensional spaces differ at relative O(h alpha'/alpha): O(1) on
    # the study's grids, halving with h. Both are fourth order (§2.3).
    m = dissertation_alpha()
    diffs = []
    for n in (201, 401, 801):
        g = equispaced_grid(n)
        x = g.snapped(m.interfaces)
        i = 3 * (n - 1) // 4
        nodes, centre = x[i - 2 : i + 3], float(x[i])
        w_jump = stencil_weights([_jumps(m)[1]], nodes, centre)
        w_seed = seed_weights(nodes, centre, SmoothEdges(m, 0.0))
        diffs.append(_difference(w_seed, w_jump))
    diffs = np.array(diffs)
    assert diffs[0] > 1.0 and np.all(np.diff(diffs) < 0)
    assert diffs[1] / diffs[2] > 1.8


@pytest.mark.parametrize("n", [100, 101, 200])
def test_the_equilibrium_is_exact_at_every_delta_on_constant_pieces(n):
    # P6: u = A + B phi_1 in every seeded window, and every plain row sees a
    # line; the residual of the seed rows on the exact solution is rounding.
    g = equispaced_grid(n)
    for delta in (0.04, 0.01, 0.0025):
        m = SmoothEdges(matlab_alpha(), delta)
        u = equilibrium_exact(m, 1.0, 0.0, g.x)
        op = seed_operator(g, m)
        rows = [i for i, _, _ in seeded_windows(g, m)]
        assert np.max(np.abs((op @ u)[rows])) * g.h**2 < 1e-14
        assert normalized_l2(solve_equilibrium(op, 1.0, 0.0), u) < 1e-12
    # Below δ ≈ h/40 the march crosses the edge in many small steps and its
    # tolerance shows: 1e-12 in the scaled residual, 4e-11 in the solution.
    m = SmoothEdges(matlab_alpha(), 1e-4)
    u = equilibrium_exact(m, 1.0, 0.0, g.x)
    assert normalized_l2(solve_equilibrium(seed_operator(g, m), 1.0, 0.0), u) < 1e-10


def test_on_eq75_the_seed_rows_are_exact_and_the_plain_rows_keep_their_error():
    # P6 as measured (§2.3): at δ = 0.04 the reach covers the whole layer,
    # every row inside it is seeded and the solve is exact; at 0.01 it covers
    # it at 101 nodes only (from 201 on a few plain rows sit in the middle of
    # the layer, x in (0.2, 0.3)); at 0.0025 the rows deeper than 20δ + 2h
    # inside the layer are plain FD4 on the sinusoid, whose pre-asymptotic
    # error (h alpha'/alpha = 0.5 at 101 nodes by the edges) sets the line;
    # at δ = 0 the six seeded rows already take the line 40× below E1.2's at
    # 101 nodes.
    m = dissertation_alpha()
    errors = {}
    for delta in (0.0, 0.04, 0.01, 0.0025):
        medium = SmoothEdges(m, delta)
        errors[delta] = np.array(
            [
                normalized_l2(
                    solve_equilibrium(seed_operator(g, medium), *DISSERTATION_BC),
                    equilibrium_exact(medium, *DISSERTATION_BC, g.x),
                )
                for g in map(equispaced_grid, (101, 201, 401))
            ]
        )
    assert np.all(errors[0.04] < 1e-11)
    assert errors[0.01][0] < 1e-12 and np.all(errors[0.01] < 1e-9)
    rates = np.log2(errors[0.0025][:-1] / errors[0.0025][1:])
    assert np.all(rates > 2.4) and rates[-1] > rates[0]
    g = equispaced_grid(101)
    e12 = normalized_l2(
        solve_equilibrium(jump_aware_operator(g, m), *DISSERTATION_BC),
        equilibrium_exact(m, *DISSERTATION_BC, g.x),
    )
    assert errors[0.0][0] < e12 / 30


def test_the_parabolic_error_is_fourth_order_at_every_delta_with_one_constant():
    # P7 on the separable solution e^{ct} v(x) (the E1.3 pattern), BD4 at
    # dt = h: the δ = 0, 0.01 and 0.0025 lines agree to three digits at 50 and
    # 100 nodes (1 % at 200, 2 % at 400, where the errors are 1e-10 and the
    # profiles v differ) and converge at fourth order (super-convergent at
    # first). The ramp problem's lines are in tests/test_heat1d_stiff.py.
    c, t_end = 1.0, 1.0

    def boundary(t):
        return np.exp(c * t), 0.0

    errors = {}
    for delta in (0.0, 0.01, 0.0025):
        m = SmoothEdges(matlab_alpha(), delta)
        errs = []
        for n in (50, 100, 200, 400):
            g = equispaced_grid(n)
            v = chebyshev_equilibrium(m, 1.0, 0.0, g.x, shift=c)
            u = bd4_march(seed_operator(g, m), v, t_end, g.h, boundary)
            errs.append(normalized_l2(u, np.exp(c * t_end) * v))
        errors[delta] = np.array(errs)
    for e in errors.values():
        assert np.all(np.log2(e[:-1] / e[1:]) > 3.7), e
        np.testing.assert_allclose(e[:2], errors[0.0][:2], rtol=3e-3)
        np.testing.assert_allclose(e, errors[0.0], rtol=3e-2)
    assert errors[0.0][0] < 2e-6 and errors[0.0][-1] < 2e-10


def test_local_truncation_is_third_order_on_the_seeded_rows_and_fourth_on_the_plain():
    # §1.6: on v with L v = c v the seeded rows' residual is O(h³) with a
    # constant that does not depend on δ (within a factor two between the two
    # widths, the max over the seeded rows), the plain rows' O(h⁴).
    c = 1.0
    constants = {}
    for delta in (0.01, 0.0025):
        m = SmoothEdges(matlab_alpha(), delta)
        seeded, plain = [], []
        for n in (100, 200, 400):
            g = equispaced_grid(n)
            v = chebyshev_equilibrium(m, 1.0, 0.0, g.x, shift=c)
            r = seed_operator(g, m) @ v - c * v
            rows = [i for i, _, _ in seeded_windows(g, m)]
            others = np.setdiff1d(np.arange(2, g.n - 2), rows)
            seeded.append(np.max(np.abs(r[rows])))
            plain.append(np.max(np.abs(r[others])))
        seeded, plain = np.array(seeded), np.array(plain)
        rates = np.log2(seeded[:-1] / seeded[1:])
        assert np.all((rates > 2.6) & (rates < 4.2)), rates
        assert np.all(np.log2(plain[:-1] / plain[1:]) > 3.7)
        constants[delta] = seeded * np.array([100, 200, 400]) ** 3
    ratio = constants[0.01] / constants[0.0025]
    assert np.all((ratio > 0.4) & (ratio < 2.5)), ratio


def test_a_thin_layer_is_one_march_through_both_edges():
    # P8, the double-cross: a layer two cells thick with both tanh edges in
    # one window; at δ = 0 the weights are E1.2's translate-twice ones, then
    # first order in δ/h, and the seed rows annihilate the exact equilibrium
    # of the smooth layer at every δ.
    g = equispaced_grid(41)
    h = g.h
    layer = PiecewiseAlpha((0.0, 2 * h), (Constant(1.0), Constant(0.1), Constant(1.0)))
    x = g.snapped(layer.interfaces)
    both = [(i, lo) for i, lo, seen in straddling_windows(g, layer) if seen == [0, 1]]
    assert len(both) == 1
    ((i, lo),) = both
    nodes, centre = x[lo : lo + 5], float(x[i])
    w_jump = stencil_weights(_jumps(layer), nodes, centre)
    diffs = [
        _difference(seed_weights(nodes, centre, SmoothEdges(layer, r * h)), w_jump)
        for r in (0.0, 0.1, 0.01, 0.001)
    ]
    assert diffs[0] < 1e-12
    np.testing.assert_allclose(diffs[1:], [0.222, 0.0186, 0.00182], rtol=0.03)
    for ratio in (0.0, 0.1, 0.5):
        m = SmoothEdges(layer, ratio * h)
        u = equilibrium_exact(m, 1.0, 0.0, g.x)
        assert np.max(np.abs(seed_operator(g, m) @ u)) < 1e-12
        seen = [s for _, _, s in seeded_windows(g, m)]
        assert [0, 1] in seen and seen == sorted(seen)


def test_seeded_windows_are_the_straddling_ones_at_delta_zero_and_grow_with_the_reach():
    g = equispaced_grid(100)
    m = matlab_alpha()
    assert seeded_windows(g, SmoothEdges(m, 0.0)) == straddling_windows(g, m)
    assert seeded_windows(g, m) == straddling_windows(g, m)
    assert edge_width(m) == 0.0 and edge_width(SmoothEdges(m, 0.0025)) == 0.0025
    # Reach 20 δ = 0.05 = 2.5 h: the windows whose span meets (−0.05, 0.05).
    rows = [i for i, _, _ in seeded_windows(g, SmoothEdges(m, 0.0025))]
    x = g.x
    assert rows == [i for i in range(2, 98) if x[i - 2] < 0.05 and x[i + 2] > -0.05]
    assert len(rows) == 8
    assert len(seeded_windows(g, SmoothEdges(m, 0.0025), reach=0.0)) == 4
    assert len(seeded_windows(g, SmoothEdges(m, 0.04))) == 84


@pytest.mark.parametrize(
    ("medium", "n"),
    [
        (matlab_alpha(), 100),
        (matlab_alpha(), 101),
        (dissertation_alpha(), 49),
        (dissertation_alpha(), 101),
    ],
)
def test_the_seed_spectrum_is_real_negative_and_damped_by_bd4(medium, n):
    # P9: like the jump-aware operator's (test_march.py), and unlike it on
    # eq. 75 at 49 nodes, where E1.2's rows put an eigenvalue at +248 and the
    # seed rows (the exact chain in the sinusoid) keep every one left of −2.
    g = equispaced_grid(n)
    for delta in (0.0, 0.0025):
        op = seed_operator(g, SmoothEdges(medium, delta))
        lam = np.linalg.eigvals(interior_operator(op).toarray())
        assert np.max(lam.real) < -0.8
        assert np.max(np.abs(lam.imag)) == 0.0
        assert np.min(lam.real) * g.h**2 == pytest.approx(-16 / 3, rel=0.01)
        assert np.max(bd4_amplification(g.h * lam)) < 1
    if n == 49:
        op = jump_aware_operator(g, medium)
        assert np.max(np.linalg.eigvals(interior_operator(op).toarray()).real) > 100


def test_the_batched_march_agrees_with_single_stencils():
    g = equispaced_grid(200)
    m = SmoothEdges(matlab_alpha(), 0.0025)
    x = g.snapped(m.interfaces)
    op = seed_operator(g, m)
    windows = seeded_windows(g, m)
    assert len(windows) > 8
    for i, lo, _ in windows:
        row = op[[i], lo : lo + 5].toarray()[0]
        assert _difference(row, seed_weights(x[lo : lo + 5], x[i], m)) < 1e-10
    # The batch marched the seeds of every row with the same node pattern.
    rows = np.array([i for i, _, _ in windows])
    phi = seed_profiles(x[rows], 2 * g.h, np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), m)
    assert phi.shape == (rows.size, 5, 5)
    np.testing.assert_array_equal(phi[:, :, 2], np.tile(np.eye(5)[0], (rows.size, 1)))


def test_build_operator_dispatches_by_name():
    g = equispaced_grid(41)
    m = SmoothEdges(matlab_alpha(), 0.01)
    for mode, build in (
        ("naive", naive_operator),
        ("direct", direct_operator),
        ("jump-aware", jump_aware_operator),
        ("seeds", seed_operator),
    ):
        assert OPERATOR_MODES[mode] is build
        assert (build_operator(g, m, mode) != build(g, m)).nnz == 0
    with pytest.raises(ValueError, match="unknown operator"):
        build_operator(g, m, "harmonic")


def test_validation():
    m = matlab_alpha()
    with pytest.raises(ValueError, match="three nodes"):
        seed_weights(np.array([0.0, 0.01]), 0.0, m)
    with pytest.raises(ValueError, match="fewer nodes"):
        seeded_windows(equispaced_grid(5), m, width=6)
