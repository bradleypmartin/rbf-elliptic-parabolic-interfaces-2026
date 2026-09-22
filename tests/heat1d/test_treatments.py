"""E3.5 (#30): the coefficient treatments on a sub-grid edge, and their FV twin."""

import numpy as np
import pytest

from heat_interfaces.heat1d.domain import (
    Constant,
    PiecewiseAlpha,
    Sinusoid,
    SmoothEdges,
    dissertation_alpha,
    eabe_alpha,
    equispaced_grid,
    matlab_alpha,
)
from heat_interfaces.heat1d.exact import (
    chebyshev_equilibrium,
    edge_resistance_deficit,
    equilibrium_exact,
    equilibrium_flux,
    inverse_alpha_integral,
    parabolic_reference,
)
from heat_interfaces.heat1d.march import ramp_boundary
from heat_interfaces.heat1d.operators import (
    jump_aware_operator,
    naive_operator,
    straddling_windows,
)
from heat_interfaces.heat1d.solve import normalized_l2, solve_equilibrium
from heat_interfaces.heat1d.stiff import seeded_windows
from heat_interfaces.heat1d.treatments import (
    LinearPiece,
    NodalAlpha,
    arithmetic_cells,
    cell_windows,
    face_conductance_operator,
    face_conductances,
    harmonic_cells,
    widened_edge,
)

BC = (1.0, 0.0)


def _constant(value: float) -> PiecewiseAlpha:
    return PiecewiseAlpha((), (Constant(value),))


def _residual(grid, medium, u) -> float:
    return float(np.max(np.abs((naive_operator(grid, medium) @ u)[2:-2])))


def test_every_treatment_is_the_identity_on_a_constant_alpha():
    g = equispaced_grid(41)
    m = _constant(0.7)
    for build in (harmonic_cells, arithmetic_cells):
        for cells in (1, 2):
            np.testing.assert_allclose(build(g, m, cells).values, 0.7, rtol=1e-14)
    widened = widened_edge(g, m, 2)
    assert widened.delta == 2 * g.h and np.all(widened.alpha(g.x) == 0.7)
    # T1-FV on a constant is alpha times the three-point second difference,
    # with zero end rows for the Dirichlet rows to replace.
    op = face_conductance_operator(g, m).toarray()
    n = g.n
    second = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1) - 2 * np.eye(n)
    second[[0, -1]] = 0.0
    np.testing.assert_allclose(op, 0.7 * second / g.h**2, rtol=1e-13)


def test_nodal_alpha_is_the_table_at_the_nodes_and_linear_between():
    g = equispaced_grid(11)
    values = 1.0 + np.arange(g.n) ** 2 / 10.0
    m = NodalAlpha(g, values)
    np.testing.assert_array_equal(m.alpha(g.x), values)
    mid = (g.x[:-1] + g.x[1:]) / 2
    np.testing.assert_allclose(m.alpha(mid), (values[:-1] + values[1:]) / 2)
    np.testing.assert_allclose(m.alpha_x(mid), np.diff(values) / g.h)
    # A node reads the slope of the cell to its right; the last node its left.
    np.testing.assert_allclose(m.alpha_x(g.x[:-1]), np.diff(values) / g.h)
    assert m.alpha_x(g.x[-1]) == pytest.approx(np.diff(values)[-1] / g.h)
    edges, pieces = m.elements()
    np.testing.assert_array_equal(edges, g.x)
    # One linear piece per cell, each a Piece: value, slope, and a Taylor
    # expansion about any point that is the interpolant's line.
    assert len(pieces) == g.n - 1
    for k, piece in enumerate(pieces):
        assert isinstance(piece, LinearPiece)
        x_mid = (g.x[k] + g.x[k + 1]) / 2
        assert piece.alpha(x_mid) == pytest.approx(m.alpha(x_mid))
        assert piece.alpha_x(x_mid) == pytest.approx(m.alpha_x(x_mid))
        np.testing.assert_allclose(
            piece.taylor(g.x[k + 1], 3), [values[k + 1], m.alpha_x(x_mid), 0.0, 0.0]
        )
    # No interfaces: nothing straddles, nothing is seeded, nothing expands.
    assert m.interfaces == ()
    assert straddling_windows(g, m) == [] and seeded_windows(g, m) == []
    with pytest.raises(ValueError, match="no interfaces"):
        m.taylor(0, "left", 4)
    # The quadrature reference of the table is the interpolant's, and the
    # naive operator samples the table to the bit.
    assert np.all(np.diff(inverse_alpha_integral(m, g.x)) > 0)
    np.testing.assert_array_equal(naive_operator(g, m).diagonal().size, g.n)
    with pytest.raises(ValueError, match="one value per node"):
        NodalAlpha(g, values[:-1])
    with pytest.raises(ValueError, match="positive"):
        NodalAlpha(g, values - values.max())


def test_cell_windows_are_clipped_at_the_ends_and_validated():
    g = equispaced_grid(21)
    lo, hi = cell_windows(g, 2)
    assert lo[0] == -1.0 and hi[-1] == 1.0
    assert hi[0] == pytest.approx(-1.0 + g.h) and lo[-1] == pytest.approx(1.0 - g.h)
    np.testing.assert_allclose(hi[1:-1] - lo[1:-1], 2 * g.h)
    lo, hi = cell_windows(g, 1)
    np.testing.assert_allclose((hi - lo)[1:-1], g.h)
    assert hi[0] - lo[0] == pytest.approx(g.h / 2) and hi[-1] - lo[-1] == pytest.approx(
        g.h / 2
    )
    for bad in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError):
            cell_windows(g, bad)
        with pytest.raises(ValueError):
            widened_edge(g, matlab_alpha(), bad)


@pytest.mark.parametrize("n", [100, 101])
@pytest.mark.parametrize("delta", [0.0, 0.01, 0.0025])
def test_the_face_conductance_scheme_is_exact_at_equilibrium(n, delta):
    # §1.8 item 3: the discrete flux equals the exact flux B at the exact
    # nodal values, at every δ and with the edge mid-cell (100) or on a node
    # (101); the same on eq. 75's two edges and its sinusoid layer.
    g = equispaced_grid(n)
    for jump in (matlab_alpha(), dissertation_alpha()):
        m = SmoothEdges(jump, delta)
        u = equilibrium_exact(m, *BC, g.x)
        op = face_conductance_operator(g, m)
        assert normalized_l2(solve_equilibrium(op, *BC), u) < 1e-13
        assert np.max(np.abs((op @ u)[1:-1])) < 1e-11
        # The flux is a difference quotient of a cumulative quadrature, so its
        # rounding is about n eps relative (2e-12 here), not eps.
        flux = face_conductances(g, m) * np.diff(u) / g.h
        np.testing.assert_allclose(flux, equilibrium_flux(m, *BC), rtol=1e-11)


def test_the_face_conductances_at_delta_zero_are_patankars_harmonic_mean():
    # With the jump mid-cell the face across it is 2ab/(a + b) and every
    # other face is its piece's value; with the jump on a node every face is
    # inside one piece and the scheme is the pieces' own.
    a, b = 1.0 / 9.0, 1.0
    g = equispaced_grid(100)
    faces = face_conductances(g, matlab_alpha())
    k = int(np.searchsorted(g.x, 0.0)) - 1
    assert faces[k] == pytest.approx(2 * a * b / (a + b))
    np.testing.assert_allclose(faces[:k], a)
    np.testing.assert_allclose(faces[k + 1 :], b)
    g = equispaced_grid(101)
    faces = face_conductances(g, matlab_alpha())
    k = int(np.searchsorted(g.x, 0.0))
    np.testing.assert_allclose(faces[:k], a)
    np.testing.assert_allclose(faces[k:], b)


def test_no_nodal_alpha_makes_dx_a_dx_exact_the_section_1_8_residuals():
    # §1.8's correction of the ticket: on the mid-cell MATLAB jump at 100
    # nodes the one- and two-cell harmonic means leave residuals 5.6 and 1.2
    # on the exact equilibrium, where the jump-aware rows and T1-FV give
    # rounding. The one-cell window ends on the jump and changes nothing.
    g = equispaced_grid(100)
    m = matlab_alpha()
    u = equilibrium_exact(m, *BC, g.x)
    assert _residual(g, harmonic_cells(g, m, 1), u) == pytest.approx(5.58, rel=0.01)
    assert _residual(g, harmonic_cells(g, m, 2), u) == pytest.approx(1.24, rel=0.01)
    np.testing.assert_allclose(harmonic_cells(g, m, 1).values, m.alpha(g.x), rtol=1e-13)
    assert _residual(g, arithmetic_cells(g, m, 2), u) > 1.0
    assert np.max(np.abs((jump_aware_operator(g, m) @ u)[2:-2])) < 1e-9
    assert np.max(np.abs((face_conductance_operator(g, m) @ u)[1:-1])) < 1e-11


def test_each_treatment_reduces_to_its_delta_zero_form():
    g = equispaced_grid(100)
    jump = matlab_alpha()
    zero = SmoothEdges(jump, 0.0)
    # (a) The smooth medium at δ = 0 is the jump bit for bit through every
    # treatment built from the medium's values.
    for build in (harmonic_cells, arithmetic_cells):
        for cells in (1, 2):
            np.testing.assert_array_equal(
                build(g, zero, cells).values, build(g, jump, cells).values
            )
    np.testing.assert_array_equal(
        face_conductances(g, zero), face_conductances(g, jump)
    )
    assert widened_edge(g, zero, 2) == widened_edge(g, jump, 2)
    # (b) As δ → 0 the treated values approach the δ = 0 ones at first order.
    gaps = {}
    for delta in (1e-3, 1e-4, 1e-5):
        m = SmoothEdges(jump, delta)
        gaps[delta] = [
            np.max(
                np.abs(
                    harmonic_cells(g, m, 2).values - harmonic_cells(g, jump, 2).values
                )
            ),
            np.max(
                np.abs(
                    arithmetic_cells(g, m, 1).values
                    - arithmetic_cells(g, jump, 1).values
                )
            ),
            np.max(np.abs(face_conductances(g, m) - face_conductances(g, jump))),
        ]
    for k in range(3):
        ratios = [gaps[1e-3][k] / gaps[1e-4][k], gaps[1e-4][k] / gaps[1e-5][k]]
        assert all(9 < r < 11 for r in ratios), (k, ratios)
    # (c) T0 is δ-independent below m h and the medium itself above it.
    for delta in (0.0, 1e-3, 0.5 * g.h):
        assert widened_edge(g, SmoothEdges(jump, delta), 1) == widened_edge(g, jump, 1)
    wide = SmoothEdges(jump, 3 * g.h)
    assert widened_edge(g, wide, 2) == wide


def test_harmonic_is_below_arithmetic_and_both_are_second_order_on_a_smooth_alpha():
    # AM–HM, strict where alpha varies; and on a smooth alpha both means are
    # alpha at the node plus alpha'' w²/6 with w the half-width, so the
    # two-cell mean is four times the one-cell one off and both are O(h²).
    sinusoid = PiecewiseAlpha((), (Sinusoid(0.5, 0.1, 2 * np.pi),))
    offsets = {}
    for n in (41, 81, 161):
        g = equispaced_grid(n)
        h1, h2 = harmonic_cells(g, sinusoid, 1), harmonic_cells(g, sinusoid, 2)
        a1, a2 = arithmetic_cells(g, sinusoid, 1), arithmetic_cells(g, sinusoid, 2)
        assert np.all(h1.values <= a1.values) and np.all(h2.values < a2.values)
        exact = sinusoid.alpha(g.x)
        # The clipped windows at the domain ends are one-sided and first
        # order; the interior nodes carry the claim.
        offsets[n] = [np.max(np.abs(t.values - exact)[2:-2]) for t in (h1, h2, a1, a2)]
    for k in range(4):
        rates = np.log2(
            [offsets[41][k] / offsets[81][k], offsets[81][k] / offsets[161][k]]
        )
        assert np.all(np.abs(rates - 2) < 0.1), (k, rates)
    assert offsets[161][1] / offsets[161][0] == pytest.approx(4.0, rel=0.02)
    assert offsets[161][3] / offsets[161][2] == pytest.approx(4.0, rel=0.02)
    # On the MATLAB jump every treated value stays between the two constants.
    g = equispaced_grid(100)
    for build in (harmonic_cells, arithmetic_cells):
        v = build(g, matlab_alpha(), 2).values
        assert np.all(v >= 1 / 9 - 1e-15) and np.all(v <= 1 + 1e-15)


def test_the_face_conductance_scheme_is_second_order_on_a_smooth_problem():
    # A manufactured solution on eq. 64's sinusoid alone: (alpha u')' = f.
    piece = Sinusoid(0.5, 0.1, 2 * np.pi)
    m = PiecewiseAlpha((), (piece,))
    u_exact = lambda x: np.sin(np.pi * x) + 0.5 * x  # noqa: E731
    errors = []
    for n in (51, 101, 201, 401):
        g = equispaced_grid(n)
        ux = np.pi * np.cos(np.pi * g.x) + 0.5
        uxx = -(np.pi**2) * np.sin(np.pi * g.x)
        f = piece.alpha(g.x) * uxx + piece.alpha_x(g.x) * ux
        u = solve_equilibrium(
            face_conductance_operator(g, m), u_exact(-1), u_exact(1), f
        )
        errors.append(np.max(np.abs(u - u_exact(g.x))))
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(np.abs(rates - 2) < 0.05), rates


def test_the_widened_edge_sits_a_resistance_deficit_from_the_true_medium():
    # On constant pieces F_δ(1) is F₀(1) − c δ to rounding, so the widened
    # medium's total resistance is c (m h − δ) below the true one.
    g = equispaced_grid(200)
    c = edge_resistance_deficit(1 / 9, 1.0)
    for delta in (0.0, 0.0025, 0.01):
        m = SmoothEdges(matlab_alpha(), delta)
        for factor in (1, 2):
            wide = widened_edge(g, m, factor)
            gap = float(
                inverse_alpha_integral(m, np.array(1.0))
                - inverse_alpha_integral(wide, np.array(1.0))
            )
            assert gap == pytest.approx(c * (factor * g.h - delta), rel=1e-10)


def test_the_treatments_see_a_smooth_edge_where_the_jump_hides_it():
    # At δ > 0 the one-cell window beside a mid-cell edge sees the tanh
    # tail: T1 differs from the naive sampling there and nowhere far away.
    g = equispaced_grid(100)
    m = SmoothEdges(matlab_alpha(), 0.01)
    t = harmonic_cells(g, m, 1).values
    sampled = m.alpha(g.x)
    near = np.abs(g.x) < 20 * 0.01 + g.h / 2
    assert np.max(np.abs(t[near] / sampled[near] - 1)) > 1e-3
    np.testing.assert_allclose(t[~near], sampled[~near], rtol=1e-13)
    # At δ = 0 the two-cell windows beside the edge are three quarters in
    # one piece and a quarter in the other: AM/HM = (1/3)/(1/7) = 7/3 at
    # x = −h/2 and (7/9)/(1/3) = 7/3 at x = +h/2.
    jump = matlab_alpha()
    a = arithmetic_cells(g, jump, 2).values
    h2 = harmonic_cells(g, jump, 2).values
    k = int(np.searchsorted(g.x, 0.0))
    assert a[k - 1] / h2[k - 1] == pytest.approx(7 / 3, rel=1e-12)
    assert a[k] / h2[k] == pytest.approx(7 / 3, rel=1e-12)
    assert h2[k - 1] == pytest.approx(1 / 7, rel=1e-12)


def test_eabe_medium_treatments_are_well_defined_on_a_varying_piece():
    # A smooth-edged eq. 64 (a sinusoid on the right): every treatment builds,
    # the two-cell harmonic mean is within the medium's range, and T1-FV is
    # still exact at equilibrium.
    g = equispaced_grid(101)
    m = SmoothEdges(eabe_alpha(), 0.005)
    v = harmonic_cells(g, m, 2).values
    assert np.all(v > 0.39) and np.all(v < 1.0 + 1e-12)
    u = equilibrium_exact(m, *BC, g.x)
    assert (
        normalized_l2(solve_equilibrium(face_conductance_operator(g, m), *BC), u)
        < 1e-13
    )


def test_nodal_alpha_reprs_differ_where_numpy_would_elide_the_edge(tmp_path):
    # Spar finding on #30: numpy summarises an array past 1000 entries with
    # its first and last three, which is never where the edge is, so two
    # different treatments at 1601 nodes printed the same. The repr keys
    # the parabolic reference cache, so it hashes the whole table instead.
    g = equispaced_grid(1601)
    m = SmoothEdges(matlab_alpha(), 0.01)
    t1, t2 = harmonic_cells(g, m, 2), arithmetic_cells(g, m, 1)
    assert np.max(np.abs(t1.values - t2.values)) > 1e-3
    assert repr(t1) != repr(t2)
    assert repr(t1) == repr(harmonic_cells(g, m, 2))
    nudged = NodalAlpha(g, t1.values * (1 + np.finfo(float).eps * (g.x > 0)))
    assert repr(nudged) != repr(t1)
    assert "..." not in repr(t1) and str(g.n) in repr(t1)
    # And the cache is not fooled: two treated media on one coarse grid
    # (cheap Chebyshev elements) give two references, the second not reused.
    g = equispaced_grid(9)
    m = SmoothEdges(matlab_alpha(), 0.05)
    kwargs = dict(
        initial=np.zeros_like,
        boundary=ramp_boundary(1.0, *BC),
        t_end=0.5,
        n_cheb=6,
        problem="t",
        cache=tmp_path / "ref",
    )
    first = parabolic_reference(harmonic_cells(g, m, 1), **kwargs)
    again = parabolic_reference(harmonic_cells(g, m, 1), **kwargs)
    other = parabolic_reference(arithmetic_cells(g, m, 1), **kwargs)
    assert again.reused and not other.reused
    assert np.max(np.abs(other.u - first.u)) > 1e-6


def test_the_chebyshev_reference_of_a_treated_medium_is_the_interpolants():
    # elements() hands the reference one LinearPiece per cell, so the
    # Chebyshev equilibrium of a NodalAlpha is its quadrature equilibrium.
    # The steepest cell has 1/alpha's pole a third of a cell past its end,
    # so 9 Chebyshev nodes leave 5e-6 and 25 leave 6e-13.
    g = equispaced_grid(21)
    t = harmonic_cells(g, SmoothEdges(matlab_alpha(), 0.02), 2)
    x = np.linspace(-1.0, 1.0, 101)
    cheb = chebyshev_equilibrium(t, *BC, x, n_cheb=24)
    quad = equilibrium_exact(t, *BC, x)
    np.testing.assert_allclose(cheb, quad, atol=1e-11)
