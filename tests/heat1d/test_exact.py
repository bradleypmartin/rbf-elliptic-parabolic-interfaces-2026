import numpy as np
import pytest

from heat_interfaces.heat1d.domain import (
    DISSERTATION_BC,
    Constant,
    PiecewiseAlpha,
    Smooth,
    SmoothEdges,
    dissertation_alpha,
    jump_alpha,
    matlab_alpha,
)
from heat_interfaces.heat1d.exact import (
    ParabolicReference,
    alpha_integral,
    chebyshev_equilibrium,
    chebyshev_lobatto,
    chebyshev_parabolic,
    edge_resistance_deficit,
    equilibrium_exact,
    equilibrium_flux,
    inverse_alpha_integral,
    parabolic_reference,
)
from heat_interfaces.heat1d.march import constant_boundary, ramp_boundary


def piecewise_linear_reference(x, a_left, a_right, x0, u_left, u_right):
    """Closed form for two constants: continuous u, continuous alpha u'."""
    f_total = (x0 + 1) / a_left + (1 - x0) / a_right
    b = (u_right - u_left) / f_total
    f = np.where(x <= x0, (x + 1) / a_left, (x0 + 1) / a_left + (x - x0) / a_right)
    return u_left + b * f, b


def test_jump_reference_is_exact_to_rounding():
    m = jump_alpha(1.0, 0.1, x0=0.3)
    x = np.linspace(-1, 1, 257)
    u, b = piecewise_linear_reference(x, 1.0, 0.1, 0.3, 1.0, 0.0)
    np.testing.assert_allclose(equilibrium_exact(m, 1.0, 0.0, x), u, atol=1e-14)
    assert equilibrium_flux(m, 1.0, 0.0) == pytest.approx(b, rel=1e-14)


def inverse_cosine_layer():
    """alpha = 1/(2 + cos pi x) on [-0.4, 0.3], 1 elsewhere: F has a closed form."""
    layer = Smooth(
        lambda x: 1 / (2 + np.cos(np.pi * x)),
        (lambda x: np.pi * np.sin(np.pi * x) / (2 + np.cos(np.pi * x)) ** 2,),
    )
    m = PiecewiseAlpha((-0.4, 0.3), (Constant(1.0), layer, Constant(1.0)))

    def f_exact(x):
        x = np.asarray(x, dtype=float)
        g = lambda t: 2 * t + np.sin(np.pi * t) / np.pi  # noqa: E731
        inside = np.clip(x, -0.4, 0.3)
        return (
            np.minimum(x, -0.4) + 1.0 + (g(inside) - g(-0.4)) + np.maximum(x - 0.3, 0.0)
        )

    return m, f_exact


def test_smooth_layer_reference_is_exact_to_rounding_with_enough_nodes():
    m, f_exact = inverse_cosine_layer()
    x = np.linspace(-1, 1, 101)
    np.testing.assert_allclose(inverse_alpha_integral(m, x), f_exact(x), atol=1e-14)
    u = equilibrium_exact(m, 2.0, -1.0, x)
    b = -3.0 / f_exact(1.0)
    np.testing.assert_allclose(u, 2.0 + b * f_exact(x), atol=1e-14)


@pytest.mark.parametrize("n_gauss", [1, 2, 3])
def test_composite_gauss_legendre_converges_at_order_2n(n_gauss):
    m, f_exact = inverse_cosine_layer()
    x = np.array([0.3])
    errs = []
    for n_panels in (2, 4, 8, 16):
        f = inverse_alpha_integral(m, x, n_gauss=n_gauss, n_panels=n_panels)
        errs.append(abs(float(f[0]) - float(f_exact(0.3))))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 2 * n_gauss - 0.3), rates


def test_evaluation_points_may_be_unsorted_and_any_shape():
    m = dissertation_alpha()
    x = np.array([[0.5, -1.0], [0.0, 0.7]])
    f = inverse_alpha_integral(m, x)
    assert f.shape == (2, 2)
    assert f[0, 1] == 0.0
    assert f[1, 0] == pytest.approx(1.0)  # alpha = 1 on [-1, 0]
    assert f[0, 0] > f[1, 0] and f[1, 1] > f[0, 0]
    with pytest.raises(ValueError):
        inverse_alpha_integral(m, np.array([1.5]))


def test_a_medium_without_interfaces_integrates_too():
    m = PiecewiseAlpha((), (Constant(0.25),))
    x = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(inverse_alpha_integral(m, x), [0.0, 4.0, 8.0])
    np.testing.assert_allclose(equilibrium_exact(m, 1.0, 0.0, x), [1.0, 0.5, 0.0])
    np.testing.assert_allclose(alpha_integral(m, x), [0.0, 0.25, 0.5])


def test_alpha_integral_is_the_same_quadrature_as_the_resistance():
    # E3.5's arithmetic mean: exact across a jump, spectral on a sinusoid,
    # and both integrals are cut at the same elements.
    x = np.array([-1.0, -0.3, 0.0, 0.2, 1.0])
    jump = jump_alpha(1.0 / 9.0, 1.0)
    np.testing.assert_allclose(
        alpha_integral(jump, x), np.where(x <= 0, (x + 1) / 9, 1 / 9 + x), rtol=1e-14
    )
    m = dissertation_alpha()
    k = 2 * np.pi
    layer = np.clip(x, 0.0, 0.5)
    expected = (x + 1) - layer + 0.1 * layer + 0.4 * (1 - np.cos(k * layer)) / k
    np.testing.assert_allclose(alpha_integral(m, x), expected, rtol=1e-13)
    s = SmoothEdges(jump, 0.01)
    total = float(alpha_integral(s, np.array(1.0)))
    # The smooth edge adds ∫ (blend − jump) = (b − a) δ ∫ (s(z) − H(z)) dz = 0
    # by the blend's symmetry, so the total is the jump's to rounding.
    assert total == pytest.approx(1 / 9 + 1, rel=1e-13)


@pytest.mark.parametrize("gap", [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-10])
def test_a_point_a_hair_from_an_interface_does_not_corrupt_the_integral(gap):
    # A thin panel next to the interface must be integrated with its own piece;
    # F(1) = 1.5 / 1 + 0.5 / 0.1 = 6.5 regardless of where else F is asked for.
    m = jump_alpha(1.0, 0.1, x0=0.5)
    f = inverse_alpha_integral(m, np.array([0.5 - gap, 0.5 + gap, 1.0]))
    assert f[-1] == pytest.approx(6.5, abs=2e-15)
    assert f[0] == pytest.approx(1.5 - gap, abs=2e-15)
    assert f[1] == pytest.approx(1.5 + 10 * gap, abs=2e-15)


def test_chebyshev_differentiation_is_exact_on_polynomials_of_the_grid_degree():
    x, d, weights = chebyshev_lobatto(12)
    assert np.all(np.diff(x) > 0) and x[0] == -1 and x[-1] == 1
    for p in range(13):
        np.testing.assert_allclose(d @ x**p, p * x ** max(p - 1, 0), atol=1e-10)
    assert weights[0] == pytest.approx(0.5) and abs(weights[1]) == pytest.approx(1.0)


def test_chebyshev_equilibrium_converges_spectrally_to_the_quadrature_reference():
    m = dissertation_alpha()
    x = np.linspace(-1, 1, 101)
    ref = equilibrium_exact(m, *DISSERTATION_BC, x)
    errs = [
        np.max(np.abs(chebyshev_equilibrium(m, *DISSERTATION_BC, x, n_cheb=n) - ref))
        for n in (16, 32, 48)
    ]
    assert errs[0] > 1e-6 and errs[1] < 1e-7 and errs[2] < 1e-11, errs


def test_chebyshev_equilibrium_is_exact_on_a_jump_between_constants():
    m = matlab_alpha()
    x = np.linspace(-1, 1, 101)
    v = chebyshev_equilibrium(m, 1.0, 0.0, x, n_cheb=8)
    np.testing.assert_allclose(v, equilibrium_exact(m, 1.0, 0.0, x), atol=1e-13)


@pytest.mark.parametrize("interfaces", [(), (0.0,), (-0.3, 0.4)])
def test_chebyshev_parabolic_reproduces_a_decaying_mode(interfaces):
    # alpha = 1 everywhere, cut into pieces or not: u = e^{-pi^2 t / 4} cos(pi x / 2).
    m = PiecewiseAlpha(interfaces, (Constant(1.0),) * (len(interfaces) + 1))
    x = np.linspace(-1, 1, 101)
    u = chebyshev_parabolic(
        m, lambda x: np.cos(np.pi * x / 2), constant_boundary(0.0, 0.0), 0.5, x, 32
    )
    exact = np.exp(-(np.pi**2) * 0.5 / 4) * np.cos(np.pi * x / 2)
    assert np.max(np.abs(u - exact)) < 1e-12


def test_chebyshev_parabolic_tracks_the_separable_solution_across_interfaces():
    # u = e^{c t} v(x) with (alpha v')' = c v: the interface matching in time.
    m = dissertation_alpha()
    c = 1.0
    x = np.linspace(-1, 1, 101)

    def initial(xx):
        return chebyshev_equilibrium(m, *DISSERTATION_BC, xx, shift=c)

    u = chebyshev_parabolic(m, initial, lambda t: (np.exp(c * t), 0.0), 0.5, x)
    assert np.max(np.abs(u - np.exp(0.5 * c) * initial(x))) < 1e-10


def test_chebyshev_parabolic_reaches_the_equilibrium_of_the_matlab_problem():
    m = matlab_alpha()
    x = np.linspace(-1, 1, 101)
    u = chebyshev_parabolic(m, np.zeros_like, ramp_boundary(1.0), 30.0, x, 32)
    assert np.max(np.abs(u - equilibrium_exact(m, 1.0, 0.0, x))) < 1e-9


# --- E3.2 (#27): references at any δ ----------------------------------------


def tanh_edge_integral(x, a, b, x0, delta):
    """``∫_{-1}^{x} dξ / (a + (b - a) s(ξ))`` for ``s = ½ (1 + tanh((ξ - x0)/δ))``.

    With ``z = (ξ - x0)/δ`` the integrand is ``(1 + e^{2z}) / (a + b e^{2z})``,
    whose antiderivative in ``ξ`` is ``δ [z/a + (a - b)/(2ab) ln(a + b e^{2z})]``.
    """

    def anti(xx):
        z = (np.asarray(xx, dtype=float) - x0) / delta
        log_term = np.logaddexp(np.log(a), np.log(b) + 2 * z)
        return delta * (z / a + (a - b) / (2 * a * b) * log_term)

    return anti(x) - anti(-1.0)


@pytest.mark.parametrize("delta", [0.1, 0.01, 0.0025, 1e-4, 1e-6])
def test_quadrature_on_a_tanh_edge_matches_the_closed_form(delta):
    m = SmoothEdges(matlab_alpha(), delta)
    x = np.linspace(-1, 1, 801)
    f = inverse_alpha_integral(m, x)
    np.testing.assert_allclose(
        f, tanh_edge_integral(x, 1 / 9, 1.0, 0.0, delta), atol=1e-14
    )
    # A contrast of a hundred, the edge off centre, the same cuts.
    m = SmoothEdges(jump_alpha(0.01, 1.0, x0=0.3), delta)
    f = inverse_alpha_integral(m, x)
    np.testing.assert_allclose(
        f, tanh_edge_integral(x, 0.01, 1.0, 0.3, delta), atol=1e-13
    )


def test_the_smooth_medium_at_delta_zero_passes_the_jump_closed_form():
    m = SmoothEdges(jump_alpha(1.0, 0.1, x0=0.3), 0.0)
    x = np.linspace(-1, 1, 257)
    u, b = piecewise_linear_reference(x, 1.0, 0.1, 0.3, 1.0, 0.0)
    np.testing.assert_allclose(equilibrium_exact(m, 1.0, 0.0, x), u, atol=1e-14)
    assert equilibrium_flux(m, 1.0, 0.0) == pytest.approx(b, rel=1e-14)
    jump = jump_alpha(1.0, 0.1, x0=0.3)
    assert np.array_equal(inverse_alpha_integral(m, x), inverse_alpha_integral(jump, x))


def test_a_smooth_edge_below_the_grid_keeps_the_jump_limit_to_first_order():
    # F_δ(1) - F_0(1) → -c δ past the edge (stiff note §1.7); the sign and the
    # order are what this pins, the constant is edge_resistance_deficit's.
    jump = matlab_alpha()
    f0 = float(inverse_alpha_integral(jump, np.array(1.0)))
    gaps = [
        f0 - float(inverse_alpha_integral(SmoothEdges(jump, d), np.array(1.0)))
        for d in (0.01, 0.001, 0.0001)
    ]
    assert gaps[0] > 0
    np.testing.assert_allclose(
        np.array(gaps[:-1]) / np.array(gaps[1:]), 10.0, rtol=1e-6
    )


def test_edge_resistance_deficit_is_the_closed_form_of_the_quadrature_gap():
    # c = (a - b) ln(a/b) / (2ab): 8.789 for 1/9 | 1, 10.36 for 1 | 0.1 (E3.3,
    # #28); symmetric, positive, zero without a contrast.
    c = edge_resistance_deficit(1 / 9, 1.0)
    assert c == pytest.approx((1 / 9 - 1) * np.log(1 / 9) / (2 / 9), rel=1e-14)
    assert c == pytest.approx(8.788898, abs=1e-6)
    assert edge_resistance_deficit(1.0, 0.1) == pytest.approx(10.361633, abs=1e-6)
    assert edge_resistance_deficit(1.0, 1 / 9) == pytest.approx(c, rel=1e-14)
    assert edge_resistance_deficit(0.7, 0.7) == 0.0
    with pytest.raises(ValueError):
        edge_resistance_deficit(1.0, 0.0)
    # On constant pieces the gap is c δ to rounding at every δ of the study,
    # in both directions of the contrast.
    for left, right in ((1 / 9, 1.0), (1.0, 0.01)):
        jump = jump_alpha(left, right, x0=0.3)
        f0 = float(inverse_alpha_integral(jump, np.array(1.0)))
        for delta in (0.04, 0.0025, 1e-4):
            f_delta = float(
                inverse_alpha_integral(SmoothEdges(jump, delta), np.array(1.0))
            )
            assert (f0 - f_delta) / delta == pytest.approx(
                edge_resistance_deficit(left, right), rel=1e-11
            )


def test_parabolic_reference_round_trips_through_the_cache(tmp_path):
    m = PiecewiseAlpha((), (Constant(1.0),))
    cache = tmp_path / "reference_d0.0025"  # a dot in the stem is not a suffix
    x = np.linspace(-1, 1, 33)
    initial = lambda xx: np.cos(np.pi * xx / 2)  # noqa: E731
    boundary = constant_boundary(0.0, 0.0)
    built = parabolic_reference(
        m, initial, boundary, 0.1, 16, problem="mode", cache=cache
    )
    for suffix in (".npz", ".json"):
        assert (tmp_path / f"reference_d0.0025{suffix}").exists()
    exact = np.exp(-(np.pi**2) * 0.1 / 4) * initial(x)
    assert np.max(np.abs(built.evaluate(x) - exact)) < 1e-11
    assert built.meta["steps"] > 0 and built.meta["elements"] == 1
    loaded = ParabolicReference.load(cache)
    assert np.array_equal(loaded.u, built.u) and loaded.meta == built.meta
    # The same request is served from the files, the recorded run time included.
    again = parabolic_reference(
        m, initial, boundary, 0.1, 16, problem="mode", cache=cache
    )
    assert again.meta == built.meta and np.array_equal(again.u, built.u)
    # A different time, resolution or problem label is solved afresh.
    other = parabolic_reference(
        m, initial, boundary, 0.2, 16, problem="mode", cache=cache
    )
    assert other.meta["t_end"] == 0.2 and not other.matches(built.meta)
    assert ParabolicReference.load(cache).meta["t_end"] == 0.2
    assert np.array_equal(
        chebyshev_parabolic(m, initial, boundary, 0.2, x, 16), other.evaluate(x)
    )


@pytest.mark.parametrize("delta", [0.04, 0.0025])
def test_chebyshev_equilibrium_on_a_smooth_edge_matches_the_quadrature(delta):
    # The elements resolve the transition; the floor is the round-off of a
    # solve whose operator norm reaches 1e11 (E3.2: 7e-12 at δ = 0.0025).
    m = SmoothEdges(matlab_alpha(), delta)
    x = np.linspace(-1, 1, 401)
    ref = equilibrium_exact(m, 1.0, 0.0, x)
    for n in (24, 32):
        v = chebyshev_equilibrium(m, 1.0, 0.0, x, n_cheb=n)
        assert np.max(np.abs(v - ref)) < 3e-11


@pytest.mark.parametrize("delta", [0.0, 0.01, 0.0025])
def test_parabolic_reference_agrees_between_two_resolutions_at_every_delta(delta):
    # The ticket's acceptance line: 1e-10 between resolutions (E3.2 measured
    # 9e-15, 6e-13 and 3e-12 for these three at 24 against 32 nodes).
    m = SmoothEdges(matlab_alpha(), delta)
    x = np.linspace(-1, 1, 401)
    u = [
        parabolic_reference(m, np.zeros_like, ramp_boundary(1.0), 2.0, n).evaluate(x)
        for n in (24, 32)
    ]
    assert np.max(np.abs(u[0] - u[1])) < 1e-10
    assert np.max(np.abs(u[1])) == pytest.approx(1.0)  # the ramped end
