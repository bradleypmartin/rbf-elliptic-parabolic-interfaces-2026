import numpy as np
import pytest

from heat_interfaces.heat1d.domain import (
    DISSERTATION_BC,
    Constant,
    PiecewiseAlpha,
    Smooth,
    dissertation_alpha,
    jump_alpha,
    matlab_alpha,
)
from heat_interfaces.heat1d.exact import (
    chebyshev_equilibrium,
    chebyshev_lobatto,
    chebyshev_parabolic,
    equilibrium_exact,
    equilibrium_flux,
    inverse_alpha_integral,
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
