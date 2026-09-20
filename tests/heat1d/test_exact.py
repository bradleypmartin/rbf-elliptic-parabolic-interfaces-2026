import numpy as np
import pytest

from heat_interfaces.heat1d.domain import (
    Constant,
    PiecewiseAlpha,
    Smooth,
    dissertation_alpha,
    jump_alpha,
)
from heat_interfaces.heat1d.exact import (
    equilibrium_exact,
    equilibrium_flux,
    inverse_alpha_integral,
)


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
