import numpy as np
import pytest

from heat_interfaces.heat2d.exact import LayeredExact, case1_exact, control_exact

Y = np.linspace(0.0, 1.0, 41)
X = np.linspace(0.0, 1.0, 17, endpoint=False)


def matlab_case1(x, y, k1=1.0, k2=0.2, yli=0.6, yui=0.8):
    """``laplaceSetup.m`` transcribed: unshifted exponentials and its 6 × 6 system."""
    c = 2 * np.pi
    e = np.exp
    a = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [e(c * yli), e(-c * yli), -e(c * yli), -e(-c * yli), 0, 0],
            [
                c * k1 * e(c * yli),
                -c * k1 * e(-c * yli),
                -c * k2 * e(c * yli),
                c * k2 * e(-c * yli),
                0,
                0,
            ],
            [0, 0, e(c * yui), e(-c * yui), -e(c * yui), -e(-c * yui)],
            [
                0,
                0,
                c * k2 * e(c * yui),
                -c * k2 * e(-c * yui),
                -c * k1 * e(c * yui),
                c * k1 * e(-c * yui),
            ],
            [0, 0, 0, 0, e(c), e(-c)],
        ]
    )
    v = np.linalg.solve(a, [0, 0, 0, 0, 0, 1])
    z1 = v[0] * e(y * c) + v[1] * e(-y * c)
    z2 = v[2] * e(y * c) + v[3] * e(-y * c)
    z3 = v[4] * e(y * c) + v[5] * e(-y * c)
    return np.sin(c * x) * np.where(y < yli, z1, np.where(y < yui, z2, z3))


def test_control_is_the_sinh_solution():
    u = control_exact()
    np.testing.assert_allclose(
        u.v(Y), np.sinh(2 * np.pi * Y) / np.sinh(2 * np.pi), atol=1e-14
    )
    np.testing.assert_allclose(u(X, 1.0), np.sin(2 * np.pi * X), atol=1e-14)
    np.testing.assert_allclose(u(X, 0.0), 0.0, atol=1e-14)
    assert u.kappas == pytest.approx([2 * np.pi])


def test_case1_matches_the_matlab_setup_and_eq_34s_conditions():
    u = case1_exact()
    xx, yy = np.meshgrid(X, Y)
    np.testing.assert_allclose(u(xx, yy), matlab_case1(xx, yy), atol=1e-13)
    # Continuity of u and of α u_y at both interfaces, from either side.
    for yb in (0.6, 0.8):
        below, above = yb - 1e-12, yb + 1e-12
        assert u.v(below) == pytest.approx(u.v(above), abs=1e-10)
        assert u.flux_y(0.3, below) == pytest.approx(u.flux_y(0.3, above), abs=1e-9)
    assert u.layer(np.array([0.59, 0.6, 0.79, 0.8, 0.81])).tolist() == [0, 1, 1, 2, 2]
    assert u.v(0.0) == pytest.approx(0.0, abs=1e-15) and u.v(1.0) == pytest.approx(1.0)


def test_every_layer_solves_its_ode():
    for u in (case1_exact(), case1_exact(1.0), LayeredExact((2.0, 0.5), (0.35,), 0.3)):
        for k, (lo, hi) in enumerate(zip(u.edges[:-1], u.edges[1:], strict=True)):
            y = np.linspace(lo + 0.02, hi - 0.02, 7)
            s = 1e-3
            vpp = (u.v(y + s) - 2 * u.v(y) + u.v(y - s)) / s**2
            np.testing.assert_allclose(vpp, u.kappas[k] ** 2 * u.v(y), rtol=1e-5)
            np.testing.assert_allclose(
                u.v_y(y), (u.v(y + s) - u.v(y - s)) / (2 * s), rtol=1e-5
            )


def test_growth_shifts_the_exponents_as_eq_86_says():
    u = case1_exact(growth=1.0)
    np.testing.assert_allclose(u.kappas**2, 4 * np.pi**2 + np.array([1.0, 5.0, 1.0]))
    assert u(0.25, 0.5, t=0.3) == pytest.approx(np.exp(0.3) * u(0.25, 0.5))
    assert u(X, 1.0, t=0.0) == pytest.approx(np.sin(2 * np.pi * X))


def test_layered_exact_refuses_malformed_layers():
    with pytest.raises(ValueError, match="one break fewer"):
        LayeredExact((1.0, 0.2), (0.3, 0.6))
    with pytest.raises(ValueError, match="positive"):
        LayeredExact((1.0, -0.2, 1.0), (0.3, 0.6))
    with pytest.raises(ValueError, match="increase"):
        LayeredExact((1.0, 0.2, 1.0), (0.6, 0.3))
