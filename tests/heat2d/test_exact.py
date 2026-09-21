import numpy as np
import pytest

from heat_interfaces.heat2d.exact import (
    LayeredExact,
    RingMode,
    case1_exact,
    control_exact,
    ring_exact,
)

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


def test_ring_mode_is_continuous_with_its_flux_and_harmonic_on_every_ring():
    for u in (ring_exact(), ring_exact(3), RingMode((0.2, 0.3), (2.0, 0.5, 1.0), 1)):
        for r in u.radii:
            below, above = r - 1e-13, r + 1e-13
            assert u.radial(below) == pytest.approx(u.radial(above), abs=1e-9)
            assert u.flux(below) == pytest.approx(u.flux(above), rel=1e-9)
        assert u.radial(u.scale_radius) == pytest.approx(1.0)
        # Harmonic: the five-point Laplacian vanishes to its own truncation.
        for x, y in ((0.55, 0.62), (0.3, 0.3), (0.72, 0.31), (0.1, 0.9)):
            s = 2e-4
            lap = u(x + s, y) + u(x - s, y) + u(x, y + s) + u(x, y - s) - 4 * u(x, y)
            assert abs(lap / s**2) < 3e-5, (x, y, lap / s**2)
    u = ring_exact()
    # Inside it is the harmonic polynomial Re (x + iy)²; across the ring R
    # climbs by about m (α_out / α_ring) (width) r: 1.05 against 0.12 inside.
    x, y = np.array([0.6, 0.3]), np.array([0.7, 0.55])
    a = u._coefficients[0, 0]
    np.testing.assert_allclose(u(x, y), a * ((x - 0.5) ** 2 - (y - 0.5) ** 2))
    assert 0.6 < u.radial(0.35) - u.radial(0.349) < 0.7
    assert u.radial(0.349) == pytest.approx(0.0775, abs=1e-3)
    assert u.ring(np.array([0.1, 0.349, 0.3495, 0.35, 0.4])).tolist() == [0, 1, 1, 2, 2]


def test_ring_mode_walk_is_stable_across_a_ring_a_billionth_wide():
    # E2.9: on EABE eq. 40's ring, 1/s wide at α = 1/(1.5 s), the pair (a, b) is
    # O(s) and evaluating a r^m + b r^-m at the outer radius cancels to an O(1)
    # climb; the walk forms the climb from expm1 increments instead, so the
    # outside coefficients keep their digits and tend to the thin-layer limit,
    # where the climb is the resistance 1.5 times the flux.
    old = ring_exact()
    np.testing.assert_allclose(
        old._coefficients,
        [
            [6.36305165e-01, 0.0],
            [4.77547026e02, -7.07520118],
            [3.34804304, 4.07473100e-02],
        ],
        rtol=1e-8,
    )
    outer = {}
    for s in (1e6, 1e9, 1e11):
        u = ring_exact(s=s)
        r1, r2 = u.radii
        assert u.radii == pytest.approx((0.35 - 1 / s, 0.35)) and u.alphas[1] == 1 / (
            1.5 * s
        )
        climb = u.radial(r2) - u._coefficients[0, 0] * r1**2
        # O(1/s) from the ring's thickness, plus the stored width's rounding (8e-8).
        assert climb == pytest.approx(1.5 * u.flux(r2), rel=3.0 / s + 2e-7)
        assert u.radial(u.scale_radius) == pytest.approx(1.0)
        outer[s] = u._coefficients[2]
    np.testing.assert_allclose(outer[1e11], outer[1e9], rtol=1e-6)


def test_ring_mode_refuses_malformed_rings():
    with pytest.raises(ValueError, match="one radius fewer"):
        RingMode((0.3,), (1.0, 2.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        RingMode((0.3,), (1.0, -2.0))
    with pytest.raises(ValueError, match="mode"):
        RingMode((0.3,), (1.0, 2.0), 0)
    with pytest.raises(ValueError, match="increase"):
        RingMode((0.3, 0.2), (1.0, 2.0, 1.0))
