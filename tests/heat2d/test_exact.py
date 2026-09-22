import numpy as np
import pytest

from heat_interfaces.heat1d.domain import OnInterval, SmoothEdges
from heat_interfaces.heat2d.domain import (
    Band,
    Constant2D,
    FlatLine,
    SineProduct,
    SmoothBand,
    build_node_set,
    case1,
    case2,
    with_smooth_edges,
)
from heat_interfaces.heat2d.exact import (
    REFERENCE_MAX_WIDTH,
    REFERENCE_N_CHEB,
    LayeredExact,
    RingMode,
    SeparableReference,
    case1_exact,
    case1_reference,
    control_exact,
    profile_medium,
    ring_exact,
    separable_reference,
)
from heat_interfaces.heat2d.operators import build_stencils, interface_aware_operator
from heat_interfaces.heat2d.rbf import BOUNDARY
from heat_interfaces.heat2d.solve import rms_error, solve_equilibrium

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
    # The last is a four-ring chain with a thin insulating ring in the middle
    # (1e-6 wide at α = 1e-6 / 1.5, resistance 1.5): the increment walk's
    # multi-hop path, which no case needs.
    chain = RingMode((0.15, 0.3, 0.3 + 1e-6, 0.45), (1.0, 3.0, 1e-6 / 1.5, 1.0, 0.5))
    for u in (
        ring_exact(),
        ring_exact(3),
        RingMode((0.2, 0.3), (2.0, 0.5, 1.0), 1),
        chain,
    ):
        # At each radius the ring above owns the point; the ring below is
        # evaluated from its own pair (a probe at r − 1e-13 would already see
        # the thin ring's climb, 4.5e5 per unit radius).
        m = u.mode
        for k, r in enumerate(u.radii):
            a, b = u._coefficients[k]
            assert a * r**m + b * r**-m == pytest.approx(u.radial(r), abs=1e-9)
            flux_below = u.alphas[k] * m * (a * r ** (m - 1) - b * r ** (-m - 1))
            assert flux_below == pytest.approx(u.flux(r), rel=1e-9)
        assert u.radial(u.scale_radius) == pytest.approx(1.0)
        # Harmonic: the five-point Laplacian vanishes to its own truncation.
        for x, y in ((0.55, 0.62), (0.3, 0.3), (0.72, 0.31), (0.1, 0.9)):
            s = 2e-4
            lap = u(x + s, y) + u(x - s, y) + u(x, y + s) + u(x, y - s) - 4 * u(x, y)
            assert abs(lap / s**2) < 3e-5, (x, y, lap / s**2)
    # Across the chain's thin ring the climb is the resistance times the flux.
    climb = chain.radial(0.3 + 1e-6 + 1e-13) - chain.radial(0.3 - 1e-13)
    assert climb == pytest.approx(1.5 * chain.flux(0.3 - 1e-13), rel=1e-4)
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


# --- the separable reference through a smooth flat band (E4.2) ---------------

STUDY_DELTAS = (0.04, 0.01, 0.005, 0.0025)
"""E4.3's edge widths (plan, #34)."""

YS = np.linspace(0.0, 1.0, 4001)


@pytest.mark.parametrize("growth", [0.0, 1.0])
def test_the_reference_at_delta_zero_is_the_analytic_case1_solution(growth):
    ref, exact = case1_reference(0.0, growth), case1_exact(growth)
    np.testing.assert_allclose(ref.v(YS), exact.v(YS), rtol=0, atol=1e-12)
    off = YS[(np.abs(YS - 0.6) > 1e-9) & (np.abs(YS - 0.8) > 1e-9)]
    np.testing.assert_allclose(ref.v_y(off), exact.v_y(off), rtol=0, atol=1e-11)
    xx, yy = np.meshgrid(X, YS[::40])
    for t in (-0.3, 0.0, 0.1):
        np.testing.assert_allclose(ref(xx, yy, t), exact(xx, yy, t), atol=1e-12)
        np.testing.assert_allclose(
            ref.flux_y(xx, yy, t), exact.flux_y(xx, yy, t), atol=1e-11
        )
    # A point on a break reads the layer above it, as LayeredExact does.
    for yb in (0.6, 0.8):
        assert ref.v_y(yb) == pytest.approx(exact.v_y(yb), abs=1e-11)
    assert ref.elements == 10 and ref.unknowns == 10 * (REFERENCE_N_CHEB - 1)


@pytest.mark.parametrize("delta", STUDY_DELTAS)
@pytest.mark.parametrize("growth", [0.0, 1.0])
def test_the_reference_is_converged_at_every_delta_of_the_study(delta, growth):
    # Stiff note §4.1: against 24 nodes on 0.05-wide elements the reference
    # agrees to 1e-12 … 2e-11, the collocation's round-off floor on the
    # δ-wide elements (E3.2's, row scaling does not move it); 5e-11 leaves
    # room for the run-to-run noise of threaded BLAS.
    ref = case1_reference(delta, growth)
    fine = case1_reference(delta, growth, 24, 0.05)
    assert np.abs(ref.v(YS) - fine.v(YS)).max() < 5e-11
    assert np.abs(ref.flux_y(0.25, YS) - fine.flux_y(0.25, YS)).max() < 1e-9
    assert ref.v(0.0) == pytest.approx(0.0, abs=1e-15)
    assert ref.v(1.0) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("delta", [0.01, 0.0025])
@pytest.mark.parametrize("growth", [0.0, 1.0])
def test_the_reference_solves_its_ode_through_the_edges(delta, growth):
    # Independent of the collocation: the flux α v′ differentiated by central
    # differences against (κ² α + c) v with the 2-D medium's own α.
    ref = case1_reference(delta, growth)
    medium = SmoothBand(case1().material, delta)
    y = np.linspace(0.02, 0.98, 20001)
    s = 1e-6
    flux = (ref.flux_y(0.25, y + s) - ref.flux_y(0.25, y - s)) / (2 * s)
    flux /= np.sin(0.5 * np.pi)
    rhs = ((2 * np.pi) ** 2 * medium.alpha(np.full_like(y, 0.25), y) + growth) * ref.v(
        y
    )
    assert np.abs(flux - rhs).max() < 1e-6 * np.abs(rhs).max()
    # And the flux is continuous where v_y jumps by the contrast at δ = 0.
    alpha = medium.alpha(np.full_like(y, 0.25), y)
    np.testing.assert_allclose(ref.flux_y(0.25, y), alpha * ref.v_y(y), atol=1e-10)


def test_the_reference_approaches_the_jump_at_first_order_in_delta():
    # H10's floor: the δ = 0 construction is off by the two references'
    # difference, O(δ) (stiff note §1.7, §4.1). sup |v_δ − v_0| / δ is 1.55
    # at δ = 1e-3 and 1.58 at 5e-4.
    v0 = case1_reference(0.0).v(YS)
    gaps = [np.abs(case1_reference(d).v(YS) - v0).max() for d in (1e-3, 5e-4)]
    assert 1.9 < gaps[0] / gaps[1] < 2.0
    assert 1.5 < gaps[1] / 5e-4 < 1.65


def test_profile_medium_is_the_smooth_band_in_y_bit_for_bit():
    for delta in (0.0, 0.01, 0.0025):
        band = SmoothBand(case1().material, delta)
        medium = profile_medium(band)
        assert isinstance(medium, OnInterval) and (medium.lo, medium.hi) == (0.0, 1.0)
        assert isinstance(medium.medium, SmoothEdges)
        assert medium.medium.delta == delta
        x = np.full_like(YS, 0.41)
        assert np.array_equal(band.alpha(x, YS), medium.alpha(YS))
        assert np.array_equal(band.gradient(x, YS)[1], medium.alpha_x(YS))
        edges, _ = medium.elements()
        assert edges[0] == 0.0 and edges[-1] == 1.0
    assert profile_medium(case1().material) == profile_medium(
        SmoothBand(case1().material, 0.0)
    )


def test_profile_medium_refuses_what_does_not_separate():
    with pytest.raises(ValueError, match="flat"):
        profile_medium(case2().material)
    sine_inside = Band(
        FlatLine(0.6), FlatLine(0.8), SineProduct(0.2, 0.1), Constant2D(1.0)
    )
    with pytest.raises(ValueError, match="constant"):
        profile_medium(SmoothBand(sine_inside, 0.01))


def test_boundary_values_are_the_papers_dirichlet_rows_at_any_time():
    ref = case1_reference(0.01, growth=1.0)
    bottom, top = ref.boundary_values()
    assert bottom == 0.0
    for t in (-0.2, 0.0, 0.1):
        np.testing.assert_array_equal(top(X, 1.0, t), np.exp(t) * np.sin(2 * np.pi * X))
        np.testing.assert_allclose(ref(X, 1.0, t), top(X, 1.0, t), atol=1e-15)
        np.testing.assert_allclose(ref(X, 0.0, t), 0.0, atol=1e-15)
    assert top(0.25, 1.0) == pytest.approx(
        1.0
    )  # t defaults to 0, as solve_equilibrium calls it
    general = separable_reference(
        Band(FlatLine(0.3), FlatLine(0.5), Constant2D(3.0), Constant2D(1.0))
    )
    exact = LayeredExact((1.0, 3.0, 1.0), (0.3, 0.5))
    np.testing.assert_allclose(general.v(YS), exact.v(YS), atol=1e-12)
    assert isinstance(general, SeparableReference)
    assert (general.n_cheb, general.max_width) == (
        REFERENCE_N_CHEB,
        REFERENCE_MAX_WIDTH,
    )


def test_the_delta_zero_reference_through_the_solver_is_e25s_case1_error():
    # The medium, the reference and its boundary values plug into the 2016
    # solver unchanged: on the smooth medium at δ = 0 the aware operator's
    # error against the reference is port notes §2.4's 1.60e-5 at 1250 nodes,
    # and it equals the error against case1_exact to the reference's 4e-13.
    domain = with_smooth_edges(case1(), 0.0)
    nodes = build_node_set(domain, 1250)
    st = build_stencils(nodes, domain, interface=BOUNDARY)
    op = interface_aware_operator(nodes, domain.material, st)
    ref = case1_reference(0.0)
    u = solve_equilibrium(op, nodes, ref.boundary_values())
    err = rms_error(u, ref(nodes.x, nodes.y))
    assert err == pytest.approx(1.60e-5, rel=0.01)
    assert err == pytest.approx(
        rms_error(u, case1_exact()(nodes.x, nodes.y)), abs=1e-12
    )
