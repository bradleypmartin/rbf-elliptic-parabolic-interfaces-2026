"""The 2-D interface algebra: coefficient operators, frames, the interface expansion,
the continuity matrices, the translated basis on curved interfaces, and the
stencil weights (dissertation §5.3, EABE §2.2.3), and the interpolation weights
of E2.6's resampling on the same system."""

import numpy as np
import pytest

from heat_interfaces.heat2d.domain import (
    STRIP,
    Band,
    Circle,
    Constant2D,
    Domain,
    FlatLine,
    SineGraph,
    SineProduct,
    build_node_set,
    case1,
)
from heat_interfaces.heat2d.interface import (
    Frame,
    LocalInterface,
    build_warp,
    coefficient_dx,
    coefficient_dy,
    coefficient_operator,
    continuity_conditions,
    continuity_matrix,
    frame_at,
    frame_change,
    interface_expansion,
    interface_stencil,
    interpolation_weights,
    local_interface,
    local_taylor,
    multiplication_matrix,
    restriction_matrix,
    stencil_weights,
    substitution_matrix,
    table_to_vector,
    translated_basis,
    translation_matrix,
    vector_to_table,
)
from heat_interfaces.heat2d.neighbors import knn
from heat_interfaces.heat2d.rbf import gaussian, polynomial_block, polynomial_exponents

P = 4
Q = len(polynomial_exponents(P))
E = polynomial_exponents(P)
RNG = np.random.default_rng(7)


def random_table(degree, rng=RNG):
    t = rng.standard_normal((degree + 1, degree + 1))
    i, j = np.indices(t.shape)
    t[i + j > degree] = 0.0
    return t


def evaluate(c, x, y, degree=P):
    return polynomial_block(np.asarray(x, float), np.asarray(y, float), degree) @ c


def position(i, j):
    return int(np.flatnonzero((E[:, 0] == i) & (E[:, 1] == j))[0])


# --- coefficient algebra ------------------------------------------------------


def test_table_and_vector_round_trip_in_the_graded_order():
    t = random_table(P)
    v = table_to_vector(t)
    assert v.shape == (Q,)
    assert v[position(2, 1)] == t[2, 1] and v[0] == t[0, 0]
    np.testing.assert_array_equal(vector_to_table(v, P), t)


def test_coefficient_derivatives_match_central_differences():
    c = RNG.standard_normal(Q)
    x, y = RNG.standard_normal(6), RNG.standard_normal(6)
    h = 1e-5
    dx = (evaluate(c, x + h, y) - evaluate(c, x - h, y)) / (2 * h)
    dy = (evaluate(c, x, y + h) - evaluate(c, x, y - h)) / (2 * h)
    np.testing.assert_allclose(evaluate(coefficient_dx(P) @ c, x, y), dx, atol=1e-8)
    np.testing.assert_allclose(evaluate(coefficient_dy(P) @ c, x, y), dy, atol=1e-8)


def test_multiplication_matrix_is_the_product_below_the_truncation():
    # A degree-1 alpha times a degree-3 polynomial fits in degree 4: exact.
    t = np.zeros((P + 1, P + 1))
    t[0, 0], t[1, 0], t[0, 1] = 0.7, -0.3, 0.2
    c = RNG.standard_normal(Q)
    c[E.sum(axis=1) > 3] = 0.0
    x, y = RNG.standard_normal(6), RNG.standard_normal(6)
    got = evaluate(multiplication_matrix(t) @ c, x, y)
    want = evaluate(table_to_vector(t), x, y) * evaluate(c, x, y)
    np.testing.assert_allclose(got, want, atol=1e-12)
    # A constant alpha is the identity times its value at every degree.
    const = np.zeros((P + 1, P + 1))
    const[0, 0] = 0.2
    np.testing.assert_array_equal(multiplication_matrix(const), 0.2 * np.eye(Q))


def test_coefficient_operator_is_div_alpha_grad_below_the_top_degree():
    # With alpha and c of degree 2, alpha c_x has degree 3 < 4 and D c is exact.
    t = random_table(2)
    t = np.pad(t, (0, P - 2))
    c = RNG.standard_normal(Q)
    c[E.sum(axis=1) > 2] = 0.0
    x, y = 0.3 * RNG.standard_normal(6), 0.3 * RNG.standard_normal(6)
    h = 1e-4
    a = table_to_vector(t)

    def flux_x(xx, yy):
        return evaluate(a, xx, yy) * evaluate(coefficient_dx(P) @ c, xx, yy)

    def flux_y(xx, yy):
        return evaluate(a, xx, yy) * evaluate(coefficient_dy(P) @ c, xx, yy)

    want = (flux_x(x + h, y) - flux_x(x - h, y)) / (2 * h)
    want += (flux_y(x, y + h) - flux_y(x, y - h)) / (2 * h)
    np.testing.assert_allclose(evaluate(coefficient_operator(t) @ c, x, y), want, 1e-6)
    # At full degree the coefficients of D c below degree p are still exact:
    # compare with the untruncated operator on degree-2p tables.
    t, c = random_table(P), RNG.standard_normal(Q)
    big = np.pad(t, (0, P))
    c_big = table_to_vector(np.pad(vector_to_table(c, P), (0, P)))
    exact = vector_to_table(coefficient_operator(big) @ c_big, 2 * P)[: P + 1, : P + 1]
    got = vector_to_table(coefficient_operator(t) @ c, P)
    i, j = np.indices(got.shape)
    low = i + j <= P - 1
    np.testing.assert_allclose(got[low], exact[low], atol=1e-12)
    assert np.abs(got[i + j == P] - exact[i + j == P]).max() > 1e-3


def test_substitution_matrix_keeps_the_polynomial_under_rotation_and_shift():
    c = RNG.standard_normal(Q)
    angle, shift = 0.7, (0.3, -0.2)
    s = substitution_matrix(P, angle, shift)
    xp, yp = RNG.standard_normal(8), RNG.standard_normal(8)
    x = np.cos(angle) * xp - np.sin(angle) * yp + shift[0]
    y = np.sin(angle) * xp + np.cos(angle) * yp + shift[1]
    np.testing.assert_allclose(evaluate(s @ c, xp, yp), evaluate(c, x, y), atol=1e-12)
    np.testing.assert_array_equal(substitution_matrix(P, 0.0), np.eye(Q))


def test_restriction_matrix_reads_the_polynomial_along_the_interface():
    f = np.array([0.0, 0.0, 0.3, -0.1, 0.0])
    c = RNG.standard_normal(Q)
    c[E.sum(axis=1) > 2] = 0.0  # degree 2 in (x, y) restricts to degree 6; keep 4
    x = np.linspace(-0.5, 0.5, 9)
    along = np.polynomial.polynomial.polyval(x, restriction_matrix(f, P) @ c)
    exact = evaluate(c, x, np.polynomial.polynomial.polyval(x, f))
    # The truncation at degree 4 drops x^5, x^6 terms of order f² x²(x², xy).
    np.testing.assert_allclose(along, exact, atol=0.3**2 * 0.5**5 * np.abs(c).max())
    flat = restriction_matrix(np.zeros(P + 1), P)
    for m in range(P + 1):
        assert flat[m, position(m, 0)] == 1.0
    assert flat.sum() == P + 1
    with pytest.raises(ValueError, match="coefficients"):
        restriction_matrix(np.zeros(3), P)


# --- continuity -----------------------------------------------------------------


def test_continuity_conditions_count_one_row_per_monomial():
    assert continuity_conditions(4) == [
        ("u", 0, 5),
        ("flux", 0, 4),
        ("u", 1, 3),
        ("flux", 1, 2),
        ("u", 2, 1),
    ]
    assert continuity_conditions(2) == [("u", 0, 3), ("flux", 0, 2), ("u", 1, 1)]
    for p in range(1, 7):
        assert sum(n for _, _, n in continuity_conditions(p)) == (p + 1) * (p + 2) // 2


def constant_table(value, degree=P):
    t = np.zeros((degree + 1, degree + 1))
    t[0, 0] = value
    return t


def test_flat_constant_continuity_matrix_is_the_matlab_continuity_creator():
    # alpha constant, flat interface: rows pick the pure x powers of D^k c and
    # of alpha d/dy D^k c, exactly the MATLAB's nullMatrix rows.
    a = 0.2
    c = continuity_matrix(constant_table(a), np.zeros(P + 1))
    assert c.shape == (Q, Q)
    for m in range(5):  # (u, 0): coefficient of x^m
        assert c[m, position(m, 0)] == 1.0 and np.count_nonzero(c[m]) == 1
    for m in range(4):  # (flux, 0): a * coefficient of x^m y
        assert c[5 + m, position(m, 1)] == a and np.count_nonzero(c[5 + m]) == 1
    # (u, 1): the constant term of the Laplacian: 2 a (x² + y²)
    assert c[9, position(2, 0)] == 2 * a and c[9, position(0, 2)] == 2 * a
    assert np.linalg.cond(c) < 1e3


def test_translation_carries_a_piecewise_linear_profile_across_a_flat_jump():
    # alpha 1 -> 0.2 across y' = 0: u = y' on the minus side becomes 5 y'.
    t = translation_matrix(constant_table(1.0), constant_table(0.2), np.zeros(P + 1))
    y = np.zeros(Q)
    y[position(0, 1)] = 1.0
    want = np.zeros(Q)
    want[position(0, 1)] = 5.0
    np.testing.assert_allclose(t @ y, want, atol=1e-13)
    # 1 and x' are continuous with continuous flux and D u = 0 already.
    for i, j in ((0, 0), (1, 0)):
        e = np.zeros(Q)
        e[position(i, j)] = 1.0
        np.testing.assert_allclose(t @ e, e, atol=1e-13)
    # x'² has D u = 2 on the minus side; on the plus side the (u, 1) row makes
    # 0.2 (2 + 2 c) = 2, so it becomes x'² + 4 y'².
    e = np.zeros(Q)
    e[position(2, 0)] = 1.0
    want = e.copy()
    want[position(0, 2)] = 4.0
    np.testing.assert_allclose(t @ e, want, atol=1e-13)


def test_translation_is_invariant_to_the_row_equilibration():
    a, b = random_table(P), random_table(P)
    a[0, 0], b[0, 0] = 1.0, 0.3
    f = np.array([0.0, 0.0, 0.1, 0.02, 0.0])
    raw = np.linalg.solve(continuity_matrix(b, f), continuity_matrix(a, f))
    np.testing.assert_allclose(translation_matrix(a, b, f), raw, rtol=1e-9, atol=1e-12)


# --- frames and the interface expansion ----------------------------------------


@pytest.mark.parametrize("curve", [SineGraph(0.6), Circle(0.35), FlatLine(0.8)])
def test_frame_puts_y_prime_along_the_normal_and_x_prime_along_the_curve(curve):
    s = 0.13
    frame = frame_at(curve, s, 0.05)
    x0, y0 = curve.point(s)
    nx, ny = curve.normal(s)
    xi, eta = frame.local(x0 + 0.01 * nx, y0 + 0.01 * ny)
    np.testing.assert_allclose([xi, eta], [0.0, 0.2], atol=1e-14)
    # A point a little further along the curve has x' > 0 (to rounding in y').
    px, py = curve.point(s + 1e-4)
    xi, eta = frame.local(px, py)
    assert abs(xi) > 0 and abs(eta) < 1e-3 * abs(xi)
    assert frame.rotate_in(nx, ny) == pytest.approx((0.0, 1.0), abs=1e-14)


def test_frame_local_wraps_x_across_the_seam():
    frame = frame_at(FlatLine(0.6), 0.99, 0.1)
    xi, eta = frame.local(0.01, 0.6)
    np.testing.assert_allclose([xi, eta], [0.2, 0.0], atol=1e-13)


def test_frame_change_is_exact_between_two_frames():
    a = Frame(0.3, 0.6, 0.4, 0.05)
    b = Frame(0.32, 0.61, -0.1, 0.05)
    c = RNG.standard_normal(Q)
    s = frame_change(a, b, P)
    x, y = 0.3 + 0.05 * RNG.standard_normal(6), 0.6 + 0.05 * RNG.standard_normal(6)
    np.testing.assert_allclose(
        evaluate(s @ c, *b.local(x, y)), evaluate(c, *a.local(x, y)), atol=1e-12
    )
    with pytest.raises(ValueError, match="scale"):
        frame_change(a, Frame(0.3, 0.6, 0.0, 0.1), P)


def test_local_taylor_is_alpha_in_the_rotated_scaled_frame():
    piece = SineProduct(0.2, 0.1)
    frame = frame_at(SineGraph(0.6), 0.21, 0.05)
    t = local_taylor(piece, frame, 5)
    xi, eta = 0.4 * RNG.standard_normal(6), 0.4 * RNG.standard_normal(6)
    c, s = np.cos(frame.angle), np.sin(frame.angle)
    x = frame.x0 + frame.scale * (c * xi - s * eta)
    y = frame.y0 + frame.scale * (s * xi + c * eta)
    got = evaluate(table_to_vector(t), xi, eta, 5)
    np.testing.assert_allclose(got, piece.alpha(x, y), atol=2e-7)
    # (0.4 * 0.05 * 2π)^6 / 6! is the degree-6 remainder.


def test_interface_expansion_matches_the_circle_and_the_sine_curvature():
    circle = Circle(0.35)
    frame = frame_at(circle, 0.1, 0.05)
    f = interface_expansion(circle, 0.1, frame, P)
    # y' = R - sqrt(R² - x'²) towards the centre, i.e. away from +normal.
    r = 0.35 / 0.05
    # f_4 carries the 9-sample fit's O(f^(9)) error, 1e-10 here.
    np.testing.assert_allclose(f, [0, 0, -1 / (2 * r), 0, -1 / (8 * r**3)], atol=1e-9)
    assert abs(f[2] + 1 / (2 * r)) < 1e-10
    sine = SineGraph(0.6)
    frame = frame_at(sine, 0.13, 0.05)
    f = interface_expansion(sine, 0.13, frame, P)
    assert f[2] == pytest.approx(0.05 * sine.curvature(0.13) / 2, rel=1e-9)
    assert abs(f[0]) < 1e-12 and abs(f[1]) < 1e-12
    flat = FlatLine(0.6)
    np.testing.assert_array_equal(
        interface_expansion(flat, 0.13, frame_at(flat, 0.13, 0.05), P), 0.0
    )


def test_local_interface_flat_variant_zeroes_the_expansion_only():
    band = case2_band()
    curved = local_interface(band, 0, 0.13, 0.62, 0.05, P, curvature=True)
    flat = local_interface(band, 0, 0.13, 0.62, 0.05, P, curvature=False)
    assert curved.frame == flat.frame
    np.testing.assert_array_equal(flat.expansion, 0.0)
    assert np.abs(curved.expansion).max() > 1e-3
    np.testing.assert_array_equal(curved.minus, flat.minus)
    np.testing.assert_array_equal(curved.plus, flat.plus)
    assert curved.degree == P
    # Region 0 of the band is outside (alpha 1), region 1 the sine product.
    assert flat.minus[0, 0] == 1.0 and flat.plus[0, 0] == pytest.approx(
        SineProduct(0.2, 0.1).alpha(*SineGraph(0.6).point(curved_foot(0.13, 0.62)))
    )


def case2_band():
    return Band(SineGraph(0.6), SineGraph(0.8), SineProduct(0.2, 0.1), Constant2D(1.0))


def curved_foot(x, y):
    return SineGraph(0.6).closest(x, y)


# --- the translated basis along curved interfaces --------------------------------


def jumps_along(band, j, x, y, curvature, xi_values, scale=0.05):
    """Largest jump in u and in alpha n·grad u over the basis at arc offsets ±xi."""
    li = local_interface(band, j, x, y, scale, P, curvature)
    regions = translated_basis({j: li}, j + 1, j, j + 1)
    cm, cp = regions[j].coefficients, regions[j + 1].coefficients
    curve = band.interfaces[j]
    s0 = curve.closest(x, y)
    out = []
    for xi in xi_values:
        s = s0 + np.array([-xi, xi]) * scale / curve.length
        px, py = curve.point(s)
        nx, ny = curve.normal(s)
        v = polynomial_block(*li.frame.local(px, py), P)
        g_xi, g_eta = li.frame.rotate_in(nx, ny)
        du = (v @ coefficient_dx(P), v @ coefficient_dy(P))
        am = band.region_piece(j).alpha(px, py)[:, None]
        ap = band.region_piece(j + 1).alpha(px, py)[:, None]
        fm = am * (g_xi[:, None] * (du[0] @ cm) + g_eta[:, None] * (du[1] @ cm))
        fp = ap * (g_xi[:, None] * (du[0] @ cp) + g_eta[:, None] * (du[1] @ cp))
        out.append((np.abs(v @ cm - v @ cp).max(), np.abs(fm - fp).max()))
    return np.array(out)


def test_curved_basis_is_continuous_to_order_p_along_the_sine_interface():
    # Along y = 0.6 + 0.02 sin 2πx the jump in u falls like xi^(p+1) and the
    # jump in normal flux like xi^p as the arc offset xi halves; with the flat
    # assumption they fall like xi² and xi, the local linear approximation's own
    # error (EABE §2.2.3, "identical for a flat interface" otherwise).
    xi = np.array([0.5, 0.25, 0.125, 0.0625])
    curved = jumps_along(case2_band(), 0, 0.13, 0.62, True, xi)
    flat = jumps_along(case2_band(), 0, 0.13, 0.62, False, xi)
    ratio_c = curved[:-1] / curved[1:]
    ratio_f = flat[:-1] / flat[1:]
    np.testing.assert_allclose(ratio_c[:, 0], 2 ** (P + 1), rtol=0.05)
    np.testing.assert_allclose(ratio_c[:, 1], 2**P, rtol=0.05)
    np.testing.assert_allclose(ratio_f[:, 0], 4.0, rtol=0.05)
    np.testing.assert_allclose(ratio_f[-1, 1], 2.0, rtol=0.05)
    assert curved[-1, 0] < 1e-3 * flat[-1, 0]


def test_curved_basis_is_continuous_along_the_ring_with_its_1500_contrast():
    band = Band(
        Circle(0.349), Circle(0.35), SineProduct(1 / 1500, 1 / 3000), Constant2D(1.0)
    )
    x, y = 0.5 + 0.36 * np.cos(0.7), 0.5 + 0.36 * np.sin(0.7)
    xi = np.array([0.5, 0.25, 0.125])
    curved = jumps_along(band, 1, x, y, True, xi)
    flat = jumps_along(band, 1, x, y, False, xi)
    # A circle's expansion is even, so the u jump starts at xi^(p+2).
    np.testing.assert_allclose(curved[:-1, 0] / curved[1:, 0], 2 ** (P + 2), rtol=0.05)
    np.testing.assert_allclose(curved[:-1, 1] / curved[1:, 1], 2**P, rtol=0.05)
    np.testing.assert_allclose(flat[:-1, 0] / flat[1:, 0], 4.0, rtol=0.05)


def test_flat_and_curved_variants_coincide_on_a_flat_interface():
    band = case1().material
    for j, (x, y) in ((0, (0.4, 0.58)), (1, (0.9, 0.83))):
        curved = local_interface(band, j, x, y, 0.04, P, curvature=True)
        flat = local_interface(band, j, x, y, 0.04, P, curvature=False)
        assert curved.frame == flat.frame
        np.testing.assert_array_equal(curved.expansion, flat.expansion)
        np.testing.assert_array_equal(curved.minus, flat.minus)
        np.testing.assert_array_equal(curved.plus, flat.plus)
        rc = translated_basis({j: curved}, j, j, j + 1)
        rf = translated_basis({j: flat}, j, j, j + 1)
        np.testing.assert_array_equal(rc[j + 1].coefficients, rf[j + 1].coefficients)


def test_translated_basis_spans_the_same_space_from_either_anchor():
    li = local_interface(case2_band(), 0, 0.13, 0.62, 0.05, P)
    from_below = translated_basis({0: li}, 0, 0, 1)
    from_above = translated_basis({0: li}, 1, 0, 1)
    stack_b = np.vstack([from_below[0].coefficients, from_below[1].coefficients])
    stack_a = np.vstack([from_above[0].coefficients, from_above[1].coefficients])
    # Same column space: each stack is the other times an invertible matrix.
    m = np.linalg.lstsq(stack_a, stack_b, rcond=None)[0]
    np.testing.assert_allclose(stack_a @ m, stack_b, atol=1e-9 * np.abs(stack_b).max())
    with pytest.raises(ValueError, match="anchor"):
        translated_basis({0: li}, 2, 0, 1)
    with pytest.raises(ValueError, match="anchor"):
        translated_basis({0: li}, 0, 0, 0)


def test_two_interface_chain_changes_frame_between_the_ring_circles():
    band = Band(Circle(0.349), Circle(0.35), Constant2D(0.01), Constant2D(1.0))
    x, y = 0.5 + 0.36 * np.cos(1.1), 0.5 + 0.36 * np.sin(1.1)
    locals_ = {
        j: local_interface(band, j, x, y, 0.05, P, curvature=False) for j in (0, 1)
    }
    regions = translated_basis(locals_, 2, 0, 2)
    assert {r.frame for r in regions.values()} == {0, 1}
    # A radial profile continuous with continuous flux: slopes 1 : 100 : 1 in
    # y' across the flat-variant ring, carried from the outside region down.
    yp = np.zeros(Q)
    yp[position(0, 1)] = 1.0
    inside = regions[1].coefficients @ yp  # in frame 1 (outer circle)
    below = regions[0].coefficients @ yp  # in frame 0 (inner circle)
    assert inside[position(0, 1)] == pytest.approx(100.0)
    # Frame 0's origin is 0.001 further in along the same radius: the profile
    # 100 y'_1 = 100 (y'_0 - 0.001/0.05) + const becomes y'_0 at slope 1.
    assert below[position(0, 1)] == pytest.approx(1.0)
    assert below[0] == pytest.approx(-100.0 * 0.001 / 0.05, rel=1e-9)


# --- stencil weights ---------------------------------------------------------------


def piecewise_linear_case1(y):
    """The 1-D equilibrium of case 1's α: slopes 1, 5, 1 with continuous α u_y."""
    return np.where(
        y < 0.6, y, np.where(y <= 0.8, 0.6 + 5 * (y - 0.6), 1.6 + (y - 0.8))
    )


def test_stencil_weights_annihilate_the_piecewise_linear_equilibrium():
    domain = case1()
    nodes = build_node_set(domain, 1250)
    idx, _ = knn(nodes.xy, 30)
    region = domain.material.region_index(nodes.x, nodes.y)
    cross = np.flatnonzero(np.ptp(region[idx], axis=1) > 0)
    assert len(cross) > 300
    u = piecewise_linear_case1(nodes.y)
    for curvature in (True, False):
        for warp in (True, False):
            residual = [
                stencil_weights(
                    nodes.xy[idx[i]],
                    domain.material,
                    P,
                    curvature=curvature,
                    warp=warp,
                )
                @ u[idx[i]]
                for i in cross[::5]
            ]
            assert np.abs(residual).max() < 1e-9
    # A stencil in one region is refused.
    inside = int(np.flatnonzero(np.ptp(region[idx], axis=1) == 0)[0])
    with pytest.raises(ValueError, match="one region"):
        stencil_weights(nodes.xy[idx[inside]], domain.material, P)
    with pytest.raises(ValueError, match="monomial count"):
        stencil_weights(nodes.xy[idx[cross[0], :10]], domain.material, P)


@pytest.mark.parametrize("warp", [True, False])
def test_stencil_weights_are_exact_on_a_matched_quadratic_across_a_circle(warp):
    # alpha 1 inside r = 0.35, 5 outside (the band's region 1 reaches r = 0.9):
    # u = r² inside and (r² - R²) / 5 + R² outside is continuous with
    # continuous alpha u_r, and div(alpha grad u) = 4 on both sides. Both
    # pieces are quadratics that satisfy the interface conditions along the
    # whole circle, so the curved basis reproduces the pair to rounding of the
    # fitted expansion; the flat assumption is off by O(1). The warp only
    # touches the Gaussian block, so it must keep the polynomial exactness.
    band = Band(Circle(0.35), Circle(0.9), Constant2D(5.0), Constant2D(1.0))
    domain = Domain(band, (Circle(0.35),), STRIP)
    nodes = build_node_set(domain, 2500, iterations=10)
    r = np.hypot(nodes.x - 0.5, nodes.y - 0.5)
    u = np.where(r < 0.35, r**2, (r**2 - 0.35**2) / 5 + 0.35**2)
    idx, _ = knn(nodes.xy, 30)
    region = band.region_index(nodes.x, nodes.y)
    lo, hi = region[idx].min(axis=1), region[idx].max(axis=1)
    cross = np.flatnonzero((lo == 0) & (hi == 1))[::4]
    assert len(cross) > 40
    residual = {
        curvature: np.array(
            [
                stencil_weights(
                    nodes.xy[idx[i]], band, P, curvature=curvature, warp=warp
                )
                @ u[idx[i]]
                - 4.0
                for i in cross
            ]
        )
        for curvature in (True, False)
    }
    assert np.abs(residual[True]).max() < 1e-6, np.abs(residual[True]).max()
    assert np.abs(residual[False]).max() > 0.1


def test_stencil_weights_refuse_near_coincident_nodes_anywhere_in_the_stencil():
    # The guard of rbf_fd_weights (#16) is shared: two non-centre nodes 1e-12
    # apart would make the Gaussian block singular without any exception.
    domain = case1()
    nodes = build_node_set(domain, 1250)
    idx, _ = knn(nodes.xy, 30)
    region = domain.material.region_index(nodes.x, nodes.y)
    i = int(np.flatnonzero(np.ptp(region[idx], axis=1) > 0)[0])
    xy = nodes.xy[idx[i]].copy()
    xy[7] = xy[12] + 1e-12
    with pytest.raises(ValueError, match="coincide or nearly so"):
        stencil_weights(xy, domain.material, P)


def concentric_band():
    """alpha 1 inside r = 0.30 and outside r = 0.35, 4 on the ring between."""
    return Band(Circle(0.30), Circle(0.35), Constant2D(4.0), Constant2D(1.0))


def matched_radial_quadratic(r):
    """Continuous, with continuous alpha u_r, and div(alpha grad u) = 4 everywhere."""
    b1 = 0.30**2 - 0.25 * 0.30**2
    b2 = 0.25 * 0.35**2 + b1 - 0.35**2
    return np.where(r < 0.30, r**2, np.where(r <= 0.35, 0.25 * r**2 + b1, r**2 + b2))


@pytest.mark.parametrize("warp", [True, False])
def test_stencil_weights_chain_across_two_curved_interfaces_end_to_end(warp):
    # Real 30-node stencils of a node set straddling the ring's midline reach
    # all three regions: the foot points, both frames, the curvature and the
    # frame change are all exercised by stencil_weights itself, and with the
    # warp the three-region stretch of build_warp too.
    band = concentric_band()
    domain = Domain(band, (Circle(0.325),), STRIP)
    nodes = build_node_set(domain, 2500, iterations=10)
    r = np.hypot(nodes.x - 0.5, nodes.y - 0.5)
    u = matched_radial_quadratic(r)
    idx, _ = knn(nodes.xy, 30)
    region = band.region_index(nodes.x, nodes.y)
    lo, hi = region[idx].min(axis=1), region[idx].max(axis=1)
    triple = np.flatnonzero((lo == 0) & (hi == 2))
    assert len(triple) > 40, len(triple)
    residual = {
        curvature: np.array(
            [
                stencil_weights(
                    nodes.xy[idx[i]], band, P, curvature=curvature, warp=warp
                )
                @ u[idx[i]]
                - 4.0
                for i in triple[::3]
            ]
        )
        for curvature in (True, False)
    }
    assert np.abs(residual[True]).max() < 1e-6, np.abs(residual[True]).max()
    assert np.abs(residual[False]).max() > 0.1


def test_chain_through_two_curved_sine_interfaces_is_continuous_at_both():
    # Two sine graphs 0.03 apart: the foot points on the two curves lie at
    # different x, so the frame change between them rotates as well as
    # shifts. Anchored above both, the basis reaches region 0 through region
    # 1; the jumps at each interface fall as xi^(p+1) in u and xi^p in flux.
    band = Band(SineGraph(0.6), SineGraph(0.63), SineProduct(0.2, 0.1), Constant2D(1))
    x, y, scale = 0.13, 0.66, 0.05
    locals_ = {j: local_interface(band, j, x, y, scale, P) for j in (0, 1)}
    assert locals_[0].frame.angle != locals_[1].frame.angle
    regions = translated_basis(locals_, 2, 0, 2)
    dx, dy = coefficient_dx(P), coefficient_dy(P)
    for j in (0, 1):
        curve = band.interfaces[j]
        s0 = curve.closest(x, y)
        jumps = []
        for xi in (0.5, 0.25, 0.125):
            s = s0 + np.array([-xi, xi]) * scale / curve.length
            px, py = curve.point(s)
            nx, ny = curve.normal(s)
            values, fluxes = [], []
            for rr in (j, j + 1):
                frame = locals_[regions[rr].frame].frame
                v = polynomial_block(*frame.local(px, py), P)
                c = regions[rr].coefficients
                g_xi, g_eta = frame.rotate_in(nx, ny)
                alpha = band.region_piece(rr).alpha(px, py)[:, None]
                values.append(v @ c)
                fluxes.append(
                    alpha
                    * (g_xi[:, None] * (v @ dx @ c) + g_eta[:, None] * (v @ dy @ c))
                )
            jumps.append(
                (
                    np.abs(values[0] - values[1]).max(),
                    np.abs(fluxes[0] - fluxes[1]).max(),
                )
            )
        jumps = np.array(jumps)
        ratio = jumps[:-1] / jumps[1:]
        np.testing.assert_allclose(ratio[:, 0], 2 ** (P + 1), rtol=0.1)
        np.testing.assert_allclose(ratio[:, 1], 2**P, rtol=0.1)


# --- warped RBFs (EABE §2.2.4) ---------------------------------------------------


def thin_flat_band():
    """alpha 1 below y = 0.6, 0.2 on the 0.03-wide band, 1 above: three flat regions."""
    return Band(FlatLine(0.6), FlatLine(0.63), Constant2D(0.2), Constant2D(1.0))


def test_warp_is_the_identity_at_the_anchor_and_balances_alpha_across_each_interface():
    band = thin_flat_band()
    x, y, scale = 0.3, 0.615, 0.05
    locals_ = {j: local_interface(band, j, x, y, scale, P) for j in (0, 1)}
    alpha = np.array([1.0, 0.2, 1.0])
    for anchor in (0, 1, 2):
        w = build_warp(locals_, anchor, 0, 2)
        assert w.frame == (anchor if anchor < 2 else 1) and w.lowest == 0
        assert w.slope[anchor] == 1.0 and w.intercept[anchor] == 0.0
        # alpha * slope is one number over the three regions: alpha d/d_eta of
        # any smooth function of eta_tilde balances at both interfaces.
        np.testing.assert_allclose(alpha * w.slope, alpha[anchor], rtol=1e-14)
        # eta_tilde is continuous at each interface's eta in the anchor frame
        # (0 for its own, ±0.03 / scale for the other of this parallel pair).
        base = locals_[w.frame].frame
        for j in (0, 1):
            other = locals_[j].frame
            eta_j = 0.0 if j == w.frame else float(base.local(other.x0, other.y0)[1])
            expected = 0.0 if j == w.frame else (1.0 if j > w.frame else -1.0) * 0.6
            assert eta_j == pytest.approx(expected, abs=1e-12)
            assert w.apply(eta_j, j) == pytest.approx(w.apply(eta_j, j + 1), abs=1e-14)
    with pytest.raises(ValueError, match="anchor"):
        build_warp(locals_, 3, 0, 2)


def test_warp_balances_the_foot_point_alphas_along_the_sine_pair_of_case2():
    # Variable alpha in the band: the balance holds at each interface with
    # that interface's own foot-point values, so the slope above the band is
    # the product of two ratios that only cancel where alpha is the same at
    # both foot points.
    band = case2_band()
    x, y = 0.13, 0.7
    locals_ = {j: local_interface(band, j, x, y, 0.05, P) for j in (0, 1)}
    w = build_warp(locals_, 1, 0, 2)
    for j in (0, 1):
        it = locals_[j]
        assert it.minus[0, 0] * w.slope[j] == pytest.approx(
            it.plus[0, 0] * w.slope[j + 1], rel=1e-14
        )
    assert w.slope[0] != pytest.approx(w.slope[2])


def test_warped_gaussian_has_continuous_value_and_alpha_normal_flux_on_the_interface():
    # EABE Fig. 6 / dissertation Fig. 5-1: alpha 1 below a flat interface, 1/2
    # above, one Gaussian centred below. As a function of the stretched
    # coordinates its value and its alpha d/dn are continuous at every point
    # of the interface, to rounding by the chain rule; the chain rule itself
    # is checked against one-sided differences on each side, and the plain
    # Gaussian is shown to fail the flux balance by the factor 2.
    band = Band(FlatLine(0.6), FlatLine(0.9), Constant2D(0.5), Constant2D(1.0))
    x, y, scale = 0.4, 0.585, 0.05
    li = local_interface(band, 0, x, y, scale, P)
    w = build_warp({0: li}, 0, 0, 1)
    assert w.slope[1] == pytest.approx(2.0)
    eps = 0.4 * scale / 0.012
    xi_c, eta_c = li.frame.local(x, y)
    eta_c = w.apply(eta_c, 0)

    def phi(px, py, side, warped=True):
        xi, eta = li.frame.local(px, py)
        et = w.apply(eta, side) if warped else eta
        return gaussian(xi - xi_c, et - eta_c, eps)

    def d_normal(px, py, side):
        # d/dn = (slope / scale) d/d_eta_tilde, in physical units.
        xi, eta = li.frame.local(px, py)
        et = w.apply(eta, side)
        g = gaussian(xi - xi_c, et - eta_c, eps)
        return w.slope[side] / scale * (-2.0 * eps**2 * (et - eta_c) * g)

    alpha = np.array([1.0, 0.5])
    px = np.linspace(0.3, 0.5, 9)
    py = np.full_like(px, 0.6)
    np.testing.assert_array_equal(phi(px, py, 0), phi(px, py, 1))
    flux = [alpha[side] * d_normal(px, py, side) for side in (0, 1)]
    np.testing.assert_allclose(flux[0], flux[1], rtol=1e-13)
    assert np.abs(flux[0]).max() > 1.0
    delta = 1e-7
    below = (phi(px, py, 0) - phi(px, py - delta, 0)) / delta
    above = (phi(px, py + delta, 1) - phi(px, py, 1)) / delta
    np.testing.assert_allclose(below, d_normal(px, py, 0), rtol=1e-4)
    np.testing.assert_allclose(above, d_normal(px, py, 1), rtol=1e-4)
    plain_below = (phi(px, py, 0, False) - phi(px, py - delta, 0, False)) / delta
    plain_above = (phi(px, py + delta, 1, False) - phi(px, py, 1, False)) / delta
    np.testing.assert_allclose(
        alpha[0] * plain_below, 2 * alpha[1] * plain_above, rtol=1e-4
    )


def test_warp_is_the_identity_when_alpha_matches_and_moves_the_weights_otherwise():
    domain = case1()
    nodes = build_node_set(domain, 1250)
    idx, _ = knn(nodes.xy, 30)
    region = domain.material.region_index(nodes.x, nodes.y)
    i = int(np.flatnonzero(np.ptp(region[idx], axis=1) > 0)[5])
    xy = nodes.xy[idx[i]]
    same = Band(FlatLine(0.6), FlatLine(0.8), Constant2D(1.0), Constant2D(1.0))
    plain = stencil_weights(xy, same, P, warp=False)
    warped = stencil_weights(xy, same, P, warp=True)
    # Only the rotation of the Gaussian offsets into the frame separates them.
    np.testing.assert_allclose(warped, plain, atol=1e-10 * np.abs(plain).max())
    plain = stencil_weights(xy, domain.material, P, warp=False)
    warped = stencil_weights(xy, domain.material, P, warp=True)
    assert np.abs(warped - plain).max() > 1e-3 * np.abs(plain).max()


def test_stencil_weights_scale_as_the_inverse_square_of_the_spacing():
    domain = case1()
    band = domain.material
    nodes = build_node_set(domain, 1250)
    idx, _ = knn(nodes.xy, 30)
    region = band.region_index(nodes.x, nodes.y)
    # A stencil across the lower interface only, away from the seam, so that
    # shrinking it about (x_c, 0.6) keeps every node's region and image.
    ok = (np.ptp(region[idx], axis=1) > 0) & (nodes.y[idx].max(axis=1) < 0.7)
    ok &= (nodes.x[idx].min(axis=1) > 0.2) & (nodes.x[idx].max(axis=1) < 0.8)
    i = int(np.flatnonzero(ok)[3])
    xy = nodes.xy[idx[i]]
    w = stencil_weights(xy, band, P)
    # Shrink the stencil about the lower interface by 2 (it stays flat).
    c = np.array([xy[0, 0], 0.6])
    xy2 = c + 0.5 * (xy - c)
    w2 = stencil_weights(xy2, band, P)
    np.testing.assert_allclose(w2, 4.0 * w, rtol=1e-8)


def test_local_interface_dataclass_holds_its_parts():
    li = local_interface(case1().material, 0, 0.3, 0.58, 0.04, 3)
    assert isinstance(li, LocalInterface) and li.degree == 3
    assert li.expansion.shape == (4,) and li.minus.shape == (4, 4)


# --- interpolation weights (E2.6) ------------------------------------------------


def circle_quadratic_setup():
    """The operator test's matched piecewise quadratic and its crossing stencils."""
    band = Band(Circle(0.35), Circle(0.9), Constant2D(5.0), Constant2D(1.0))
    domain = Domain(band, (Circle(0.35),), STRIP)
    nodes = build_node_set(domain, 2500, iterations=10)

    def u(x, y):
        r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
        return np.where(r2 < 0.35**2, r2, (r2 - 0.35**2) / 5 + 0.35**2)

    idx, _ = knn(nodes.xy, 30)
    region = band.region_index(nodes.x, nodes.y)
    lo, hi = region[idx].min(axis=1), region[idx].max(axis=1)
    cross = np.flatnonzero((lo == 0) & (hi == 1))[::4]
    assert len(cross) > 40
    return band, nodes, u, idx, cross


@pytest.mark.parametrize("warp", [True, False])
def test_interpolation_weights_are_exact_on_the_matched_quadratic_off_the_nodes(warp):
    # Points halfway between the centre and its first five neighbours, on both
    # sides of the circle: the curved basis reproduces the pair to rounding,
    # the flat assumption misses by the local linear approximation's error.
    band, nodes, u, idx, cross = circle_quadratic_setup()
    worst = {True: 0.0, False: 0.0}
    for i in cross:
        xy = nodes.xy[idx[i]]
        px, py = 0.5 * (xy[0, 0] + xy[1:6, 0]), 0.5 * (xy[0, 1] + xy[1:6, 1])
        for curvature in (True, False):
            w = interpolation_weights(
                xy, band, P, (px, py), curvature=curvature, warp=warp
            )
            assert w.shape == (5, 30)
            err = np.abs(w @ u(xy[:, 0], xy[:, 1]) - u(px, py)).max()
            worst[curvature] = max(worst[curvature], err)
    assert worst[True] < 1e-9, worst
    assert worst[False] > 1e-5, worst


def test_interpolation_weights_give_the_unit_vector_on_a_node_and_refuse_far_points():
    # A case-1 stencil across y = 0.6 reaches regions 0 and 1; a point above
    # y = 0.8 lies in region 2, which it has no basis for.
    domain = case1()
    nodes = build_node_set(domain, 1250, iterations=20)
    idx, _ = knn(nodes.xy, 30)
    region = domain.material.region_index(nodes.x, nodes.y)
    lo, hi = region[idx].min(axis=1), region[idx].max(axis=1)
    i = int(np.flatnonzero((lo == 0) & (hi == 1))[0])
    xy = nodes.xy[idx[i]]
    w = interpolation_weights(xy, domain.material, P, (xy[3:4, 0], xy[3:4, 1]))
    assert np.abs(w[0] - np.eye(30)[3]).max() < 1e-10
    with pytest.raises(ValueError, match="does not reach"):
        interpolation_weights(xy, domain.material, P, (xy[:1, 0], np.array([0.9])))
    with pytest.raises(ValueError, match="matching 1-D"):
        interpolation_weights(xy, domain.material, P, (np.zeros(2), np.zeros(3)))


def test_interface_stencil_exposes_the_blocks_stencil_weights_solves():
    band, nodes, _, idx, cross = circle_quadratic_setup()
    xy = nodes.xy[idx[cross[1]]]
    st = interface_stencil(xy, band, P)
    assert st.anchor == int(band.region_index(xy[:1, 0], xy[:1, 1])[0])
    assert sorted(st.regions) == [0, 1] and st.warp is not None
    assert st.polynomial_block().shape == (30, Q)
    a = st.gaussian_block()
    assert a.shape == (30, 30) and np.allclose(a, a.T) and np.allclose(np.diag(a), 1)
    # The nodes' own Gaussian coordinates are the stored ones.
    xi, eta = st.gaussian_coordinates(xy[:, 0], xy[:, 1])
    np.testing.assert_array_equal(xi, st.xi)
    np.testing.assert_array_equal(eta, st.eta)
    # Without the warp the coordinates are the global offsets over the radius.
    plain = interface_stencil(xy, band, P, warp=False)
    assert plain.warp is None
    r = np.hypot(xy[:, 0] - xy[0, 0], xy[:, 1] - xy[0, 1])
    np.testing.assert_allclose(np.hypot(plain.xi, plain.eta), r / r.max(), atol=1e-15)
    with pytest.raises(ValueError, match="one region"):
        interface_stencil(nodes.xy[idx[0]][:1].repeat(30, axis=0) + 1e-3, band, P)
