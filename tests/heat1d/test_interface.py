import numpy as np
import pytest
from numpy.polynomial import Polynomial

from heat_interfaces.fd_weights import fornberg_weights
from heat_interfaces.heat1d.domain import (
    PiecewiseAlpha,
    Sinusoid,
    dissertation_alpha,
    eabe_alpha,
)
from heat_interfaces.heat1d.interface import (
    Jump,
    coefficient_dx,
    coefficient_operator,
    continuity_conditions,
    continuity_matrix,
    multiplication_matrix,
    region_index,
    shift_matrix,
    stencil_weights,
    translated_basis,
    translation_matrix,
)

# The papers' printed matrices, as strings so each entry is checked to the
# precision it was printed with (half a unit in the last digit).
EABE_EQ_18 = [
    ["0.5", "0", "0", "0", "0"],
    ["0.63", "0.5", "0", "0", "0"],
    ["0", "0.63", "0.5", "0", "0"],
    ["-4.13", "0", "0.63", "0.5", "0"],
    ["0", "-4.13", "0", "0.63", "0.5"],
]
EABE_EQ_21 = [
    ["0", "0.5", "0", "0", "0"],
    ["0", "0.6", "1.0", "0", "0"],
    ["0", "0", "1.3", "1.5", "0"],
    ["0", "-4.1", "0", "1.9", "2.0"],
    ["0", "0", "-8.3", "0", "2.5"],
]
EABE_EQ_23_RIGHT = [
    ["1", "0", "0", "0", "0"],
    ["0", "0.5", "0", "0", "0"],
    ["0", "0.6", "1.0", "0", "0"],
    ["0", "0", "1.3", "1.5", "0"],
    ["0", "-12", "1.6", "7.5", "6.0"],
]
DISSERTATION_EQ_74 = [
    ["1", "0", "0", "0", "0"],
    ["0", "0.5", "0", "0", "0"],
    ["0", "0.31", "0.5", "0", "0"],
    ["0", "0", "0.21", "0.25", "0"],
    ["0", "-0.52", "0.07", "0.31", "0.25"],
]


def assert_matches_printed(actual, printed):
    for row_a, row_p in zip(actual, printed, strict=True):
        for a, p in zip(row_a, row_p, strict=True):
            decimals = len(p.split(".")[1]) if "." in p else 0
            assert abs(a - float(p)) <= 0.5 * 10.0**-decimals + 1e-12, (a, p)


def eabe_taylor(degree=4):
    m = eabe_alpha()
    return m.taylor(0, "left", degree), m.taylor(0, "right", degree)


def test_coefficient_dx_is_eq_13():
    dx = coefficient_dx(4)
    expected = np.zeros((5, 5))
    expected[[0, 1, 2, 3], [1, 2, 3, 4]] = [1, 2, 3, 4]
    np.testing.assert_array_equal(dx, expected)
    c = np.array([3.0, -1.0, 0.5, 2.0, -0.25])
    np.testing.assert_allclose(dx @ c, Polynomial(c).deriv().coef.tolist() + [0.0])


def test_multiplication_matrices_are_eq_17_and_18():
    a_left, a_right = eabe_taylor()
    np.testing.assert_array_equal(multiplication_matrix(a_left), np.eye(5))
    assert_matches_printed(multiplication_matrix(a_right), EABE_EQ_18)
    # It is multiplication truncated to P_4.
    c = np.array([1.0, -2.0, 0.5, 0.0, 1.5])
    product = (Polynomial(a_right) * Polynomial(c)).coef[:5]
    np.testing.assert_allclose(multiplication_matrix(a_right) @ c, product)


def test_flux_matrix_is_eq_21():
    _, a_right = eabe_taylor()
    assert_matches_printed(
        multiplication_matrix(a_right) @ coefficient_dx(4), EABE_EQ_21
    )


def test_continuity_conditions_interleave_temperature_and_flux():
    assert continuity_conditions(4) == [
        ("u", 0),
        ("flux", 0),
        ("u", 1),
        ("flux", 1),
        ("u", 2),
    ]
    assert continuity_conditions(5)[-1] == ("flux", 2)
    assert continuity_conditions(6)[-1] == ("u", 3)
    assert len(continuity_conditions(6)) == 7


def test_continuity_matrices_are_eq_23():
    a_left, a_right = eabe_taylor()
    np.testing.assert_allclose(continuity_matrix(a_left), np.diag([1, 1, 2, 6, 24]))
    c_right = continuity_matrix(a_right)
    assert_matches_printed(c_right, EABE_EQ_23_RIGHT)
    # The closed forms behind the printed digits.
    a0, a1, a2, a3 = a_right[:4]
    np.testing.assert_allclose(
        c_right[4],
        [
            0,
            2 * a1 * a2 + 6 * a0 * a3,
            4 * a1**2 + 12 * a0 * a2,
            24 * a0 * a1,
            24 * a0**2,
        ],
    )


def test_translation_is_eq_74():
    a_left, a_right = eabe_taylor()
    u_left = translation_matrix(a_right, a_left)
    assert_matches_printed(u_left, DISSERTATION_EQ_74)
    # Translating back is the inverse.
    np.testing.assert_allclose(
        translation_matrix(a_left, a_right) @ u_left, np.eye(5), atol=1e-12
    )


def _operator(alpha: Polynomial, p: Polynomial) -> Polynomial:
    return (alpha * p.deriv()).deriv()


@pytest.mark.parametrize(
    ("medium", "interface", "anchor"),
    [
        (eabe_alpha(), 0, 1),
        (eabe_alpha(), 0, 0),
        (dissertation_alpha(), 0, 1),
        (dissertation_alpha(), 1, 0),
        (dissertation_alpha(), 1, 1),
    ],
)
def test_translated_basis_keeps_u_flux_and_their_time_derivatives_continuous(
    medium, interface, anchor
):
    # Independent check by polynomial arithmetic with alpha expanded to degree
    # 12 on each side: the k = 0..2 temperature and k = 0..1 flux conditions
    # of eq. 54-55 hold for every basis function at the interface.
    xi = medium.interfaces[interface]
    jump = Jump(
        xi, medium.taylor(interface, "left", 4), medium.taylor(interface, "right", 4)
    )
    left, right = translated_basis([jump], anchor)
    alpha_left = Polynomial(medium.pieces[interface].taylor(xi, 12))
    alpha_right = Polynomial(medium.pieces[interface + 1].taylor(xi, 12))
    np.testing.assert_array_equal((left, right)[anchor].coefficients, np.eye(5))
    for j in range(5):
        p_left = Polynomial(left.coefficients[:, j])
        p_right = Polynomial(right.coefficients[:, j])
        d_left, d_right = _operator(alpha_left, p_left), _operator(alpha_right, p_right)
        pairs = [
            (p_left, p_right),
            (alpha_left * p_left.deriv(), alpha_right * p_right.deriv()),
            (d_left, d_right),
            (alpha_left * d_left.deriv(), alpha_right * d_right.deriv()),
            (_operator(alpha_left, d_left), _operator(alpha_right, d_right)),
        ]
        for lhs, rhs in pairs:
            # The layer's k = 2 rows cancel terms near 1e3 (a_3 ~ 10 at x = 0.5).
            assert lhs(0.0) == pytest.approx(rhs(0.0), abs=1e-9)


def _interface_values(alpha_derivatives, p: Polynomial, t: float) -> list[float]:
    """``u, alpha u_x, D u, alpha (D u)_x, D^2 u`` at ``t`` (eq. 54-55 through k = 2).

    Written out from ``D u = alpha' u' + alpha u''`` with alpha's derivatives
    supplied directly, so nothing here shares code with ``interface.py``.
    """
    a0, a1, a2, a3 = alpha_derivatives
    d = [p.deriv(k)(t) if k else p(t) for k in range(5)]
    du = a1 * d[1] + a0 * d[2]
    du_x = a2 * d[1] + 2 * a1 * d[2] + a0 * d[3]
    du_xx = a3 * d[1] + 3 * a2 * d[2] + 3 * a1 * d[3] + a0 * d[4]
    return [d[0], a0 * d[1], du, a0 * du_x, a1 * du_x + a0 * du_xx]


@pytest.mark.parametrize("anchor", [0, 1, 2])
def test_two_jump_basis_is_continuous_at_both_interfaces_with_varying_alpha(anchor):
    # alpha varies on all three regions, so the re-centring and the second
    # translation both carry non-trivial information (a piecewise-constant
    # alpha would let a wrong shift pass, since its solution is piecewise
    # linear whatever the basis).
    m = PiecewiseAlpha(
        (0.0, 0.3),
        (
            Sinusoid(2.0, 0.5, 3.0),
            Sinusoid(0.5, 0.1, 2 * np.pi),
            Sinusoid(1.5, -0.3, 4.0, phase=0.7),
        ),
    )
    jumps = [
        Jump(xi, m.taylor(k, "left", 4), m.taylor(k, "right", 4))
        for k, xi in enumerate(m.interfaces)
    ]
    regions = translated_basis(jumps, anchor)
    assert len(regions) == 3
    np.testing.assert_array_equal(regions[anchor].coefficients, np.eye(5))
    assert sum(np.allclose(r.coefficients, np.eye(5)) for r in regions) == 1
    for k, xi in enumerate(m.interfaces):
        values = {}
        for side, region, piece in (
            ("left", regions[k], m.pieces[k]),
            ("right", regions[k + 1], m.pieces[k + 1]),
        ):
            derivatives = piece.taylor(xi, 3) * np.array([1, 1, 2, 6])
            t = xi - jumps[region.jump].position
            values[side] = np.array(
                [
                    _interface_values(
                        derivatives, Polynomial(region.coefficients[:, j]), t
                    )
                    for j in range(5)
                ]
            )
        np.testing.assert_allclose(
            values["left"], values["right"], rtol=1e-10, atol=1e-9
        )


def test_shift_matrix_recentres_without_changing_the_polynomial():
    rng = np.random.default_rng(3)
    c = rng.normal(size=5)
    delta = 0.37
    shifted = shift_matrix(delta, 4) @ c
    x = rng.uniform(-1, 1, 7)
    np.testing.assert_allclose(Polynomial(shifted)(x - delta), Polynomial(c)(x))
    np.testing.assert_array_equal(shift_matrix(0.0, 4), np.eye(5))


def test_two_interface_basis_spans_the_same_space_for_every_anchor():
    m = dissertation_alpha()
    jumps = [
        Jump(xi, m.taylor(k, "left", 4), m.taylor(k, "right", 4))
        for k, xi in enumerate(m.interfaces)
    ]
    x = np.linspace(-0.5, 1.0, 9)
    which = region_index(jumps, x)

    def sample(anchor):
        regions = translated_basis(jumps, anchor)
        return np.vstack(
            [regions[r].evaluate(jumps, x[[i]]) for i, r in enumerate(which)]
        )

    phi = [sample(anchor) for anchor in range(3)]
    for other in phi[1:]:
        coefficients, *_ = np.linalg.lstsq(phi[0], other, rcond=None)
        residual = np.abs(phi[0] @ coefficients - other).max(axis=0)
        assert np.all(residual < 1e-11 * np.abs(other).max(axis=0))
        assert np.linalg.matrix_rank(coefficients) == 5


def test_region_index_puts_a_point_on_a_jump_to_its_right():
    jumps = [Jump(0.0, np.ones(5), np.ones(5)), Jump(0.5, np.ones(5), np.ones(5))]
    np.testing.assert_array_equal(
        region_index(jumps, np.array([-1.0, 0.0, 0.2, 0.5, 0.9])), [0, 1, 1, 2, 2]
    )


def test_stencil_weights_reduce_to_fornberg_when_alpha_is_linear_across_the_jump():
    # alpha = 2 + 0.3 (x - 0.3) on both sides: no interface, and no truncation
    # in Dx M Dx since alpha phi' has degree at most four.
    a = np.array([2.0, 0.3, 0.0, 0.0, 0.0])
    jump = Jump(0.3, a, a)
    nodes = np.arange(-2.0, 3.0)
    for centre in nodes:
        t = centre - jump.position
        w = fornberg_weights(centre, nodes, 2)
        expected = (a[0] + a[1] * t) * w[2] + a[1] * w[1]
        np.testing.assert_allclose(
            stencil_weights([jump], nodes, centre), expected, atol=1e-12
        )


@pytest.mark.parametrize("centre", [-1.5, -0.5, 0.5, 1.5])
def test_stencil_weights_are_exact_on_the_piecewise_linear_equilibrium(centre):
    # alpha = 1 | 4 at 0 with the nodes offset by half a spacing (Fig. 4-3's
    # layout): u = x on the left, x / 4 on the right, has (alpha u')' = 0.
    jump = Jump(0.0, np.array([1.0, 0, 0, 0, 0]), np.array([4.0, 0, 0, 0, 0]))
    nodes = np.array([-1.5, -0.5, 0.5, 1.5, 2.5])
    u = np.where(nodes < 0, nodes, nodes / 4)
    w = stencil_weights([jump], nodes, centre)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert w @ u == pytest.approx(0.0, abs=1e-12)
    assert np.any(w != 0)


def test_stencil_weights_are_invariant_to_the_units_of_the_problem():
    # Scaling positions by h and coefficients a_k by h^k is what
    # jump_aware_operator does; the weights come back divided by h^2.
    a_left, a_right = eabe_taylor()
    h = 0.01
    nodes = h * np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
    w = stencil_weights([Jump(0.0, a_left, a_right)], nodes, nodes[2])
    powers = h ** np.arange(5)
    scaled = stencil_weights(
        [Jump(0.0, a_left * powers, a_right * powers)], nodes / h, nodes[2] / h
    )
    np.testing.assert_allclose(scaled / h**2, w, rtol=1e-9)


def test_coefficient_operator_is_exact_below_the_top_degree():
    _, a_right = eabe_taylor()
    c = np.array([0.0, 1.0, -0.5, 2.0, 0.25])
    exact = _operator(Polynomial(a_right), Polynomial(c)).coef
    truncated = coefficient_operator(a_right) @ c
    np.testing.assert_allclose(truncated[:4], exact[:4])
    assert truncated[4] != pytest.approx(exact[4])


def test_validation():
    jump = Jump(0.0, np.ones(5), np.ones(5))
    with pytest.raises(ValueError):
        translated_basis([], 0)
    with pytest.raises(ValueError):
        translated_basis([jump], 2)
    with pytest.raises(ValueError):
        stencil_weights([jump], np.arange(4.0), 0.0)
