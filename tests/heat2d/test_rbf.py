import numpy as np
import pytest

from heat_interfaces.heat2d.domain import build_node_set, case1
from heat_interfaces.heat2d.neighbors import knn, offsets
from heat_interfaces.heat2d.rbf import (
    OPERATORS,
    ORDER,
    StencilSpec,
    augmented_solve,
    gaussian,
    gaussian_derivative,
    polynomial_count,
    polynomial_exponents,
    polynomial_rhs,
    rbf_fd_weights,
    rbf_interpolation_weights,
)


def monomial_derivative(i, j, op, x0, y0):
    """``L (x^i y^j)`` at ``(x0, y0)`` by hand."""

    def d(n, k, z):
        # k-th derivative of z^n
        c = 1.0
        for r in range(k):
            c *= n - r
        return c * z ** max(n - k, 0) if n >= k else 0.0

    table = {
        "dx": d(i, 1, x0) * d(j, 0, y0),
        "dy": d(i, 0, x0) * d(j, 1, y0),
        "dxx": d(i, 2, x0) * d(j, 0, y0),
        "dyy": d(i, 0, x0) * d(j, 2, y0),
        "dxy": d(i, 1, x0) * d(j, 1, y0),
    }
    table["lap"] = table["dxx"] + table["dyy"]
    return table[op]


@pytest.fixture(scope="module")
def stencils():
    """42-node stencils of a 900-node case-1 set, as offsets."""
    nodes = build_node_set(case1(), 900)
    idx, _ = knn(nodes.xy, 42)
    return offsets(nodes.xy, idx)


def test_polynomial_exponents_are_graded_with_x_powers_first():
    e = polynomial_exponents(5)
    assert e.shape == (21, 2) and polynomial_count(5) == 21
    np.testing.assert_array_equal(
        e[:6], [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    )
    assert np.all(e.sum(axis=1)[:-1] <= e.sum(axis=1)[1:])
    assert polynomial_exponents(0).shape == (1, 2)
    with pytest.raises(ValueError):
        polynomial_exponents(-1)


def test_stencil_spec_refuses_fewer_nodes_than_monomials():
    with pytest.raises(ValueError, match="21 monomials"):
        StencilSpec(20, 5)
    assert StencilSpec(21, 5).size == 21


def test_polynomial_rhs_picks_the_monomial_the_operator_flattens():
    rhs = polynomial_rhs(3, ("dx", "lap", "dxy"))
    e = polynomial_exponents(3)
    assert rhs.shape == (10, 3)
    assert rhs[(e == (1, 0)).all(axis=1), 0] == 1.0 and rhs[:, 0].sum() == 1.0
    assert rhs[(e == (2, 0)).all(axis=1), 1] == 2.0
    assert rhs[(e == (0, 2)).all(axis=1), 1] == 2.0 and rhs[:, 1].sum() == 4.0
    assert rhs[(e == (1, 1)).all(axis=1), 2] == 1.0 and rhs[:, 2].sum() == 1.0
    with pytest.raises(ValueError, match="unknown operator"):
        polynomial_rhs(3, ("curl",))


@pytest.mark.parametrize("op", OPERATORS)
def test_gaussian_derivative_matches_central_differences(op):
    # φ_j(x) = exp(-ε²|x - x_j|²) evaluated at x_c; (dx, dy) = x_j - x_c.
    eps, dx, dy, s = 1.7, 0.3, -0.45, 1e-4

    def phi(cx, cy):
        return gaussian(dx - cx, dy - cy, eps)

    fd = {
        "dx": (phi(s, 0) - phi(-s, 0)) / (2 * s),
        "dy": (phi(0, s) - phi(0, -s)) / (2 * s),
        "dxx": (phi(s, 0) - 2 * phi(0, 0) + phi(-s, 0)) / s**2,
        "dyy": (phi(0, s) - 2 * phi(0, 0) + phi(0, -s)) / s**2,
        "dxy": (phi(s, s) - phi(s, -s) - phi(-s, s) + phi(-s, -s)) / (4 * s**2),
    }
    fd["lap"] = fd["dxx"] + fd["dyy"]
    got = gaussian_derivative(np.array(dx), np.array(dy), np.array(eps), op)
    assert got == pytest.approx(fd[op], rel=1e-6, abs=1e-9)


def test_weights_reproduce_every_monomial_through_degree_five_to_rounding(stencils):
    dx, dy = stencils
    w = rbf_fd_weights(dx, dy, OPERATORS, 5)
    assert w.shape == (6, len(dx), 42)
    x0, y0 = 0.3, -0.7  # the stencil's centre in the polynomial's coordinates
    for i, j in polynomial_exponents(5):
        p = (dx + x0) ** i * (dy + y0) ** j
        scale = np.abs(w).sum(axis=2) * np.abs(p).max(axis=1)
        for c, op in enumerate(OPERATORS):
            got = np.einsum("mk,mk->m", w[c], p)
            err = np.abs(got - monomial_derivative(i, j, op, x0, y0)) / scale[c]
            assert err.max() < 1e-13, (op, i, j, err.max())


def test_weights_reproduce_degree_four_and_not_five_at_degree_four(stencils):
    dx, dy = stencils
    dx, dy = dx[:50, :30], dy[:50, :30]
    w = rbf_fd_weights(dx, dy, ("lap",), 4)[0]
    quartic = np.einsum("mk,mk->m", w, (dx + 0.2) ** 2 * (dy - 0.1) ** 2)
    exact = 2 * (-0.1) ** 2 + 2 * 0.2**2
    assert np.abs(quartic - exact).max() < 1e-13 * np.abs(w).sum(axis=1).max()
    quintic = np.einsum("mk,mk->m", w, (dx + 0.2) ** 5)
    assert np.abs(quintic - 20 * 0.2**3).max() > 1e-6


def test_weights_scale_as_the_inverse_power_of_the_spacing(stencils):
    dx, dy = stencils
    w = rbf_fd_weights(dx[:20], dy[:20], OPERATORS, 5)
    ws = rbf_fd_weights(0.01 * dx[:20], 0.01 * dy[:20], OPERATORS, 5)
    for c, op in enumerate(OPERATORS):
        # Rounding in dx / R differs between the two scalings and the stencil
        # systems amplify it by their condition number (about 1e6).
        expect = w[c] / 0.01 ** ORDER[op]
        np.testing.assert_allclose(
            ws[c], expect, rtol=0, atol=1e-9 * np.abs(expect).max()
        )


def test_laplacian_is_about_fourth_order_on_a_smooth_field():
    errs = []
    for n in (1250, 5000):
        nodes = build_node_set(case1(), n)
        idx, _ = knn(nodes.xy, 42)
        dx, dy = offsets(nodes.xy, idx)
        w = rbf_fd_weights(dx, dy, ("lap",), 5)[0]
        f = np.sin(2 * np.pi * nodes.x) * np.exp(nodes.y)
        lap = (1 - 4 * np.pi**2) * f
        inside = (nodes.y > 0.1) & (nodes.y < 0.9)  # one-sided rows are E2.2's zone
        e = np.einsum("mk,mk->m", w, f[idx]) - lap
        errs.append(np.sqrt(np.mean(e[inside] ** 2)))
    # h halves between the two counts; degree 5 gives at least h^4.
    assert errs[0] / errs[1] > 2**3.5, errs


def test_augmented_solve_returns_only_the_node_weights():
    rng = np.random.default_rng(3)
    xi = rng.uniform(-1, 1, (4, 12))
    eta = rng.uniform(-1, 1, (4, 12))
    xi[:, 0] = eta[:, 0] = 0.0
    a = gaussian(
        xi[:, :, None] - xi[:, None, :], eta[:, :, None] - eta[:, None, :], 1.0
    )
    p = np.stack([np.ones_like(xi), xi, eta], axis=-1)
    b = gaussian_derivative(xi, eta, np.array(1.0), "dx")[..., None]
    w = augmented_solve(a, p, b, polynomial_rhs(1, ("dx",)))
    assert w.shape == (4, 12, 1)
    np.testing.assert_allclose(np.einsum("mk,mk->m", w[..., 0], xi), 1.0, atol=1e-12)
    np.testing.assert_allclose(w[..., 0].sum(axis=1), 0.0, atol=1e-12)


def test_weights_refuse_bad_stencils(stencils):
    dx, dy = stencils
    with pytest.raises(ValueError, match="unknown operator"):
        rbf_fd_weights(dx, dy, ("grad",), 5)
    with pytest.raises(ValueError, match="same shape"):
        rbf_fd_weights(dx, dy[:, :10], ("dx",), 5)
    with pytest.raises(ValueError, match="cannot carry"):
        rbf_fd_weights(dx[:, :20], dy[:, :20], ("dx",), 5)
    # An exact duplicate of the centre, and a neighbour a rounding error away
    # from another neighbour, are both refused; the node sets never make either.
    twin = dx[:3].copy()
    twin[:, 1] = twin[:, 0]
    twin_y = dy[:3].copy()
    twin_y[:, 1] = twin_y[:, 0]
    with pytest.raises(ValueError, match="coincide"):
        rbf_fd_weights(twin, twin_y, ("dx",), 3)
    near = dx[:3].copy()
    near[:, 2] = near[:, 3] + 1e-13
    near_y = dy[:3].copy()
    near_y[:, 2] = near_y[:, 3]
    with pytest.raises(ValueError, match="nearly so"):
        rbf_fd_weights(near, near_y, ("dx",), 3)


# --- interpolation weights (E2.6) ---------------------------------------------


def smooth(x, y):
    return np.sin(2 * np.pi * x) * np.exp(y)


def interpolation_setup(n, rng):
    """A case-1 set with 42-node stencils and a point half a spacing off each centre."""
    nodes = build_node_set(case1(), n, iterations=20)
    idx, dist = knn(nodes.xy, 42)
    dx, dy = offsets(nodes.xy, idx)
    theta = rng.uniform(0.0, 2 * np.pi, nodes.n)
    ex, ey = 0.5 * dist[:, 1] * np.cos(theta), 0.5 * dist[:, 1] * np.sin(theta)
    return nodes, idx, dx, dy, ex, ey


def test_interpolation_weights_reproduce_monomials_off_centre():
    nodes, idx, dx, dy, ex, ey = interpolation_setup(1250, np.random.default_rng(3))
    w = rbf_interpolation_weights(dx, dy, ex, ey, 5)
    assert w.shape == (nodes.n, 42)
    for i, j in polynomial_exponents(5):
        p = dx**i * dy**j
        got = np.einsum("ij,ij->i", w, p)
        scale = (np.abs(w) * np.abs(p)).sum(axis=1) + np.abs(ex**i * ey**j)
        assert np.abs(got - ex**i * ey**j).max() <= 1e-12 * scale.max()


def test_interpolation_weights_are_the_unit_vector_on_the_centre():
    _, _, dx, dy, _, _ = interpolation_setup(900, np.random.default_rng(4))
    w = rbf_interpolation_weights(dx, dy, np.zeros(len(dx)), np.zeros(len(dx)), 5)
    assert np.abs(w - np.eye(42)[0]).max() < 1e-10


def test_interpolation_is_at_least_fifth_order_on_a_smooth_field():
    errs = []
    for n in (1250, 5000):
        nodes, idx, dx, dy, ex, ey = interpolation_setup(n, np.random.default_rng(5))
        w = rbf_interpolation_weights(dx, dy, ex, ey, 5)
        got = np.einsum("ij,ij->i", w, smooth(nodes.x, nodes.y)[idx])
        errs.append(np.sqrt(np.mean((got - smooth(nodes.x + ex, nodes.y + ey)) ** 2)))
    assert errs[1] < 1e-7 and errs[0] / errs[1] > 2**5, errs


def test_interpolation_weights_refuse_mismatched_shapes():
    _, _, dx, dy, ex, ey = interpolation_setup(900, np.random.default_rng(6))
    with pytest.raises(ValueError, match=r"\(m,\)"):
        rbf_interpolation_weights(dx, dy, ex[:-1], ey[:-1], 5)
    with pytest.raises(ValueError, match="monomials"):
        rbf_interpolation_weights(dx[:, :10], dy[:, :10], ex, ey, 5)
