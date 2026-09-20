from math import factorial

import numpy as np
import pytest

from heat_interfaces.fd_weights import fornberg_weights


def vandermonde_weights(z, x, k):
    """Weights for f^(k)(z) by solving the moment conditions directly."""
    x = np.asarray(x, dtype=float)
    n = x.size
    lhs = (x[None, :] - z) ** np.arange(n)[:, None]
    rhs = np.zeros(n)
    if k < n:
        rhs[k] = factorial(k)
    return np.linalg.solve(lhs, rhs)


@pytest.mark.parametrize(
    ("z", "x", "k", "expected"),
    [
        (0, [-1, 0, 1], 1, [-0.5, 0.0, 0.5]),
        (0, [-1, 0, 1], 2, [1.0, -2.0, 1.0]),
        (0, [-2, -1, 0, 1, 2], 1, np.array([1, -8, 0, 8, -1]) / 12),
        (0, [-2, -1, 0, 1, 2], 2, np.array([-1, 16, -30, 16, -1]) / 12),
        (0, [0, 1, 2, 3, 4], 1, np.array([-25, 48, -36, 16, -3]) / 12),
        (1, [0, 1, 2, 3, 4], 1, np.array([-3, -10, 18, -6, 1]) / 12),
        (0, [0, 1, 2, 3, 4], 2, np.array([35, -104, 114, -56, 11]) / 12),
    ],
)
def test_tabulated_weights(z, x, k, expected):
    w = fornberg_weights(z, np.array(x, dtype=float), k)
    assert w.shape == (k + 1, len(x))
    np.testing.assert_allclose(w[k], expected, atol=1e-13)


def test_interpolation_row_is_the_lagrange_basis_and_sums_to_one():
    x = np.array([-1.0, -0.3, 0.2, 0.9])
    z = 0.37
    w = fornberg_weights(z, x, 0)[0]
    assert w.sum() == pytest.approx(1.0)
    for j, xj in enumerate(x):
        others = np.delete(x, j)
        assert w[j] == pytest.approx(np.prod((z - others) / (xj - others)))


def test_matches_moment_conditions_on_scattered_nodes():
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(-1, 1, 7))
    z = 0.123
    w = fornberg_weights(z, x, 4)
    for k in range(5):
        np.testing.assert_allclose(w[k], vandermonde_weights(z, x, k), rtol=1e-9)


def test_rows_beyond_the_node_count_are_zero_and_scaling_is_h_to_minus_k():
    x = np.array([0.0, 1.0, 2.0])
    w = fornberg_weights(1.0, x, 4)
    assert np.all(w[3:] == 0)
    h = 0.01
    ws = fornberg_weights(h, h * x, 2)
    np.testing.assert_allclose(ws[1], w[1] / h)
    np.testing.assert_allclose(ws[2], w[2] / h**2)
