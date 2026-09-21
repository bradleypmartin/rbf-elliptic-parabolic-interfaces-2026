import numpy as np
import pytest

from heat_interfaces.heat2d.neighbors import (
    knn,
    nearest_spacing,
    offsets,
    periodic_dx,
    periodic_tree,
    wrap_x,
)

SEAM = np.array([[0.01, 0.5], [0.99, 0.5], [0.5, 0.01], [0.5, 0.99], [0.5, 0.5]])


def test_wrap_and_periodic_difference():
    np.testing.assert_allclose(wrap_x([-0.25, 0.0, 1.0, 1.75]), [0.75, 0.0, 0.0, 0.75])
    np.testing.assert_allclose(
        periodic_dx([0.98, -0.98, 0.4, -0.5]), [-0.02, 0.02, 0.4, -0.5]
    )


def test_knn_wraps_in_x_and_not_in_y():
    idx, dist = knn(SEAM, 2)
    assert idx[0, 0] == 0 and dist[0, 0] == 0.0
    assert idx[0, 1] == 1 and dist[0, 1] == pytest.approx(0.02)
    assert idx[1, 1] == 0
    # The nodes 0.02 apart across y = 0 / y = 1 are not neighbours.
    assert idx[2, 1] == 4 and dist[2, 1] == pytest.approx(0.49)
    assert idx[3, 1] == 4


def test_knn_with_queries_and_k_one_keep_two_dimensions():
    idx, dist = knn(SEAM, 1, query=np.array([[0.995, 0.5]]))
    assert idx.shape == dist.shape == (1, 1)
    assert idx[0, 0] == 1 and dist[0, 0] == pytest.approx(0.005)
    idx, _ = knn(SEAM, 3, query=np.array([[0.02, 0.5]]))
    assert set(idx[0, :2]) == {0, 1}


def test_offsets_take_the_nearest_image():
    idx, _ = knn(SEAM, 2)
    dx, dy = offsets(SEAM, idx)
    assert dx[0, 1] == pytest.approx(-0.02) and dy[0, 1] == 0.0
    assert dx[1, 1] == pytest.approx(0.02)
    dx, dy = offsets(SEAM, idx[:1], centre=np.array([[0.5, 0.5]]))
    assert dx[0, 1] == pytest.approx(0.49) and dy[0, 1] == 0.0


def test_tree_refuses_nodes_off_the_period():
    with pytest.raises(ValueError, match="wrap_x"):
        periodic_tree(np.array([[1.0, 0.5]]))
    with pytest.raises(ValueError, match="wrap_x"):
        periodic_tree(np.array([[-0.1, 0.5]]))
    with pytest.raises(ValueError, match="k = 7"):
        knn(SEAM, 7)


def test_nearest_spacing_on_a_lattice_is_the_lattice_spacing():
    x, y = np.meshgrid(np.arange(10) / 10, np.arange(5) / 10 + 0.3)
    xy = np.column_stack([x.ravel(), y.ravel()])
    np.testing.assert_allclose(nearest_spacing(xy), 0.1)
