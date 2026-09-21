"""The shared figure helpers: the surface plot's hole mask, view and title."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

from heat_interfaces.plotting import surface_over_nodes  # noqa: E402


def _grid_nodes(m: int = 21) -> SimpleNamespace:
    x, y = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
    return SimpleNamespace(x=x.ravel(), y=y.ravel())


def _drawn_polygons(fig) -> int:
    """One polygon per triangle ``plot_trisurf`` kept, known once drawn."""
    fig.canvas.draw()
    (poly,) = fig.axes[0].collections
    return len(poly.get_paths())


def test_surface_drops_exactly_the_triangles_whose_centroid_is_in_the_hole():
    nodes = _grid_nodes()
    tri = mtri.Triangulation(nodes.x, nodes.y)
    cx = nodes.x[tri.triangles].mean(axis=1)
    cy = nodes.y[tri.triangles].mean(axis=1)
    inside = np.hypot(cx - 0.5, cy - 0.5) < 0.2
    assert 0 < inside.sum() < len(inside)

    fig = surface_over_nodes(nodes, nodes.x, hole=(0.5, 0.5, 0.2), title="hole")
    assert _drawn_polygons(fig) == int((~inside).sum())
    ax = fig.axes[0]
    assert ax.get_title() == "hole" and (ax.elev, ax.azim) == (32, -128)
    plt.close(fig)

    fig = surface_over_nodes(nodes, nodes.x)
    assert _drawn_polygons(fig) == len(tri.triangles)
    assert fig.axes[0].get_title() == ""
    plt.close(fig)
