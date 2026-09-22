"""Figure style shared by the drivers and the manuscript.

Blue is the interface-aware method (and, from E3 on, the seeds); orange is
the naive baseline, as in the wave-equation companion; purple is the δ = 0
construction of the stiff-edge study (E1.2's rows applied to a smooth edge,
the baseline the seeds are measured against). ``use_print_style`` sets the
manuscript's sizes; the drivers' defaults are for the screen.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

AWARE = "#1f77b4"
NAIVE = "#ff7f0e"
CONSTRUCTION = "#9467bd"
REFERENCE = "#7f7f7f"


def use_print_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.2,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def surface_over_nodes(nodes, u: np.ndarray, *, hole=None, title: str = ""):
    """``u`` as a surface over a scattered node set, a Dirichlet hole cut out.

    ``hole`` is ``(cx, cy, r)``: triangles whose centroid lies inside it are
    masked, which is how the case-3 drivers draw the cooling disc (the Fig. 13
    and Fig. 5-15 twins). Pyplot is imported here so that the drivers choose
    the backend before it loads.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(nodes.x, nodes.y)
    if hole is not None:
        cx, cy, r = hole
        centroid_x = nodes.x[tri.triangles].mean(axis=1)
        centroid_y = nodes.y[tri.triangles].mean(axis=1)
        tri.set_mask(np.hypot(centroid_x - cx, centroid_y - cy) < r)
    fig = plt.figure(figsize=(7.0, 5.2))
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(
        tri, u, cmap="viridis", linewidth=0.0, antialiased=False, shade=False
    )
    ax.view_init(elev=32, azim=-128)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title(title)
    fig.tight_layout()
    return fig
