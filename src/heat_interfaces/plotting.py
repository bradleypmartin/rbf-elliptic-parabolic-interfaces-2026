"""Figure style shared by the drivers and the manuscript.

Blue is the interface-aware method (and, from E3 on, the seeds); orange is
the naive baseline, as in the wave-equation companion; purple is the δ = 0
construction of the stiff-edge study (E1.2's rows applied to a smooth edge,
the baseline the seeds are measured against). ``use_print_style`` sets the
manuscript's chrome and sizes (``scripts/paper_figures.py``, E5.3); the
drivers' defaults are for the screen.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import numpy as np

AWARE = "#1f77b4"
NAIVE = "#ff7f0e"
CONSTRUCTION = "#9467bd"
REFERENCE = "#7f7f7f"

INK = "#222222"
INK_MUTED = "#6b6b6b"
GRID = "#e3e3e3"

TEXTWIDTH = 360.0 / 72.27
"""The manuscript's text width in inches: amsart at 11 pt sets ``\\textwidth`` to
360 pt (measured with tectonic, E5.3). A print figure drawn this wide goes in at
its natural size, so its fonts are the sizes ``use_print_style`` sets."""


def use_print_style() -> None:
    """The manuscript's figures (E5.3): serif with Computer Modern mathtext, 8 pt
    beside the 11 pt body, 7 pt ticks and legends, thin recessive axes.

    The palette is the drivers'; only the chrome differs. Pins matplotlib's PDF
    ``CreationDate`` through ``SOURCE_DATE_EPOCH`` so that a regenerated figure is
    byte-identical to the committed one (``paper_figures.py --check``), the figure
    analogue of ``main.tex``'s fixed ``\\date``: an assignment, not a default, since
    an inherited value would break the guarantee. matplotlib reads it when the PDF
    is written, so any time before ``savefig`` will do.
    """
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.size": 8.0,
            "mathtext.fontset": "cm",
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.0,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
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
