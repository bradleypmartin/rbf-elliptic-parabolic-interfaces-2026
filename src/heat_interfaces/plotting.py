"""Figure style shared by the drivers and the manuscript.

Blue is the interface-aware method (and, from E3 on, the seeds); orange is
the naive baseline, as in the wave-equation companion. ``use_print_style``
sets the manuscript's sizes; the drivers' defaults are for the screen.
"""

from __future__ import annotations

import matplotlib as mpl

AWARE = "#1f77b4"
NAIVE = "#ff7f0e"
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
