"""E2.1 (#15): the 2-D node sets, twins of dissertation Fig. 5-3 and EABE Fig. 12.

Three figures under ``outputs/`` for the node sets of the three 2016 cases
(``heat_interfaces.heat2d.domain``): fixed rows straddling every interface
in the MATLAB's hexagonal layout, Dirichlet rows on ``y = 0``, ``y = 1`` and,
for case 3, on the circle ``r = 0.05``, the rest scattered by repulsion.

``heat2d_nodes_case1.png`` (Fig. 5-3): 2500 nodes about the flat band.

``heat2d_nodes_case2.png``: the same count about the sine pair
``0.6, 0.8 + 0.02 sin 2πx`` (EABE Fig. 8's band).

``heat2d_nodes_case3.png`` (EABE Fig. 12a/b): the whole domain with the
ring and the cooling circle, and the zoom ``[0.73, 0.79] × [0.24, 0.30]``
of Fig. 12b, where the ring's two interfaces pass between one straddling
pair.

The report prints, per case, the counts of each kind of node and the
nearest-neighbour spacing of the free nodes in units of the row spacing
``h = 1 / round(0.95 √N)``; ``docs/port-notes.md`` §2.1 carries the table.

    uv run python scripts/heat2d_nodesets.py                 # < 1 s
    uv run python scripts/heat2d_nodesets.py --n 10000 --cases 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from heat_interfaces.heat2d import (  # noqa: E402
    CASES,
    Circle,
    Domain,
    NodeSet,
    build_node_set,
    nearest_spacing,
)
from heat_interfaces.plotting import REFERENCE  # noqa: E402

ZOOM = ((0.73, 0.79), (0.24, 0.30))
"""EABE Fig. 12b's window."""


def draw_curve(ax, curve, **kw) -> None:
    """A curve as a line; graphs are drawn over one period, circles closed."""
    s = np.linspace(0.0, 1.0, 721)
    x, y = curve.point(s)
    if not isinstance(curve, Circle):
        x = s  # unwrapped, so the line does not jump at the seam
    ax.plot(x, y, **kw)


def plot_nodes(ax, nodes: NodeSet, domain: Domain, markersize: float = 2.5) -> None:
    """Black dots as in the 2016 figures, interfaces dashed, holes dotted."""
    ax.plot(nodes.x, nodes.y, ".k", markersize=markersize)
    for curve in domain.material.interfaces:
        draw_curve(ax, curve, color=REFERENCE, linestyle="--", linewidth=0.8)
    for hole in domain.holes:
        draw_curve(ax, hole, color=REFERENCE, linestyle=":", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def figure_full(nodes: NodeSet, domain: Domain, title: str):
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    plot_nodes(ax, nodes, domain)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def figure_case3(nodes: NodeSet, domain: Domain, title: str):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 4.8))
    plot_nodes(a, nodes, domain)
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_title(title)
    plot_nodes(b, nodes, domain, markersize=7)
    (x0, x1), (y0, y1) = ZOOM
    b.set_xlim(x0, x1)
    b.set_ylim(y0, y1)
    b.set_title("the ring between one straddling pair (Fig. 12b)")
    fig.tight_layout()
    return fig


def report(case: int, nodes: NodeSet, seconds: float) -> dict[str, float]:
    """One row of the table: counts and free-node spacing in units of ``h``."""
    spacing = nearest_spacing(nodes.xy) / nodes.h
    free = spacing[nodes.free]
    return {
        "case": case,
        "n": nodes.n,
        "h": nodes.h,
        "per_row": len(nodes.straddle_rows[0].index),
        "straddle": sum(len(r.index) for r in nodes.straddle_rows),
        "dirichlet": sum(len(r.index) for r in nodes.dirichlet_rows),
        "free": int(nodes.free.sum()),
        "min": free.min(),
        "median": float(np.median(free)),
        "max": free.max(),
        "seconds": seconds,
    }


def print_table(rows: list[dict[str, float]]) -> None:
    print(
        "case      n       h  per row  straddle  Dirichlet   free   "
        "NN/h min  median   max    time"
    )
    for r in rows:
        print(
            f"{r['case']:>4} {r['n']:>6}  {r['h']:.4f}  {r['per_row']:>7}  "
            f"{r['straddle']:>8}  {r['dirichlet']:>9}  {r['free']:>5}   "
            f"{r['min']:>8.3f}  {r['median']:>6.3f}  {r['max']:>4.2f}  "
            f"{r['seconds']:>5.2f} s"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=2500, help="nodes per set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cases", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    bad = sorted(set(args.cases) - set(CASES))
    if bad:
        parser.error(f"unknown cases {bad}; the cases are {sorted(CASES)}")
    args.outputs.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in args.cases:
        domain = CASES[case]()
        t0 = time.perf_counter()
        nodes = build_node_set(
            domain, args.n, seed=args.seed, iterations=args.iterations
        )
        rows.append(report(case, nodes, time.perf_counter() - t0))
        title = f"case {case}: {nodes.n} nodes"
        if case == 3:
            fig = figure_case3(nodes, domain, title)
        else:
            fig = figure_full(nodes, domain, title)
        fig.savefig(args.outputs / f"heat2d_nodes_case{case}.png", dpi=150)
        plt.close(fig)
    print_table(rows)


if __name__ == "__main__":
    main()
