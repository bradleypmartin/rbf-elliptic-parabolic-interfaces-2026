"""E2.4 (#18): warped RBFs across interfaces (EABE §2.2.4) and the warp-and-straddle
ablation of EABE Fig. 11.

Two tables on stdout and two figures under ``outputs/``.

``heat2d_warp_case1.png``: RMS error against the node count on case 1
(``α = 0.2`` on ``0.6 ≤ y ≤ 0.8``, analytic solution, EABE eq. 32–34) for the
interface-aware operator in the four combinations of warped / plain Gaussians
and node sets with / without the straddling rows, next to EABE Fig. 7's
markers. Case 1 has the exact solution, so this is the clean measure of what
the warp and the rows buy; the table adds the interface-group sizes and the
order per halving of ``h``.

The case-2 table runs the same four combinations on case 2 (the sine pair,
EABE eq. 35–36), which has no analytic solution, at three resolutions: the
group sizes, the build and solve times, the largest ``|u|`` (the Dirichlet
data bound it by 1) and how far the warp moves the solution on the same node
set, RMS and largest ``|u_warp − u_plain|``. The errors against the
160,000-node reference, Fig. 11 proper, are E2.6's (#20).

``heat2d_warped_rbf.png``: the twin of EABE Fig. 6 / dissertation Fig. 5-1,
contours of one Gaussian centred a unit below a flat interface through the
origin at slope 0.2, ``α = 1`` below it and ``1/2`` above, plain and warped.

    uv run python scripts/heat2d_warp.py                              # ~45 s
    uv run python scripts/heat2d_warp.py --counts 1250 2500 5000 10000 20000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from math import atan, cos, sin
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from heat_interfaces.heat2d import (  # noqa: E402
    BOUNDARY,
    Domain,
    Warp,
    build_node_set,
    build_stencils,
    case1,
    case1_exact,
    case2,
    gaussian,
    interface_aware_operator,
    interface_crossings,
    rms_error,
    solve_equilibrium,
)
from heat_interfaces.plotting import AWARE, REFERENCE  # noqa: E402

FIG7 = {
    1250: 1.0e-5,
    2500: 2.6e-6,
    5000: 5.5e-7,
    10000: 8e-8,
    20000: 1.8e-8,
    40000: 6e-9,
}
"""EABE Fig. 7 read off the rendered page (2026-09-21, as in heat2d_interface.py)."""

VARIANTS = (
    ("warp+rows", True, True),
    ("plain+rows", False, True),
    ("warp,none", True, False),
    ("plain,none", False, False),
)
"""``(name, warp, straddle)``: Fig. 11's two lines are the first and the last."""


def top(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)


def node_sets(domain: Domain, n: int, seed: int, iterations: int):
    """``{True: with rows, False: without}``: node sets and interface stencils."""
    out = {}
    for straddle in (True, False):
        dom = domain if straddle else replace(domain, straddle=())
        nodes = build_node_set(dom, n, seed=seed, iterations=iterations)
        out[straddle] = (dom, nodes, build_stencils(nodes, dom, interface=BOUNDARY))
    return out


def run(dom, nodes, stencils, warp: bool) -> tuple[np.ndarray, float, float]:
    """``(u, build seconds, solve seconds)`` of one variant on one node set."""
    t0 = time.perf_counter()
    op = interface_aware_operator(nodes, dom.material, stencils, warp=warp)
    t1 = time.perf_counter()
    u = solve_equilibrium(op, nodes, [0.0, top])
    return u, t1 - t0, time.perf_counter() - t1


# --- case 1: the four variants against the analytic solution ---------------------


def sweep_case1(
    counts: tuple[int, ...], seed: int = 0, iterations: int = 100
) -> list[dict[str, float]]:
    """One row per count: RMS errors of the four variants on case 1."""
    domain, exact = case1(), case1_exact()
    rows = []
    for n in counts:
        sets = node_sets(domain, n, seed, iterations)
        row: dict[str, float] = {"n": sets[True][1].n, "h": sets[True][1].h}
        for straddle, (dom, nodes, stencils) in sets.items():
            row[f"group-{straddle}"] = len(stencils.groups[-1].rows)
            reference = exact(nodes.x, nodes.y)
            for name, warp, s in VARIANTS:
                if s == straddle:
                    u, row[f"build-{name}"], _ = run(dom, nodes, stencils, warp)
                    row[name] = rms_error(u, reference)
        rows.append(row)
    return rows


def rates(rows: list[dict[str, float]], name: str) -> list[float]:
    """Order per halving of ``h`` between consecutive rows (``nan`` for the first)."""
    out = [float("nan")]
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        out.append(np.log(a[name] / b[name]) / np.log(a["h"] / b["h"]))
    return out


def print_case1(rows: list[dict[str, float]]) -> None:
    print(
        "case 1: warped vs plain Gaussians, with and without the straddling rows"
        " (RMS error vs the analytic solution; 'group' is the interface group"
        " with rows / without)"
    )
    head = "     n       h   group(rows/none) |"
    head += "".join(f" {name:>11s}  order |" for name, _, _ in VARIANTS)
    print(head + "  EABE Fig. 7")
    orders = {name: rates(rows, name) for name, _, _ in VARIANTS}
    for i, row in enumerate(rows):
        line = (
            f"{int(row['n']):6d}  {row['h']:.4f}   {int(row['group-True']):5d} /"
            f" {int(row['group-False']):5d} |"
        )
        for name, _, _ in VARIANTS:
            o = orders[name][i]
            line += f"   {row[name]:9.2e}  {'    -' if np.isnan(o) else f'{o:5.2f}'} |"
        fig7 = FIG7.get(int(row["n"]))
        print(line + (f"    {fig7:8.1e}" if fig7 else "           -"))
    print("ratio to Fig. 7:", end="")
    for name, _, _ in VARIANTS:
        ratios = [
            f"{row[name] / FIG7[int(row['n'])]:.1f}"
            for row in rows
            if int(row["n"]) in FIG7
        ]
        print(f"  {name} {', '.join(ratios)};", end="")
    print()


def figure_case1(rows: list[dict[str, float]]):
    n = np.array([r["n"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    styles = {
        "warp+rows": ("o-", AWARE, "warped RBFs, straddling rows"),
        "plain+rows": ("o-", REFERENCE, "plain RBFs, straddling rows"),
        "warp,none": ("s--", AWARE, "warped RBFs, no rows"),
        "plain,none": ("s--", REFERENCE, "plain RBFs, no rows"),
    }
    for name, _, _ in VARIANTS:
        marker, color, label = styles[name]
        ax.loglog(
            n,
            [r[name] for r in rows],
            marker,
            color=color,
            markersize=4,
            markerfacecolor="none" if "none" in name else color,
            label=label,
        )
    fig7 = sorted(FIG7.items())
    ax.loglog(
        [k for k, _ in fig7],
        [v for _, v in fig7],
        "s:",
        color="k",
        markerfacecolor="none",
        markersize=5,
        linewidth=0.7,
        label="EABE 2017 Fig. 7 (read off)",
    )
    e4 = rows[0]["warp+rows"]
    ax.loglog(n, e4 * (n / n[0]) ** -2.0, "k-.", linewidth=0.7, label="4th order")
    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("RMS error in u")
    ax.set_title("case 1: warped vs plain RBFs, with and without straddling rows")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --- case 2: the ablation runs ----------------------------------------------------


def sweep_case2(
    counts: tuple[int, ...], seed: int = 0, iterations: int = 100
) -> list[dict[str, float]]:
    """One row per count and node set: sizes, times, ``max |u|``, the warp's effect."""
    domain = case2()
    rows = []
    for n in counts:
        for straddle, (dom, nodes, stencils) in node_sets(
            domain, n, seed, iterations
        ).items():
            group = stencils.groups[-1]
            row: dict[str, float] = {
                "n": nodes.n,
                "h": nodes.h,
                "rows": straddle,
                "group": len(group.rows),
                "crossing": int(
                    interface_crossings(nodes, dom.material, group.index).sum()
                ),
            }
            u = {}
            for warp in (True, False):
                u[warp], row[f"build-{warp}"], row[f"solve-{warp}"] = run(
                    dom, nodes, stencils, warp
                )
                row[f"max-{warp}"] = float(np.abs(u[warp]).max())
            row["moved-rms"] = rms_error(u[True], u[False])
            row["moved-max"] = float(np.abs(u[True] - u[False]).max())
            rows.append(row)
    return rows


def print_case2(rows: list[dict[str, float]]) -> None:
    print(
        "\ncase 2: the four combinations at three resolutions (no analytic"
        " solution; the errors against the 160,000-node reference are E2.6's)"
    )
    print(
        "     n       h  rows  group  cross |  build warp  plain   solve |"
        "  max|u| warp    plain |  warp - plain  rms       max"
    )
    for r in rows:
        print(
            f"{int(r['n']):6d}  {r['h']:.4f}  {'yes' if r['rows'] else ' no':>4s}"
            f"  {int(r['group']):5d}  {int(r['crossing']):5d} |"
            f"  {r['build-True']:8.1f}s  {r['build-False']:4.1f}s"
            f"  {r['solve-True']:4.1f}s |"
            f"  {r['max-True']:11.6f}  {r['max-False']:.6f} |"
            f"  {r['moved-rms']:18.2e}  {r['moved-max']:8.2e}"
        )


# --- the warped Gaussian of EABE Fig. 6 --------------------------------------------


def figure_warped_rbf(slope_ratio: float = 2.0, shape: float = 0.4):
    """Contours of a Gaussian below ``y = 0.2 x``, plain and warped across it.

    ``α = 1`` below the line and ``1/2`` above, so the normal coordinate above
    it is stretched by ``α_below / α_above = 2``; the Gaussian is centred a
    unit below the line with ``ε = 0.4``, as in the papers' figure. The frame
    is rotated by hand: ``Frame.local`` wraps ``x`` with the strip's period,
    and this picture spans eight of them.
    """
    warp = Warp(0, 0, np.array([1.0, slope_ratio]), np.zeros(2))
    xi_c, eta_c = 0.0, -1.0
    x, y = np.meshgrid(np.linspace(-4, 4, 321), np.linspace(-4, 4, 321))
    c, s = cos(atan(0.2)), sin(atan(0.2))
    xi, eta = c * x + s * y, -s * x + c * y
    region = (eta > 0).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), sharey=True)
    levels = np.linspace(0.05, 0.95, 10)
    for ax, warped in zip(axes, (False, True), strict=True):
        et = warp.apply(eta, region) if warped else eta
        phi = gaussian(xi - xi_c, et - eta_c, shape)
        ax.contour(x, y, phi, levels=levels, colors=AWARE, linewidths=0.8)
        ax.plot([-4, 4], [-0.8, 0.8], "k-", linewidth=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_title("plain Gaussian" if not warped else "warped across the interface")
        ax.text(-3.8, 3.4, "α = 1/2", fontsize=9)
        ax.text(-3.8, -3.7, "α = 1", fontsize=9)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.suptitle("EABE Fig. 6: a Gaussian (ε = 0.4) centred below y = 0.2 x")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--counts", type=int, nargs="+", default=[1250, 2500, 5000, 10000]
    )
    parser.add_argument(
        "--case2-counts", type=int, nargs="+", default=[1250, 2500, 5000]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    if len(args.counts) < 2 or any(n < 300 for n in args.counts):
        parser.error("give at least two counts of 300 nodes or more")
    if any(n < 300 for n in args.case2_counts):
        parser.error("the case-2 counts need 300 nodes or more")
    args.outputs.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rows = sweep_case1(tuple(args.counts), args.seed, args.iterations)
    print_case1(rows)
    print(f"({time.perf_counter() - t0:.1f} s for the case-1 sweep)")
    fig = figure_case1(rows)
    fig.savefig(args.outputs / "heat2d_warp_case1.png", dpi=150)
    plt.close(fig)

    t0 = time.perf_counter()
    print_case2(sweep_case2(tuple(args.case2_counts), args.seed, args.iterations))
    print(f"({time.perf_counter() - t0:.1f} s for the case-2 sweep)")

    fig = figure_warped_rbf()
    fig.savefig(args.outputs / "heat2d_warped_rbf.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
