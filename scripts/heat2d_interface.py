"""E2.3 (#17): interface-aware stencils on case 1, basis continuity along curved
interfaces, and the conditioning of the continuity matrices.

Three tables on stdout and one figure under ``outputs/`` for the translated
polynomial bases of ``heat_interfaces.heat2d.interface`` (dissertation §5.3,
EABE §2.2.3) and the operator of ``heat2d.operators.interface_aware_operator``.

``heat2d_interface_convergence.png``: RMS error against the node count on
case 1 (``α = 0.2`` on ``0.6 ≤ y ≤ 0.8``, EABE eq. 32–34) for the naive
``Dx A Dx + Dy A Dy`` of E2.2 and for the interface-aware operator, the
twin of EABE Fig. 7 (six markers, 1250 to 40,000 nodes: 1.0e-5, 2.6e-6,
5.5e-7, 8e-8, 1.8e-8, 6e-9, read off the rendered page). The convergence
table adds the size of the interface group, how many of its stencils
cross, the order per halving of ``h`` and the times to build the operator
and to solve. Case 1's interfaces are flat, so the flat and
curvature-included variants give the same matrix; the table prints their
largest difference. The Gaussians are the plain ones (``warp=False``), the
setting of port notes §2.3; E2.4's ``heat2d_warp.py`` adds the warp.

The continuity table evaluates the translated basis (degree 4) along the
sine interface of case 2 and along the outer circle of case 3's ring at arc
offsets ``ξ = 1/2, 1/4, 1/8, 1/16`` stencil radii from the foot point and
prints the largest jump in ``u`` and in ``α n·∇u`` over the basis, for the
curvature-included and the flat variants: the curved jumps fall by ``2^(p+1)``
(u) and ``2^p`` (flux) per halving, the flat ones by 4 and 2.

The conditioning table takes every crossing stencil of a case-2 node set at
each count, builds its continuity matrices in units of the stencil radius
(epic #4, item 5) and prints the median and largest condition numbers of
``C_minus``, ``C_plus`` and of the translation ``C_plus^{-1} C_minus``, to
show there is no trend with N.

    uv run python scripts/heat2d_interface.py                       # ~30 s
    uv run python scripts/heat2d_interface.py --counts 2500 5000 10000 20000
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
    BOUNDARY,
    Band,
    Circle,
    Constant2D,
    SineGraph,
    SineProduct,
    build_node_set,
    build_stencils,
    case1,
    case1_exact,
    case2,
    coefficient_dx,
    coefficient_dy,
    continuity_matrix,
    interface_aware_operator,
    interface_crossings,
    local_interface,
    naive_operator,
    polynomial_block,
    rms_error,
    solve_equilibrium,
    translated_basis,
    translation_matrix,
)
from heat_interfaces.plotting import AWARE, NAIVE  # noqa: E402

DEGREE = BOUNDARY.degree

FIG7 = {
    1250: 1.0e-5,
    2500: 2.6e-6,
    5000: 5.5e-7,
    10000: 8e-8,
    20000: 1.8e-8,
    40000: 6e-9,
}
"""EABE Fig. 7 read off the rendered page (2026-09-21)."""


def top(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)


# --- case 1 --------------------------------------------------------------------


def sweep(
    counts: tuple[int, ...], seed: int = 0, iterations: int = 100
) -> list[dict[str, float]]:
    """One row per count: naive and interface-aware RMS errors on case 1, timings."""
    domain = case1()
    exact = case1_exact()
    rows = []
    for n in counts:
        nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
        reference = exact(nodes.x, nodes.y)
        row: dict[str, float] = {"n": nodes.n, "h": nodes.h}
        t0 = time.perf_counter()
        plain = build_stencils(nodes, domain)
        u = solve_equilibrium(
            naive_operator(nodes, domain.material, plain), nodes, [0.0, top]
        )
        row["naive"] = rms_error(u, reference)
        row["naive-seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        stencils = build_stencils(nodes, domain, interface=BOUNDARY)
        group = stencils.groups[-1]
        row["group"] = len(group.rows)
        row["crossing"] = int(
            interface_crossings(nodes, domain.material, group.index).sum()
        )
        op = interface_aware_operator(nodes, domain.material, stencils, warp=False)
        row["build-seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        u = solve_equilibrium(op, nodes, [0.0, top])
        row["aware"] = rms_error(u, reference)
        row["solve-seconds"] = time.perf_counter() - t0
        flat = interface_aware_operator(
            nodes, domain.material, stencils, curvature=False, warp=False
        )
        row["flat-diff"] = float(abs(op - flat).max())
        rows.append(row)
    return rows


def rates(rows: list[dict[str, float]], name: str) -> list[float]:
    """Order per halving of ``h`` between consecutive rows (``nan`` for the first)."""
    out = [float("nan")]
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        out.append(np.log(a[name] / b[name]) / np.log(a["h"] / b["h"]))
    return out


def print_convergence(rows: list[dict[str, float]]) -> None:
    print("case 1: naive Dx A Dx + Dy A Dy against the interface-aware operator")
    print(
        "     n       h  group  cross |      naive  order   time |      aware  order"
        "  build  solve  flat-diff |  EABE Fig. 7"
    )
    rn, ra = rates(rows, "naive"), rates(rows, "aware")
    for i, row in enumerate(rows):
        order = [("    -" if np.isnan(r[i]) else f"{r[i]:5.2f}") for r in (rn, ra)]
        fig7 = FIG7.get(int(row["n"]))
        fig7s = f"{fig7:9.1e}" if fig7 else "        -"
        print(
            f"{int(row['n']):6d}  {row['h']:.4f}  {int(row['group']):5d}  "
            f"{int(row['crossing']):5d} | {row['naive']:10.2e}  {order[0]}  "
            f"{row['naive-seconds']:4.1f}s | {row['aware']:10.2e}  {order[1]}  "
            f"{row['build-seconds']:4.1f}s  {row['solve-seconds']:4.1f}s  "
            f"{row['flat-diff']:9.1e} | {fig7s}"
        )


def figure_convergence(rows: list[dict[str, float]]):
    n = np.array([r["n"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.loglog(
        n, [r["naive"] for r in rows], "o-", color=NAIVE, markersize=4, label="naive"
    )
    ax.loglog(
        n,
        [r["aware"] for r in rows],
        "o-",
        color=AWARE,
        markersize=4,
        label="interface-aware (translated basis)",
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
    e1 = rows[0]["naive"]
    ax.loglog(n, e1 * (n / n[0]) ** -0.5, "k--", linewidth=0.7, label="1st order")
    e4 = rows[0]["aware"]
    ax.loglog(n, e4 * (n / n[0]) ** -2.0, "k-.", linewidth=0.7, label="4th order")
    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("RMS error in u")
    ax.set_title("case 1: two flat interfaces, α = 0.2 on 0.6 ≤ y ≤ 0.8")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --- basis continuity along curved interfaces -----------------------------------

CURVES = {
    "case-2 sine y = 0.6 + 0.02 sin 2πx": (
        Band(SineGraph(0.6), SineGraph(0.8), SineProduct(0.2, 0.1), Constant2D(1.0)),
        0,
        (0.13, 0.62),
    ),
    "case-3 ring r = 0.35 (α 1/1500 : 1)": (
        Band(
            Circle(0.349),
            Circle(0.35),
            SineProduct(1 / 1500, 1 / 3000),
            Constant2D(1.0),
        ),
        1,
        (0.5 + 0.36 * np.cos(0.7), 0.5 + 0.36 * np.sin(0.7)),
    ),
}

OFFSETS = (0.5, 0.25, 0.125, 0.0625)


def basis_jumps(
    band: Band, j: int, x: float, y: float, curvature: bool, scale: float = 0.05
) -> np.ndarray:
    """``(len(OFFSETS), 2)``: largest jump in ``u`` and in ``α n·∇u`` over the basis."""
    li = local_interface(band, j, x, y, scale, DEGREE, curvature)
    regions = translated_basis({j: li}, j + 1, j, j + 1)
    cm, cp = regions[j].coefficients, regions[j + 1].coefficients
    curve = band.interfaces[j]
    s0 = curve.closest(x, y)
    dx, dy = coefficient_dx(DEGREE), coefficient_dy(DEGREE)
    out = []
    for xi in OFFSETS:
        s = s0 + np.array([-xi, xi]) * scale / curve.length
        px, py = curve.point(s)
        nx, ny = curve.normal(s)
        v = polynomial_block(*li.frame.local(px, py), DEGREE)
        g_xi, g_eta = li.frame.rotate_in(nx, ny)
        am = band.region_piece(j).alpha(px, py)[:, None]
        ap = band.region_piece(j + 1).alpha(px, py)[:, None]
        fm = am * (g_xi[:, None] * (v @ dx @ cm) + g_eta[:, None] * (v @ dy @ cm))
        fp = ap * (g_xi[:, None] * (v @ dx @ cp) + g_eta[:, None] * (v @ dy @ cp))
        out.append((np.abs(v @ cm - v @ cp).max(), np.abs(fm - fp).max()))
    return np.array(out)


def continuity_table() -> dict[str, dict[str, np.ndarray]]:
    return {
        name: {
            "curved": basis_jumps(band, j, x, y, True),
            "flat": basis_jumps(band, j, x, y, False),
        }
        for name, (band, j, (x, y)) in CURVES.items()
    }


def print_continuity(table: dict[str, dict[str, np.ndarray]]) -> None:
    print(
        f"\nbasis continuity along curved interfaces (degree {DEGREE}, scale 0.05):"
        " largest jump over the basis at arc offset ξ (stencil radii)"
    )
    head = "interface / variant                        "
    head += "".join(f" | ξ = {xi:<6.4f} u      flux  " for xi in OFFSETS)
    print(head)
    for name, variants in table.items():
        for variant, jumps in variants.items():
            line = f"{name:34s} {variant:7s}"
            for ju, jf in jumps:
                line += f" | {ju:9.2e} {jf:9.2e}     "
            print(line)
            ratio = jumps[:-1] / jumps[1:]
            print(
                f"{'':42s}   ratios per halving: u "
                + " ".join(f"{r:5.1f}" for r in ratio[:, 0])
                + "   flux "
                + " ".join(f"{r:5.1f}" for r in ratio[:, 1])
            )


# --- conditioning of the continuity matrices --------------------------------------


def conditioning(counts: tuple[int, ...], seed: int = 0) -> list[dict[str, float]]:
    """Condition numbers of the continuity and translation matrices on case 2.

    ``outside`` is the side with ``α ≡ 1`` (region 0 of the lower interface,
    region 2 of the upper), ``inside`` the band's ``0.2 + 0.1 sin 2πx sin 2πy``;
    the translation is taken from outside to inside.
    """
    domain = case2()
    band = domain.material
    rows = []
    for n in counts:
        nodes = build_node_set(domain, n, seed=seed)
        stencils = build_stencils(nodes, domain, interface=BOUNDARY)
        group = stencils.groups[-1]
        cross = interface_crossings(nodes, band, group.index)
        region = band.region_index(nodes.x, nodes.y)
        conds: dict[str, list[float]] = {"outside": [], "inside": [], "translation": []}
        for idx in group.index[cross]:
            xy = nodes.xy[idx]
            dx = np.mod(xy[:, 0] - xy[0, 0] + 0.5, 1.0) - 0.5
            scale = float(np.hypot(dx, xy[:, 1] - xy[0, 1]).max())
            lo, hi = int(region[idx].min()), int(region[idx].max())
            for j in range(lo, hi):
                li = local_interface(band, j, xy[0, 0], xy[0, 1], scale, DEGREE)
                outside, inside = (li.minus, li.plus) if j == 0 else (li.plus, li.minus)
                conds["outside"].append(
                    np.linalg.cond(continuity_matrix(outside, li.expansion))
                )
                conds["inside"].append(
                    np.linalg.cond(continuity_matrix(inside, li.expansion))
                )
                conds["translation"].append(
                    np.linalg.cond(translation_matrix(outside, inside, li.expansion))
                )
        row: dict[str, float] = {
            "n": nodes.n,
            "h": nodes.h,
            "stencils": int(cross.sum()),
        }
        for name, values in conds.items():
            row[name + "-median"] = float(np.median(values))
            row[name + "-max"] = float(np.max(values))
        rows.append(row)
    return rows


def print_conditioning(rows: list[dict[str, float]]) -> None:
    print(
        "\ncontinuity matrices on case 2, units of the stencil radius:"
        " condition numbers over the crossing stencils"
    )
    print(
        "     n       h  stencils |  C outside (α=1) median    max |"
        "  C inside (band) median    max |  translation median     max"
    )
    for r in rows:
        print(
            f"{int(r['n']):6d}  {r['h']:.4f}  {int(r['stencils']):8d} |"
            f"  {r['outside-median']:22.1f} {r['outside-max']:6.1f} |"
            f"  {r['inside-median']:22.1f} {r['inside-max']:6.1f} |"
            f"  {r['translation-median']:18.1f} {r['translation-max']:7.1f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--counts", type=int, nargs="+", default=[1250, 2500, 5000, 10000]
    )
    parser.add_argument(
        "--conditioning-counts", type=int, nargs="+", default=[1250, 5000, 20000]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    if len(args.counts) < 2 or any(n < 300 for n in args.counts):
        parser.error("give at least two counts of 300 nodes or more")
    if any(n < 300 for n in args.conditioning_counts):
        parser.error("the conditioning counts need 300 nodes or more")
    args.outputs.mkdir(parents=True, exist_ok=True)

    rows = sweep(tuple(args.counts), args.seed, args.iterations)
    print_convergence(rows)
    fig = figure_convergence(rows)
    fig.savefig(args.outputs / "heat2d_interface_convergence.png", dpi=150)
    plt.close(fig)

    print_continuity(continuity_table())

    t0 = time.perf_counter()
    print_conditioning(conditioning(tuple(args.conditioning_counts), args.seed))
    print(f"({time.perf_counter() - t0:.1f} s for the conditioning sweep)")


if __name__ == "__main__":
    main()
