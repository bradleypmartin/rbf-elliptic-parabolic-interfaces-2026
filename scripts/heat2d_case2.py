"""E2.6 (#20): case 2, the sine pair (EABE eq. 35–36): FD4 / flat / curved
convergence against a fine reference (EABE Fig. 10, dissertation Fig. 5-9), the
warp-and-straddle ablation (Fig. 11 / 5-10) and error against wall-clock
(dissertation Fig. 5-11).

Three figures under ``outputs/`` and five tables on stdout.

The reference is the interface-aware solution (curvature, warped Gaussians,
straddling rows) on ``--reference-n`` nodes, 160,000 in 2016, solved once and
cached as ``outputs/heat2d_case2_reference_n<N>_seed<s>.npz`` with its node set
and timings (``heat2d.resample.Reference``). Every coarse solution is compared
with it at the coarse nodes through ``heat2d.resample.resample``: the fine
set's own stencils, with the translated basis and warped Gaussians where they
cross an interface, since a blind interpolant across the kink of ``u`` is first
order there. The same resampling applied to the analytic solution of case 1,
sampled on a node set of ``--check-n`` nodes (the reference's count by
default, 0 to skip), measures what the resampling itself contributes, aware
and blind, at the coarse nodes and on the FD4 grids.

``heat2d_case2_convergence.png``: RMS error against the node count for the
Cartesian FD4 baseline ``Dx A Dx + Dy A Dy`` (``heat2d.fd4``, the MATLAB
``FDheat1.m`` form), the flat-interface RBF-FD variant and the
curvature-included one, next to Fig. 10's markers read off the rendered page.
``--fd4-counts`` may run the grid past the RBF-FD counts, since it is cheap
(plan D7).

``heat2d_case2_ablation.png``: the curvature-included operator with warped
Gaussians on the straddled node sets against plain Gaussians on node sets
built without the rows (``dataclasses.replace(domain, straddle=())``), the two
lines of Fig. 11.

``heat2d_case2_performance.png``: the same errors against the wall-clock of
node set (or grid), stencils, operator and solve on this machine, FD4 and
RBF-FD, with the two lines of Fig. 5-11 (MATLAB backslash on a 2.7 GHz Core
i7, 2016) for their shape only; its RBF/FD4 hybrid is not ported.

    uv run python scripts/heat2d_case2.py                              # ~55 s
    uv run python scripts/heat2d_case2.py --reference-n 160000 \\
        --counts 1250 2500 5000 10000 20000 40000 80000 \\
        --fd4-counts 1250 2500 5000 10000 20000 40000 80000 160000 320000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from heat_interfaces.heat2d import (  # noqa: E402
    BOUNDARY,
    Domain,
    NodeSet,
    Reference,
    Stencils,
    build_node_set,
    build_stencils,
    cartesian_grid,
    case1,
    case1_exact,
    case2,
    fd4_operator,
    interface_aware_operator,
    reference_solution,
    resample,
    rms_error,
    solve_equilibrium,
)
from heat_interfaces.plotting import AWARE, NAIVE, REFERENCE  # noqa: E402

FIG10 = {
    "fd4": {
        1250: 6.0e-3,
        2500: 5.5e-3,
        5000: 2.5e-3,
        10000: 2.2e-3,
        20000: 1.2e-3,
        40000: 8.5e-4,
        80000: 3.3e-4,
    },
    "flat": {
        1250: 5.2e-4,
        2500: 3.4e-4,
        5000: 2.4e-4,
        10000: 1.5e-4,
        20000: 1.05e-4,
        40000: 7.5e-5,
        80000: 5.0e-5,
    },
    "curved": {
        1250: 2.6e-5,
        2500: 8.5e-6,
        5000: 1.4e-6,
        10000: 3.9e-7,
        20000: 9.3e-8,
        40000: 3.1e-8,
        80000: 9.3e-9,
    },
}
"""EABE Fig. 10 (= dissertation Fig. 5-9) read off the rendered page (2026-09-21).

Seven markers per line, 1250 to 80,000 nodes, against a 160,000-node reference;
reading accuracy about ±15 %.
"""

FIG11_NONE = {
    1250: 1.7e-4,
    2500: 4.8e-5,
    5000: 2.0e-5,
    10000: 8.4e-6,
    20000: 6.0e-6,
    40000: 1.9e-7,
    80000: 3.1e-8,
}
"""Fig. 11's "no warp; no straddling" line; its other line is ``FIG10["curved"]``."""

FIG511 = {
    "fd4": (
        (0.012, 7.4e-3),
        (0.036, 6.3e-3),
        (0.066, 3.2e-3),
        (0.16, 2.7e-3),
        (0.31, 1.4e-3),
        (0.75, 8.5e-4),
        (1.5, 5.2e-4),
        (3.7, 5.0e-4),
        (9.0, 4.0e-4),
        (21.0, 4.8e-4),
        (51.0, 4.3e-4),
    ),
    "rbf": (
        (1.8, 4.0e-5),
        (3.9, 1.1e-5),
        (7.4, 2.7e-6),
        (14.0, 7.2e-7),
        (26.0, 1.7e-7),
        (51.0, 4.5e-8),
        (100.0, 6.2e-9),
        (200.0, 1.5e-9),
    ),
}
"""Dissertation Fig. 5-11 read off: ``(seconds, error)`` of its FD4 and RBF-only lines.

MATLAB backslash on a 2.7 GHz Core i7; node set, operator and solve. The
"RBF/FD4 hybrid" line (a node set that turns Cartesian away from the
interface) is not ported.
"""

VARIANTS = (
    ("flat", True, False, True),
    ("curved", True, True, True),
    ("none", False, True, False),
)
"""``(name, straddling rows, curvature, warp)``.

Fig. 10's two RBF-FD lines are ``flat`` and ``curved``; Fig. 11 compares
``curved`` ("with warp and straddling") with ``none`` ("no warp; no straddling").
"""


def top(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Eq. 36's Dirichlet row at ``y = 1``."""
    return np.sin(2 * np.pi * x)


# --- the reference ---------------------------------------------------------------


def reference_path(outputs: Path, n: int, seed: int) -> Path:
    return outputs / f"heat2d_case2_reference_n{n}_seed{seed}.npz"


def reference(
    domain: Domain, n: int, seed: int, iterations: int, outputs: Path
) -> tuple[Reference, bool]:
    """The cached fine solution, solved and saved if absent; ``True`` when reused."""
    path = reference_path(outputs, n, seed)
    if path.exists():
        ref = Reference.load(path)
        if ref.meta.get("iterations") == iterations and ref.meta.get("n") == n:
            return ref, True
    ref = reference_solution(domain, n, [0.0, top], seed=seed, iterations=iterations)
    ref.save(path)
    return ref, False


def error_against(
    ref: Reference, stencils: Stencils, material, nodes: NodeSet, u: np.ndarray
) -> tuple[float, float]:
    """``(RMS error, seconds)`` of ``u`` against the reference read at ``nodes``."""
    t0 = time.perf_counter()
    u_ref = resample(ref.u, ref.nodes, stencils, material, nodes.x, nodes.y)
    return rms_error(u, u_ref), time.perf_counter() - t0


def reference_error_estimate(
    rows: list[dict[str, float]], ref_n: int, name: str = "curved"
) -> tuple[float, int]:
    """The reference's own error from the finest ``name`` point, if ``e ∝ N⁻²``.

    The measured difference ``m(N) = e(N) − e(R)`` with ``e(R) = q e(N)``,
    ``q = (N / R)²``, gives ``e(R) = q m / (1 − q)``: a third of the measured
    80,000-node error against a 160,000-node reference.
    """
    finest = max(rows, key=lambda r: r["n"])
    q = (finest["n"] / ref_n) ** 2
    return finest[name] * q / (1.0 - q), int(finest["n"])


# --- the sweeps ------------------------------------------------------------------


def sweep(
    counts: tuple[int, ...],
    ref: Reference,
    ref_stencils: Stencils,
    seed: int = 0,
    iterations: int = 100,
) -> list[dict[str, float]]:
    """One row per count: the three RBF-FD variants' errors, group sizes and times.

    ``<name>-nodes`` is the node set with its stencils (shared by ``flat`` and
    ``curved``), ``<name>-build`` the operator, ``<name>-solve`` the solve and
    ``<name>-resample`` the reading of the reference at the nodes.
    """
    domain = case2()
    rows = []
    for n in counts:
        sets = {}
        row: dict[str, float] = {}
        for straddle in (True, False):
            dom = domain if straddle else replace(domain, straddle=())
            t0 = time.perf_counter()
            nodes = build_node_set(dom, n, seed=seed, iterations=iterations)
            stencils = build_stencils(nodes, dom, interface=BOUNDARY)
            sets[straddle] = (dom, nodes, stencils, time.perf_counter() - t0)
            row[f"group-{straddle}"] = int(stencils.near_interface.sum())
        row["n"], row["h"] = sets[True][1].n, sets[True][1].h
        for name, straddle, curvature, warp in VARIANTS:
            dom, nodes, stencils, nodes_seconds = sets[straddle]
            t0 = time.perf_counter()
            op = interface_aware_operator(
                nodes, dom.material, stencils, curvature=curvature, warp=warp
            )
            t1 = time.perf_counter()
            u = solve_equilibrium(op, nodes, [0.0, top])
            t2 = time.perf_counter()
            row[name], row[f"{name}-resample"] = error_against(
                ref, ref_stencils, domain.material, nodes, u
            )
            row[f"{name}-nodes"] = nodes_seconds
            row[f"{name}-build"] = t1 - t0
            row[f"{name}-solve"] = t2 - t1
        rows.append(row)
    return rows


def fd4_sweep(
    counts: tuple[int, ...], ref: Reference, ref_stencils: Stencils
) -> list[dict[str, float]]:
    """One row per count: the Cartesian FD4 error and its grid, solve and read times."""
    domain = case2()
    rows = []
    for n in counts:
        t0 = time.perf_counter()
        grid = cartesian_grid(domain, n)
        op = fd4_operator(grid, domain.material)
        t1 = time.perf_counter()
        u = solve_equilibrium(op, grid, [0.0, top])
        t2 = time.perf_counter()
        err, seconds = error_against(ref, ref_stencils, domain.material, grid, u)
        rows.append(
            {
                "n": grid.n,
                "h": grid.h,
                "fd4": err,
                "fd4-build": t1 - t0,
                "fd4-solve": t2 - t1,
                "fd4-resample": seconds,
            }
        )
    return rows


def resampling_check(
    fine_n: int, counts: tuple[int, ...], seed: int = 0, iterations: int = 100
) -> list[dict[str, float]]:
    """Case 1: its analytic solution on a fine set, read back at coarse nodes and grids.

    ``aware`` uses the fine set's stencils with the interface group, ``blind``
    the same set without it; RMS and largest error against the analytic values.
    """
    domain, exact = case1(), case1_exact()
    t0 = time.perf_counter()
    fine = build_node_set(domain, fine_n, seed=seed, iterations=iterations)
    u = exact(fine.x, fine.y)
    stencils = {
        "aware": build_stencils(fine, domain, interface=BOUNDARY),
        "blind": build_stencils(fine, domain),
    }
    setup = time.perf_counter() - t0
    rows = []
    for n in counts:
        coarse = build_node_set(domain, n, seed=seed, iterations=iterations)
        row: dict[str, float] = {
            "n": coarse.n,
            "h": coarse.h,
            "fine-n": fine.n,
            "fine-h": fine.h,
            "setup": setup,
        }
        for label, pts in (("nodes", coarse), ("grid", cartesian_grid(domain, n))):
            ref = exact(pts.x, pts.y)
            for kind, st in stencils.items():
                t0 = time.perf_counter()
                got = resample(u, fine, st, domain.material, pts.x, pts.y)
                row[f"{label}-{kind}"] = rms_error(got, ref)
                row[f"{label}-{kind}-max"] = float(np.abs(got - ref).max())
                row[f"{label}-{kind}-seconds"] = time.perf_counter() - t0
        rows.append(row)
    return rows


# --- tables ----------------------------------------------------------------------


def rates(rows: list[dict[str, float]], name: str) -> list[float]:
    """Order per halving of ``h`` between consecutive rows (``nan`` for the first)."""
    out = [float("nan")]
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        out.append(np.log(a[name] / b[name]) / np.log(a["h"] / b["h"]))
    return out


def _order(value: float) -> str:
    return "    -" if np.isnan(value) else f"{value:5.2f}"


def _marker(table: dict[int, float], n: int) -> str:
    value = table.get(n)
    return f"{value:8.1e}" if value else "       -"


def _ratios(rows: list[dict[str, float]], name: str, table: dict[int, float]) -> str:
    found = [
        f"{r[name] / table[int(r['n'])]:.1f}" for r in rows if int(r["n"]) in table
    ]
    return ", ".join(found) if found else "-"


def total_seconds(row: dict[str, float], name: str) -> float:
    """Node set (or grid), operator and solve; the resampling is not the method's."""
    keys = ("nodes", "build", "solve") if name != "fd4" else ("build", "solve")
    return float(sum(row[f"{name}-{k}"] for k in keys))


def print_reference(ref: Reference, reused: bool, rows: list[dict[str, float]]) -> None:
    s = ref.meta["seconds"]
    how = "cached" if reused else "solved now"
    print(
        f"reference: {ref.nodes.n} nodes, h = {ref.nodes.h:.5f}, interface group"
        f" {ref.meta['interface_group']}, node set {s['nodes']:.1f} s, operator"
        f" {s['operator']:.1f} s, solve {s['solve']:.1f} s ({how});"
        f" max |u| = {np.abs(ref.u).max():.9f}"
    )
    estimate, n = reference_error_estimate(rows, ref.nodes.n)
    finest = next(r["curved"] for r in rows if r["n"] == n)
    print(
        f"its own error, from the {n}-node curved point at fourth order:"
        f" about {estimate:.1e} (that point's measured error is {finest:.2e})"
    )


def print_convergence(
    rows: list[dict[str, float]], fd4_rows: list[dict[str, float]]
) -> None:
    print(
        "\ncase 2, RBF-FD against the reference (RMS error at the coarse nodes,"
        " order per halving of h; 'group' is the interface group with rows /"
        " without; times are the curved operator's)"
    )
    print(
        "     n       h  group(rows/none) |       flat  order |     curved  order"
        "  nodes  build  solve  read |       none  order |  Fig. 10 flat   curved"
        "  Fig. 11 none"
    )
    orders = {name: rates(rows, name) for name, _, _, _ in VARIANTS}
    for i, r in enumerate(rows):
        n = int(r["n"])
        groups = f"{int(r['group-True']):5d} / {int(r['group-False']):5d}"
        times = "  ".join(
            f"{r[f'curved-{k}']:4.1f}s" for k in ("nodes", "build", "solve", "resample")
        )
        line = f"{n:6d}  {r['h']:.4f}   {groups} |"
        line += f"  {r['flat']:9.2e}  {_order(orders['flat'][i])} |"
        line += f"  {r['curved']:9.2e}  {_order(orders['curved'][i])}  {times} |"
        line += f"  {r['none']:9.2e}  {_order(orders['none'][i])} |"
        line += f"  {_marker(FIG10['flat'], n)} {_marker(FIG10['curved'], n)}"
        line += f"  {_marker(FIG11_NONE, n)}"
        print(line)
    print(
        f"ratio to 2016: flat {_ratios(rows, 'flat', FIG10['flat'])};"
        f" curved {_ratios(rows, 'curved', FIG10['curved'])};"
        f" none {_ratios(rows, 'none', FIG11_NONE)}"
    )
    print("\ncase 2, Cartesian FD4 (Dx A Dx + Dy A Dy on m (m + 1) grid points)")
    print("     n       h |        fd4  order  build  solve  read |  Fig. 10 FD4")
    fd4_orders = rates(fd4_rows, "fd4")
    ratios = []
    for i, r in enumerate(fd4_rows):
        n = int(r["n"])
        marker = nearest_marker(n)
        times = "  ".join(
            f"{r[f'fd4-{k}']:4.1f}s" for k in ("build", "solve", "resample")
        )
        print(
            f"{n:6d}  {r['h']:.4f} |  {r['fd4']:9.2e}  {_order(fd4_orders[i])}"
            f"  {times} |  {_marker(FIG10['fd4'], marker)} (at {marker})"
        )
        ratios.append(f"{r['fd4'] / FIG10['fd4'][marker]:.1f}")
    print(f"ratio to 2016 (FD4 at the nearest marker count): {', '.join(ratios)}")


def nearest_marker(n: int) -> int:
    """The Fig. 10 count nearest an FD4 grid's ``m (m + 1)`` (1260 for 1250)."""
    return min(FIG10["fd4"], key=lambda k: abs(np.log(k / n)))


def print_performance(
    rows: list[dict[str, float]], fd4_rows: list[dict[str, float]]
) -> None:
    print("\nwall-clock (s) on this machine: node set + stencils, operator, solve")
    print("  RBF-FD curved:      n   nodes  build  solve  total |  error")
    for r in rows:
        stages = "  ".join(
            f"{r[f'curved-{k}']:5.1f}" for k in ("nodes", "build", "solve")
        )
        total = total_seconds(r, "curved")
        print(
            f"                 {int(r['n']):6d}  {stages}  {total:5.1f} |"
            f"  {r['curved']:.2e}"
        )
    print("  FD4:                n   grid+op  solve  total |  error")
    for r in fd4_rows:
        stages = f"{r['fd4-build']:7.2f}  {r['fd4-solve']:5.1f}"
        total = total_seconds(r, "fd4")
        print(
            f"                 {int(r['n']):6d}  {stages}  {total:5.1f} |"
            f"  {r['fd4']:.2e}"
        )


def print_check(rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    r0 = rows[0]
    print(
        f"\nresampling check on case 1: the analytic solution on {int(r0['fine-n'])}"
        f" nodes (h = {r0['fine-h']:.5f}; {r0['setup']:.1f} s to build with both"
        " stencil sets) read at coarse nodes and FD4 grid points; RMS / max error"
    )
    print(
        "     n       h |  nodes aware        max   time |  nodes blind        max |"
        "   grid aware        max |   grid blind        max"
    )
    for r in rows:
        cells = [f"{int(r['n']):6d}  {r['h']:.4f} |"]
        for label in ("nodes-aware", "nodes-blind", "grid-aware", "grid-blind"):
            cells.append(f"  {r[label]:11.2e}  {r[label + '-max']:9.2e}")
            if label == "nodes-aware":
                cells.append(f"  {r[label + '-seconds']:4.1f}s |")
            elif label != "grid-blind":
                cells.append(" |")
        print("".join(cells))


# --- figures ---------------------------------------------------------------------


def _markers(ax, table: dict[int, float], marker: str, label: str) -> None:
    items = sorted(table.items())
    ax.loglog(
        [k for k, _ in items],
        [v for _, v in items],
        marker + ":",
        color="k",
        markerfacecolor="none",
        markersize=6,
        linewidth=0.7,
        label=label,
    )


def figure_convergence(rows: list[dict[str, float]], fd4_rows: list[dict[str, float]]):
    n = np.array([r["n"] for r in rows], dtype=float)
    n_fd4 = np.array([r["n"] for r in fd4_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.loglog(
        n_fd4,
        [r["fd4"] for r in fd4_rows],
        "s-",
        color=NAIVE,
        markersize=5,
        label="FD4, Dx A Dx + Dy A Dy",
    )
    ax.loglog(
        n,
        [r["flat"] for r in rows],
        "o-",
        color=REFERENCE,
        markersize=5,
        label="RBF-FD, flat interface",
    )
    ax.loglog(
        n,
        [r["curved"] for r in rows],
        "^-",
        color=AWARE,
        markersize=5,
        label="RBF-FD, curvature included",
    )
    _markers(ax, FIG10["fd4"], "s", "EABE Fig. 10 FD4 (read off)")
    _markers(ax, FIG10["flat"], "o", "Fig. 10 flat int.")
    _markers(ax, FIG10["curved"], "^", "Fig. 10 curv. incl.")
    e1, e4 = fd4_rows[0]["fd4"], rows[0]["curved"]
    ax.loglog(
        n_fd4, e1 * (n_fd4 / n_fd4[0]) ** -0.5, "k-.", linewidth=0.7, label="1st order"
    )
    ax.loglog(n, e4 * (n / n[0]) ** -2.0, "k--", linewidth=0.7, label="4th order")
    ticks = sorted({*n.astype(int), *FIG10["curved"]})
    ax.set_xticks(ticks, [str(k) for k in ticks], fontsize=7, rotation=45)
    ax.set_xticks([], minor=True)
    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("RMS error in u against the fine reference")
    ax.set_title("case 2: FD4, flat and curved RBF-FD (EABE Fig. 10 twin)")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


def figure_ablation(rows: list[dict[str, float]]):
    n = np.array([r["n"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.loglog(
        n,
        [r["none"] for r in rows],
        "o--",
        color=AWARE,
        markersize=5,
        markerfacecolor="none",
        label="no warp, no straddling rows",
    )
    ax.loglog(
        n,
        [r["curved"] for r in rows],
        "^-",
        color=AWARE,
        markersize=5,
        label="warped RBFs, straddling rows",
    )
    _markers(ax, FIG11_NONE, "o", "EABE Fig. 11 no warp; no straddling (read off)")
    _markers(ax, FIG10["curved"], "^", "Fig. 11 with warp and straddling")
    e4 = rows[0]["curved"]
    ax.loglog(n, e4 * (n / n[0]) ** -2.0, "k--", linewidth=0.7, label="4th order")
    ticks = sorted({*n.astype(int), *FIG11_NONE})
    ax.set_xticks(ticks, [str(k) for k in ticks], fontsize=7, rotation=45)
    ax.set_xticks([], minor=True)
    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("RMS error in u against the fine reference")
    ax.set_title("case 2: the warp-and-straddle ablation (EABE Fig. 11 twin)")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


def figure_performance(rows: list[dict[str, float]], fd4_rows: list[dict[str, float]]):
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.loglog(
        [total_seconds(r, "fd4") for r in fd4_rows],
        [r["fd4"] for r in fd4_rows],
        "s-",
        color=NAIVE,
        markersize=5,
        label="FD4 (this machine)",
    )
    ax.loglog(
        [total_seconds(r, "curved") for r in rows],
        [r["curved"] for r in rows],
        "^-",
        color=AWARE,
        markersize=5,
        label="RBF-FD, curvature included (this machine)",
    )
    for name, marker, label in (("fd4", "s", "FD4"), ("rbf", "^", "RBF only")):
        pts = FIG511[name]
        ax.loglog(
            [t for t, _ in pts],
            [e for _, e in pts],
            marker + ":",
            color="k",
            markerfacecolor="none",
            markersize=6,
            linewidth=0.7,
            label=f"dissertation Fig. 5-11 {label} (2016, MATLAB, Core i7)",
        )
    ax.set_xlabel("node set + operator + solve (s)")
    ax.set_ylabel("RMS error in u against the fine reference")
    ax.set_title("case 2: error against wall-clock (Fig. 5-11 twin)")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


# --- main ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--counts", type=int, nargs="+", default=[1250, 2500, 5000, 10000]
    )
    parser.add_argument("--fd4-counts", type=int, nargs="+", default=None)
    parser.add_argument("--reference-n", type=int, default=40000)
    parser.add_argument("--check-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    fd4_counts = args.counts if args.fd4_counts is None else args.fd4_counts
    check_n = args.reference_n if args.check_n is None else args.check_n
    if len(args.counts) < 2 or any(n < 300 for n in args.counts):
        parser.error("give at least two counts of 300 nodes or more")
    if any(n < 300 for n in fd4_counts):
        parser.error("the FD4 counts need 300 nodes or more")
    if args.reference_n <= max(args.counts):
        parser.error("the reference must have more nodes than every RBF-FD count")
    args.outputs.mkdir(parents=True, exist_ok=True)

    domain = case2()
    t0 = time.perf_counter()
    ref, reused = reference(
        domain, args.reference_n, args.seed, args.iterations, args.outputs
    )
    ref_stencils = ref.stencils(domain)
    print(f"({time.perf_counter() - t0:.1f} s for the reference)")

    t0 = time.perf_counter()
    rows = sweep(tuple(args.counts), ref, ref_stencils, args.seed, args.iterations)
    fd4_rows = fd4_sweep(tuple(fd4_counts), ref, ref_stencils)
    print_reference(ref, reused, rows)
    print_convergence(rows, fd4_rows)
    print_performance(rows, fd4_rows)
    print(f"({time.perf_counter() - t0:.1f} s for the sweeps)")
    for name, fig in (
        ("convergence", figure_convergence(rows, fd4_rows)),
        ("ablation", figure_ablation(rows)),
        ("performance", figure_performance(rows, fd4_rows)),
    ):
        fig.savefig(args.outputs / f"heat2d_case2_{name}.png", dpi=150)
        plt.close(fig)

    if check_n:
        t0 = time.perf_counter()
        print_check(
            resampling_check(check_n, tuple(args.counts), args.seed, args.iterations)
        )
        print(f"({time.perf_counter() - t0:.1f} s for the resampling check)")


if __name__ == "__main__":
    main()
