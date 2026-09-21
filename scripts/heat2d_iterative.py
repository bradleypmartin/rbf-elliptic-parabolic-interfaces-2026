"""E2.8 (#22): iterative solvers on the case-3 control problem and on case 3 itself
(dissertation §5.4.4 and Appendix B, EABE §3.3.2): ``gmres`` and ``bicgstab``
against SuperLU, unpreconditioned, with Appendix B's diagonal-dominance
preconditioner, and with ``spilu``; Fig. 5-16 / 5-17 / 5-18 twins with our
timings, Fig. B-1's DDR histograms, and the tables behind them.

Stencils are the section's 19 nodes / degree 3 everywhere (``rbf.ITERATIVE``),
curvature and warped Gaussians on where they cross the ring. Both problems use
case 3's node layout (the straddling rows on the ring's midline, the cooling
disc cut out); the control has ``α ≡ 1`` and no interface group at all, the
plain RBF-FD Laplacian, "no interfaces are actually present". Every solve is
of the reduced interior system (``heat2d.solve.reduced_system``): the
identity-row form breaks BiCGSTAB at its first step on either problem (the
``solve`` module docstring), which is not the effect the section is about.

Errors are RMS against a fine reference read at the nodes through its own
stencils (``heat2d.resample``): case 3's cached
``outputs/heat2d_case3_reference_n<N>_seed<s>.npz`` from E2.7 and the
control's twin ``heat2d_case3_control_reference_…``, solved on the first run.
The tolerance is ``|r| <= rtol |b|`` with ``rtol = 1e-8`` (a flag), three
orders below the discretisation error's relative size; the tables give each
iterative solution's distance from the direct one so that choice is visible.

``heat2d_iterative_performance.png``: error against time to solution, one
panel each for the control (Fig. 5-16), case 3 (Fig. 5-17), case 3 with
Appendix B's ``P A u = P f`` (Fig. 5-18) and case 3 with ``spilu`` (its
``MMD_AT_PLUS_A`` ordering; ``--ilu-ordering COLAMD`` shows SuperLU's default
failing from 20,000 nodes); filled
markers include the preconditioner's build, hollow ones are the solver alone;
the 2016 markers are read off the rendered pages (the dissertation gives no
node counts for them; five per line, taken to be 1250 to 20,000).
``heat2d_iterative_iterations.png``: inner iterations against the node count
for every method and preconditioner (SciPy's restarted GMRES(20) added).
``heat2d_iterative_ddr.png``: Fig. B-1's histograms, the DDR of the rows whose
stencils cross the ring and of every interior row, before and after the three
sweeps, at ``--ddr-n`` nodes.

    uv run python scripts/heat2d_iterative.py                        # ~1.5 min
    uv run python scripts/heat2d_iterative.py --reference-n 160000 \\
        --counts 1250 2500 5000 10000 20000                          # ~5 min
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
    ILU_ORDERING,
    INTERIOR,
    ITERATIVE,
    NEIGHBOURS,
    RING,
    SWEEPS,
    Band,
    Circle,
    Constant2D,
    Domain,
    NodeSet,
    ReducedSystem,
    Reference,
    Stencils,
    build_node_set,
    build_stencils,
    case3,
    diagonal_dominance_ratio,
    dominance_preconditioner,
    ilu_preconditioner,
    interface_aware_operator,
    laplacian_operator,
    neighbour_table,
    reduced_system,
    reference_solution,
    resample,
    rms_error,
    solve_iterative,
)
from heat_interfaces.plotting import AWARE, NAIVE  # noqa: E402

FIG2016 = {
    "control": {
        "direct": [
            (0.04, 1.3e-3),
            (0.06, 6.0e-4),
            (0.15, 2.0e-4),
            (0.35, 6.5e-5),
            (0.7, 1.4e-5),
        ],
        "gmres": [
            (0.012, 1.3e-3),
            (0.07, 6.0e-4),
            (0.2, 2.0e-4),
            (0.7, 6.5e-5),
            (3.5, 1.6e-5),
        ],
        "bicgstab": [
            (0.02, 1.4e-3),
            (0.04, 6.0e-4),
            (0.1, 2.0e-4),
            (0.35, 6.5e-5),
            (1.2, 1.6e-5),
        ],
    },
    "case3-none": {
        "direct": [
            (0.04, 1.0e-3),
            (0.06, 4.0e-4),
            (0.15, 1.3e-4),
            (0.35, 5.0e-5),
            (0.8, 1.4e-5),
        ],
        "gmres": [
            (0.9, 1.0e-3),
            (3.0, 4.0e-4),
            (15.0, 1.3e-4),
            (35.0, 5.0e-5),
            (250.0, 1.6e-5),
        ],
    },
    "case3-appendix-b": {
        "direct": [(0.09, 4.0e-4), (0.2, 1.3e-4), (0.35, 5.0e-5), (0.9, 1.4e-5)],
        "gmres": [(0.08, 4.0e-4), (0.2, 1.4e-4), (0.7, 5.0e-5), (4.5, 1.6e-5)],
        "bicgstab": [(0.05, 4.0e-4), (0.2, 1.4e-4), (0.45, 5.0e-5), (1.6, 1.4e-5)],
    },
}
"""Dissertation Fig. 5-16, 5-17 and 5-18 (= EABE Fig. 16–18): ``(seconds, error)``.

Read off the rendered pages on 2026-09-21, about ±15 % in both coordinates.
Five markers per line (Fig. 5-18 shows the last four; the first of each line
is above its axis); bicgstab is absent from Fig. 5-17 because it "completely
failed to work". The counts are not stated; the error levels match a 1250 …
20,000 sequence, which the tables below check.
"""

METHODS = (
    ("gmres", "gmres", None),
    ("gmres(20)", "gmres", 20),
    ("bicgstab", "bicgstab", None),
)
"""``(label, SciPy method, restart)``: full GMRES (MATLAB's default), GMRES(20),
BiCGSTAB.

The restarted variant is SciPy's own default restart length.
"""

PRECONDITIONERS = ("none", "appendix-b", "spilu")
PROBLEMS = ("control", "case3")
STYLE = {"gmres": ("^", AWARE), "gmres(20)": ("v", AWARE), "bicgstab": ("s", NAIVE)}
DASH = {"none": "-", "appendix-b": "--", "spilu": ":"}
DIRECT_STYLE = ("o", "k")


def edge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Eq. 39's data on both rows, ``sin 6πx`` at ``y = 0`` and ``y = 1``."""
    return np.sin(6 * np.pi * x)


VALUES = [edge, edge, 0.0]


def control_domain() -> Domain:
    """Case 3's geometry with ``α ≡ 1``: the control problem of §5.4.4."""
    band = Band(Circle(RING[0]), Circle(RING[1]), Constant2D(1.0), Constant2D(1.0))
    return replace(case3(), material=band)


DOMAINS = {"control": control_domain(), "case3": case3()}


# --- references ------------------------------------------------------------------


def reference_path(outputs: Path, problem: str, n: int, seed: int) -> Path:
    tag = "case3" if problem == "case3" else "case3_control"
    return outputs / f"heat2d_{tag}_reference_n{n}_seed{seed}.npz"


def reference(
    problem: str, n: int, seed: int, iterations: int, outputs: Path
) -> tuple[Reference, bool]:
    """The cached fine solution (E2.7's file for case 3), solved and saved if absent."""
    path = reference_path(outputs, problem, n, seed)
    if path.exists():
        ref = Reference.load(path)
        if ref.meta.get("iterations") == iterations and ref.meta.get("n") == n:
            return ref, True
    ref = reference_solution(
        DOMAINS[problem], n, VALUES, seed=seed, iterations=iterations
    )
    ref.save(path)
    return ref, False


# --- operators -------------------------------------------------------------------


def operator(problem: str, nodes: NodeSet, warp: bool = True):
    """``(operator, stencils)`` on 19-node / degree-3 stencils.

    The control is the plain Laplacian with no interface group; case 3 the
    §5.3 operator with curvature and, by default, the warped Gaussians.
    """
    domain = DOMAINS[problem]
    if problem == "control":
        stencils = build_stencils(nodes, domain, interior=ITERATIVE, boundary=ITERATIVE)
        return laplacian_operator(nodes, stencils), stencils
    stencils = build_stencils(
        nodes, domain, interior=ITERATIVE, boundary=ITERATIVE, interface=ITERATIVE
    )
    return interface_aware_operator(
        nodes, domain.material, stencils, warp=warp
    ), stencils


def ddr_summary(ddr: np.ndarray, group: np.ndarray) -> dict[str, float]:
    """Min, median and the fraction below 1 over all rows and over ``group``."""
    out = {
        "min": ddr.min(),
        "median": float(np.median(ddr)),
        "below1": np.mean(ddr < 1),
    }
    if group.any():
        out.update(
            {
                "group-min": ddr[group].min(),
                "group-median": float(np.median(ddr[group])),
                "group-below1": np.mean(ddr[group] < 1),
            }
        )
    else:
        out.update(
            {"group-min": np.nan, "group-median": np.nan, "group-below1": np.nan}
        )
    return out


# --- the sweep -------------------------------------------------------------------


def sweep(
    problem: str,
    counts: tuple[int, ...],
    ref: Reference,
    ref_stencils: Stencils,
    args: argparse.Namespace,
) -> list[dict]:
    """One row per count with the direct solve, the DDR before and after, and every
    ``(method, preconditioner)`` solve's iterations, time, error and distance."""
    material = DOMAINS[problem].material
    rows = []
    for n in counts:
        t0 = time.perf_counter()
        nodes = build_node_set(case3(), n, seed=args.seed, iterations=args.iterations)
        op, stencils = operator(problem, nodes, warp=not args.no_warp)
        system = reduced_system(op, nodes, VALUES)
        build = time.perf_counter() - t0
        u_ref = resample(ref.u, ref.nodes, ref_stencils, material, nodes.x, nodes.y)
        t0 = time.perf_counter()
        u_direct = system.solve_direct()
        direct = time.perf_counter() - t0
        group = stencils.near_interface[system.interior]
        row: dict = {
            "n": nodes.n,
            "h": nodes.h,
            "interior": system.n,
            "group": int(group.sum()),
            "build": build,
            "nnz": system.a.nnz / system.n,
            "direct": direct,
            "direct-error": rms_error(u_direct, u_ref),
            "ddr": ddr_summary(diagonal_dominance_ratio(system.a), group),
        }
        t0 = time.perf_counter()
        table = neighbour_table(nodes, system.interior, args.neighbours)
        p = dominance_preconditioner(system.a, table, sweeps=args.sweeps)
        pre = system.left_preconditioned(p)
        row["appendix-b-build"] = time.perf_counter() - t0
        row["appendix-b-nnz"] = pre.a.nnz / pre.n
        row["ddr-after"] = ddr_summary(diagonal_dominance_ratio(pre.a), group)
        m, row["spilu-build"] = ilu_preconditioner(system, permc_spec=args.ilu_ordering)
        systems: dict[str, tuple[ReducedSystem, object]] = {
            "none": (system, None),
            "appendix-b": (pre, None),
            "spilu": (system, m),
        }
        for label, method, restart in METHODS:
            for pc, (sys_, m_) in systems.items():
                res = solve_iterative(
                    sys_,
                    method,
                    rtol=args.rtol,
                    maxiter=args.maxiter,
                    restart=restart,
                    m=m_,
                )
                row[(label, pc)] = {
                    "iterations": res.iterations,
                    "seconds": res.seconds,
                    "info": res.info,
                    "residual": res.residual,
                    "error": rms_error(res.u, u_ref),
                    "distance": float(
                        np.linalg.norm(res.u - u_direct) / np.linalg.norm(u_direct)
                    ),
                }
        rows.append(row)
    return rows


def ddr_variants(n: int, seed: int, iterations: int) -> list[tuple[str, dict]]:
    """The DDR of other operators on the same node set, for the record.

    Case 3 with plain Gaussians (the E2.7 note on #22 asked), and E2.7's own
    42 / 5 and 30 / 4 stencils on case 3 and on the control.
    """
    nodes = build_node_set(case3(), n, seed=seed, iterations=iterations)
    out = []
    for label, problem, specs, warp in (
        ("case 3, 19 / 3, warp on (this study)", "case3", (ITERATIVE, ITERATIVE), True),
        ("case 3, 19 / 3, plain Gaussians", "case3", (ITERATIVE, ITERATIVE), False),
        ("case 3, 42 / 5 and 30 / 4 (E2.7)", "case3", (INTERIOR, BOUNDARY), True),
        ("control, 19 / 3", "control", (ITERATIVE, ITERATIVE), True),
        ("control, 42 / 5 and 30 / 4", "control", (INTERIOR, BOUNDARY), True),
    ):
        domain = DOMAINS[problem]
        interior, boundary = specs
        if problem == "control":
            stencils = build_stencils(
                nodes, domain, interior=interior, boundary=boundary
            )
            op = laplacian_operator(nodes, stencils)
        else:
            stencils = build_stencils(
                nodes, domain, interior=interior, boundary=boundary, interface=boundary
            )
            op = interface_aware_operator(nodes, domain.material, stencils, warp=warp)
        system = reduced_system(op, nodes, VALUES)
        group = stencils.near_interface[system.interior]
        out.append((label, ddr_summary(diagonal_dominance_ratio(system.a), group)))
    return out


def ddr_histograms(
    n: int, seed: int, iterations: int, neighbours: int, sweeps: int
) -> dict[str, np.ndarray]:
    """DDR of case 3's rows at ``n`` nodes before and after the sweeps (Fig. B-1)."""
    nodes = build_node_set(case3(), n, seed=seed, iterations=iterations)
    op, stencils = operator("case3", nodes)
    system = reduced_system(op, nodes, VALUES)
    group = stencils.near_interface[system.interior]
    before = diagonal_dominance_ratio(system.a)
    table = neighbour_table(nodes, system.interior, neighbours)
    p = dominance_preconditioner(system.a, table, sweeps=sweeps)
    after = diagonal_dominance_ratio(system.left_preconditioned(p).a)
    return {"before": before, "after": after, "group": group, "n": nodes.n}


# --- tables ----------------------------------------------------------------------


def _flag(cell: dict) -> str:
    return " " if cell["info"] == 0 else ("!" if cell["info"] > 0 else "x")


def print_setup(problem: str, rows: list[dict], ordering: str) -> None:
    print(
        f"\n{problem}: 19-node / degree-3 stencils, the reduced interior system;"
        " 'group' the rows whose stencils cross the ring; DDR min / median /"
        " fraction below 1 over all interior rows and over the group, before and"
        " after Appendix B's sweeps (P's build time and the nonzeros per row"
        f" of P A); spilu's build time ({ordering} ordering)"
    )
    print(
        "     n  interior  group  build  nnz/row  direct  direct err |"
        "  DDR all: min  med  <1 | group: min  med  <1 |"
        "  P build  nnz/row | after all: min  med  <1 | group: min  med  <1 |  ilu"
    )
    for r in rows:
        d, a = r["ddr"], r["ddr-after"]

        def trio(s: dict, key: str) -> str:
            mn, md, b1 = s[key + "min"], s[key + "median"], s[key + "below1"]
            return f"{mn:5.3f} {md:5.3f} {b1:4.2f}"

        print(
            f"{int(r['n']):6d}  {int(r['interior']):8d}  {int(r['group']):5d}"
            f"  {r['build']:4.1f}s  {r['nnz']:7.1f}  {r['direct']:5.2f}s"
            f"  {r['direct-error']:10.2e} |  {trio(d, '')} | {trio(d, 'group-')} |"
            f"  {r['appendix-b-build']:5.2f}s  {r['appendix-b-nnz']:7.1f} |"
            f"  {trio(a, '')} | {trio(a, 'group-')} |  {r['spilu-build']:4.2f}s"
        )


def print_solves(problem: str, rows: list[dict]) -> None:
    print(
        f"\n{problem}: iterations and solver seconds per method (full GMRES,"
        " GMRES(20), BiCGSTAB) and preconditioner (none, Appendix B, spilu);"
        " '!' did not converge within the cap, 'x' broke down; 'err' is the RMS"
        " error against the reference and 'dist' the relative distance from the"
        " direct solution; 'residual' is the unweighted |b - A u| / |b|"
    )
    header = "     n |"
    for label, _, _ in METHODS:
        for pc in PRECONDITIONERS:
            header += f" {label + ' ' + pc:>20s} |"
    print(header)
    for r in rows:
        line = f"{int(r['n']):6d} |"
        for label, _, _ in METHODS:
            for pc in PRECONDITIONERS:
                c = r[(label, pc)]
                line += f" {c['iterations']:5d}{_flag(c)} {c['seconds']:6.2f}s"
                line += f" {c['distance']:6.0e} |"
        print(line)
    print(
        "largest unweighted residual |b - A u| / |b| over the three methods, per"
        " preconditioner (Appendix B's solver tests |P b - P A u| <= rtol |P b|):"
    )
    for r in rows:
        worst = {
            pc: max(r[(label, pc)]["residual"] for label, _, _ in METHODS)
            for pc in PRECONDITIONERS
        }
        print(
            f"{int(r['n']):6d}  "
            + "  ".join(f"{pc} {v:.1e}" for pc, v in worst.items())
        )
    print("errors against the reference (direct first, then each solve):")
    for r in rows:
        errs = [f"{r['direct-error']:.2e}"]
        for label, _, _ in METHODS:
            for pc in PRECONDITIONERS:
                errs.append(f"{r[(label, pc)]['error']:.2e}")
        print(f"{int(r['n']):6d}  " + "  ".join(errs))
    none = [r[("gmres", "none")]["iterations"] for r in rows]
    appb = [r[("gmres", "appendix-b")]["iterations"] for r in rows]
    bi_none = [r[("bicgstab", "none")]["iterations"] for r in rows]
    bi_appb = [r[("bicgstab", "appendix-b")]["iterations"] for r in rows]
    print(
        "iteration ratios none / Appendix B: gmres "
        + ", ".join(f"{a / b:.2f}" for a, b in zip(none, appb, strict=True))
        + "; bicgstab "
        + ", ".join(f"{a / b:.2f}" for a, b in zip(bi_none, bi_appb, strict=True))
    )


def print_comparison(results: dict[str, list[dict]]) -> None:
    c, s = results["control"], results["case3"]
    print(
        "\ncase 3 / control, unpreconditioned: iterations and seconds, gmres then"
        " bicgstab; and the 2016 gmres time ratio read off Fig. 5-17 / 5-16"
    )
    ratio2016 = [
        t3 / tc
        for (t3, _), (tc, _) in zip(
            FIG2016["case3-none"]["gmres"], FIG2016["control"]["gmres"], strict=True
        )
    ]
    for i, (rc, rs) in enumerate(zip(c, s, strict=True)):
        g3, gc = rs[("gmres", "none")], rc[("gmres", "none")]
        b3, bc = rs[("bicgstab", "none")], rc[("bicgstab", "none")]
        marker = f"{ratio2016[i]:.0f}" if i < len(ratio2016) else "-"
        print(
            f"{int(rs['n']):6d}  gmres {g3['iterations'] / gc['iterations']:.2f}x its"
            f"  {g3['seconds'] / gc['seconds']:.2f}x s |  bicgstab"
            f"  {b3['iterations'] / bc['iterations']:.2f}x its"
            f"  {b3['seconds'] / bc['seconds']:.2f}x s |  2016 gmres time ratio"
            f" {marker}x"
        )


def print_variants(variants: list[tuple[str, dict]], n: int) -> None:
    print(
        f"\nDDR of other operators on the same {n}-node set (min / median / <1; group)"
    )
    for label, s in variants:
        grp = (
            f"{s['group-min']:5.3f} {s['group-median']:5.3f} {s['group-below1']:4.2f}"
            if np.isfinite(s["group-min"])
            else "no group"
        )
        print(
            f"  {label:38s} {s['min']:5.3f} {s['median']:5.3f} {s['below1']:4.2f}"
            f" | {grp}"
        )


def print_histograms(h: dict[str, np.ndarray]) -> None:
    g = h["group"]
    print(
        f"\nFig. B-1 twin at {int(h['n'])} nodes: DDR of the {int(g.sum())} rows"
        f" crossing the ring before {np.median(h['before'][g]):.3f}"
        f" (min {h['before'][g].min():.3f})"
        f" and after {np.median(h['after'][g]):.3f} (min {h['after'][g].min():.3f});"
        f" all {len(g)} interior rows before {np.median(h['before']):.3f} and after"
        f" {np.median(h['after']):.3f}, fraction below 1 {np.mean(h['before'] < 1):.2f}"
        f" → {np.mean(h['after'] < 1):.2f}"
    )


# --- figures ---------------------------------------------------------------------


def _panel(ax, rows: list[dict], pc: str, readings: dict | None, title: str) -> None:
    ax.loglog(
        [r["direct"] for r in rows],
        [r["direct-error"] for r in rows],
        DIRECT_STYLE[0] + "--",
        color=DIRECT_STYLE[1],
        markersize=5,
        label="SuperLU (direct)",
    )
    for label, _, _ in METHODS:
        if label == "gmres(20)":
            continue
        marker, colour = STYLE[label]
        build = 0.0 if pc == "none" else None
        seconds = np.array([r[(label, pc)]["seconds"] for r in rows])
        errors = [r[(label, pc)]["error"] for r in rows]
        ok = np.array([r[(label, pc)]["info"] == 0 for r in rows])
        if build is None:
            total = seconds + np.array([r[f"{pc}-build"] for r in rows])
            ax.loglog(
                total, errors, marker + "-", color=colour, markersize=5, label=label
            )
            ax.loglog(
                seconds,
                errors,
                marker,
                color=colour,
                markersize=5,
                markerfacecolor="none",
                label=f"{label}, solver alone",
            )
        else:
            ax.loglog(
                seconds, errors, marker + "-", color=colour, markersize=5, label=label
            )
        for t, e, good in zip(seconds, errors, ok, strict=True):
            if not good:
                ax.annotate(
                    "×", (t, e), fontsize=9, color=colour, ha="center", va="center"
                )
    if readings:
        for name, pts in readings.items():
            marker = DIRECT_STYLE[0] if name == "direct" else STYLE[name][0]
            ax.loglog(
                [t for t, _ in pts],
                [e for _, e in pts],
                marker + ":",
                color="k",
                markerfacecolor="none",
                markersize=6,
                linewidth=0.7,
                label=f"2016 {'backslash' if name == 'direct' else name} (read off)",
            )
    for r in rows:
        ax.annotate(
            str(int(r["n"])),
            (r["direct"], r["direct-error"]),
            fontsize=6,
            xytext=(-4, -8),
            textcoords="offset points",
            ha="right",
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time to solution (s)")
    ax.set_ylabel("RMS error against the fine reference")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=6, loc="lower left")


def figure_performance(results: dict[str, list[dict]], ordering: str):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    _panel(
        axes[0, 0],
        results["control"],
        "none",
        FIG2016["control"],
        "control problem, unpreconditioned (Fig. 5-16 twin)",
    )
    _panel(
        axes[0, 1],
        results["case3"],
        "none",
        FIG2016["case3-none"],
        "case 3, unpreconditioned (Fig. 5-17 twin)",
    )
    _panel(
        axes[1, 0],
        results["case3"],
        "appendix-b",
        FIG2016["case3-appendix-b"],
        "case 3, Appendix B's P A u = P f (Fig. 5-18 twin)",
    )
    _panel(
        axes[1, 1],
        results["case3"],
        "spilu",
        None,
        f"case 3, spilu (drop 1e-4, fill 10, {ordering})",
    )
    xlo = min(ax.get_xlim()[0] for ax in axes.ravel())
    xhi = max(ax.get_xlim()[1] for ax in axes.ravel())
    ylo = min(ax.get_ylim()[0] for ax in axes.ravel())
    yhi = max(ax.get_ylim()[1] for ax in axes.ravel())
    for ax in axes.ravel():
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
    fig.suptitle(
        "case 3 solved iteratively: 19-node / degree-3 stencils (dissertation §5.4.4)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def figure_iterations(results: dict[str, list[dict]]):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharey=True)
    for ax, problem in zip(axes, PROBLEMS, strict=True):
        rows = results[problem]
        n = [int(r["n"]) for r in rows]
        for label, _, _ in METHODS:
            marker, colour = STYLE[label]
            for pc in PRECONDITIONERS:
                its = [r[(label, pc)]["iterations"] for r in rows]
                ax.loglog(
                    n,
                    its,
                    marker + DASH[pc],
                    color=colour,
                    markersize=5,
                    markerfacecolor="none" if label == "gmres(20)" else colour,
                    linewidth=1.0,
                    label=f"{label}, {pc}",
                )
        ax.loglog(
            n,
            np.sqrt(n) * rows[0][("gmres", "none")]["iterations"] / np.sqrt(n[0]),
            "k--",
            linewidth=0.6,
            label="∝ √N",
        )
        ax.set_xticks(n, [str(k) for k in n], fontsize=7)
        ax.set_xticks([], minor=True)
        ax.set_xlabel("number of nodes N")
        ax.set_title(
            "control problem" if problem == "control" else "case 3", fontsize=9
        )
        ax.grid(True, which="both", linewidth=0.3)
    axes[0].set_ylabel("inner iterations to |r| ≤ rtol |b|")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def figure_ddr(h: dict[str, np.ndarray]):
    bins = np.arange(0.0, 1.6 + 0.01, 0.01)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for ax, (title, mask) in zip(
        axes,
        (
            (
                f"rows whose stencils cross the ring ({int(h['group'].sum())})",
                h["group"],
            ),
            (
                f"all interior rows ({len(h['group'])})",
                np.ones(len(h["group"]), dtype=bool),
            ),
        ),
        strict=True,
    ):
        for key, colour, label in (
            ("before", NAIVE, "before"),
            ("after", AWARE, "after three sweeps"),
        ):
            values = np.clip(h[key][mask], 0, bins[-1])
            counts, _ = np.histogram(values, bins)
            ax.step(
                bins[:-1],
                counts / counts.max(),
                where="post",
                color=colour,
                linewidth=1.0,
                label=label,
            )
        ax.axvline(1.0, color="k", linewidth=0.5, linestyle=":")
        ax.set_xlim(0, bins[-1])
        ax.set_xlabel("diagonal dominance ratio (bins of 0.01)")
        ax.set_ylabel("count / fullest bin")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(
        f"case 3 at {int(h['n'])} nodes: DDR before and after Appendix B's"
        " preconditioning (Fig. B-1 twin)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


# --- main ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--counts", type=int, nargs="+", default=[1250, 2500, 5000, 10000]
    )
    parser.add_argument("--reference-n", type=int, default=40000)
    parser.add_argument("--ddr-n", type=int, default=10000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--maxiter", type=int, default=3000)
    parser.add_argument("--sweeps", type=int, default=SWEEPS)
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS)
    parser.add_argument(
        "--no-warp", action="store_true", help="plain Gaussians across the ring"
    )
    parser.add_argument(
        "--ilu-ordering",
        default=ILU_ORDERING,
        help="spilu's permc_spec; COLAMD, SuperLU's default, fails from 20,000 nodes",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    if len(args.counts) < 2 or any(n < 900 for n in args.counts):
        parser.error("give at least two counts of 900 nodes or more")
    if args.reference_n <= max(args.counts):
        parser.error("the reference must have more nodes than every count")
    args.outputs.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {}
    for problem in PROBLEMS:
        t0 = time.perf_counter()
        ref, reused = reference(
            problem, args.reference_n, args.seed, args.iterations, args.outputs
        )
        ref_stencils = ref.stencils(DOMAINS[problem])
        s = ref.meta["seconds"]
        print(
            f"{problem} reference: {ref.nodes.n} nodes, node set {s['nodes']:.1f} s,"
            f" operator {s['operator']:.1f} s, solve {s['solve']:.1f} s"
            f" ({'cached' if reused else 'solved now'};"
            f" {time.perf_counter() - t0:.1f} s)"
        )
        t0 = time.perf_counter()
        results[problem] = sweep(problem, tuple(args.counts), ref, ref_stencils, args)
        print_setup(problem, results[problem], args.ilu_ordering)
        print_solves(problem, results[problem])
        print(f"({time.perf_counter() - t0:.1f} s for the {problem} sweep)")
    print_comparison(results)

    t0 = time.perf_counter()
    print_variants(ddr_variants(args.ddr_n, args.seed, args.iterations), args.ddr_n)
    h = ddr_histograms(
        args.ddr_n, args.seed, args.iterations, args.neighbours, args.sweeps
    )
    print_histograms(h)
    print(f"({time.perf_counter() - t0:.1f} s for the DDR tables)")

    for name, fig in (
        (
            "heat2d_iterative_performance.png",
            figure_performance(results, args.ilu_ordering),
        ),
        ("heat2d_iterative_iterations.png", figure_iterations(results)),
        ("heat2d_iterative_ddr.png", figure_ddr(h)),
    ):
        fig.savefig(args.outputs / name, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
