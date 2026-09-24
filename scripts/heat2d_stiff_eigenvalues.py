"""E4.5 (#36), the seed rows in the global matrix: dominance, solvers, spectra.

The medium is E4.2's ``SmoothBand(case1().material, δ)`` and the problem the
elliptic one of E4.3, ``u = sin 2πx v(y)`` with ``v`` the separable Chebyshev
reference (``case1_reference(δ, 0)``), ``u = 0`` on ``y = 0`` and
``sin 2πx`` on ``y = 1``. Four operators run on it (stiff note §4.4):

* *naive* ``Dx A Dx + Dy A Dy`` with α at the nodes (E4.3's baseline),
  factored with ``PRODUCT_ORDERING``;
* *direct*, ``α ∇² + ∇α · ∇`` on the smooth medium with no interface group:
  the right method once the edge is resolved and H5's ``δ ≫ h`` end;
* *construction*, ``interface_aware_operator`` warped, whose crossing rows
  read the pieces as if δ were 0 (E4.3's second line);
* *seeds*, ``seed_operator`` on the rows that see the edge (``seeded_rows``,
  reach ``20 δ``), Gaussians in ``(ξ, φ₀₁(η))``;
* *seeds-plain*, the same rows with plain Gaussians (H7's ablation, and
  E2.9's warning that on a ring the warp decides the sign of the spectrum).

``--mode rows`` (H5, plan §3.3's solvability question) reports per δ/h and
operator: how many rows the method replaces; the diagonal dominance ratio
(Appendix B eq. 93) least, median and fraction below one over those rows and
over every interior row; SuperLU's factor-and-solve time, its residual, the
solution's RMS error against the reference and a one-norm condition estimate
of the reduced interior matrix; and ``gmres`` / ``bicgstab`` to
``|r| <= rtol |b|``, unpreconditioned, with Appendix B's ``P`` and with
``spilu`` (``MMD_AT_PLUS_A``). Everything is on the *reduced* system
(``heat2d.solve.reduced_system``): the identity-row form breaks BiCGSTAB at
its first step (port notes §2.8).

``--mode spectra`` (H6) is the Fig. 5-6 twin with a seeds panel: the dense
interior spectrum of each operator at ``--spectrum-n`` nodes per δ of the
study, with the count of eigenvalues in the right half-plane and BD4's
largest root modulus at ``dt = h``. The eigenvalues are cached per (n, δ,
seed) in ``outputs/heat2d_stiff_spectra_n<N>_d<δ>_s<seed>.npz``; the row
tables cache in ``outputs/heat2d_stiff_rows.json``.

Figures: ``heat2d_stiff_spectra.png`` (the four spectra at ``--figure-delta``
with BD4's boundary at ``dt = h``) and ``heat2d_stiff_dominance.png`` (the
DDR and the iteration counts against δ/h), ``_n<N>`` before ``.png`` off the
default counts. Every run writes its tables to a results file (E4.10,
``results_cache``): ``heat2d_stiff_eigenvalues.json`` for the default run,
``heat2d_stiff_eigenvalues_rows_n<N>.json`` and ``…_spectra_n<N>.json`` for
the others, under ``--outputs`` and, with ``--data-dir``, there too.

    uv run python scripts/heat2d_stiff_eigenvalues.py       # 2.7 min cold, 1.4 s cached
    uv run python scripts/heat2d_stiff_eigenvalues.py --mode rows --n 10000   # 6.7 min
    uv run python scripts/heat2d_stiff_eigenvalues.py --mode spectra \
        --spectrum-n 4900 --deltas 0 0.005 --figure-delta 0.005               # 2.1 min
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import LinearOperator, onenormest, splu  # noqa: E402

from heat_interfaces.heat1d.domain import TANH_REACH  # noqa: E402
from heat_interfaces.heat1d.march import (  # noqa: E402
    bd4_amplification,
    bd4_stability_boundary,
)
from heat_interfaces.heat2d import (  # noqa: E402
    BOUNDARY,
    INTERFACE_KIND,
    NEIGHBOURS,
    PRODUCT_ORDERING,
    SWEEPS,
    NodeSet,
    SmoothBand,
    Stencils,
    build_node_set,
    build_stencils,
    case1,
    case1_reference,
    diagonal_dominance_ratio,
    direct_operator,
    dominance_preconditioner,
    ilu_preconditioner,
    interface_aware_operator,
    interface_crossings,
    interior_eigenvalues,
    naive_operator,
    neighbour_table,
    reduced_system,
    rms_error,
    seed_operator,
    seeded_rows,
    solve_iterative,
)
from heat_interfaces.plotting import AWARE, CONSTRUCTION, NAIVE, REFERENCE  # noqa: E402
from heat_interfaces.results_cache import ResultsCache, finite  # noqa: E402

RATIOS = (8.0, 1.0, 0.125, 1.0 / 64.0, 0.0)
"""δ/h of H5's table: two resolved widths, a marginal one, and two unresolved."""

STUDY_DELTAS = (0.0, 0.04, 0.01, 0.005, 0.0025)
"""E4.3's widths, absolute, so the spectra sit beside its growing-mode table."""

LABELS = ("naive", "direct", "construction", "seeds", "seeds-plain")
"""The operators, in the order the tables print them."""

SEEDS_PLAIN = "#7fb3d5"
"""The seeds' ablation: the same blue as the seeds, lighter (``plotting``'s key)."""

STYLE = {
    "naive": (NAIVE, "o"),
    "direct": (REFERENCE, "d"),
    "construction": (CONSTRUCTION, "s"),
    "seeds": (AWARE, "^"),
    "seeds-plain": (SEEDS_PLAIN, "v"),
}

METHODS = ("gmres", "bicgstab")
PRECONDITIONERS = ("none", "appendix-b", "spilu")

N = 2500
"""The row study's node count: E4.4's set, 576 crossing stencils at ``h = 1/48``."""

SPECTRUM_N = 1600
"""The spectra's: a dense eigenvalue problem on about 1500 interior rows."""

FIGURE_DELTA = 0.005
"""Which δ the spectrum figure draws (``h/4`` at 1600 nodes)."""

CACHE_VERSION = 1
"""Bump after any change to an operator, the march or the seeded-row rule: both
caches carry it and a mismatch is recomputed (the E4.1 trap on the knee cache)."""

ROW_CACHE = "heat2d_stiff_rows.json"
ROW_CACHE_META = {
    "study": "E4.5 seed rows: dominance, SuperLU and the iterative solvers",
    "version": CACHE_VERSION,
    "reach": TANH_REACH,
}
"""What the whole file is: another study's, or another reach's, is discarded.

Everything a *row* depends on goes in ``row_key`` instead, so that changing a
solver flag recomputes the rows it moves and keeps the rest.
"""

RESULTS = "heat2d_stiff_eigenvalues"
"""The results files' stem (``results_cache``, E4.10): ``<stem>.json`` for the
default ``--mode all``, ``<stem>_rows_n<N>.json`` and ``<stem>_spectra_n<N>.json``
for the documented 10,000- and 4900-node runs, under ``--outputs`` and
``--data-dir``."""

NOT_APPLICABLE = frozenset()
"""No placeholder is written as null: ``dominance`` of an empty row group is
NaN, which no documented run has, and a NaN in a results file stops its write
(``results_cache.finite``) rather than hiding a degenerate matrix (E4.10, /spar)."""

VALUES = (0.0, lambda x, y: np.sin(2.0 * np.pi * x))
"""The elliptic problem's Dirichlet rows: 0 on ``y = 0``, ``sin 2πx`` on ``y = 1``."""


# --- the operators ------------------------------------------------------------------


def stencil_groups(nodes: NodeSet, medium: SmoothBand) -> dict[str, Stencils]:
    """The three stencil sets one δ needs: plain, the δ = 0 group, the reach group."""
    domain = replace(case1(), material=medium)
    return {
        "plain": build_stencils(nodes, domain),
        "crossing": build_stencils(nodes, domain, interface=BOUNDARY),
        "reach": build_stencils(nodes, domain, interface=BOUNDARY, reach=TANH_REACH),
    }


def operator(
    label: str, nodes: NodeSet, medium: SmoothBand, groups: dict[str, Stencils]
) -> tuple[sp.csr_array, np.ndarray, str | None]:
    """``(L, the rows it replaces, SuperLU's ordering)`` for one operator.

    The replaced rows are the seeded ones for the seed operators and the
    crossing ones for the rest, so that every line's "group" columns are read
    on the rows that method treats; at δ = 0 the two sets are the same.
    """
    if label == "naive":
        op = naive_operator(nodes, medium, groups["plain"])
        return op, group_rows(nodes, medium, groups["crossing"], 0.0), PRODUCT_ORDERING
    if label == "direct":
        op = direct_operator(nodes, medium, groups["plain"])
        return op, group_rows(nodes, medium, groups["crossing"], 0.0), None
    if label == "construction":
        op = interface_aware_operator(nodes, medium, groups["crossing"])
        return op, group_rows(nodes, medium, groups["crossing"], 0.0), None
    warp = label == "seeds"
    op = seed_operator(nodes, medium, groups["reach"], warp=warp)
    return op, group_rows(nodes, medium, groups["reach"], TANH_REACH), None


def group_rows(
    nodes: NodeSet, medium: SmoothBand, stencils: Stencils, reach: float
) -> np.ndarray:
    """The rows of the interface group whose own nodes see the edge at ``reach``."""
    mask = np.zeros(nodes.n, dtype=bool)
    for g in stencils.groups:
        if g.kind != INTERFACE_KIND:
            continue
        seen = (
            seeded_rows(nodes, medium, g.index, reach)
            if reach
            else interface_crossings(nodes, medium, g.index)
        )
        mask[g.rows[seen]] = True
    return mask


# --- the row study (H5) -------------------------------------------------------------


def dominance(ddr: np.ndarray, group: np.ndarray) -> dict[str, float]:
    """Least, median and the fraction below one, over ``group`` and over all rows."""
    out = {
        "all-least": float(ddr.min()),
        "all-median": float(np.median(ddr)),
        "all-below1": float(np.mean(ddr < 1)),
    }
    if group.any():
        out.update(
            {
                "least": float(ddr[group].min()),
                "median": float(np.median(ddr[group])),
                "below1": float(np.mean(ddr[group] < 1)),
            }
        )
    else:
        out.update({"least": np.nan, "median": np.nan, "below1": np.nan})
    return out


def condition_estimate(a: sp.sparray) -> float:
    """``|A|₁ |A⁻¹|₁`` by Higham and Tisseur's estimator on the LU factors.

    ``onenormest`` is a lower bound on each factor, so the product is one on
    the condition number: read it as "at least", and compare it across the
    operators of one table, not against a textbook bound.
    """
    lu = splu(sp.csc_array(a))
    inverse = LinearOperator(
        a.shape,
        matvec=lu.solve,
        rmatvec=lambda x: lu.solve(x, "T"),
        matmat=lambda x: lu.solve(x),
        rmatmat=lambda x: lu.solve(x, "T"),
        dtype=float,
    )
    return float(onenormest(sp.csc_array(a)) * onenormest(inverse))


def row_key(label: str, ratio: float, n: int, args: argparse.Namespace) -> str:
    """The cache key of one line: every argument its numbers depend on.

    The node set (``n``, ``seed``, the repulsion ``iterations``), the width
    ``ratio``, the solvers' ``rtol`` and ``maxiter``, and Appendix B's
    ``neighbours`` and ``sweeps``, which move the preconditioned DDR and
    every iteration count.
    """
    return (
        f"rows r{ratio:g} n{n} s{args.seed} i{args.iterations} "
        f"rtol{args.rtol:g} m{args.maxiter} k{args.neighbours} w{args.sweeps} "
        f"{label}"
    )


def row_entry(
    label: str,
    nodes: NodeSet,
    medium: SmoothBand,
    groups: dict[str, Stencils],
    reference,
    args: argparse.Namespace,
) -> dict:
    """One line of H5's table: the build, the DDR, SuperLU and the six solves."""
    t0 = time.perf_counter()
    op, group, permc = operator(label, nodes, medium, groups)
    entry: dict = {"h": nodes.h, "build": time.perf_counter() - t0}
    system = reduced_system(op, nodes, VALUES)
    inside = group[system.interior]
    entry["rows"] = int(group.sum())
    entry["nnz"] = system.a.nnz / system.n
    entry.update(dominance(diagonal_dominance_ratio(system.a), inside))
    t0 = time.perf_counter()
    u = np.asarray(
        sp.linalg.spsolve(sp.csc_array(system.a), system.b, permc_spec=permc)
    )
    entry["direct"] = time.perf_counter() - t0
    entry["residual"] = system.residual(u)
    entry["error"] = rms_error(system.expand(u), reference(nodes.x, nodes.y, 0.0))
    entry["cond"] = condition_estimate(system.a)

    t0 = time.perf_counter()
    table = neighbour_table(nodes, system.interior, args.neighbours)
    p = dominance_preconditioner(system.a, table, sweeps=args.sweeps)
    pre = system.left_preconditioned(p)
    entry["appendix-b-build"] = time.perf_counter() - t0
    entry.update(
        {
            f"b-{k}": v
            for k, v in dominance(diagonal_dominance_ratio(pre.a), inside).items()
        }
    )
    m, entry["spilu-build"] = ilu_preconditioner(system)
    systems = {"none": (system, None), "appendix-b": (pre, None), "spilu": (system, m)}
    for method in METHODS:
        for pc, (sys_, m_) in systems.items():
            res = solve_iterative(
                sys_, method, rtol=args.rtol, maxiter=args.maxiter, m=m_
            )
            entry[f"{method}/{pc}"] = {
                "iterations": res.iterations,
                "seconds": res.seconds,
                "info": res.info,
                "residual": res.residual,
                "distance": float(
                    np.linalg.norm(res.u - system.expand(u))
                    / np.linalg.norm(system.expand(u))
                ),
            }
    return entry


def row_sweep(
    n: int, ratios: Sequence[float], cache: dict, args: argparse.Namespace
) -> list[dict]:
    """One row per (δ/h, operator), computing only what the cache lacks."""
    rows = []
    nodes = None
    for ratio in ratios:
        needed = [
            label
            for label in args.labels
            if row_key(label, ratio, n, args) not in cache
        ]
        if needed:
            if nodes is None:
                nodes = build_node_set(
                    case1(), n, seed=args.seed, iterations=args.iterations
                )
            delta = ratio * nodes.h
            medium = SmoothBand(case1().material, delta)
            groups = stencil_groups(nodes, medium)
            reference = case1_reference(delta)
            for label in needed:
                t0 = time.perf_counter()
                cache[row_key(label, ratio, n, args)] = row_entry(
                    label, nodes, medium, groups, reference, args
                )
                print(
                    f"  (δ/h = {ratio:g}, {label}: {time.perf_counter() - t0:.1f} s)",
                    flush=True,
                )
        for label in args.labels:
            entry = cache[row_key(label, ratio, n, args)]
            rows.append({"ratio": ratio, "label": label, "n": n, **entry})
    return rows


def load_cache(outputs: Path) -> dict:
    path = outputs / ROW_CACHE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if data.get("meta") != ROW_CACHE_META:
        return {}
    return dict(data["entries"])


def save_cache(outputs: Path, cache: dict) -> None:
    data = {"meta": ROW_CACHE_META, "entries": dict(sorted(cache.items()))}
    (outputs / ROW_CACHE).write_text(json.dumps(data, indent=1) + "\n")


def print_rows(rows: list[dict], n: int) -> None:
    h = rows[0]["h"]
    print(
        f"\nthe seed rows in the matrix at {n} nodes, h = {h:.4f} "
        f"(δ = δ/h × h; the reduced interior system)"
    )
    print(
        "  δ/h  operator       rows   nnz    DDR least  median  <1     "
        "all least  median    cond    build   SuperLU   error"
    )
    for r in rows:
        print(
            f"{r['ratio']:5g}  {r['label']:13s} {r['rows']:5d}  {r['nnz']:5.1f}  "
            f"{r['least']:9.3f} {r['median']:7.3f}  {r['below1']:4.2f}  "
            f"{r['all-least']:9.3f} {r['all-median']:7.3f}  {r['cond']:7.1e}  "
            f"{r['build']:6.1f}s  {r['direct']:6.2f}s  {r['error']:8.2e}"
        )
    print(
        "\n  Appendix B's P on the same rows (DDR after three sweeps), and the "
        "iterations to |r| <= rtol |b|"
    )
    print(
        "  δ/h  operator       P least  median   gmres  +P  +ilu   "
        "bicgstab  +P  +ilu   P build  ilu build"
    )
    for r in rows:
        cells = [
            f"{r[f'{m}/{pc}']['iterations']:4d}"
            + ("*" if r[f"{m}/{pc}"]["info"] else " ")
            for m in METHODS
            for pc in PRECONDITIONERS
        ]
        print(
            f"{r['ratio']:5g}  {r['label']:13s} {r['b-least']:7.3f} "
            f"{r['b-median']:7.3f}  " + " ".join(cells) + f"   "
            f"{r['appendix-b-build']:6.2f}s  {r['spilu-build']:6.2f}s"
        )
    print("  (* the solver stopped on its iteration cap or broke down)")


# --- the spectra (H6) ---------------------------------------------------------------


def spectra_path(
    outputs: Path, n: int, delta: float, seed: int, iterations: int
) -> Path:
    """Where one (n, δ) spectrum is cached; ``iterations`` moves the node set."""
    return outputs / f"heat2d_stiff_spectra_n{n}_d{delta:g}_i{iterations}_s{seed}.npz"


def spectra(n: int, delta: float, args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Every operator's interior eigenvalues at one (n, δ), cached as an ``npz``."""
    path = spectra_path(args.outputs, n, delta, args.seed, args.iterations)
    if path.exists():
        data = np.load(path)
        fresh = data.get("version", np.array(0)) == CACHE_VERSION
        if fresh and all(label in data for label in args.labels):
            return {label: data[label] for label in (*args.labels, "h")}
    nodes = build_node_set(case1(), n, seed=args.seed, iterations=args.iterations)
    medium = SmoothBand(case1().material, delta)
    groups = stencil_groups(nodes, medium)
    out: dict[str, np.ndarray] = {
        "h": np.array(nodes.h),
        "version": np.array(CACHE_VERSION),
    }
    for label in args.labels:
        t0 = time.perf_counter()
        op, _, _ = operator(label, nodes, medium, groups)
        out[label] = interior_eigenvalues(op, nodes)
        print(
            f"  (δ = {delta:g}, {label}: {time.perf_counter() - t0:.1f} s)", flush=True
        )
    np.savez_compressed(path, **out)
    return out


def spectrum_rows(
    spec: dict[str, np.ndarray], delta: float, labels: Sequence[str]
) -> list[dict]:
    """Per operator: the right edge, the scaled extremes and BD4 at ``dt = h``."""
    h = float(spec["h"])
    rows = []
    for label in labels:
        lam = spec[label]
        rows.append(
            {
                "delta": delta,
                "label": label,
                "h": h,
                "count": int(lam.size),
                "complex": int(np.sum(np.abs(lam.imag) > 1e-8 * np.abs(lam).max())),
                "max_re": float(lam.real.max()),
                "positive": int(np.sum(lam.real > 0)),
                "min_re_h2": float(lam.real.min() * h**2),
                "max_im_h2": float(np.abs(lam.imag).max() * h**2),
                "bd4": float(bd4_amplification(h * lam).max()),
            }
        )
    return rows


def print_spectra(rows: list[dict], n: int) -> None:
    print(
        f"\ninterior spectra at {n} nodes, h = {rows[0]['h']:.4f} "
        "(BD4's largest root modulus at dt = h)"
    )
    print(
        "    δ   δ/h  operator       eigs  complex   max Re  Re > 0  "
        "h² min Re  h² max |Im|   BD4 max |ζ|"
    )
    for r in rows:
        ratio = r["delta"] / r["h"]
        print(
            f"{r['delta']:6.4f} {ratio:5.2f}  {r['label']:13s} {r['count']:5d} "
            f"{r['complex']:8d}  {r['max_re']:8.2f}  {r['positive']:5d}  "
            f"{r['min_re_h2']:9.2f}  {r['max_im_h2']:11.3f}  {r['bd4']:11.3f}"
        )


def figure_spectra(spec: dict[str, np.ndarray], delta: float, labels: Sequence[str]):
    """Fig. 5-6's twin at one δ: each operator's cloud, then the zoom with BD4."""
    h = float(spec["h"])
    fig = plt.figure(figsize=(9.0, 7.0))
    grid = fig.add_gridspec(2, len(labels), height_ratios=[1.0, 1.4])
    for k, label in enumerate(labels):
        ax = fig.add_subplot(grid[0, k])
        lam = spec[label]
        ax.plot(lam.real / 1e4, lam.imag / 1e3, ".", color=STYLE[label][0], ms=2)
        ax.axvline(0.0, color="0.6", lw=0.6)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Re λ / 10⁴")
        if k == 0:
            ax.set_ylabel("Im λ / 10³")
    zoom = fig.add_subplot(grid[1, :])
    for label in reversed(labels):
        lam = spec[label]
        # The right edge goes in the legend: a growing mode sits far outside
        # this window, where the slow modes and BD4's boundary are.
        zoom.plot(
            lam.real,
            lam.imag,
            ".",
            color=STYLE[label][0],
            ms=3,
            label=f"{label} (max Re {lam.real.max():.4g})",
        )
    curve = bd4_stability_boundary(np.linspace(0.0, 2 * np.pi, 721)) / h
    zoom.plot(curve.real, curve.imag, "k-", lw=1.0, label="BD4 boundary, dt = h")
    zoom.axvline(0.0, color="0.6", lw=0.6)
    span = 3.0 / h
    zoom.set_xlim(-span, 0.2 * span)
    zoom.set_ylim(-span, span)
    zoom.set_xlabel("Re λ")
    zoom.set_ylabel("Im λ")
    zoom.set_title(
        f"case 1, δ = {delta:g} ({delta / h:.2f} h): BD4 is stable outside the curve",
        fontsize=9,
    )
    zoom.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def figure_dominance(rows: list[dict], labels: Sequence[str]):
    """H5 at a glance: the group's DDR and the iteration counts against δ/h."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ratios = sorted({r["ratio"] for r in rows if r["ratio"]})
    # Where δ = 0 goes on a log axis; a sweep of δ = 0 alone draws it at 1.
    floor = min(ratios) / 4.0 if ratios else 1.0
    for label in labels:
        colour, marker = STYLE[label]
        line = [r for r in rows if r["label"] == label]
        x = [r["ratio"] or floor for r in line]
        axes[0].plot(
            x, [r["least"] for r in line], marker + "-", color=colour, label=label
        )
        axes[0].plot(x, [r["median"] for r in line], marker + "--", color=colour)
        axes[1].plot(
            x,
            [r["gmres/none"]["iterations"] for r in line],
            marker + "-",
            color=colour,
            label=label,
        )
        axes[1].plot(
            x,
            [r["bicgstab/none"]["iterations"] for r in line],
            marker + "--",
            color=colour,
        )
    for ax, ylabel, title in (
        (axes[0], "DDR", "diagonal dominance (solid: least, dashed: median)"),
        (axes[1], "inner iterations", "gmres (solid) and bicgstab (dashed)"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("δ / h  (leftmost: δ = 0)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="both", linewidth=0.3)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    return fig


# --- the driver ---------------------------------------------------------------------


def figure_name(what: str, n: int, default: int) -> str:
    """``heat2d_stiff_<what>.png`` at the default count, ``…_n<N>.png`` at another,
    so that the documented 10,000- and 4900-node runs keep the default figures."""
    return (
        f"heat2d_stiff_{what}.png" if n == default else f"heat2d_stiff_{what}_n{n}.png"
    )


def run_rows(args: argparse.Namespace) -> list[dict]:
    cache = load_cache(args.outputs)
    t0 = time.perf_counter()
    rows = row_sweep(args.n, args.ratios, cache, args)
    save_cache(args.outputs, cache)
    print_rows(rows, args.n)
    print(f"\nrow study {time.perf_counter() - t0:.1f} s")
    figure = args.outputs / figure_name("dominance", args.n, N)
    figure_dominance(rows, args.labels).savefig(figure)
    plt.close("all")
    print(f"wrote {figure}")
    return rows


def run_spectra(args: argparse.Namespace) -> list[dict]:
    t0 = time.perf_counter()
    rows = []
    for delta in args.deltas:
        spec = spectra(args.spectrum_n, delta, args)
        rows.extend(spectrum_rows(spec, delta, args.labels))
    print_spectra(rows, args.spectrum_n)
    print(f"\nspectra {time.perf_counter() - t0:.1f} s")
    spec = spectra(args.spectrum_n, args.figure_delta, args)
    figure = args.outputs / figure_name("spectra", args.spectrum_n, SPECTRUM_N)
    figure_spectra(spec, args.figure_delta, args.labels).savefig(figure)
    plt.close("all")
    print(f"wrote {figure}")
    return rows


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=("all", "rows", "spectra"), default="all")
    parser.add_argument("--n", type=int, default=N, help="the row study's node count")
    parser.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    parser.add_argument("--deltas", type=float, nargs="+", default=list(STUDY_DELTAS))
    parser.add_argument("--labels", nargs="+", default=list(LABELS), choices=LABELS)
    parser.add_argument("--spectrum-n", type=int, default=SPECTRUM_N)
    parser.add_argument("--figure-delta", type=float, default=FIGURE_DELTA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--maxiter", type=int, default=3000)
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS)
    parser.add_argument("--sweeps", type=int, default=SWEEPS)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="also write the run's results file here (paper/data, E5.3)",
    )
    args = parser.parse_args(argv)
    if any(r < 0 for r in args.ratios):
        parser.error("δ/h must be non-negative")
    if args.figure_delta not in args.deltas:
        args.deltas = [*args.deltas, args.figure_delta]
    args.outputs.mkdir(parents=True, exist_ok=True)
    results = ResultsCache(RESULTS, vars(args))
    start = time.perf_counter()
    tables: dict = {}
    if args.mode in ("all", "rows"):
        t0 = time.perf_counter()
        tables["rows"] = run_rows(args)
        results.time("rows", time.perf_counter() - t0)
    if args.mode in ("all", "spectra"):
        t0 = time.perf_counter()
        tables["spectra"] = run_spectra(args)
        results.time("spectra", time.perf_counter() - t0)
    for name, table in tables.items():
        results.add(name, finite(table, NOT_APPLICABLE, name))
    results.time("total", time.perf_counter() - start)
    name = results_name(args)
    paths = [args.outputs / name]
    if args.data_dir is not None:
        paths.append(args.data_dir / name)
    results.write(*paths)
    return tables


def results_name(args: argparse.Namespace) -> str:
    """The run's results file: the default run's, or the mode and its node count."""
    if args.mode == "all":
        return f"{RESULTS}.json"
    n = args.n if args.mode == "rows" else args.spectrum_n
    return f"{RESULTS}_{args.mode}_n{n}.json"


if __name__ == "__main__":
    main()
