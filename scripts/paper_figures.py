"""Regenerate the manuscript's figures and table fragments (E5.3, #44).

House rule (``paper/README.md``): ``paper/figures/`` comes from committed
scripts and is never hand-edited. Every figure here and every ``tab_*.tex``
fragment (``paper_tables.py``) is drawn from the results files in
``paper/data/`` alone: the documented runs' tables and the arrays their
figures need, which ``scripts/paper_data.py`` regenerates. The figures are
the notes' (stiff note §2.5's four and §5.1's table), laid out again for the
manuscript: print style (``plotting.use_print_style``), ``TEXTWIDTH`` wide
so that ``\\includegraphics`` sets them at their natural size, the notes'
colours and markers. ``SOURCE_DATE_EPOCH`` is pinned, so a regenerated file
is byte-identical to the committed one; ``--check`` draws everything into a
temporary directory and compares bytes with ``paper/figures/``, the gate
``paper/make_arxiv.py`` runs. Names carry no dot but the extension's, since
``\\includegraphics`` would read ``a0.02`` as one.

    uv run python scripts/paper_figures.py            # every file, ~30 s
    uv run python scripts/paper_figures.py --check    # byte identity
    uv run python scripts/paper_figures.py --only heat2d_stiff_knee.pdf
"""

from __future__ import annotations

import argparse
import filecmp
import logging
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_tables import TABLES  # noqa: E402

import heat1d_stiff as d1  # noqa: E402
import heat2d_ring as ring  # noqa: E402
import heat2d_stiff as d2  # noqa: E402
import heat2d_stiff_eigenvalues as eig  # noqa: E402
from heat_interfaces.heat1d.domain import X_MAX, X_MIN  # noqa: E402
from heat_interfaces.heat1d.march import bd4_stability_boundary  # noqa: E402
from heat_interfaces.plotting import (  # noqa: E402
    AWARE,
    CONSTRUCTION,
    NAIVE,
    REFERENCE,
    TEXTWIDTH,
    use_print_style,
)
from heat_interfaces.results_cache import float_keys, read_results  # noqa: E402

# fontTools reads the pinned SOURCE_DATE_EPOCH = 0 into each embedded font's
# head table and says so once per font; the date is the point.
logging.getLogger("fontTools").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data"
FIGURES_DIR = ROOT / "paper" / "figures"


class Data:
    """The results files of a data directory, read once each."""

    def __init__(self, root: Path = DATA):
        self.root = Path(root)

    @cache  # noqa: B019 (one Data per run; the files do not change under it)
    def tables(self, name: str) -> dict:
        return read_results(self.root / name)["tables"]


def by_delta(table: dict) -> dict[float, list[dict]]:
    """``{δ: rows}`` from a table keyed by ``%g`` strings, in the file's order."""
    return float_keys(table)


# --- shared chrome ----------------------------------------------------------------


def _figure(height: float, **kwargs):
    return plt.figure(figsize=(TEXTWIDTH, height), layout="constrained", **kwargs)


def _counts(ax) -> None:
    """A log axis of node counts ticked at the counts drawn, every other one past four.

    Called after the lines are drawn; a vertical marker line (two equal x) is not
    a count.
    """
    counts = sorted(
        {
            int(round(x))
            for line in ax.get_lines()
            if len(set(np.atleast_1d(line.get_xdata()))) > 1
            for x in line.get_xdata()
        }
    )
    ticks = counts if len(counts) <= 4 else counts[::2]
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(ticks, [str(t) for t in ticks])


def _legend(fig, handles: Sequence, ncol: int) -> None:
    fig.legend(
        handles=list(handles),
        loc="outside lower center",
        ncol=ncol,
        handlelength=1.8,
        columnspacing=1.2,
    )


def _key(colour: str, label: str, **kwargs) -> Line2D:
    return Line2D([], [], color=colour, label=label, **kwargs)


def _line_key(label: str, **kwargs) -> Line2D:
    """A legend entry in a 2-D line's ``STYLE``."""
    return Line2D([], [], label=label, **_line(label), **kwargs)


def _delta_keys(deltas: Sequence[float], markers: Sequence[str]) -> list[Line2D]:
    keys = []
    for k, delta in enumerate(d for d in deltas if d > 0):
        keys.append(
            Line2D(
                [],
                [],
                color="k",
                marker=markers[k % len(markers)],
                ls="",
                label=rf"$\delta = {delta:g}$",
            )
        )
    return keys


def _delta_label(delta: float) -> str:
    return "the jump" if delta == 0.0 else rf"$\delta = {delta:g}$"


# --- one dimension (stiff note §2) --------------------------------------------------

MEDIUM_TITLE = {"matlab": r"$1/9\,|\,1$", "eq75": "eq. 75"}


def heat1d_knee(data: Data):
    """§2.2–§2.3: naive, the δ = 0 construction and the seeds against n."""
    tables = data.tables("heat1d_stiff.json")
    fig = _figure(4.2)
    axes = fig.subplots(2, 2, sharex="col", sharey="row")
    problems = (
        ("equilibrium", "knee/equilibrium"),
        (rf"ramp, $t = {d1.T_END:g}$", "knee/ramp"),
    )
    for i, medium in enumerate(("matlab", "eq75")):
        for j, (problem, name) in enumerate(problems):
            ax = axes[i, j]
            results = by_delta(tables[name][medium])
            positive = [d for d in results if d > 0]
            for delta, rows in results.items():
                n = [r["n"] for r in rows]
                if delta == 0.0:
                    style = dict(lw=0.7, alpha=0.8)
                else:
                    marker = d1.MARKERS[positive.index(delta) % len(d1.MARKERS)]
                    style = dict(marker=marker, ms=3, lw=0.9)
                    ax.loglog(n, [r["floor"] for r in rows], ":", color=CONSTRUCTION)
                    ax.axvline(
                        (X_MAX - X_MIN) / delta + 1, color=REFERENCE, lw=0.5, ls="--"
                    )
                for label, colour in d1.COLOURS.items():
                    ax.loglog(n, [r[label] for r in rows], color=colour, **style)
            ax.set_title(f"{MEDIUM_TITLE[medium]}, {problem}")
            _counts(ax)
            if i == 1:
                ax.set_xlabel("nodes $n$")
        axes[i, 0].set_ylabel(r"$\|e\|_2 / \|u\|_2$")
    handles = [
        _key(NAIVE, r"naive $D_x A D_x$"),
        _key(CONSTRUCTION, r"$\delta = 0$ construction"),
        _key(AWARE, "seeds"),
        _key(CONSTRUCTION, "floor", ls=":"),
        _key(REFERENCE, r"$h = \delta$", ls="--", lw=0.5),
        _key("k", r"$\delta = 0$ (jump)", lw=0.7),
        *_delta_keys(list(by_delta(tables["knee/ramp"]["matlab"])), d1.MARKERS),
    ]
    _legend(fig, handles, 5)
    return fig


def heat1d_treatments(data: Data):
    """§2.4: the comparators on the ramp problem, a row per medium, a column per δ."""
    tables = data.tables("heat1d_stiff.json")
    ramp = tables["comparators/ramp"]
    deltas = sorted(d for d in by_delta(ramp["matlab"]) if d > 0)
    fig = _figure(3.7)
    axes = fig.subplots(2, len(deltas), sharex=True, sharey="row")
    for i, medium in enumerate(("matlab", "eq75")):
        results = by_delta(ramp[medium])
        for j, delta in enumerate(deltas):
            ax = axes[i, j]
            rows = results[delta]
            n = [r["n"] for r in rows]
            for label, (colour, ls) in d1.COMPARATOR_STYLE.items():
                ax.loglog(n, [r[label] for r in rows], color=colour, ls=ls, lw=0.9)
            ax.axvline((X_MAX - X_MIN) / delta + 1, color=REFERENCE, lw=0.5, ls="--")
            ax.set_title(rf"{MEDIUM_TITLE[medium]}, $\delta = {delta:g}$")
            _counts(ax)
            if i == 1:
                ax.set_xlabel("nodes $n$")
        axes[i, 0].set_ylabel(r"$\|e\|_2 / \|u\|_2$")
    handles = [
        _key(colour, label, ls=ls)
        for label, (colour, ls) in d1.COMPARATOR_STYLE.items()
    ]
    _legend(fig, handles, 4)
    return fig


def heat1d_seeds(data: Data):
    """§2.5: the seeds across P4's window against the monomials and E1.2's basis."""
    tables = data.tables("heat1d_stiff.json")
    meta, curves = tables["seed_functions"], tables["seed_functions/curves"]
    xi = np.asarray(curves["xi"])
    seeds = {r: np.asarray(v) for r, v in by_delta(curves["seeds"]).items()}
    alpha = {r: np.asarray(v) for r, v in by_delta(curves["alpha"]).items()}
    ratios = [r for r in seeds if r > 0]
    styles = ["-", "--", "-.", ":"]
    fig = _figure(4.4)
    grid = fig.add_gridspec(3, 2, height_ratios=(0.8, 1.6, 1.6))
    top = fig.add_subplot(grid[0, :])
    for j, ratio in enumerate(ratios):
        top.plot(xi, alpha[ratio], color=AWARE, ls=styles[j % 4])
    top.plot(xi, alpha[0.0], color=CONSTRUCTION, lw=0.8)
    top.set_ylabel(r"$\alpha$")
    top.set_title(
        f"{MEDIUM_TITLE[meta['medium']]} medium, {meta['n']} nodes,"
        r" $\xi = (x - x_e) / h_s$"
    )
    nodes = np.asarray(curves["xi_nodes"])
    for k in range(1, 5):
        ax = fig.add_subplot(grid[1 + (k - 1) // 2, (k - 1) % 2])
        ax.plot(xi, curves["monomials"][k], color=REFERENCE, ls="--", lw=0.8)
        ax.plot(xi, curves["jump"][k], color=CONSTRUCTION, lw=0.9)
        for j, ratio in enumerate(ratios):
            ax.plot(xi, seeds[ratio][k], color=AWARE, ls=styles[j % 4])
        ax.plot(nodes, np.interp(nodes, xi, seeds[ratios[0]][k]), "o", color=AWARE)
        for xc in curves["xi_edges"]:
            if -1 <= xc <= 1:
                ax.axvline(xc, color=REFERENCE, lw=0.5, ls=":")
        ax.set_title(rf"$\phi_{k}$")
        if k >= 3:
            ax.set_xlabel(r"$\xi$")
    handles = [
        _key(REFERENCE, r"monomial $\xi^k$", ls="--", lw=0.8),
        _key(CONSTRUCTION, r"translated basis ($\delta = 0$)"),
        *[
            _key(AWARE, rf"seeds, $\delta/h = {r:g}$", ls=styles[j % 4])
            for j, r in enumerate(ratios)
        ],
        _key(AWARE, "nodes", marker="o", ls=""),
    ]
    _legend(fig, handles, 3)
    return fig


def heat1d_snapshot(data: Data):
    """§2.5: the ramp solution at h = 8δ, the four operators, and their errors."""
    tables = data.tables("heat1d_stiff.json")
    meta, curves = tables["snapshot"], tables["snapshot/curves"]
    x = np.asarray(curves["x"])
    x_fine, u_fine = np.asarray(curves["x_fine"]), np.asarray(curves["u_fine"])
    solutions = {k: np.asarray(v) for k, v in curves["solutions"].items()}
    fig = _figure(2.5)
    left, right = fig.subplots(1, 2)
    left.plot(x_fine, u_fine, color=REFERENCE, lw=0.9)
    for label, (colour, marker) in d1.SNAPSHOT_STYLE.items():
        left.plot(x, solutions[label], marker, color=colour, ms=2.2, mfc="none", lw=0)
    for xc in curves["centres"]:
        left.axvline(xc, color=REFERENCE, lw=0.5, ls="--")
        right.axvline(xc, color=REFERENCE, lw=0.5, ls="--")
    left.set_xlabel("$x$")
    left.set_ylabel("$u$")
    h, delta = meta["h"], meta["delta"]
    left.set_title(
        rf"$t = {d1.T_END:g}$, {meta['n']} nodes, $\delta = {delta:g}$"
        rf" ($h = {h / delta:.3g}\,\delta$)"
    )
    xc = float(curves["centres"][0])
    span = 4 * h
    inset = left.inset_axes((0.55, 0.5, 0.42, 0.42))
    window = np.abs(x_fine - xc) <= span
    inset.plot(x_fine[window], u_fine[window], color=REFERENCE, lw=0.9)
    near = np.abs(x - xc) <= span
    for label, (colour, marker) in d1.SNAPSHOT_STYLE.items():
        inset.plot(
            x[near], solutions[label][near], marker, color=colour, ms=2.2, mfc="none"
        )
    inset.axvline(xc, color=REFERENCE, lw=0.5, ls="--")
    inset.set_title(r"the edge, $\pm 4h$", fontsize=6)
    inset.tick_params(labelsize=5)
    inset.grid(False)
    # The errors as the driver stored them: the seeds' 3e-8 is below six figures of
    # u, so the difference of the stored solutions would be rounding.
    errors = {k: np.abs(np.asarray(v)) for k, v in curves["errors"].items()}
    interior = slice(1, -1)
    for label, (colour, _marker) in d1.SNAPSHOT_STYLE.items():
        right.semilogy(x[interior], errors[label][interior], color=colour, lw=0.8)
    right.set_xlabel("$x$")
    right.set_ylabel(r"$|u_h - u|$, interior nodes")
    right.set_ylim(bottom=1e-12)
    handles = [_key(REFERENCE, "reference", lw=0.9)] + [
        _key(colour, label, marker=marker, mfc="none", ms=3)
        for label, (colour, marker) in d1.SNAPSHOT_STYLE.items()
    ]
    _legend(fig, handles, 5)
    return fig


# --- two dimensions (stiff note §4, §5) -----------------------------------------------


def _h_equals_delta(delta: float) -> float:
    """The count at which ``h = 1/round(0.95 √N)`` equals δ (port notes §2.10)."""
    return (1.0 / (0.95 * delta)) ** 2


def _line(label: str) -> dict:
    colour, marker = d2._style(label)
    return {"color": colour, "marker": marker}


def _seed_line(data: Data) -> dict[str, dict[float, list[tuple[int, float]]]]:
    """The seeds' RMS at every δ of the flat sweep, δ = 0 to 160,000 nodes (§4.5)."""
    sweep = data.tables("heat2d_stiff_seeds.json")["sweep"]
    jump = data.tables("heat2d_stiff_seeds_jump.json")["sweep"]
    out: dict[str, dict[float, list[tuple[int, float]]]] = {}
    for problem in ("elliptic", "parabolic"):
        lines = {d: rows for d, rows in by_delta(sweep[problem]).items() if d > 0}
        lines = {0.0: by_delta(jump[problem])[0.0], **lines}
        out[problem] = {
            d: [(r["n"], r["seeds/rms"]) for r in rows] for d, rows in lines.items()
        }
    return out


def heat2d_knee(data: Data):
    """§4.2: the naive knee, the construction's floor, the seeds, the h/δ collapse."""
    knee = data.tables("heat2d_stiff_naive.json")["knee"]
    seeds = _seed_line(data)
    fig = _figure(4.9)
    grid = fig.add_gridspec(2, 6, height_ratios=(1.35, 1.0))
    top = [fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3:])]
    bottom = [fig.add_subplot(grid[1, 2 * k : 2 * k + 2]) for k in range(3)]
    names = {"elliptic": "equilibrium", "parabolic": rf"parabolic, $t = {d2.T_END:g}$"}
    for ax, problem in zip(top, ("elliptic", "parabolic"), strict=True):
        lines = by_delta(knee[problem])
        positive = [d for d in lines if d > 0]
        first = next(iter(lines.values()))
        n = [r["n"] for r in first]
        ax.loglog(n, [r["uniform"] for r in first], color=REFERENCE, lw=0.8)
        for delta, rows in lines.items():
            n = [r["n"] for r in rows]
            if delta == 0.0:
                ax.loglog(n, [r["naive/rms"] for r in rows], color=NAIVE, lw=0.7)
                continue
            marker = d2.MARKERS[positive.index(delta) % len(d2.MARKERS)]
            ax.loglog(n, [r["naive/rms"] for r in rows], color=NAIVE, marker=marker)
            ax.loglog(
                n,
                [r["construction/rms"] for r in rows],
                color=CONSTRUCTION,
                marker=marker,
                lw=0.8,
            )
            ax.loglog(n, [r["floor"] for r in rows], ":", color=CONSTRUCTION, lw=0.7)
            at = _h_equals_delta(delta)
            if n[0] <= at <= n[-1]:
                ax.axvline(at, color=REFERENCE, lw=0.5, ls="--")
        for delta, points in seeds[problem].items():
            m, e = zip(*points, strict=True)
            if delta == 0.0:
                ax.loglog(m, e, color=AWARE, lw=0.7)
            else:
                marker = d2.MARKERS[positive.index(delta) % len(d2.MARKERS)]
                ax.loglog(m, e, color=AWARE, marker=marker, ms=2.5, lw=0.8)
        ax.set_title(f"case 1, {names[problem]}")
        ax.set_xlabel("nodes $N$")
        _counts(ax)
    top[0].set_ylabel("RMS error in $u$")
    elliptic = by_delta(knee["elliptic"])
    positive = [d for d in elliptic if d > 0]
    for ax, quantity, title in (
        (bottom[0], "naive/flux", "flux error, first pair"),
        (bottom[1], "naive/vs_jump", "RMS ÷ the jump's"),
        (bottom[2], "naive/rms", "RMS error in $u$"),
    ):
        if quantity != "naive/vs_jump":
            jump = [r[quantity] for r in elliptic[0.0]]
            ax.axhspan(min(jump), max(jump), color=NAIVE, alpha=0.15, lw=0)
        for delta in positive:
            rows = elliptic[delta]
            ax.loglog(
                [r["h_over_delta"] for r in rows],
                [r[quantity] for r in rows],
                color=NAIVE,
                marker=d2.MARKERS[positive.index(delta) % len(d2.MARKERS)],
                ms=2.5,
                lw=0.7,
            )
        ax.axvline(1.0, color=REFERENCE, lw=0.5, ls="--")
        ax.set_xlabel(r"$h / \delta$")
        ax.set_title(title)
        ax.minorticks_off()
    bottom[0].set_ylabel("naive, equilibrium")
    handles = [
        _key(NAIVE, "naive"),
        _key(CONSTRUCTION, r"$\delta = 0$ construction"),
        _key(AWARE, "seeds"),
        _key(CONSTRUCTION, "floor", ls=":", lw=0.7),
        _key(REFERENCE, r"naive, $\alpha \equiv 1$", lw=0.8),
        _key("k", "the jump", lw=0.7),
        *_delta_keys(list(elliptic), d2.MARKERS),
    ]
    _legend(fig, handles, 5)
    return fig


def _seed_figure(
    data: Data, name: str, top_lines: Sequence[str], seeds: str, plain: str
):
    """§4.5–§4.7: three widths, the seeds at every δ, the ratios, and the warp."""
    sweep = data.tables(name)["sweep"]
    elliptic = by_delta(sweep["elliptic"])
    parabolic = by_delta(sweep["parabolic"])
    positive = [d for d in elliptic if d > 0]
    widths = sorted(positive)
    shown = list(dict.fromkeys([widths[0], widths[len(widths) // 2], widths[-1]]))
    fig = _figure(4.4)
    axes = fig.subplots(2, 3)
    for ax, delta in zip(axes[0], shown, strict=False):
        rows = elliptic[delta]
        n = [r["n"] for r in rows]
        for label in top_lines:
            ax.loglog(n, [r[f"{label}/rms"] for r in rows], lw=0.9, **_line(label))
        ax.loglog(n, [r["floor"] for r in rows], ":", color=CONSTRUCTION, lw=0.7)
        ax.set_title(rf"equilibrium, $\delta = {delta:g}$")
        _counts(ax)
    axes[0][0].set_ylabel("RMS error in $u$")
    ax = axes[1][0]
    colour = d2._style(seeds)[0]
    for delta in elliptic:
        marker = d2._marker(delta, positive)
        for results, ls in ((elliptic, "-"), (parabolic, "--")):
            rows = results[delta]
            ax.loglog(
                [r["n"] for r in rows],
                [r[f"{seeds}/rms"] for r in rows],
                ls,
                color=colour,
                marker=marker,
                ms=2.5,
                lw=0.8 if marker else 0.6,
            )
    rows = elliptic[0.0]
    n = np.array([r["n"] for r in rows], dtype=float)
    ax.loglog(n, rows[0][f"{seeds}/rms"] * (n / n[0]) ** -2.0, "-.", color=REFERENCE)
    ax.set_title("seeds, every $\\delta$")
    ax.set_xlabel("nodes $N$")
    ax.set_ylabel("RMS error in $u$")
    _counts(ax)
    ax = axes[1][1]
    for delta in positive:
        rows = elliptic[delta]
        marker = d2._marker(delta, positive)
        for label, c in (("naive", NAIVE), ("direct", REFERENCE)):
            ax.loglog(
                [r["h_over_delta"] for r in rows],
                [r[f"{seeds}/rms"] / r[f"{label}/rms"] for r in rows],
                color=c,
                marker=marker,
                ms=2.5,
                lw=0.7,
            )
    ax.axhline(1.0, color="k", lw=0.5)
    ax.axvline(1.0, color=REFERENCE, lw=0.5, ls="--")
    ax.set_title("seeds ÷ naive, ÷ direct")
    ax.set_xlabel(r"$h / \delta$")
    ax.minorticks_off()
    ax = axes[1][2]
    for delta in elliptic:
        rows = elliptic[delta]
        ax.semilogx(
            [r["n"] for r in rows],
            [r[f"{plain}/rms"] / r[f"{seeds}/rms"] for r in rows],
            color=d2._style(plain)[0],
            marker=d2._marker(delta, positive),
            ms=2.5,
            lw=0.8,
        )
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_title("plain ÷ warped")
    ax.set_xlabel("nodes $N$")
    _counts(ax)
    handles = [_line_key(label, lw=0.9, ms=3) for label in top_lines]
    handles += [
        _key(CONSTRUCTION, "floor", ls=":", lw=0.7),
        _key(REFERENCE, "$h^4$", ls="-.", lw=0.8),
        _key(colour, "parabolic", ls="--", lw=0.8),
        _key("k", r"$\delta = 0$", lw=0.6),
        *_delta_keys(list(elliptic), d2.MARKERS),
    ]
    _legend(fig, handles, 4)
    return fig


def heat2d_seeds(data: Data):
    """§4.5: the flat sweep on case 1."""
    lines = ("naive", "construction", "direct", "seeds")
    return _seed_figure(data, "heat2d_stiff_seeds.json", lines, "seeds", "seeds-plain")


def heat2d_seeds_curved(data: Data):
    """§4.6: route (a), the flat seeds along the foot point's normal, on case 2."""
    lines = ("naive", "construction", "construction-flat", "direct", "seeds")
    return _seed_figure(
        data, "heat2d_stiff_seeds_a0.02_sine.json", lines, "seeds", "seeds-plain"
    )


def heat2d_tangential_curved(data: Data):
    """§4.7: the tangential chain on case 2."""
    lines = ("naive", "construction", "direct", "seeds", "tangential")
    return _seed_figure(
        data,
        "heat2d_stiff_seeds_tangential_a0.02_sine.json",
        lines,
        "tangential",
        "tangential-plain",
    )


def _treatment_figure(data: Data, name: str, seeds: str):
    """§4.9: every line at three widths, then each family over the naive product."""
    parabolic = by_delta(data.tables(name)["sweep"]["parabolic"])
    positive = [d for d in parabolic if d > 0]
    widths = sorted(positive)
    shown = list(dict.fromkeys([0.0, widths[0], widths[-1]]))
    lines = ["naive", *d2.TREATMENT_LABELS, seeds]

    def style(label: str) -> dict:
        if label in d2.TREATMENT_LABELS:
            return d2._treatment_line(label)
        return {"color": d2._style(label)[0], "marker": d2._style(label)[1]}

    fig = _figure(4.4)
    axes = fig.subplots(2, 3)
    for ax, delta in zip(axes[0], shown, strict=False):
        rows = parabolic[delta]
        n = [r["n"] for r in rows]
        for label in lines:
            ax.loglog(
                n,
                [r[f"{label}/rms"] for r in rows],
                ms=2.5,
                lw=0.8,
                zorder=3 if label == "naive" else 2,
                **style(label),
            )
        ax.set_title(f"parabolic, {_delta_label(delta)}")
        _counts(ax)
    axes[0][0].set_ylabel("RMS error in $u$")
    for ax, family in zip(axes[1], ("harmonic", "arithmetic", "widened"), strict=True):
        for label in d2.TREATMENT_LABELS:
            if not label.startswith(family):
                continue
            s = d2._treatment_line(label)
            for delta in positive:
                rows = parabolic[delta]
                ax.loglog(
                    [r["h_over_delta"] for r in rows],
                    [r[f"{label}/rms"] / r["naive/rms"] for r in rows],
                    color=s["color"],
                    ls=s["ls"],
                    mfc=s["mfc"],
                    marker=d2._marker(delta, positive),
                    ms=2.5,
                    lw=0.7,
                )
        ax.axhline(1.0, color="k", lw=0.5)
        ax.axvline(1.0, color=REFERENCE, lw=0.5, ls="--")
        ax.set_title(f"{family} ÷ naive")
        ax.set_xlabel(r"$h / \delta$")
        ax.minorticks_off()
    axes[1][0].set_ylabel("RMS ÷ naive RMS")
    handles = [
        Line2D([], [], label=label, ms=3, lw=0.8, **style(label)) for label in lines
    ]
    handles += _delta_keys(positive, d2.MARKERS)
    _legend(fig, handles, 4)
    return fig


def heat2d_treatments(data: Data):
    """§4.9 on case 1."""
    return _treatment_figure(data, "heat2d_stiff_treatments.json", "seeds")


def heat2d_treatments_curved(data: Data):
    """§4.9 on case 2."""
    return _treatment_figure(
        data, "heat2d_stiff_treatments_a0.02_sine.json", "tangential"
    )


def heat2d_snapshot(data: Data):
    """§5.1: the four regimes on one node set, ``|e|`` over the strip and in y."""
    tables = data.tables("heat2d_stiff_snapshot.json")
    rows, field = tables["snapshot"], tables["snapshot/field"]
    x, y = np.asarray(field["x"]), np.asarray(field["y"])
    errors = {k: np.abs(np.asarray(v)) for k, v in field["errors"].items()}
    tri = mtri.Triangulation(x, y)
    lows = [np.percentile(errors[r["operator"]], 5) for r in rows]
    norm = LogNorm(vmin=max(min(lows), 1e-12), vmax=max(r["max"] for r in rows))
    fig = _figure(4.0)
    top, bottom = fig.subfigures(2, 1, height_ratios=(1.15, 1.0))
    axes = top.subplots(1, len(rows), sharey=True)
    image = None
    for ax, r in zip(axes, rows, strict=True):
        e = errors[r["operator"]]
        image = ax.tripcolor(
            tri,
            np.maximum(e, norm.vmin),
            norm=norm,
            cmap="viridis",
            shading="gouraud",
            rasterized=True,
        )
        for yc in (0.6, 0.8):
            ax.axhline(yc, color="w", lw=0.5, ls="--")
        ax.set_aspect("equal")
        ax.set_title(f"{r['operator']}\nRMS {r['rms']:.2e}".replace("e-0", "e-"))
        ax.set_xlabel("$x$")
        ax.grid(False)
    axes[0].set_ylabel("$y$")
    top.colorbar(image, ax=list(axes), shrink=0.6, label="$|e|$")
    ax = bottom.subplots()
    for r in rows:
        colour = d2._style(r["operator"])[0]
        ax.semilogy(
            y,
            errors[r["operator"]],
            ".",
            color=colour,
            ms=1.2,
            rasterized=True,
            label=r["operator"],
        )
    ax.axvspan(0.6, 0.8, color=REFERENCE, alpha=0.12, lw=0)
    ax.set_xlabel("$y$")
    ax.set_ylabel("$|e|$ at the nodes")
    ax.set_ylim(bottom=norm.vmin)
    bottom.legend(markerscale=5, ncol=4, loc="outside lower center")
    return fig


def heat2d_spectra(data: Data):
    """§4.4: every operator's interior spectrum at one δ, and the zoom with BD4."""
    figure = data.tables("heat2d_stiff_eigenvalues.json")["spectra/figure"]
    h, delta = figure["h"], figure["delta"]
    eigs = {
        label: np.asarray(v["re"]) + 1j * np.asarray(v["im"])
        for label, v in figure["eigenvalues"].items()
    }
    fig = _figure(4.2)
    grid = fig.add_gridspec(2, len(eigs), height_ratios=(1.0, 1.6))
    for k, (label, lam) in enumerate(eigs.items()):
        ax = fig.add_subplot(grid[0, k])
        ax.plot(
            lam.real * h**2,
            lam.imag * h**2,
            ".",
            color=eig.STYLE[label][0],
            ms=1.0,
            rasterized=True,
        )
        ax.axvline(0.0, color=REFERENCE, lw=0.5)
        ax.set_title(label)
        ax.set_xlabel(r"$h^2\,\mathrm{Re}\,\lambda$")
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel(r"$h^2\,\mathrm{Im}\,\lambda$")
    zoom = fig.add_subplot(grid[1, :])
    for label in reversed(list(eigs)):
        lam = eigs[label]
        zoom.plot(
            lam.real,
            lam.imag,
            ".",
            color=eig.STYLE[label][0],
            ms=1.6,
            rasterized=True,
            label=rf"{label} (max Re ${lam.real.max():.4g}$)",
        )
    curve = bd4_stability_boundary(np.linspace(0.0, 2 * np.pi, 721)) / h
    zoom.plot(curve.real, curve.imag, "k-", lw=0.8, label="BD4 boundary, $dt = h$")
    zoom.axvline(0.0, color=REFERENCE, lw=0.5)
    span = 3.0 / h
    zoom.set_xlim(-span, 0.2 * span)
    zoom.set_ylim(-span, span)
    zoom.set_xlabel(r"$\mathrm{Re}\,\lambda$")
    zoom.set_ylabel(r"$\mathrm{Im}\,\lambda$")
    zoom.set_title(
        rf"case 1, $N = {figure['n']}$, $\delta = {delta:g}$ ($= {delta / h:.2f}\,h$)"
    )
    zoom.legend(markerscale=4, loc="upper left", fontsize=6)
    return fig


def heat2d_dominance(data: Data):
    """§4.4, H5: the seeded rows' diagonal dominance and the iterations against δ/h."""
    rows = data.tables("heat2d_stiff_eigenvalues.json")["rows"]
    fig = _figure(2.3)
    axes = fig.subplots(1, 2)
    ratios = sorted({r["ratio"] for r in rows if r["ratio"]})
    floor = min(ratios) / 4.0
    for label in eig.LABELS:
        colour, marker = eig.STYLE[label]
        line = [r for r in rows if r["label"] == label]
        x = [r["ratio"] or floor for r in line]
        axes[0].plot(x, [r["least"] for r in line], marker + "-", color=colour, ms=3)
        axes[0].plot(x, [r["median"] for r in line], marker + "--", color=colour, ms=3)
        axes[1].plot(
            x,
            [r["gmres/none"]["iterations"] for r in line],
            marker + "-",
            color=colour,
            ms=3,
        )
        axes[1].plot(
            x,
            [r["bicgstab/none"]["iterations"] for r in line],
            marker + "--",
            color=colour,
            ms=3,
        )
    n = rows[0]["n"]
    for ax, ylabel, title in (
        (axes[0], "DDR", "least (solid), median (dashed)"),
        (axes[1], "iterations", "gmres (solid), bicgstab (dashed)"),
    ):
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xlabel(rf"$\delta / h$ (leftmost: $\delta = 0$), $N = {n}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].set_yscale("log")
    handles = [
        _key(eig.STYLE[label][0], label, marker=eig.STYLE[label][1], ms=3)
        for label in eig.LABELS
    ]
    _legend(fig, handles, 5)
    return fig


def heat2d_ring_convergence(data: Data):
    """§4.8: EABE Fig. 19's twin with the seeds, at every s."""
    results = by_delta(data.tables("heat2d_ring_results.json")["convergence"])
    fig = _figure(2.9)
    ax = fig.subplots()
    colours = ring._colours(tuple(results))
    for s, rows in results.items():
        n = [r["seeds"]["n"] for r in rows]
        marker = ring.MARKERS.get(s, "s")
        ax.loglog(
            n,
            [r["construction"]["full"] for r in rows],
            marker + "--",
            color=colours[s],
            mfc="none",
            ms=3.5,
            lw=0.7,
        )
        ax.loglog(n, [r["seeds"]["full"] for r in rows], marker + "-", color=colours[s])
        ax.loglog(n, [r["seeds15"]["full"] for r in rows], ":", color=colours[s])
    first = next(iter(results.values()))
    n = np.array([r["seeds"]["n"] for r in first], dtype=float)
    ax.loglog(n, first[0]["seeds"]["full"] * (n / n[0]) ** -2.0, "k-.", lw=0.6)
    ax.set_xlabel("nodes $N$")
    ax.set_ylabel("RMS error in $u$")
    ax.set_title(r"eq. 40's ring at $\delta = 0$")
    _counts(ax)
    handles = [
        _key(colours[s], ring.power(s), marker=ring.MARKERS.get(s, "s"), ms=3)
        for s in results
    ]
    handles += [
        _key("k", "seeds"),
        _key("k", "E2.3", ls="--", marker="o", mfc="none", ms=3, lw=0.7),
        _key("k", "seeds without the flux seeds", ls=":"),
        _key("k", "$h^4$", ls="-.", lw=0.6),
    ]
    _legend(fig, handles, 5)
    return fig


def heat2d_ring_conditioning(data: Data):
    """§4.8: EABE Fig. 20's twin: exactness and the stencil systems' conditioning."""
    rows = data.tables("heat2d_ring_results.json")["conditioning"]
    zero = [r for r in rows if r["delta"] == 0.0]
    s = np.array([r["s"] for r in zero])
    fig = _figure(2.6)
    left, right = fig.subplots(1, 2)
    left.loglog(s, [r["e23-worst"] for r in zero], "o-", color=CONSTRUCTION, ms=3)
    left.loglog(s, [r["stored-worst"] for r in zero], "^-", color=NAIVE, ms=3)
    left.loglog(s, [r["seeds-worst"] for r in zero], "s-", color=AWARE, ms=3)
    deltas = sorted({r["delta"] for r in rows if r["delta"] > 0.0}, reverse=True)
    for d, alpha in zip(deltas, np.linspace(0.8, 0.35, len(deltas)), strict=True):
        line = [r for r in rows if r["delta"] == d]
        left.loglog(
            [r["s"] for r in line],
            [r["seeds-worst"] for r in line],
            "s:",
            color=AWARE,
            alpha=alpha,
            ms=3,
        )
    left.loglog(s, ring.E29_RESIDUAL * s, "k:", lw=0.6)
    left.set_xlabel("$s$")
    left.set_ylabel("worst relative residual")
    left.set_title("exactness on the matched profile")
    right.loglog(s, [r["e23-block-mean"] for r in zero], "o-", color=CONSTRUCTION, ms=3)
    right.loglog(
        s, [r["e23-system-mean"] for r in zero], "o--", color=CONSTRUCTION, ms=3
    )
    right.loglog(s, [r["block-raw-mean"] for r in zero], "s-", color=AWARE, ms=3)
    right.loglog(s, [r["block-scaled-mean"] for r in zero], "s:", color=AWARE, ms=3)
    right.loglog(s, [r["system-mean"] for r in zero], "s--", color=AWARE, ms=3)
    right.set_xlabel("$s$")
    right.set_ylabel("mean 2-norm condition number")
    right.set_title(rf"stencil systems, $\delta = 0$, $N = {int(zero[0]['n'])}$")
    for ax in (left, right):
        ax.minorticks_off()
    handles = [
        _key(CONSTRUCTION, "E2.3 (block; dashed: system)", marker="o", ms=3),
        _key(NAIVE, "seeds on the stored radii", marker="^", ms=3),
        _key(
            AWARE,
            r"seeds, $\delta = 0$ (block raw; dotted: scaled; dashed: system)",
            marker="s",
            ms=3,
        ),
        _key(
            AWARE, r"seeds, $\delta > 0$ (lighter: narrower)", ls=":", marker="s", ms=3
        ),
        _key("k", r"$1.5 \times 10^{-18}\, s$ (E2.9)", ls=":", lw=0.6),
    ]
    _legend(fig, handles, 2)
    return fig


def heat2d_ring_smooth(data: Data):
    """§4.8, §4.10: the smooth ring, the probe on the seeded rows and the far field."""
    table = data.tables("heat2d_ring_results.json")["smooth"]
    results = {}
    for key, rows in table.items():
        s, d = key.split("|")
        results[(float(s), float(d))] = rows
    deltas = sorted({d for _, d in results}, key=lambda d: (d != 0.0, -d))
    s_values = sorted({s for s, _ in results})
    styles = dict(zip(s_values, ("-", "--", ":", "-."), strict=False))
    fig = _figure(3.9)
    axes = fig.subplots(2, len(deltas), sharey="row")
    plain = ring.error_field("seeds", "seeds-plain")
    for j, d in enumerate(deltas):
        for s in s_values:
            rows = results.get((s, d))
            if not rows:
                continue
            labels = (*ring.LABELS, "seeds-warp") if d > 0.0 else ring.LABELS
            for lab in labels:
                n = [r[f"probe-{lab}"]["n"] for r in rows]
                axes[0, j].loglog(
                    n,
                    [r[f"probe-{lab}"]["seeded"] for r in rows],
                    "o" + styles[s],
                    color=ring.COLOURS[lab],
                    ms=2,
                    lw=0.7,
                )
                if f"error-{lab}" in rows[0]:
                    axes[1, j].loglog(
                        n,
                        [r[f"error-{lab}"]["far"] for r in rows],
                        "o" + styles[s],
                        color=ring.COLOURS[lab],
                        ms=2,
                        lw=0.7,
                    )
            if plain in rows[0]:
                axes[1, j].loglog(
                    [r[plain]["n"] for r in rows],
                    [r[plain]["far"] for r in rows],
                    "s" + styles[s],
                    color=AWARE,
                    mfc="none",
                    ms=3,
                    lw=0.6,
                )
        axes[0, j].set_title(f"probe, {_delta_label(d)}")
        axes[1, j].set_title(f"far field, {_delta_label(d)}")
        axes[1, j].set_visible(d > 0.0)
        for ax in axes[:, j]:
            _counts(ax)
            ax.tick_params(labelsize=6)
        axes[1, j].set_xlabel("$N$")
    axes[0, 0].set_ylabel("RMS of $Lu$, seeded rows")
    axes[1, 1].set_ylabel("RMS error, far field")
    handles = [
        _key(ring.COLOURS[lab], ring.NAMES[lab], marker="o", ms=2)
        for lab in (*ring.LABELS, "seeds-warp")
    ]
    handles += [_key("k", ring.power(s), ls=styles[s]) for s in s_values]
    handles.append(
        _key(
            AWARE,
            "seeds against the plain-built run",
            marker="s",
            mfc="none",
            ls="",
            ms=3,
        )
    )
    _legend(fig, handles, 4)
    return fig


# --- the registry --------------------------------------------------------------------


@dataclass(frozen=True)
class Figure:
    """A manuscript figure: its drawer, the results files it reads, where it belongs."""

    draw: Callable[[Data], plt.Figure]
    files: tuple[str, ...]
    section: str


_RING = ("heat2d_ring_results.json",)
_EIG = ("heat2d_stiff_eigenvalues.json",)

FIGURES: dict[str, Figure] = {
    "heat1d_stiff_knee.pdf": Figure(
        heat1d_knee, ("heat1d_stiff.json",), "stiff §2.2–§2.3"
    ),
    "heat1d_stiff_treatments.pdf": Figure(
        heat1d_treatments, ("heat1d_stiff.json",), "stiff §2.4"
    ),
    "heat1d_stiff_seeds.pdf": Figure(
        heat1d_seeds, ("heat1d_stiff.json",), "stiff §2.5"
    ),
    "heat1d_stiff_snapshot.pdf": Figure(
        heat1d_snapshot, ("heat1d_stiff.json",), "stiff §2.5"
    ),
    "heat2d_stiff_knee.pdf": Figure(
        heat2d_knee,
        (
            "heat2d_stiff_naive.json",
            "heat2d_stiff_seeds.json",
            "heat2d_stiff_seeds_jump.json",
        ),
        "stiff §4.2, §5.1",
    ),
    "heat2d_stiff_spectra.pdf": Figure(heat2d_spectra, _EIG, "stiff §4.4"),
    "heat2d_stiff_dominance.pdf": Figure(heat2d_dominance, _EIG, "stiff §4.4"),
    "heat2d_stiff_seeds.pdf": Figure(
        heat2d_seeds, ("heat2d_stiff_seeds.json",), "stiff §4.5"
    ),
    "heat2d_stiff_seeds_curved.pdf": Figure(
        heat2d_seeds_curved, ("heat2d_stiff_seeds_a0.02_sine.json",), "stiff §4.6"
    ),
    "heat2d_stiff_tangential_curved.pdf": Figure(
        heat2d_tangential_curved,
        ("heat2d_stiff_seeds_tangential_a0.02_sine.json",),
        "stiff §4.7",
    ),
    "heat2d_ring_convergence.pdf": Figure(heat2d_ring_convergence, _RING, "stiff §4.8"),
    "heat2d_ring_conditioning.pdf": Figure(
        heat2d_ring_conditioning, _RING, "stiff §4.8"
    ),
    "heat2d_ring_smooth.pdf": Figure(heat2d_ring_smooth, _RING, "stiff §4.8, §4.10"),
    "heat2d_stiff_treatments.pdf": Figure(
        heat2d_treatments, ("heat2d_stiff_treatments.json",), "stiff §4.9"
    ),
    "heat2d_stiff_treatments_curved.pdf": Figure(
        heat2d_treatments_curved,
        ("heat2d_stiff_treatments_a0.02_sine.json",),
        "stiff §4.9",
    ),
    "heat2d_stiff_snapshot.pdf": Figure(
        heat2d_snapshot, ("heat2d_stiff_snapshot.json",), "stiff §5.1"
    ),
}


def write(names: Sequence[str], out_dir: Path, data: Data) -> list[Path]:
    """Each named figure or fragment into ``out_dir``; the paths written."""
    use_print_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        path = out_dir / name
        if name in FIGURES:
            fig = FIGURES[name].draw(data)
            fig.savefig(path)
            plt.close(fig)
        else:
            path.write_text(TABLES[name].write(data))
        written.append(path)
    return written


def check(names: Sequence[str], data: Data, committed: Path = FIGURES_DIR) -> list[str]:
    """What stands between ``committed`` and a fresh draw; empty when nothing.

    Each named file must exist there and match a fresh draw byte for byte, and
    every ``*.pdf`` and ``tab_*.tex`` there must be one this script registers.
    """
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for path in write(names, Path(tmp), data):
            old = committed / path.name
            if not old.exists():
                bad.append(f"missing {path.name}")
            elif not filecmp.cmp(path, old, shallow=False):
                bad.append(f"differs {path.name}")
    present = {p.name for p in committed.glob("*.pdf")}
    present |= {p.name for p in committed.glob("tab_*.tex")}
    for name in sorted(present - set(FIGURES) - set(TABLES)):
        bad.append(f"not generated {name} (remove it, or register it)")
    return bad


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    everything = [*FIGURES, *TABLES]
    parser.add_argument("--only", nargs="+", choices=everything, metavar="FILE")
    parser.add_argument(
        "--check", action="store_true", help="draw elsewhere and compare the bytes"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    names = args.only or everything
    data = Data(args.data_dir)
    t0 = time.perf_counter()
    if args.check:
        bad = check(names, data, args.out_dir)
        for line in bad:
            print(f"  {line} ({args.out_dir})")
        print(f"{len(names) - len(bad)} of {len(names)} identical")
        return 1 if bad else 0
    for path in write(names, args.out_dir, data):
        print(f"  {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    print(f"{len(names)} files in {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
