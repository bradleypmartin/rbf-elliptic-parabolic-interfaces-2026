"""The manuscript's table fragments, from ``paper/data`` alone (E5.3, #44).

Imported by ``scripts/paper_figures.py``, which writes each ``TABLES`` entry to
``paper/figures/<name>`` and byte-compares it under ``--check``; this module has
no command line of its own. Each fragment is one booktabs ``tabular`` (no float,
no caption: the drafting tickets wrap it) headed by a ``% GENERATED`` line that
names the results files and the notes section it mirrors, so nobody edits it by
hand. The fragments mirror the notes' tables (``docs/stiff-diffusion.md``) so that
the numbers of the closing statements (§2.5's four, §5.1's snapshot, §5.3's
eleven) can be read off them: errors as ``\\sci{m}{e}`` (``paper/main.tex``) to
the notes' three figures, fitted orders to two decimals, ratios to two or three
figures. Fits are least squares of log(error) against log(h) with the rows' own
``h`` (the 2-D convention, port notes §2.10); the 1-D rates are the 1-D driver's,
``log2`` of successive errors per doubling. Everything is ASCII, since arXiv's
pdfTeX reads the fragments (``paper/make_arxiv.py``).

Every fragment fits the 360 pt text width at ``\\footnotesize`` (checked by
building them all with tectonic, E5.3); a few are wider at ``\\small``, so the
table environments that wrap them set ``\\footnotesize``. The ``%`` lines after
the GENERATED line say what the caption must (what each column is, which norm,
over which counts); they are the drafting tickets' to turn into prose.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from heat_interfaces.results_cache import float_keys

SCRIPT = "scripts/paper_figures.py"


@dataclass(frozen=True)
class Table:
    """A fragment: its writer (``data -> text``), the files it reads, its section."""

    write: Callable[[Any], str]
    files: tuple[str, ...]
    section: str


# --- numbers ------------------------------------------------------------------------


def mantissa(x: float, sig: int = 2) -> tuple[str, int]:
    """``x`` as ``(mantissa text, exponent)`` to ``sig`` figures, as ``sci`` rounds."""
    exp = math.floor(math.log10(abs(x)))
    mant = round(x / 10**exp, sig - 1)
    if abs(mant) >= 10:
        exp += 1
        mant = round(x / 10**exp, sig - 1)
    return f"{mant:.{sig - 1}f}", exp


def sci(x: float | None, sig: int = 3) -> str:
    r"""``1.598e-5`` as ``$\sci{1.60}{-5}$`` to ``sig`` figures; None as ``--``."""
    if x is None:
        return "--"
    if x == 0:
        return "$0$"
    mant, exp = mantissa(x, sig)
    return rf"$\sci{{{mant}}}{{{exp}}}$"


def fixed(x: float | None, digits: int = 2) -> str:
    """A fitted order or a rate to ``digits`` decimals, minus signs as math."""
    if x is None:
        return "--"
    text = f"{x:.{digits}f}"
    return f"${text}$" if text.startswith("-") else text


def figs(x: float | None, sig: int = 3) -> str:
    """A ratio to ``sig`` figures, plainly written from 0.001 to 10^5, else ``sci``."""
    if x is None:
        return "--"
    if x == 0 or not 1e-3 <= abs(x) < 1e5:
        return sci(x, 2)
    exp = math.floor(math.log10(abs(x)))
    digits = sig - 1 - exp
    value = round(x, digits)
    if value != 0 and math.floor(math.log10(abs(value))) > exp:  # 0.9996 -> 1.00
        digits -= 1
        value = round(x, digits)
    return f"{value:.{max(digits, 0)}f}"


def ratio(x: float | None) -> str:
    """A ratio as the notes print one: three figures from 0.1, two below, and
    whole numbers from 100 to 10^5 (``6742``, not ``6740``)."""
    if x is None:
        return "--"
    if 100 <= abs(x) < 1e5:
        return f"{x:.0f}"
    return figs(x, 3 if abs(x) >= 0.1 else 2)


def fit(h: Sequence[float], e: Sequence[float]) -> float:
    """Least-squares order of ``e`` in ``h`` (port notes §2.10)."""
    return float(np.polyfit(np.log(h), np.log(e), 1)[0])


def rates(values: Sequence[float]) -> list[float | None]:
    """The 1-D driver's rates: ``log2`` of successive values, None for the first."""
    return [None] + [math.log2(a / b) for a, b in zip(values, values[1:], strict=False)]


def delta_text(delta: float) -> str:
    return "jump" if delta == 0.0 else f"{delta:g}"


def per_h(ratio_: float) -> str:
    """``delta/h`` as the notes write it: 8, 1, 1/8, 1/64, 0."""
    if ratio_ == 0:
        return "0"
    if ratio_ >= 1:
        return f"{ratio_:g}"
    return f"1/{1 / ratio_:g}"


def power(s: float) -> str:
    return f"$10^{{{round(math.log10(s))}}}$"


def count(n: int) -> str:
    """A node count with a thin-space thousands separator from 10,000."""
    return f"{n:,}".replace(",", "{,}") if n >= 10000 else str(n)


# --- assembly -----------------------------------------------------------------------


def fragment(
    table: Table,
    colspec: str,
    head: Iterable[str],
    body: Iterable[str],
    comments: Iterable[str] = (),
) -> str:
    """The fragment text: the GENERATED line, ``comments`` as ``%`` lines (what the
    caption must say), then one booktabs tabular."""
    files = ", ".join(f"paper/data/{f}" for f in table.files)
    lines = [
        f"% GENERATED by {SCRIPT} from {files} ({table.section}); do not edit.",
        *(f"% {c}" for c in comments),
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        *head,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def cols(first: str, rest: str, gap: str | None = None) -> str:
    """A column spec without outer padding; ``gap`` fixes the space between columns
    (the default is twice ``\\tabcolsep``, 12 pt), for the wide tables."""
    if gap is None:
        return f"@{{}}{first}{rest}@{{}}"
    sep = rf"@{{\hspace{{{gap}}}}}"
    return f"@{{}}{first}" + "".join(sep + c for c in rest) + "@{}"


def row(*cells: str) -> str:
    return " & ".join(cells) + r" \\"


def group(columns: int, text: str, rule: bool = False) -> list[str]:
    """A block's heading row across the table, after a midrule unless first."""
    return ([r"\midrule"] if rule else []) + [
        rf"\multicolumn{{{columns}}}{{l}}{{\emph{{{text}}}}} \\"
    ]


def note(columns: int, text: str) -> list[str]:
    return [r"\midrule", rf"\multicolumn{{{columns}}}{{l}}{{{text}}} \\"]


def rows_by_delta(table: dict) -> dict[float, list[dict]]:
    return float_keys(table)


# --- one dimension (stiff note §2) --------------------------------------------------

# The manuscript's names (its §2 notation), not the notes' "MATLAB" and "eq. 75".
MEDIA = (("matlab", r"two-constant medium $1/9\,|\,1$"), ("eq75", "sinusoidal medium"))
H1D = "heat1d_stiff.json"
CONSTRUCTION_1D = "δ = 0 construction"  # the 1-D driver's label, a row key


def tab_1d_knee(data: Any) -> str:
    """§2.2: naive and the δ = 0 construction on the ramp problem, both media."""
    table = TABLES["tab_1d_knee.tex"]
    ramp = data.tables(H1D)["knee/ramp"]
    deltas = (0.04, 0.01, 0.0025)
    head = [
        row(
            "",
            r"\multicolumn{3}{c}{naive $D_x A D_x$}",
            r"\multicolumn{3}{c}{$\delta = 0$ construction}",
        ),
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        row("$n$", *(rf"$\delta = {d:g}$" for d in deltas * 2)),
    ]
    body = []
    for k, (medium, title) in enumerate(MEDIA):
        lines = rows_by_delta(ramp[medium])
        body += group(7, f"{title}, ramp problem at $t = 2$", rule=k > 0)
        for i, r in enumerate(lines[deltas[0]]):
            naive = [sci(lines[d][i]["naive"]) for d in deltas]
            built = [sci(lines[d][i][CONSTRUCTION_1D]) for d in deltas]
            body.append(row(str(r["n"]), *naive, *built))
        floors = [sci(lines[d][-1]["floor"]) for d in deltas]
        body.append(row("floor", "", "", "", *floors))
    constants = data.tables(H1D)["floor_constants/matlab"]
    measured = sorted({f"{c['measured']:.5f}" for c in constants})
    body += note(
        7,
        r"two-constant medium: deficit $c\,\delta$,"
        rf" $c = {'$, $'.join(measured)}$ measured,"
        rf" ${constants[0]['closed']:.5f}$ closed form",
    )
    comments = [
        "floor: the two references' difference at the nodes, at the finest n;",
        "errors are ||e||_2 / ||u||_2 against the Chebyshev reference at each delta.",
    ]
    return fragment(table, cols("r", "rrrrrr", "6pt"), head, body, comments)


def tab_1d_seeds(data: Any) -> str:
    """§2.3 (P7): the seed operator on the ramp problem at every δ, with rates."""
    table = TABLES["tab_1d_seeds.tex"]
    ramp = data.tables(H1D)["knee/ramp"]
    head = [
        row("$n$", *(rf"$\delta = {d:g}$" for d in (0, 0.04, 0.01, 0.0025))),
    ]
    body = []
    for k, (medium, title) in enumerate(MEDIA):
        lines = rows_by_delta(ramp[medium])
        body += group(5, f"{title}, ramp problem, error (rate)", rule=k > 0)
        columns = {d: [r["seeds"] for r in rows] for d, rows in lines.items()}
        rate = {d: rates(v) for d, v in columns.items()}
        for i, r in enumerate(lines[0.0]):
            cells = []
            for d in (0.0, 0.04, 0.01, 0.0025):
                q = rate[d][i]
                cells.append(
                    sci(columns[d][i]) + ("" if q is None else f" ({fixed(q)})")
                )
            body.append(row(str(r["n"]), *cells))
    comments = ["rates: log2 of successive errors, per doubling of n (the driver's)."]
    return fragment(table, cols("r", "llll", "6pt"), head, body, comments)


def tab_1d_equilibrium(data: Any) -> str:
    """§2.3 (P6), §2.4, §2.5 statement 4: the equilibrium problem's remark."""
    table = TABLES["tab_1d_equilibrium.tex"]
    tables = data.tables(H1D)
    knee = tables["knee/equilibrium"]
    comparators = tables["comparators/equilibrium"]
    head = [
        row(
            "",
            r"\multicolumn{2}{c}{$\delta = 0.01$}",
            r"\multicolumn{3}{c}{seeds}",
            "T1-FV",
        ),
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-7}",
        row(
            "$n$",
            "naive",
            "construction",
            r"$\delta = 0$",
            r"$\delta = 0.01$",
            r"$\delta = 0.0025$",
            r"$\delta = 0$",
        ),
    ]
    body = []
    for k, (medium, title) in enumerate(MEDIA):
        lines = rows_by_delta(knee[medium])
        fv = rows_by_delta(comparators[medium])
        body += group(7, f"{title}, equilibrium, $\\|e\\|_2 / \\|u\\|_2$", rule=k > 0)
        for i, r in enumerate(lines[0.0]):
            body.append(
                row(
                    str(r["n"]),
                    sci(lines[0.01][i]["naive"]),
                    sci(lines[0.01][i][CONSTRUCTION_1D]),
                    sci(lines[0.0][i]["seeds"], 2),
                    sci(lines[0.01][i]["seeds"], 2),
                    sci(lines[0.0025][i]["seeds"], 2),
                    sci(fv[0.0][i]["T1-FV"], 2),
                )
            )
    return fragment(table, "@{}rrrrrrr@{}", head, body)


COMPARATOR_ORDER = (
    ("seeds", "seeds"),
    ("T1-FV", "T1-FV"),
    ("T1 harmonic 2c", "T1, two cells"),
    ("naive", "naive"),
    ("T1 harmonic 1c", "T1, one cell"),
    ("T2 arithmetic 1c", "T2, one cell"),
    ("T0 widened m=1", "T0, $m = 1$"),
    ("T0 widened m=2", "T0, $m = 2$"),
)


def tab_1d_treatments(data: Any) -> str:
    """§2.4 (P10): the comparators on the ramp at δ = 0.0025, in ranking order."""
    table = TABLES["tab_1d_treatments.tex"]
    ramp = data.tables(H1D)["comparators/ramp"]
    counts = max(len(rows_by_delta(ramp[m])[0.0025]) for m, _ in MEDIA)
    head = [row("", rf"\multicolumn{{{counts}}}{{c}}{{$\|e\|_2 / \|u\|_2$}}")]
    body = []
    for k, (medium, title) in enumerate(MEDIA):
        rows = rows_by_delta(ramp[medium])[0.0025]
        body += group(counts + 1, f"{title}, ramp problem, $\\delta = 0.0025$", k > 0)
        ns = [str(r["n"]) for r in rows]
        body.append(row("$n$", *ns, *[""] * (counts - len(ns))))
        body.append(rf"\cmidrule(lr){{2-{counts + 1}}}")
        for label, text in COMPARATOR_ORDER:
            cells = [sci(r[label], 2) for r in rows]
            body.append(row(text, *cells, *[""] * (counts - len(cells))))
    comments = [
        "rows in the ranking of stiff note 2.5 statement 3; errors to two figures,",
        "as that statement quotes them. T1: harmonic cell mean over one or two",
        "cells; T2: arithmetic; T0: the edge widened to max(delta, m h); T1-FV:",
        "the conservative scheme with exact face conductances.",
    ]
    return fragment(table, cols("l", "r" * counts, "6pt"), head, body, comments)


def tab_1d_snapshot(data: Any) -> str:
    """§2.5: the snapshot at h = 8δ, the four operators."""
    table = TABLES["tab_1d_snapshot.tex"]
    snap = data.tables(H1D)["snapshot"]
    names = {
        "naive": "naive $D_x A D_x$",
        CONSTRUCTION_1D: r"$\delta = 0$ construction",
        "T1-FV": "T1-FV",
        "seeds": "seeds",
    }
    head = [
        row(
            "operator",
            r"$\|e\|_2 / \|u\|_2$",
            r"$\max |e|$",
            "at $x$",
            r"share within $2h$",
        )
    ]
    body = [
        row(
            names[r["label"]],
            sci(r["error"]),
            sci(r["max"]),
            f"${r['at']:.3f}$",
            f"{r['local']:.2f}",
        )
        for r in snap["rows"]
    ]
    return fragment(table, "@{}lrrrr@{}", head, body)


# --- two dimensions (stiff note §4, §5) ---------------------------------------------

LINES = {
    "naive": "naive",
    "construction": "construction",
    "construction-flat": "construction, flat",
    "direct": "direct",
    "direct-reach": "direct, seed stencils",
    "seeds": "seeds",
    "seeds-plain": "seeds, plain",
    "tangential": "tangential",
    "tangential-plain": "tangential, plain",
}


def tab_2d_references(data: Any) -> str:
    """§4.1, §4.6: the references' own accuracy, case 1 and case 2."""
    table = TABLES["tab_2d_references.tex"]
    flat = data.tables("heat2d_stiff_references.json")["references"]
    curved = data.tables("heat2d_stiff_references_a0.02_sine.json")["references"]
    case1 = {(r["delta"], r["growth"]): r for r in flat}
    deltas = list(dict.fromkeys(r["delta"] for r in flat))
    case2: dict[float, list[dict]] = {}
    for r in curved:
        case2.setdefault(r["delta"], []).append(r)
    head = [
        row(
            "",
            r"\multicolumn{4}{c}{case 1: Chebyshev in $y$}",
            r"\multicolumn{2}{c}{case 2: Fourier $\times$ Chebyshev}",
        ),
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
        row(
            r"$\delta$",
            "finer, $c = 0$",
            "finer, $c = 1$",
            r"$\sup |v_\delta - v_0|$",
            r"$\div \delta$",
            "finer in $x$",
            "finer in $y$",
        ),
    ]
    body = []
    for d in deltas:
        c0, c1 = case1[(d, 0.0)], case1.get((d, 1.0))
        c2 = case2.get(d)
        distance = c0["distance"]
        body.append(
            row(
                f"{d:g}",
                sci(c0["agreement"], 2),
                sci(c1["agreement"], 2) if c1 else "--",
                sci(distance, 2 if d == 0 else 3),
                "--" if d == 0 else f"{distance / d:.3f}",
                sci(max(r["n_x"] for r in c2), 2) if c2 else "--",
                sci(max(r["elements_check"] for r in c2), 2) if c2 else "--",
            )
        )
    e26 = next(r for r in curved if r["delta"] == 0.0 and "e26_rms" in r)
    body += note(
        7,
        rf"case 2, $\delta = 0$, vs.\ a 160{{,}}000-node jump-aware run:"
        rf" {sci(e26['e26_rms'])}",
    )
    comments = [
        "finer: the largest difference from a finer reference (case 1: in y, at",
        "c = 0 and c = 1; case 2: in x and in y, the larger over c); at delta = 0",
        "the sup |v_delta - v_0| column is the distance to the analytic solution;",
        "case 2's last line: RMS from a 160,000-node jump-aware run (port E2.6).",
    ]
    return fragment(table, cols("l", "rrrrrr", "7pt"), head, body, comments)


# Five widths of eight counts are more rows than a page holds, so each §4.2
# table is two fragments (#51): the jump and the wider edges, then the thinner.
THIN_BELOW = 0.01


def wide(d: float) -> bool:
    return d == 0 or d >= THIN_BELOW


def thin(d: float) -> bool:
    return not wide(d)


def tab_2d_knee(data: Any) -> str:
    """§4.2 (H10): the naive product against the jump's line, the wider widths."""
    return _knee(data, "tab_2d_knee.tex", wide)


def tab_2d_knee_thin(data: Any) -> str:
    """§4.2 (H10): the same, the thinner widths."""
    return _knee(data, "tab_2d_knee_thin.tex", thin)


def _knee(data: Any, name: str, keep: Callable[[float], bool]) -> str:
    table = TABLES[name]
    knee = data.tables("heat2d_stiff_naive.json")["knee"]
    elliptic = {d: r for d, r in rows_by_delta(knee["elliptic"]).items() if keep(d)}
    parabolic = rows_by_delta(knee["parabolic"])
    head = [
        row(
            "",
            "",
            "",
            r"\multicolumn{2}{c}{$\div$ jump}",
            "",
            "",
        ),
        r"\cmidrule(lr){4-5}",
        row("$N$", r"$h/\delta$", "RMS", "ell.", "par.", "par./ell.", "flux"),
    ]
    body = []
    for k, (d, rows) in enumerate(elliptic.items()):
        title = "the jump" if d == 0 else rf"$\delta = {d:g}$"
        body += group(7, title, rule=k > 0)
        for r, p in zip(rows, parabolic[d], strict=True):
            body.append(
                row(
                    count(r["n"]),
                    "--" if d == 0 else f"{r['h_over_delta']:.2f}",
                    sci(r["naive/rms"]),
                    "--" if d == 0 else ratio(r["naive/vs_jump"]),
                    "--" if d == 0 else ratio(p["naive/vs_jump"]),
                    ratio(p["naive/rms"] / r["naive/rms"]),
                    ratio(r["naive/flux"]),
                )
            )
        h = [r["h"] for r in rows]
        body.append(
            row("fit", "", fixed(fit(h, [r["naive/rms"] for r in rows])), *[""] * 4)
        )
    comments = [
        "the naive product Dx A Dx + Dy A Dy, equilibrium unless marked: RMS over",
        "all nodes; jump: its RMS over the jump's on the same nodes (ell., par.);",
        "par./ell.: the parabolic RMS over the equilibrium one; flux: the relative",
        "flux error on the first pair of rows off the edge; fit: over every count.",
    ]
    return fragment(table, cols("r", "rrrrrr", "7pt"), head, body, comments)


def tab_2d_floor(data: Any) -> str:
    """§4.2 (H10): the δ = 0 construction on its floor, the wider widths."""
    return _floor(data, "tab_2d_floor.tex", wide)


def tab_2d_floor_thin(data: Any) -> str:
    """§4.2 (H10): the same, the thinner widths."""
    return _floor(data, "tab_2d_floor_thin.tex", thin)


def _floor(data: Any, name: str, keep: Callable[[float], bool]) -> str:
    table = TABLES[name]
    naive = data.tables("heat2d_stiff_naive.json")["knee"]["elliptic"]
    knee = {d: r for d, r in rows_by_delta(naive).items() if keep(d)}
    head = [
        row(
            "$N$",
            r"$h/\delta$",
            "RMS",
            "floor",
            r"$\div$ floor",
            r"$\div$ naive",
        )
    ]
    body = []
    for k, (d, rows) in enumerate(knee.items()):
        title = "the jump: the jump-aware operator" if d == 0 else rf"$\delta = {d:g}$"
        body += group(6, title, rule=k > 0)
        for r in rows:
            body.append(
                row(
                    count(r["n"]),
                    "--" if d == 0 else f"{r['h_over_delta']:.2f}",
                    sci(r["construction/rms"]),
                    "--" if d == 0 else sci(r["floor"]),
                    "--" if d == 0 else f"{r['construction/rms'] / r['floor']:.3f}",
                    ratio(r["construction/rms"] / r["naive/rms"]),
                )
            )
    comments = [
        "the jump-aware rows reading the pieces as if delta were 0, equilibrium:",
        "RMS over all nodes; floor: the two references' difference at the nodes;",
        "naive: the naive product's RMS on the same nodes.",
    ]
    return fragment(table, cols("r", "rrrrr", "8pt"), head, body, comments)


def tab_2d_seeds(data: Any) -> str:
    """§4.5 (H4): the seeds at every width and count, the fits, the jump to 160,000."""
    table = TABLES["tab_2d_seeds.tex"]
    sweep = data.tables("heat2d_stiff_seeds.json")["sweep"]
    jump = data.tables("heat2d_stiff_seeds_jump.json")["sweep"]
    head = [
        row(
            "$N$",
            r"$\delta = 0$",
            r"$\delta = 0.04$",
            r"$0.01$",
            r"$0.005$",
            r"$0.0025$",
            r"max/min",
        )
    ]
    body = []
    for k, problem in enumerate(("elliptic", "parabolic")):
        lines = rows_by_delta(sweep[problem])
        deltas = [0.0, *sorted((d for d in lines if d > 0), reverse=True)]
        long = rows_by_delta(jump[problem])[0.0]
        title = "equilibrium" if problem == "elliptic" else "parabolic, $t = 0.1$"
        body += group(7, f"{title}, RMS error of the seeds", rule=k > 0)
        n_rows = len(lines[0.0])
        for i in range(n_rows):
            values = [lines[d][i]["seeds/rms"] for d in deltas]
            r = lines[0.0][i]
            body.append(
                row(
                    count(r["n"]),
                    *(sci(v) for v in values),
                    f"{max(values) / min(values):.2f}",
                )
            )
        for r in long[n_rows:]:
            body.append(row(count(r["n"]), sci(r["seeds/rms"]), *["--"] * 5))
        fits = [
            fixed(fit([r["h"] for r in lines[d]], [r["seeds/rms"] for r in lines[d]]))
            for d in deltas
        ]
        body.append(row("fit", *fits, ""))
        body.append(
            row(
                f"fit, {len(long)} $N$",
                fixed(fit([r["h"] for r in long], [r["seeds/rms"] for r in long])),
                *[""] * 5,
            )
        )
    comments = [
        "max/min: the largest over the smallest of the five widths at one count;",
        "fit: least squares of log(RMS) against log(h), h = 1/round(0.95 sqrt N),",
        "over 1250-40,000 nodes; fit, 8 N: the jump's over all eight counts.",
    ]
    return fragment(table, cols("r", "rrrrrr", "7pt"), head, body, comments)


def tab_2d_baselines(data: Any) -> str:
    """§4.5 (H8): every line at 40,000 nodes, equilibrium."""
    table = TABLES["tab_2d_baselines.tex"]
    lines = rows_by_delta(data.tables("heat2d_stiff_seeds.json")["sweep"]["elliptic"])
    labels = ("naive", "construction", "direct", "direct-reach", "seeds", "seeds-plain")
    names = ("naive", "constr.", "direct", "direct, 30/4", "seeds", "plain")
    head = [row(r"$\delta$", r"$h/\delta$", *names)]
    body = []
    for d, rows in lines.items():
        r = rows[-1]
        body.append(
            row(
                delta_text(d),
                "--" if d == 0 else f"{r['h_over_delta']:.2f}",
                *(sci(r[f"{label}/rms"]) for label in labels),
            )
        )
    n = next(iter(lines.values()))[-1]["n"]
    comments = [
        f"equilibrium, RMS error at N = {n} (h = {lines[0.0][-1]['h']:.4f});",
        "direct, 30/4: the direct operator on the seeds' own stencils; plain:",
        "the seed rows with plain Gaussians.",
    ]
    return fragment(table, cols("l", "rrrrrrr", "5pt"), head, body, comments)


def tab_2d_resolved(data: Any) -> str:
    """§4.5 (H8): the resolved edge, δ = 0.04, the seeds against the smooth lines."""
    table = TABLES["tab_2d_resolved.tex"]
    sweep = data.tables("heat2d_stiff_seeds.json")["sweep"]
    ell = rows_by_delta(sweep["elliptic"])[0.04]
    par = rows_by_delta(sweep["parabolic"])[0.04]
    head = [
        row(
            "$N$",
            r"$\delta / h$",
            "seeds",
            r"$\div$ direct",
            r"$\div$ direct, 30/4",
            r"$\div$ naive",
            r"$\div$ naive, par.",
        )
    ]
    body = []
    for r, p in zip(ell, par, strict=True):
        body.append(
            row(
                count(r["n"]),
                f"{1 / r['h_over_delta']:.2f}",
                sci(r["seeds/rms"]),
                ratio(r["seeds/rms"] / r["direct/rms"]),
                ratio(r["seeds/rms"] / r["direct-reach/rms"]),
                ratio(r["seeds/rms"] / r["naive/rms"]),
                ratio(p["seeds/rms"] / p["naive/rms"]),
            )
        )
    h = [r["h"] for r in ell]
    seeds = fit(h, [r["seeds/rms"] for r in ell])
    naive = fit(h, [r["naive/rms"] for r in ell])
    body += note(7, rf"fits: seeds {seeds:.2f}, naive {naive:.2f}")
    comments = [
        "delta = 0.04, equilibrium unless marked par.; the seeds' RMS over the",
        "direct operator's (on the 42/5 and on the seeds' own 30/4 stencils) and",
        f"over the naive product's; fits over {ell[0]['n']}-{ell[-1]['n']} nodes.",
    ]
    return fragment(table, cols("r", "rrrrrr", "7pt"), head, body, comments)


EIG = "heat2d_stiff_eigenvalues.json"
EIG_ROWS = "heat2d_stiff_eigenvalues_rows_n10000.json"
EIG_4900 = "heat2d_stiff_eigenvalues_spectra_n4900.json"


def _range(values: Sequence[float], text: Callable[[float], str]) -> str:
    lo, hi = text(min(values)), text(max(values))
    return lo if lo == hi else f"{lo}--{hi}"


def sci_range(values: Sequence[float], sig: int = 2) -> str:
    r"""``min``--``max`` in ``\sci``, written with one exponent when they share it."""
    (m_lo, e_lo), (m_hi, e_hi) = mantissa(min(values), sig), mantissa(max(values), sig)
    if (m_lo, e_lo) == (m_hi, e_hi):
        return sci(min(values), sig)
    if e_lo == e_hi:
        return rf"$({m_lo}$--${m_hi}) \times 10^{{{e_lo}}}$"
    return f"{sci(min(values), sig)}--{sci(max(values), sig)}"


def tab_2d_solvability(data: Any) -> str:
    """§4.4 (H5): the seeded rows' dominance, conditioning and solvers against δ/h."""
    table = TABLES["tab_2d_solvability.tex"]
    head = [
        row(
            r"$\delta / h$",
            "rows",
            "DDR least",
            "median",
            r"cond$/10^3$",
            "gmres",
            "bicgstab",
        )
    ]
    body = []
    for k, name in enumerate((EIG, EIG_ROWS)):
        rows = data.tables(name)["rows"]
        n = rows[0]["n"]
        body += group(7, f"$N = {count(n)}$, the seeds", rule=k > 0)
        for r in (r for r in rows if r["label"] == "seeds"):
            body.append(
                row(
                    per_h(r["ratio"]),
                    count(r["rows"]),
                    f"{r['least']:.3f}",
                    f"{r['median']:.3f}",
                    figs(r["cond"] / 1e3, 2),
                    str(r["gmres/none"]["iterations"]),
                    str(r["bicgstab/none"]["iterations"]),
                )
            )
        for label in ("construction", "direct", "naive"):
            line = [r for r in rows if r["label"] == label]
            body.append(
                row(
                    rf"\emph{{{label}}}",
                    _range([r["rows"] for r in line], str),
                    _range([r["least"] for r in line], lambda v: f"{v:.3f}"),
                    _range([r["median"] for r in line], lambda v: f"{v:.3f}"),
                    _range([r["cond"] / 1e3 for r in line], lambda v: figs(v, 2)),
                    _range([r["gmres/none"]["iterations"] for r in line], str),
                    _range([r["bicgstab/none"]["iterations"] for r in line], str),
                )
            )
    comments = [
        "the seeded rows' diagonal dominance ratio (least, median), the matrix's",
        "condition estimate, unpreconditioned gmres and bicgstab iterations; the",
        "italic lines: the other operators' range over delta/h. No timings: they",
        "are not reproducible from paper/data (the notes quote SuperLU's).",
    ]
    return fragment(table, cols("l", "rrrrrr", "6pt"), head, body, comments)


def tab_2d_spectra(data: Any) -> str:
    """§4.4 (H6): the interior spectra at 1600 and 4900 nodes."""
    table = TABLES["tab_2d_spectra.tex"]
    head = [
        row(
            r"$\delta$",
            r"$\delta / h$",
            "operator",
            r"$\max \mathrm{Re}\,\lambda$",
            r"$\mathrm{Re} > 0$",
            r"$h^2 \min \mathrm{Re}$",
            r"$h^2 \max |\mathrm{Im}|$",
            "BD4",
        )
    ]
    body = []
    picks = {
        EIG: lambda r: r["delta"] == 0.0 or r["label"] == "seeds",
        EIG_4900: lambda r: r["label"] in ("construction", "seeds", "seeds-plain"),
    }
    for k, (name, keep) in enumerate(picks.items()):
        rows = [r for r in data.tables(name)["spectra"] if keep(r)]
        rows.sort(key=lambda r: (r["delta"], r["label"] != "seeds"))
        n = round((1 / rows[0]["h"]) ** 2 / 0.95**2)
        n = min((1600, 2500, 4900), key=lambda m: abs(m - n))
        body += group(8, f"$N = {count(n)}$, interior eigenvalues", rule=k > 0)
        for r in rows:
            body.append(
                row(
                    f"{r['delta']:g}",
                    f"{r['delta'] / r['h']:.2f}",
                    LINES[r["label"]],
                    fixed(r["max_re"], 2),
                    str(r["positive"]),
                    fixed(r["min_re_h2"], 2),
                    f"{r['max_im_h2']:.3f}",
                    f"{r['bd4']:.3f}",
                )
            )
    return fragment(table, "@{}rrlrrrrr@{}", head, body)


FLAT = "heat2d_stiff_seeds.json"
CURVED = "heat2d_stiff_seeds_tangential_a0.02_sine.json"
SPLIT_A = "heat2d_stiff_seeds_tangential_a0.02_constant.json"
SPLIT_B = "heat2d_stiff_seeds_tangential_a0_sine.json"


def tab_2d_warp(data: Any) -> str:
    """§4.5, §4.7 (H7): the warp's factor, plain over warped Gaussians, per count."""
    table = TABLES["tab_2d_warp.tex"]
    blocks = (
        ("case 1: the seeds, equilibrium", FLAT, "elliptic", "seeds"),
        ("case 1: the seeds, parabolic", FLAT, "parabolic", "seeds"),
        ("case 2: the tangential chain, equilibrium", CURVED, "elliptic", "tangential"),
    )
    first = rows_by_delta(data.tables(FLAT)["sweep"]["elliptic"])
    counts = [r["n"] for r in first[0.0]]
    head = [row(r"$\delta$", *(count(n) for n in counts))]
    body = []
    for k, (title, name, problem, warped) in enumerate(blocks):
        lines = rows_by_delta(data.tables(name)["sweep"][problem])
        body += group(len(counts) + 1, title, rule=k > 0)
        for d, rows in lines.items():
            values = [r[f"{warped}-plain/rms"] / r[f"{warped}/rms"] for r in rows]
            body.append(row(delta_text(d), *(ratio(v) for v in values)))
    comments = [
        "the RMS error of the seed rows with plain Gaussians over that with the",
        "warped ones (above 1: the warp wins).",
    ]
    return fragment(table, cols("l", "r" * len(counts)), head, body, comments)


def tab_2d_curved(data: Any) -> str:
    """§4.6–§4.7 (H9, H15): case 2 at 40,000 nodes, and every line's fitted order."""
    table = TABLES["tab_2d_curved.tex"]
    tables = data.tables(CURVED)
    lines = rows_by_delta(tables["sweep"]["elliptic"])
    flat = {
        (r["delta"], r["n"]): r for r in tables["flat"] if r["problem"] == "elliptic"
    }
    labels = ("naive", "construction", "seeds", "tangential", "tangential-plain")
    names = ("naive", "constr.", "frozen", "tangential", "tang., plain")
    head = [row(r"$\delta$", r"$h/\delta$", *names, r"$\div$ flat")]
    body = group(8, "case 2, equilibrium, RMS error at the finest count")
    for d, rows in lines.items():
        r = rows[-1]
        f = flat[(d, r["n"])]
        body.append(
            row(
                delta_text(d),
                "--" if d == 0 else f"{r['h_over_delta']:.2f}",
                *(sci(r[f"{label}/rms"]) for label in labels),
                ratio(f["curved/tangential"] / f["flat/tangential"]),
            )
        )
    first, last = (
        next(iter(lines.values()))[0]["n"],
        next(iter(lines.values()))[-1]["n"],
    )
    body += group(8, f"fitted order, {count(first)}--{count(last)}", rule=True)
    for d, rows in lines.items():
        h = [r["h"] for r in rows]
        body.append(
            row(
                delta_text(d),
                "",
                *(fixed(fit(h, [r[f"{label}/rms"] for r in rows])) for label in labels),
                "",
            )
        )
    comments = [
        "frozen: the flat seeds along the foot point's normal; tang. / flat:",
        "the tangential chain over the flat seeds (case 1) at the same (delta, N).",
    ]
    return fragment(table, cols("l", "rrrrrrr", "5pt"), head, body, comments)


def tab_2d_curved_ratios(data: Any) -> str:
    """§4.6–§4.7: route (a) and the tangential chain over the flat seeds, per count."""
    table = TABLES["tab_2d_curved_ratios.tex"]
    counts = [
        r["n"] for r in rows_by_delta(data.tables(CURVED)["sweep"]["elliptic"])[0.0]
    ]
    head = [row(r"$\delta$", *(count(n) for n in counts))]

    def ratios(name: str, over: str, under: str) -> dict[float, list[float]]:
        out: dict[float, dict[int, float]] = {}
        for r in data.tables(name)["flat"]:
            if r["problem"] == "elliptic":
                out.setdefault(r["delta"], {})[r["n"]] = r[over] / r[under]
        return {d: [v[n] for n in counts] for d, v in out.items()}

    blocks = (
        (
            "case 2: the frozen profile $\\div$ flat seeds",
            CURVED,
            "curved/seeds",
            "flat/seeds",
        ),
        (
            "case 2: tangential $\\div$ flat seeds",
            CURVED,
            "curved/tangential",
            "flat/tangential",
        ),
        (
            "A, the curvature alone: tangential $\\div$ flat seeds",
            SPLIT_A,
            "curved/tangential",
            "flat/tangential",
        ),
        (
            "B, $\\alpha$ along the edge: tangential $\\div$ flat seeds",
            SPLIT_B,
            "curved/tangential",
            "flat/tangential",
        ),
    )
    body = []
    for k, (title, name, over, under) in enumerate(blocks):
        body += group(len(counts) + 1, title, rule=k > 0)
        for d, values in ratios(name, over, under).items():
            body.append(row(delta_text(d), *(ratio(v) for v in values)))
    curved = rows_by_delta(data.tables(CURVED)["sweep"]["elliptic"])
    flat = rows_by_delta(data.tables(FLAT)["sweep"]["elliptic"])
    body += group(
        len(counts) + 1, "case 2: tangential $\\div$ flat seeds, max norm", True
    )
    for d, rows in curved.items():
        values = [
            r["tangential/max"] / f["seeds/max"]
            for r, f in zip(rows, flat[d], strict=True)
        ]
        body.append(row(delta_text(d), *(ratio(v) for v in values)))
    sweep = curved[0.0]
    body += group(
        len(counts) + 1, "case 2, the jump: tangential $\\div$ jump-aware", True
    )
    body.append(
        row(
            "jump",
            *(ratio(r["tangential/rms"] / r["construction/rms"]) for r in sweep),
        )
    )
    return fragment(table, "@{}l" + "r" * len(counts) + "@{}", head, body)


RING = "heat2d_ring_results.json"


def tab_2d_probe(data: Any) -> str:
    """§4.6–§4.7 (H9, H15): the crossing rows on the true solution, per geometry."""
    table = TABLES["tab_2d_probe.tex"]
    geometries = (("case 2", CURVED), ("A", SPLIT_A), ("B", SPLIT_B))
    lines = (
        ("tangential", "tangential"),
        ("seeds", "frozen"),
        ("construction", "jump-aware"),
    )
    first = rows_by_delta(data.tables(CURVED)["sweep"]["elliptic"])[0.0]
    head = [
        row(
            "",
            r"$\delta$",
            "",
            f"$N = {count(first[0]['n'])}$",
            f"${count(first[-1]['n'])}$",
            "fit",
        )
    ]
    body = []
    for k, (title, name) in enumerate(geometries):
        sweep = rows_by_delta(data.tables(name)["sweep"]["elliptic"])
        if k:
            body.append(r"\midrule")
        for d, rows in sweep.items():
            for key, text in lines:
                if key == "construction" and d > 0:
                    continue  # it reads the smooth edge as a jump: not a probe
                values = [r[f"{key}/probe_crossing"] for r in rows]
                body.append(
                    row(
                        title,
                        delta_text(d),
                        text,
                        sci(values[0]),
                        sci(values[-1]),
                        fixed(fit([r["h"] for r in rows], values)),
                    )
                )
    comments = [
        "RMS of L u over the crossing rows, u the equilibrium reference at the",
        "nodes: the first and the finest count, and the least-squares order in h.",
        "A: the sine pair with alpha constant inside (curvature alone); B: flat",
        "lines with alpha varying along them.",
    ]
    return fragment(table, cols("l", "llrrr"), head, body, comments)


def tab_2d_circles(data: Any) -> str:
    """§4.7 (H14): the tangential chain on concentric circles."""
    table = TABLES["tab_2d_circles.tex"]
    circles = data.tables("heat2d_stiff_tangential.json")["tangential"]["circles"]
    head = [
        row(
            "$N$",
            "rows",
            "jump-aware",
            "frozen",
            "tangential",
            r"jump-aware $\div$ tang.",
        )
    ]
    body = [
        row(
            count(r["n"]),
            count(r["rows"]),
            sci(r["construction"]),
            sci(r["seeds"]),
            sci(r["tangential"]),
            ratio(r["construction"] / r["tangential"]),
        )
        for r in circles
    ]
    h = [r["h"] for r in circles]
    body.append(
        row(
            "fit",
            "",
            *(
                fixed(fit(h, [r[k] for r in circles]))
                for k in ("construction", "seeds")
            ),
            fixed(fit(h, [r["tangential"] for r in circles])),
            "",
        )
    )
    comments = [
        "the 0.25 and 0.35 circles about the strip's centre: RMS of L u over the",
        "crossing rows on the exact equilibrium through both.",
    ]
    return fragment(table, cols("r", "rrrrr"), head, body, comments)


def tab_2d_ring(data: Any) -> str:
    """§4.8 (H11): Fig. 19's twin as ratios to E2.3, with and without flux seeds."""
    table = TABLES["tab_2d_ring.tex"]
    convergence = float_keys(data.tables(RING)["convergence"])
    counts = [r["n"] for r in next(iter(convergence.values()))]
    fit_to = [n for n in counts if n <= 40000]
    last = count(counts[-1])
    head = [
        row(
            "",
            rf"\multicolumn{{{len(counts)}}}{{c}}{{ratio to jump-aware, $N$ =}}",
            r"\multicolumn{2}{c}{fit to 40{,}000}",
            "RMS at",
        ),
        rf"\cmidrule(lr){{2-{len(counts) + 1}}}"
        rf"\cmidrule(lr){{{len(counts) + 2}-{len(counts) + 3}}}",
        row("$s$", *(count(n) for n in counts), "j.-aware", "line", last),
    ]
    width = len(counts) + 4
    body = []
    for k, (line, title) in enumerate(
        (("seeds", "the seeds"), ("seeds15", "the seeds without the flux seeds"))
    ):
        body += group(width, f"{title}, $\\delta = 0$", rule=k > 0)
        for s, rows in convergence.items():
            kept = [r for r in rows if r["n"] in fit_to]
            h = [r["seeds"]["h"] for r in kept]
            body.append(
                row(
                    power(s),
                    *(ratio(r[line]["full"] / r["construction"]["full"]) for r in rows),
                    fixed(fit(h, [r["construction"]["full"] for r in kept])),
                    fixed(fit(h, [r[line]["full"] for r in kept])),
                    sci(rows[-1][line]["full"], 2),
                )
            )
    comments = [
        "RMS errors against the port's per-s references at delta = 0: the line",
        "over the jump-aware operator's at each count, the two fits over",
        "1250-40,000 nodes, and the line's",
        "own RMS error at the finest count.",
    ]
    return fragment(table, cols("l", "r" * (width - 1), "6pt"), head, body, comments)


def tab_2d_ring_conditioning(data: Any) -> str:
    """§4.8 (H11): EABE Fig. 20's twin, exactness and the stencil systems at δ = 0."""
    table = TABLES["tab_2d_ring_conditioning.tex"]
    rows = [r for r in data.tables(RING)["conditioning"] if r["delta"] == 0.0]
    head = [
        row(
            "",
            r"\multicolumn{2}{c}{worst residual}",
            r"\multicolumn{3}{c}{seeds: mean cond}",
            r"\multicolumn{2}{c}{jump-aware: mean cond}",
        ),
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
        row(
            "$s$", "seeds", "jump-aware", "block", "scaled", "system", "block", "system"
        ),
    ]
    body = [
        row(
            power(r["s"]),
            sci(r["seeds-worst"]),
            sci(r["e23-worst"]),
            sci(r["block-raw-mean"], 2),
            sci(r["block-scaled-mean"], 2),
            sci(r["system-mean"], 2),
            sci(r["e23-block-mean"], 2),
            sci(r["e23-system-mean"], 2),
        )
        for r in rows
    ]
    comments = [
        f"N = {int(rows[0]['n'])}, delta = 0: the worst relative residual on the",
        "matched radial profile over every seeded row, and the mean 2-norm condition",
        "numbers of the seed block (raw, columns scaled), the seed stencil system,",
        "the jump-aware operator's P block and its system.",
    ]
    return fragment(table, cols("l", "rrrrrrr", "6pt"), head, body, comments)


def tab_2d_ring_smooth(data: Any) -> str:
    """§4.8, §4.10: the smooth ring, the far-field error against the fine seed runs."""
    table = TABLES["tab_2d_ring_smooth.tex"]
    tables = data.tables(RING)
    smooth = {}
    for key, rows in tables["smooth"].items():
        s, d = (float(v) for v in key.split("|"))
        if d > 0:
            smooth[(d, s)] = rows
    pairs = sorted(smooth, key=lambda p: (-p[0], p[1]))
    width = len(pairs) + 1
    head = [
        row(r"$\delta$", *(f"{d:g}" for d, _ in pairs)),
        row("$s$", *(power(s) for _, s in pairs)),
    ]
    counts = [r["n"] for r in smooth[pairs[0]]]

    def far(label: str, i: int) -> list[str]:
        return [sci(smooth[p][i][f"error-{label}"]["far"], 2) for p in pairs]

    def fits(label: str, upto: int) -> list[str]:
        out = []
        for p in pairs:
            kept = [r[label] for r in smooth[p] if r["n"] <= upto and label in r]
            h, e = [r["h"] for r in kept], [r["far"] for r in kept]
            out.append(fixed(fit(h, e)) if len(kept) > 1 else "--")
        return out

    body = group(width, "the seeds (the diagonal rule), RMS error away from the ring")
    for i, n in enumerate(counts):
        body.append(row(count(n), *far("seeds", i)))
    body.append(row(r"fit to 20{,}000", *fits("error-seeds", 20000)))
    body.append(row(rf"fit to {count(counts[-1])}", *fits("error-seeds", counts[-1])))
    body += group(width, "the warp on every row", rule=True)
    for i, n in enumerate(counts):
        body.append(row(count(n), *far("seeds-warp", i)))
    body += group(width, "the rule over the better of warp and plain Gaussians", True)
    for i, n in enumerate(counts):
        cells = []
        for p in pairs:
            r = smooth[p][i]
            fired = r["error-seeds"].get("plain", 0) > 0
            best = min(r["error-seeds-warp"]["far"], r["error-seeds-plain"]["far"])
            cells.append(ratio(r["error-seeds"]["far"] / best) if fired else "--")
        body.append(row(count(n), *cells))
    body += group(width, f"the others at $N = {count(counts[-1])}$", rule=True)
    for label, text in (
        ("construction", "constr."),
        ("naive", "naive"),
        ("direct", "direct"),
    ):
        body.append(row(text, *far(label, -1)))
    plain = "error-seeds|seeds-plain"
    body.append(
        row(
            r"vs.\ plain-built",
            *(
                sci(smooth[p][-1][plain]["far"], 2) if plain in smooth[p][-1] else "--"
                for p in pairs
            ),
        )
    )
    body.append(row(r"\quad its fit", *fits(plain, counts[-1])))
    for key, gap in tables.get("fine-gap", {}).items():
        s, d = (float(v) for v in key.split("|"))
        body += note(
            width,
            rf"fine runs at $s = {power(s)[1:-1]}$, $\delta = {d:g}$ differ by"
            rf" {sci(gap['rms'])}",
        )
    comments = [
        "RMS error on the far field (the nodes away from the ring) against the",
        "fine seed run at 160,000 nodes. The rule: the diagonal rule, warped",
        "or plain Gaussians per row; -- where no row takes plain ones (the rule",
        "is the warp there). vs. plain-built: the seeds against the fine run",
        "built with plain Gaussians; the note: the two fine runs' RMS difference.",
    ]
    return fragment(table, cols("l", "r" * len(pairs), "7pt"), head, body, comments)


TREATMENTS = (
    "harmonic-0.5h",
    "harmonic-1h",
    "arithmetic-0.5h",
    "arithmetic-1h",
    "widened-1h",
    "widened-2h",
)
TREATMENT_NAMES = ("H $h/2$", "H $h$", "A $h/2$", "A $h$", "W $h$", "W $2h$")
JUMP_FROM = 5000
"""The count from which §4.9 reads the treatments at the jump (the coarser sets
are where the radius-h means still gain)."""

TREAT = (
    ("case 1", "heat2d_stiff_treatments.json", "seeds"),
    ("case 2", "heat2d_stiff_treatments_a0.02_sine.json", "tangential"),
)


def tab_2d_treatments(data: Any) -> str:
    """§4.9 (H12): every treatment over sampling at 40,000 nodes, equilibrium."""
    table = TABLES["tab_2d_treatments.tex"]
    head = [
        row(
            "",
            "",
            "",
            r"\multicolumn{6}{c}{treatment RMS $\div$ naive RMS}",
            "",
        ),
        r"\cmidrule(lr){4-9}",
        row(r"$\delta$", r"$h/\delta$", "naive", *TREATMENT_NAMES, "seeds"),
    ]
    body = []
    for k, (title, name, seeds) in enumerate(TREAT):
        lines = rows_by_delta(data.tables(name)["sweep"]["elliptic"])
        n = next(iter(lines.values()))[-1]["n"]
        body += group(10, f"{title}, equilibrium, $N = {count(n)}$", rule=k > 0)
        for d, rows in lines.items():
            r = rows[-1]
            body.append(
                row(
                    delta_text(d),
                    "--" if d == 0 else f"{r['h_over_delta']:.2f}",
                    sci(r["naive/rms"]),
                    *(ratio(r[f"{t}/rms"] / r["naive/rms"]) for t in TREATMENTS),
                    sci(r[f"{seeds}/rms"]),
                )
            )
    comments = [
        "H: harmonic, A: arithmetic mean of alpha over a disc of radius h/2 or h;",
        "W: the edge widened to max(delta, m h); seeds: the tangential chain on",
        "case 2.",
    ]
    return fragment(table, cols("l", "r" * 9, "6pt"), head, body, comments)


def tab_2d_treatments_best(data: Any) -> str:
    """§4.9 (H12): the best treatment's gain over sampling per count, and the seeds'."""
    table = TABLES["tab_2d_treatments_best.tex"]
    head = [
        row(
            "",
            r"\multicolumn{4}{c}{case 1}",
            r"\multicolumn{4}{c}{case 2}",
            "seeds",
        ),
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-10}",
        row("$N$", *(["ell.", "par.", "ell., max", "par., max"] * 2), "orders"),
    ]
    gains: dict[int, list[str]] = {}
    for _title, name, _seeds in TREAT:
        sweep = data.tables(name)["sweep"]
        for problem, norm in (
            ("elliptic", "rms"),
            ("parabolic", "rms"),
            ("elliptic", "max"),
            ("parabolic", "max"),
        ):
            best: dict[int, float] = {}
            for rows in rows_by_delta(sweep[problem]).values():
                for r in rows:
                    g = max(r[f"naive/{norm}"] / r[f"{t}/{norm}"] for t in TREATMENTS)
                    best[r["n"]] = max(best.get(r["n"], 0.0), g)
            for n, g in best.items():
                gains.setdefault(n, []).append(f"{g:.2f}")
    lead = {
        r["n"]: r["orders"]
        for r in data.tables(TREAT[0][1])["h12/parabolic/rms"]
        if r["delta"] == 0.0
    }
    body = [
        row(count(n), *cells, f"{lead[n]:.1f}" if n in lead else "--")
        for n, cells in gains.items()
    ]
    comments = [
        "the largest naive error over a treatment's, over every delta and all six",
        "treatments (above 1: some treatment beats sampling); the 1250-node",
        "parabolic column carries the naive product's growing mode (stiff 4.2);",
        "seeds: log10 of the best treatment over the seeds, case 1, parabolic,",
        "RMS, at the jump.",
    ]
    return fragment(table, cols("r", "r" * 9, "6pt"), head, body, comments)


def tab_2d_treatments_jump(data: Any) -> str:
    """§4.9 (H12): at the jump, each treatment over sampling, from 5000 nodes."""
    table = TABLES["tab_2d_treatments_jump.tex"]
    head = [
        row(
            "",
            r"\multicolumn{2}{c}{case 1}",
            r"\multicolumn{2}{c}{case 2}",
        ),
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        row("treatment", "equilibrium", "parabolic", "equilibrium", "parabolic"),
    ]
    names = dict(zip(TREATMENTS, TREATMENT_NAMES, strict=True))
    body = []
    for t in TREATMENTS:
        cells = []
        for _title, name, _seeds in TREAT:
            sweep = data.tables(name)["sweep"]
            for problem in ("elliptic", "parabolic"):
                jump = rows_by_delta(sweep[problem])[0.0]
                values = [
                    r[f"{t}/rms"] / r["naive/rms"] for r in jump if r["n"] >= JUMP_FROM
                ]
                cells.append(_range(values, ratio))
        body.append(row(names[t], *cells))
    comments = [
        f"at the jump, from {JUMP_FROM} nodes: the range over the counts of each",
        "treatment's RMS error over the naive product's (above 1: worse than",
        "sampling). H, A: harmonic, arithmetic disc means; W: the widened edge.",
    ]
    return fragment(table, cols("l", "rrrr"), head, body, comments)


def tab_2d_crossovers(data: Any) -> str:
    """§4.9: the h/δ at which the half-spacing disc means cross sampling."""
    table = TABLES["tab_2d_crossovers.tex"]
    deltas = (0.01, 0.005, 0.0025)
    head = [
        row(
            "",
            *(rf"\multicolumn{{2}}{{c}}{{$\delta = {d:g}$}}" for d in deltas),
        ),
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        row("", *(["H $h/2$", "A $h/2$"] * len(deltas))),
    ]
    body = []
    for title, name, _seeds in TREAT:
        tables = data.tables(name)
        for problem in ("elliptic", "parabolic"):
            cross = tables[f"crossovers/{problem}"]
            cells = []
            for d in deltas:
                for t in ("harmonic-0.5h", "arithmetic-0.5h"):
                    values = float_keys(cross[t]).get(d) or []
                    cells.append(", ".join(f"{v:.2f}" for v in values) or "--")
            body.append(row(f"{title}, {problem}", *cells))
    comments = [
        "the h/delta at which each half-spacing disc mean's RMS error crosses the",
        "naive product's (H: harmonic, A: arithmetic); -- where it never does.",
    ]
    return fragment(table, cols("l", "rrrrrr"), head, body, comments)


def tab_2d_snapshot(data: Any) -> str:
    """§5.1: the four regimes on one node set."""
    table = TABLES["tab_2d_snapshot.tex"]
    rows = data.tables("heat2d_stiff_snapshot.json")["snapshot"]
    names = {
        "naive": r"naive $D_x A D_x + D_y A D_y$",
        "construction": r"$\delta = 0$ construction",
        "direct": r"direct $\alpha \nabla^2 + \nabla \alpha \cdot \nabla$",
        "seeds": "seeds",
    }
    head = [row("operator", "RMS", r"$\max |e|$", "at $(x, y)$", r"share within $2h$")]
    body = [
        row(
            names[r["operator"]],
            sci(r["rms"]),
            sci(r["max"]),
            f"$({r['x']:.3f}, {r['y']:.3f})$",
            f"{r['near_share']:.2f}",
        )
        for r in rows
    ]
    return fragment(table, "@{}lrrrr@{}", head, body)


# --- the registry ---------------------------------------------------------------------

_TREAT_FILES = tuple(name for _t, name, _s in TREAT)

TABLES: dict[str, Table] = {
    "tab_1d_knee.tex": Table(tab_1d_knee, (H1D,), "stiff 2.2; 2.5 statement 1"),
    "tab_1d_seeds.tex": Table(tab_1d_seeds, (H1D,), "stiff 2.3; 2.5 statement 2"),
    "tab_1d_equilibrium.tex": Table(
        tab_1d_equilibrium, (H1D,), "stiff 2.3, 2.4; 2.5 statement 4"
    ),
    "tab_1d_treatments.tex": Table(
        tab_1d_treatments, (H1D,), "stiff 2.4; 2.5 statement 3"
    ),
    "tab_1d_snapshot.tex": Table(tab_1d_snapshot, (H1D,), "stiff 2.5"),
    "tab_2d_references.tex": Table(
        tab_2d_references,
        ("heat2d_stiff_references.json", "heat2d_stiff_references_a0.02_sine.json"),
        "stiff 4.1, 4.6; 5.3 statement 1",
    ),
    "tab_2d_knee.tex": Table(
        tab_2d_knee, ("heat2d_stiff_naive.json",), "stiff 4.2; 5.3 statement 2"
    ),
    "tab_2d_knee_thin.tex": Table(
        tab_2d_knee_thin, ("heat2d_stiff_naive.json",), "stiff 4.2; 5.3 statement 2"
    ),
    "tab_2d_floor.tex": Table(
        tab_2d_floor, ("heat2d_stiff_naive.json",), "stiff 4.2; 5.3 statement 3"
    ),
    "tab_2d_floor_thin.tex": Table(
        tab_2d_floor_thin,
        ("heat2d_stiff_naive.json",),
        "stiff 4.2; 5.3 statement 3",
    ),
    "tab_2d_solvability.tex": Table(
        tab_2d_solvability, (EIG, EIG_ROWS), "stiff 4.4; 5.3 statement 6"
    ),
    "tab_2d_spectra.tex": Table(
        tab_2d_spectra, (EIG, EIG_4900), "stiff 4.4; 5.3 statement 6"
    ),
    "tab_2d_seeds.tex": Table(
        tab_2d_seeds,
        ("heat2d_stiff_seeds.json", "heat2d_stiff_seeds_jump.json"),
        "stiff 4.5; 5.3 statement 4",
    ),
    "tab_2d_baselines.tex": Table(
        tab_2d_baselines, ("heat2d_stiff_seeds.json",), "stiff 4.5; 5.3 statement 4"
    ),
    "tab_2d_resolved.tex": Table(
        tab_2d_resolved, ("heat2d_stiff_seeds.json",), "stiff 4.5; 5.3 statement 4"
    ),
    "tab_2d_warp.tex": Table(
        tab_2d_warp, (FLAT, CURVED), "stiff 4.5, 4.7; 5.3 statement 7"
    ),
    "tab_2d_curved.tex": Table(
        tab_2d_curved, (CURVED,), "stiff 4.6-4.7; 5.3 statement 8"
    ),
    "tab_2d_curved_ratios.tex": Table(
        tab_2d_curved_ratios,
        (CURVED, SPLIT_A, SPLIT_B, FLAT),
        "stiff 4.6-4.7; 5.3 statement 8",
    ),
    "tab_2d_probe.tex": Table(
        tab_2d_probe, (CURVED, SPLIT_A, SPLIT_B), "stiff 4.6-4.7; 5.3 statement 8"
    ),
    "tab_2d_circles.tex": Table(
        tab_2d_circles,
        ("heat2d_stiff_tangential.json",),
        "stiff 4.7; 5.3 statement 8",
    ),
    "tab_2d_ring.tex": Table(tab_2d_ring, (RING,), "stiff 4.8; 5.3 statement 9"),
    "tab_2d_ring_conditioning.tex": Table(
        tab_2d_ring_conditioning, (RING,), "stiff 4.8; 5.3 statement 9"
    ),
    "tab_2d_ring_smooth.tex": Table(
        tab_2d_ring_smooth, (RING,), "stiff 4.8, 4.10; 5.3 statement 9"
    ),
    "tab_2d_treatments.tex": Table(
        tab_2d_treatments, _TREAT_FILES, "stiff 4.9; 5.3 statement 10"
    ),
    "tab_2d_treatments_best.tex": Table(
        tab_2d_treatments_best, _TREAT_FILES, "stiff 4.9; 5.3 statement 10"
    ),
    "tab_2d_treatments_jump.tex": Table(
        tab_2d_treatments_jump, _TREAT_FILES, "stiff 4.9; 5.3 statement 10"
    ),
    "tab_2d_crossovers.tex": Table(
        tab_2d_crossovers, _TREAT_FILES, "stiff 4.9; 5.3 statement 10"
    ),
    "tab_2d_snapshot.tex": Table(
        tab_2d_snapshot, ("heat2d_stiff_snapshot.json",), "stiff 5.1"
    ),
}
