"""Check every cache-backed number the manuscript quotes (E5.3, #44).

The manuscript quotes the notes (``docs/stiff-diffusion.md``, canonical for the
stiff-edge study), and the notes quote the documented runs, whose tables are
committed as results files under ``paper/data/`` (``scripts/paper_data.py``).
This script is the scripted half of the check that the two still agree: for
every number the notes' closing statements quote and the results files hold
or determine (stiff note §2.5's four statements, §5.1's snapshot table, §5.3's
eleven statements) it recomputes the value from the files' unrounded records
and fails if it no longer rounds to the quoted figure. Ratios, rates and fits
are recomputed as the drivers print them: 1-D rates per doubling of the node
count, ``log2`` of successive errors; 2-D fits the least-squares slope of
``log e`` against ``log h`` over exactly the counts the note names, with ``h =
1/round(0.95 √N)`` the rows' own ``h`` (port notes §2.10).

A quoted figure is a string as the notes print it: ``"2.63e-3"``, ``"0.52"``,
``"34"``. ``=`` passes when the value rounds to it, within half a unit in its
last digit; the trailing zeros of an integer figure are not significant, but
the tolerance is capped at 6 % of the figure, so a ``"200"`` cannot pass a
25 % drift (the rule of the companion wave manuscript's number check, whose
idea, not code, this is: plan D2). ``<=`` and ``>=`` are bounds at the quoted
precision (the value rounds to at most, or at least, the figure), for the
notes' "below", "up to", "at most" and "within" claims. A range such as
"0.86–0.94" is two ``=`` checks, on the smallest and the largest of the values
the note ranges over; a round-number range ("a 100–200× drop") is two bounds.

Before any number, the gate: ``paper_data.verify`` (every file the documented
run's, from its command, a clean tree, a commit in HEAD's history) must pass,
and ``paper_data.stale`` is reported, not enforced: a file whose run's code
changed since its commit may or may not have moved (the working caches key on
labels, stiff note §5.1).

Each check names where the quoted figure stands, ``stiff §5.3 (4)``: the notes
section and statement, or for the construction's own tables that §3 of the
manuscript quotes, the section and the prediction of the notes' §1.9 that the
table answers (``stiff §2.3 P4``). The drafting tickets (E5.4–E5.9, #45–#50) prepend the
manuscript section when the text quotes a number (``§6.3; stiff §5.3 (4)``)
and add a check for any cache-backed number the text quotes that is not here;
numbers the notes take from elsewhere (scratch runs, the port notes, timings)
are listed as skipped under each statement's block and carry their ``%
TRACE`` to the notes, not to a results file.

    uv run python scripts/paper_numbers.py            # one line per check
    uv run python scripts/paper_numbers.py --quiet    # failures and the summary
    uv run python scripts/paper_numbers.py --data-dir <dir>
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from heat_interfaces.results_cache import read_results  # noqa: E402
from paper_data import DATA, stale, verify  # noqa: E402

# --- the rule -----------------------------------------------------------------------


def unit(quoted: str) -> float:
    """A unit in the last significant digit of ``quoted``, capped (the docstring)."""
    q = float(quoted)
    mantissa, _, exponent = quoted.lower().partition("e")
    power = int(exponent) if exponent else 0
    digits = mantissa.lstrip("+-")
    if "." in digits:
        return 10.0 ** (power - len(digits.split(".")[1]))
    significant = digits.rstrip("0")
    zeros = len(digits) - len(significant)
    size = 10.0 ** (power + zeros)
    return min(size, 0.12 * abs(q)) if zeros else size


def rounds_to(value: float, quoted: str) -> bool:
    """True when ``value`` rounds to the figure ``quoted``."""
    return abs(value - float(quoted)) <= 0.5 * unit(quoted) + 1e-12 * abs(value)


def holds(value: float, quoted: str, kind: str) -> bool:
    if not math.isfinite(value):
        return False
    if kind == "=":
        return rounds_to(value, quoted)
    half = 0.5 * unit(quoted) + 1e-12 * abs(value)
    if kind == "<=":
        return value <= float(quoted) + half
    if kind == ">=":
        return value >= float(quoted) - half
    raise ValueError(f"kind {kind!r}")


@dataclass(frozen=True)
class Check:
    where: str
    what: str
    value: float
    quoted: str
    kind: str = "="

    @property
    def ok(self) -> bool:
        return holds(self.value, self.quoted, self.kind)

    def line(self) -> str:
        mark = "ok  " if self.ok else "FAIL"
        sign = "≈" if self.kind == "=" else self.kind
        value = f"{self.value:.6g} {sign} {self.quoted}"
        return f"{mark} {self.where:<18} {self.what}: {value}"


# --- the files ------------------------------------------------------------------------


class Files:
    """The results files of a data directory, by name, read once."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._tables: dict[str, dict] = {}

    def __call__(self, name: str, table: str) -> Any:
        if name not in self._tables:
            self._tables[name] = read_results(self.root / name)["tables"]
        return self._tables[name][table]


def key(delta: float) -> str:
    """A δ or an s as ``jsonable`` wrote it."""
    return f"{delta:g}"


def cited(where: str, *sections: str) -> str:
    """``where`` with the manuscript's sections that quote the figure in front.

    ``cited("stiff §5.3 (4)", "abstract", "§1")`` is ``"abstract, §1; stiff
    §5.3 (4)"``: the drafting tickets (E5.4–E5.9) tag each check whose figure
    the text quotes, so a failure names the sentence to revisit.
    """
    return f"{', '.join(sections)}; {where}"


def column(rows: Sequence[dict], name: str, counts: Iterable[int] | None = None):
    """``{n: row[name]}`` over ``counts`` (every row when None)."""
    wanted = None if counts is None else set(counts)
    out = {r["n"]: r[name] for r in rows if wanted is None or r["n"] in wanted}
    if wanted is not None and set(out) != wanted:
        raise KeyError(f"{name}: counts {sorted(wanted - set(out))} missing")
    return out


def at(rows: Sequence[dict], n: int) -> dict:
    """The row of count ``n``."""
    (row,) = [r for r in rows if r["n"] == n]
    return row


def fit(rows: Sequence[dict], name: str, counts: Iterable[int] | None = None) -> float:
    """The least-squares order of ``name`` in ``h`` (the 2-D drivers' ``_fit``)."""
    wanted = None if counts is None else set(counts)
    use = [r for r in rows if wanted is None or r["n"] in wanted]
    if wanted is not None and {r["n"] for r in use} != wanted:
        raise KeyError(f"{name}: counts missing from the fit")
    h = np.log([r["h"] for r in use])
    e = np.log([r[name] for r in use])
    return float(np.polyfit(h, e, 1)[0])


def rates_per_doubling(values: Sequence[float]) -> list[float]:
    """``log2`` of successive errors: the 1-D driver's rate columns."""
    return [math.log2(a / b) for a, b in zip(values, values[1:], strict=False)]


def rate_in_h(a: dict, b: dict, name: str) -> float:
    """Order per halving of ``h`` between two rows (the 2-D drivers' ``_rate``)."""
    return math.log(a[name] / b[name]) / math.log(a["h"] / b["h"])


class Book:
    """The checks as they are made: ``eq``, ``bounds`` and ``span`` append."""

    def __init__(self) -> None:
        self.checks: list[Check] = []

    def eq(self, where: str, what: str, value: float, quoted: str) -> None:
        self.checks.append(Check(where, what, float(value), quoted, "="))

    def le(self, where: str, what: str, value: float, quoted: str) -> None:
        self.checks.append(Check(where, what, float(value), quoted, "<="))

    def ge(self, where: str, what: str, value: float, quoted: str) -> None:
        self.checks.append(Check(where, what, float(value), quoted, ">="))

    def span(
        self, where: str, what: str, values: Iterable[float], low: str, high: str
    ) -> None:
        """A quoted range ``low–high``: the extremes of ``values`` round to its ends."""
        values = [float(v) for v in values]
        self.eq(where, f"{what} (least)", min(values), low)
        self.eq(where, f"{what} (largest)", max(values), high)

    def within(
        self, where: str, what: str, values: Iterable[float], low: str, high: str
    ) -> None:
        """A quoted envelope ``low–high``: every value lies within it."""
        values = [float(v) for v in values]
        self.ge(where, f"{what} (least)", min(values), low)
        self.le(where, f"{what} (largest)", max(values), high)

    def each(
        self, where: str, what: str, values: Sequence[float], quoted: Sequence[str]
    ) -> None:
        if len(values) != len(quoted):
            raise ValueError(
                f"{where} {what}: {len(values)} values, {len(quoted)} quoted"
            )
        for i, (v, q) in enumerate(zip(values, quoted, strict=True)):
            self.eq(where, f"{what} [{i}]", v, q)


# --- stiff note §2.5: what the 1-D study states -----------------------------------

ONE_D = "heat1d_stiff.json"
MATLAB_COUNTS = (50, 100, 200, 400, 800, 1600)
EQ75_COUNTS = (101, 201, 401, 801, 1601)
CONSTRUCTION_1D = "δ = 0 construction"


def knee_1d(f: Files, problem: str, medium: str, delta: float) -> list[dict]:
    """A line of the 1-D knee: ``problem`` is ``equilibrium`` or ``ramp``."""
    return f(ONE_D, f"knee/{problem}")[medium][key(delta)]


def comparators_1d(f: Files, problem: str, medium: str, delta: float) -> list[dict]:
    return f(ONE_D, f"comparators/{problem}")[medium][key(delta)]


def one_d(f: Files, b: Book) -> None:
    # (1) The knee (§2.2). h = 2δ and h = δ/2 are the counts 100 → 400 and
    # 400 → 1600 on the MATLAB grids (h = 2/(n − 1)), 101 → 401 and 401 → 1601 on
    # eq. 75's.
    w = "stiff §2.5 (1)"
    drops = []
    for medium, delta, (at_2, at_half), quoted in (
        ("matlab", 0.01, (100, 400), ("1.8e-4", "1.5e-6")),
        ("matlab", 0.0025, (400, 1600), ("4.4e-5", "2.2e-7")),
        ("eq75", 0.01, (101, 401), ("5.7e-3", "7.4e-6")),
        ("eq75", 0.0025, (401, 1601), ("1.0e-3", "1.0e-6")),
    ):
        naive = column(knee_1d(f, "ramp", medium, delta), "naive")
        what = f"naive, {medium} ramp, δ = {delta:g}"
        b.eq(w, f"{what}, h = 2δ", naive[at_2], quoted[0])
        b.eq(w, f"{what}, h = δ/2", naive[at_half], quoted[1])
        drops.append((medium, naive[at_2] / naive[at_half]))
    b.within(
        cited(w, "§1"),
        "the drop across the knee, MATLAB",
        [d for m, d in drops if m == "matlab"],
        "100",
        "200",
    )
    # E5.6 (#47): §1 had "800 to 1000 times", a round range the ratios miss at
    # its low end; §1 and §4.2 now quote the measured ones.
    b.span(
        cited(w, "§4.2"),
        "the drop across the knee, eq. 75",
        [d for m, d in drops if m == "eq75"],
        "770",
        "980",
    )
    b.each(
        cited(w, "§4.2"),
        "the drop across the knee, MATLAB 0.01, 0.0025, eq. 75 0.01, 0.0025",
        [d for _, d in drops],
        ["117", "199", "771", "977"],
    )
    for row in f(ONE_D, "floor_constants/matlab"):
        b.eq(
            cited(w, "§2.1", "§4.3"),
            f"c, closed form, δ = {row['delta']:g}",
            row["closed"],
            "8.79",
        )
        b.eq(
            cited(w, "§4.3"),
            f"c measured ÷ closed − 1 (six digits), δ = {row['delta']:g}",
            abs(row["measured"] / row["closed"] - 1),
            "0.000000",
        )
        b.eq(
            cited(w, "§4.3"),
            f"the flux's shift c δ / F₀(1) over δ, δ = {row['delta']:g}",
            row["closed"] / row["f0"],
            "0.88",
        )
        b.eq(cited(w, "§4.3"), "F₀(1), the jump's resistance", row["f0"], "10")
    # E5.4's correction: the floor is not c δ itself but first order in δ,
    # 0.73–0.75 δ at the finest count of §2.2's equilibrium table.
    b.span(
        cited(w, "§4.3"),
        "the floor over δ at 1600, MATLAB equilibrium",
        [
            at(knee_1d(f, "equilibrium", "matlab", d), 1600)["floor"] / d
            for d in (0.04, 0.01, 0.0025)
        ],
        "0.73",
        "0.75",
    )
    b.eq(
        cited(w, "§1", "§4.3"),
        "construction at 1600, δ = 0.01, equilibrium",
        at(knee_1d(f, "equilibrium", "matlab", 0.01), 1600)[CONSTRUCTION_1D],
        "0.11",
    )
    b.eq(
        w,
        "construction at 1600, δ = 0.01, ramp",
        at(knee_1d(f, "ramp", "matlab", 0.01), 1600)[CONSTRUCTION_1D],
        "0.04",
    )
    # "On its floor while h ≳ 2δ" (P3 as corrected in §2.2), the two-constant
    # medium, both problems; eq. 75's coarse rows carry the sinusoid's own
    # error on top (1.05–2.3× the floor there), so the text names the medium.
    b.within(
        cited(w, "§1", "§4.3"),
        "construction ÷ floor while h ≥ 2δ, MATLAB, both problems",
        [
            r[CONSTRUCTION_1D] / r["floor"]
            for problem in ("equilibrium", "ramp")
            for d in (0.04, 0.01, 0.0025)
            for r in knee_1d(f, problem, "matlab", d)
            if r["h"] >= 2 * d * (1 - 1e-9)
        ],
        "0.93",
        "1.00",
    )

    # (2) The seeds' rates (§2.3). Skipped: the spatial error at dt → h/8
    # (1.47e-8 … 4.5e-12, rates 3.8, 3.8, 4.0, the 0.3 % and 1.8 %) and the time
    # error's share, E4.10's roundtable scratch (§5.6); P6's "then the direct
    # solve's own growth", P4's 2e-14 in the weights is the δ = 0 row of
    # weights_vs_jump (below), the DOP853 floor's row residual and solution
    # (1e-12, 4e-11, §2.3's scratch ladder).
    w = "stiff §2.5 (2)"
    lines = {
        d: column(knee_1d(f, "ramp", "matlab", d), "seeds")
        for d in (0.0, 0.04, 0.01, 0.0025)
    }
    zero = [lines[0.0][n] for n in MATLAB_COUNTS]
    b.each(
        cited(w, "§4.4"),
        "seeds' rates, MATLAB ramp, δ = 0, 50 → 400",
        rates_per_doubling(zero[:4]),
        ["6.4", "4.2", "4.05"],
    )
    b.span(
        cited(w, "§4.4"),
        "seeds' rate 400 → 800 at every δ, MATLAB ramp",
        [math.log2(lines[d][400] / lines[d][800]) for d in lines],
        "4.0",
        "4.1",
    )
    b.eq(cited(w, "§4.4"), "seeds at 50, δ = 0", lines[0.0][50], "2.5e-6")
    b.eq(cited(w, "§4.4"), "seeds at 800, δ = 0", lines[0.0][800], "6e-12")
    b.eq(w, "seeds at 1600, δ = 0.04", lines[0.04][1600], "2e-11")
    for delta, low, high in (
        (0.0025, "0.1", "0.3"),
        (0.01, "1", "1.5"),
        (0.04, "3", "5"),
    ):
        b.span(
            cited(w, "§1", "§4.4") if delta == 0.04 else cited(w, "§4.4"),
            f"seeds off the δ = 0 line, %, δ = {delta:g}, 50–400",
            [100 * abs(lines[delta][n] / lines[0.0][n] - 1) for n in MATLAB_COUNTS[:4]],
            low,
            high,
        )
    b.le(
        w,
        "seeds at equilibrium to 400 nodes, every δ (MATLAB)",
        max(
            r["seeds"]
            for d in (0.0, 0.04, 0.01, 0.0025)
            for r in knee_1d(f, "equilibrium", "matlab", d)
            if r["n"] <= 400
        ),
        "1e-12",
    )
    jump = [r for r in f(ONE_D, "weights_vs_jump/matlab") if r["ratio"] == 0.0]
    b.eq(
        cited(w, "§3.2"),
        "seed weights against E1.2's at δ = 0 (P4)",
        jump[0]["difference"],
        "7e-16",
    )
    functions = f(ONE_D, "seed_functions")["rows"]
    (limit,) = [r for r in functions if r["ratio"] == 0.0]
    b.eq(
        cited(w, "§3.2"),
        "seed functions against E1.2's at δ = 0",
        max(limit["vs_jump"]),
        "3e-15",
    )
    eq75 = column(knee_1d(f, "ramp", "eq75", 0.0), "seeds")
    eq75_rates = rates_per_doubling([eq75[n] for n in EQ75_COUNTS])
    b.eq(cited(w, "§4.4"), "eq. 75 seeds' first rate, δ = 0", eq75_rates[0], "2.7")
    b.eq(cited(w, "§4.4"), "eq. 75 seeds' last rate, δ = 0", eq75_rates[-1], "3.75")
    b.eq(cited(w, "§4.4"), "eq. 75 seeds at 101, δ = 0", eq75[101], "1.3e-4")
    b.eq(cited(w, "§4.4"), "eq. 75 seeds at 1601, δ = 0", eq75[1601], "1.4e-8")
    wide = column(knee_1d(f, "ramp", "eq75", 0.04), "seeds")
    b.each(
        cited(w, "§4.4"),
        "eq. 75 seeds, δ = 0.04, 101–801",
        [wide[n] for n in EQ75_COUNTS[:4]],
        ["2.6e-8", "1.5e-9", "1.1e-10", "2e-11"],
    )
    b.each(
        cited(w, "§4.4"),
        "eq. 75 seeds' rates, δ = 0.04, 101–401",
        rates_per_doubling([wide[n] for n in EQ75_COUNTS[:3]]),
        ["4.1", "3.8"],
    )
    references = f(ONE_D, "references")
    # Corrected by E5.3: the notes had 2e-12 (δ = 0.0025's); the worst is δ =
    # 0.01's 4.8e-12, §2.1's own 5e-12. Round-off, quoted to one figure.
    for medium, quoted in (("matlab", "5e-12"), ("eq75", "6e-11")):
        b.eq(
            cited(w, "§4.1"),
            f"the references' own error, {medium} (worst agreement)",
            max(r["agreement"] for r in references if r["medium"] == medium),
            quoted,
        )

    # (3) The comparator ranking (§2.4). Skipped: the max-norm reading of the
    # two-cell harmonic mean (not in the results file; §2.4's scratch).
    w = "stiff §2.5 (3)"
    row = at(comparators_1d(f, "ramp", "matlab", 0.0025), 200)
    for label, quoted in (
        ("seeds", "1.6e-9"),
        ("T1-FV", "8.0e-6"),
        ("T1 harmonic 2c", "8.9e-5"),
        ("naive", "7.9e-4"),
        ("T1 harmonic 1c", "4.0e-4"),
        ("T2 arithmetic 1c", "4.6e-4"),
        ("T0 widened m=1", "2.2e-3"),
        ("T0 widened m=2", "5.0e-3"),
    ):
        # §4.4 quotes the seeds and the naive operator at h = 4δ too.
        quoting = ("§4.4", "§4.5") if label in ("seeds", "naive") else ("§4.5",)
        b.eq(
            cited(w, *quoting),
            f"{label} at 200, δ = 0.0025, MATLAB ramp",
            row[label],
            quoted,
        )
    fv = column(comparators_1d(f, "ramp", "matlab", 0.0), "T1-FV")
    b.eq(cited(w, "§4.5"), "T1-FV at 50, MATLAB ramp, δ = 0", fv[50], "1.31e-4")
    b.eq(cited(w, "§4.5"), "T1-FV at 1600, MATLAB ramp, δ = 0", fv[1600], "1.26e-7")
    b.eq(
        cited(w, "§4.5"),
        "T1-FV's order 50 → 1600, δ = 0",
        math.log2(fv[50] / fv[1600]) / 5,
        "2.00",
    )
    # δ = 0's is LITERATURE.md §6b's caveat, which the abstract and §1 quote.
    for delta, quoted, where in (
        (0.0, "801", cited(w, "§1", "§4.5", "§7.1", "§7.3")),
        (0.0025, "201", cited(w, "§4.5")),
    ):
        rows = comparators_1d(f, "ramp", "eq75", delta)
        b.eq(
            where,
            f"eq. 75: the first count where the seeds lead T1-FV, δ = {delta:g}",
            min(r["n"] for r in rows if r["seeds"] < r["T1-FV"]),
            quoted,
        )
    two_cells = column(comparators_1d(f, "ramp", "matlab", 0.0), "T1 harmonic 2c")
    b.eq(
        cited(w, "§4.5"),
        "T1 (two cells)' last rate, MATLAB ramp, δ = 0",
        rates_per_doubling([two_cells[n] for n in MATLAB_COUNTS])[-1],
        "1.5",
    )

    # (4) The elliptic remark (§1.8, §2.2, §2.4). The manuscript's remark in §4.5
    # quotes every figure of it.
    w = cited("stiff §2.5 (4)", "§4.5")
    seeds = column(knee_1d(f, "equilibrium", "matlab", 0.01), "seeds")
    b.eq(w, "seeds at equilibrium, 50 nodes, δ = 0.01", seeds[50], "1.9e-14")
    b.eq(w, "seeds at equilibrium, 1600 nodes, δ = 0.01", seeds[1600], "1.7e-11")
    # T1-FV's "1e-14 → 6.5e-12" is §2.4's envelope over δ (P10): the least at
    # 50 nodes, the largest at 1600 (δ = 0's).
    fv = [
        comparators_1d(f, "equilibrium", "matlab", d) for d in (0.0, 0.04, 0.01, 0.0025)
    ]
    b.eq(
        w,
        "T1-FV at equilibrium, 50, least over δ",
        min(at(r, 50)["T1-FV"] for r in fv),
        "1e-14",
    )
    b.eq(
        w,
        "T1-FV at equilibrium, 1600, largest over δ",
        max(at(r, 1600)["T1-FV"] for r in fv),
        "6.5e-12",
    )
    for delta, quoted in ((0.0, ("9.7e-5", "1.2e-8")), (0.0025, ("5.0e-6", "3.6e-10"))):
        eq75 = column(knee_1d(f, "equilibrium", "eq75", delta), "seeds")
        b.eq(
            w, f"eq. 75 seeds at equilibrium, 101, δ = {delta:g}", eq75[101], quoted[0]
        )
        b.eq(
            w,
            f"eq. 75 seeds at equilibrium, 1601, δ = {delta:g}",
            eq75[1601],
            quoted[1],
        )
    fv = column(comparators_1d(f, "equilibrium", "eq75", 0.0), "T1-FV")
    b.eq(w, "eq. 75 T1-FV at equilibrium, 101, δ = 0", fv[101], "3e-14")
    b.eq(w, "eq. 75 T1-FV at equilibrium, 1601, δ = 0", fv[1601], "1.4e-11")
    naive = column(knee_1d(f, "equilibrium", "matlab", 0.01), "naive")
    b.eq(w, "naive at equilibrium, h = 4δ (50), δ = 0.01", naive[50], "7.6e-3")
    b.eq(w, "naive at equilibrium, h = δ/8 (1600), δ = 0.01", naive[1600], "9.3e-9")


def seeds_1d(f: Files, b: Book) -> None:
    """The construction's own measurements that §3 of the manuscript quotes.

    Stiff §2.3's P4–P7 tables, at the stencil of the seed-function figure (P4's
    window: 200 nodes, the ``1/9 | 1`` edge half a cell right of ``x_e``) and at
    eq. 75's edge on a node (201 nodes). E5.5 (#46) added them. Skipped, and
    traced to the notes: the constant-α condition number 23.5 and the 24 of
    δ ≥ 10 h, the double-cross (7.6e-15; 22 %, 1.9 %, 0.18 %), E1.2's h³
    moments, the separable solution's truncation rates, eq. 75's first order in
    h, the march floor's solution error and every count and timing of the
    march (tests and scratch runs, not the results file).
    """
    # P4: the weights against E1.2's, first order in δ/h. §2.3's table has
    # 0.465 at δ/h = 1/2, which is 46 %, not §1.4's scratch 47 %.
    w = "stiff §2.3 P4"
    rows = {r["ratio"]: r for r in f(ONE_D, "weights_vs_jump/matlab")}
    ratios = (1.0, 0.5, 0.1, 0.01, 0.001)
    b.each(
        cited(w, "§3.2"),
        "seed weights off E1.2's, %, δ/h = 1 … 0.001",
        [100 * rows[r]["difference"] for r in ratios],
        ["84", "46", "11", "1.1", "0.11"],
    )
    eq75 = {r["ratio"]: r for r in f(ONE_D, "weights_vs_jump/eq75")}
    b.eq(
        cited(w, "§3.2"),
        "eq. 75 weights off E1.2's at δ = 0, edge on a node, 201",
        eq75[0.0]["difference"],
        "1.25",
    )
    b.span(
        cited(w, "§3.2"),
        "eq. 75 weights off E1.2's, δ/h = 0.001 … 1",
        [eq75[r]["difference"] for r in ratios],
        "1.26",
        "2.24",
    )
    # P5: the stencil solve's condition number in ξ, as built.
    w = "stiff §2.3 P5"
    b.eq(cited(w, "§3.1"), "cond A at δ = 0, P4's window", rows[0.0]["cond"], "90")
    b.eq(cited(w, "§3.1"), "cond A at δ = h/2, P4's window", rows[0.5]["cond"], "138")
    # P6: the seeded rows' residual h² max |L_h u| on the exact equilibrium at 200
    # nodes: rounding while the march crosses the edge in a few steps, the
    # march's floor below.
    w = "stiff §2.3 P6"
    residuals = {r["ratio"]: r["seeds"] for r in f(ONE_D, "row_residuals/matlab")}
    b.le(
        cited(w, "§3.1"),
        "seeded rows' residual, δ/h ≥ 0.1",
        max(residuals[r] for r in (1.0, 0.5, 0.1)),
        "2.5e-16",
    )
    b.eq(
        cited(w, "§3.1"),
        "seeded rows' residual, δ/h = 0.01",
        residuals[0.01],
        "2.4e-13",
    )
    b.eq(
        cited(w, "§3.1"),
        "seeded rows' residual, δ/h = 0.001",
        residuals[0.001],
        "5.8e-13",
    )
    # P6–P7 on eq. 75 at δ = 0: E1.2's operator (the construction's column at δ =
    # 0) over the seed operator, 101–1601 nodes.
    for problem, high, where, quoting in (
        ("equilibrium", "44", "stiff §2.3 P6", ("§3.2",)),
        ("ramp", "38", "stiff §2.3 P7", ("§3.2", "§4.4")),
    ):
        line = knee_1d(f, problem, "eq75", 0.0)
        b.span(
            cited(where, *quoting),
            f"eq. 75 E1.2 over seeds at δ = 0, {problem}",
            [r[CONSTRUCTION_1D] / r["seeds"] for r in line],
            "6",
            high,
        )


PROBLEMS_1D = ("equilibrium", "ramp")
MIDDLE_1D = ("naive", "T1 harmonic 1c", "T2 arithmetic 1c")
NODAL_1D = ("T1 harmonic 1c", "T1 harmonic 2c", "T2 arithmetic 1c")


def results_1d(f: Files, b: Book) -> None:
    """The 1-D study's own tables that §4 of the manuscript quotes.

    Keyed by the prediction of the notes' §1.9 that each table answers: P2, the
    knee, and P3, the construction (stiff §2.2); P7 and P9, the seeds (§2.3);
    P10, the treatments (§2.4); and the snapshot (§2.5). E5.6 (#47) added them.
    Skipped, and traced to the notes: the construction's eigenvalues at +248 and
    +8.8 on 49 and 53 nodes (not in the results file), the spatial error at
    dt → h/8 and the snapshot seeds' 1.47e-8 (E4.10's roundtable scratch, §5.6),
    the two-cell mean's max-norm error (§2.4's scratch), the rows seeded with
    the reach widened (P6–P7 scratch), the quadrature against the closed form
    (tests), and the medium's α′/α = 25 (arithmetic).
    """
    # P2: the naive operator's knee (§4.2).
    w = cited("stiff §2.2 P2", "§4.2")
    for medium, low, high, skip in (
        ("matlab", "0.98", "1.02", 0),
        ("eq75", "1.00", "1.01", 1),
    ):
        b.span(
            w,
            f"naive's rates at δ = 0, {medium}, both problems",
            [
                q
                for p in PROBLEMS_1D
                for q in rates_per_doubling(
                    [r["naive"] for r in knee_1d(f, p, medium, 0.0)]
                )[skip:]
            ],
            low,
            high,
        )
    ratios, rates = [], []
    for medium in ("matlab", "eq75"):
        for p in PROBLEMS_1D:
            jump = column(knee_1d(f, p, medium, 0.0), "naive")
            for d in (0.04, 0.01, 0.0025):
                rows = [
                    r for r in knee_1d(f, p, medium, d) if r["h"] >= 4 * d * (1 - 1e-9)
                ]
                ratios += [r["naive"] / jump[r["n"]] for r in rows]
                rates += rates_per_doubling([r["naive"] for r in rows])
    b.span(w, "naive ÷ the jump's while h ≥ 4δ", ratios, "0.43", "1.05")
    b.span(w, "naive's rates while h ≥ 4δ", rates, "0.9", "1.6")
    # The doubling that carries the knee: h = δ → δ/2 with the edge mid-cell,
    # h = 2δ → δ with the edges on nodes.
    for medium, pairs, low, high in (
        ("matlab", {0.01: (200, 400), 0.0025: (800, 1600)}, "5.9", "6.8"),
        ("eq75", {0.01: (101, 201), 0.0025: (401, 801)}, "5.2", "5.4"),
    ):
        steep = []
        for p in PROBLEMS_1D:
            for d, (n1, n2) in pairs.items():
                naive = column(knee_1d(f, p, medium, d), "naive")
                steep.append(math.log2(naive[n1] / naive[n2]))
        b.span(w, f"naive's steepest doubling, {medium}", steep, low, high)
    below = [
        q
        for medium in ("matlab", "eq75")
        for p in PROBLEMS_1D
        for d in (0.04, 0.01, 0.0025)
        for q in rates_per_doubling(
            [r["naive"] for r in knee_1d(f, p, medium, d) if r["h"] <= 0.51 * d]
        )
    ]
    b.span(w, "naive's rates from h = δ/2", below, "3.95", "4.23")

    # P3: the δ = 0 construction (§4.3).
    w = cited("stiff §2.2 P3", "§4.3")
    rows = knee_1d(f, "equilibrium", "matlab", 0.0025)
    b.le(
        w,
        "construction ÷ floor − 1, MATLAB equilibrium, δ = 0.0025, 50–200",
        max(abs(r[CONSTRUCTION_1D] / r["floor"] - 1) for r in rows if r["n"] <= 200),
        "0.0005",
    )
    b.eq(
        w,
        "the dip at h = 2δ: construction ÷ floor, 400, δ = 0.0025",
        at(rows, 400)[CONSTRUCTION_1D] / at(rows, 400)["floor"],
        "0.93",
    )
    widths = (0.04, 0.01, 0.0025)
    b.span(
        w,
        "the ramp floor over δ at 1600, MATLAB",
        [at(knee_1d(f, "ramp", "matlab", d), 1600)["floor"] / d for d in widths],
        "0.29",
        "0.29",
    )
    b.each(
        w,
        "eq. 75's equilibrium floor over δ at 1601",
        [at(knee_1d(f, "equilibrium", "eq75", d), 1601)["floor"] / d for d in widths],
        ["0.76", "1.20", "1.44"],
    )
    constants = {r["delta"]: r for r in f(ONE_D, "floor_constants/eq75")}
    b.each(
        w,
        "eq. 75's measured deficit ÷ its limit",
        [constants[d]["measured"] / constants[d]["closed"] for d in widths],
        ["0.44", "0.74", "0.91"],
    )
    b.eq(w, "eq. 75's limit, the two edges' c", constants[0.04]["closed"], "20.7")
    b.span(
        w,
        "eq. 75 construction ÷ floor while h ≥ 2δ, both problems",
        [
            r[CONSTRUCTION_1D] / r["floor"]
            for p in PROBLEMS_1D
            for d in widths
            for r in knee_1d(f, p, "eq75", d)
            if r["h"] >= 2 * d * (1 - 1e-9)
        ],
        "1.05",
        "2.3",
    )
    # Resolved: the error follows h/δ, not δ (the counts at h ≈ δ/2).
    half = [
        at(knee_1d(f, "equilibrium", "matlab", d), n)
        for d, n in ((0.04, 100), (0.01, 400), (0.0025, 1600))
    ]
    b.span(
        w,
        "construction at h ≈ δ/2, MATLAB equilibrium, every δ",
        [r[CONSTRUCTION_1D] for r in half],
        "0.0685",
        "0.0687",
    )
    b.each(
        w,
        "… over its floor, δ = 0.04, 0.01, 0.0025",
        [r[CONSTRUCTION_1D] / r["floor"] for r in half],
        ["2.4", "9.2", "37"],
    )
    for medium, n, quoted in (
        ("matlab", 1600, ("0.125", "0.048")),
        ("eq75", 1601, ("0.22", "0.22")),
    ):
        for p, q in zip(PROBLEMS_1D, quoted, strict=True):
            b.eq(
                w,
                f"construction at {n}, δ = 0.04, {medium} {p}",
                at(knee_1d(f, p, medium, 0.04), n)[CONSTRUCTION_1D],
                q,
            )
    b.span(
        w,
        "the construction's rates at δ = 0 on eq. 75 (E1.2's operator)",
        [
            q
            for p in PROBLEMS_1D
            for q in rates_per_doubling(
                [r[CONSTRUCTION_1D] for r in knee_1d(f, p, "eq75", 0.0)]
            )
        ],
        "3.88",
        "4.12",
    )

    # P7: the seeds on the ramp problem (§4.4).
    w = cited("stiff §2.3 P7", "§4.4")
    ramp = knee_1d(f, "ramp", "matlab", 0.0025)
    b.eq(w, "h/δ at 50, δ = 0.0025", at(ramp, 50)["h"] / 0.0025, "16")
    wide = knee_1d(f, "ramp", "matlab", 0.04)
    b.eq(w, "h/δ at 400, δ = 0.04", at(wide, 400)["h"] / 0.04, "0.125")
    b.eq(
        w,
        "construction at 200, δ = 0.0025, MATLAB ramp",
        at(ramp, 200)[CONSTRUCTION_1D],
        "7.2e-4",
    )
    b.eq(
        w,
        "eq. 75 seeds at 201, δ = 0.01 (the plain rows)",
        at(knee_1d(f, "ramp", "eq75", 0.01), 201)["seeds"],
        "1.4e-8",
    )
    thin = rates_per_doubling([r["seeds"] for r in knee_1d(f, "ramp", "eq75", 0.0025)])
    b.eq(w, "eq. 75 seeds' first rate, δ = 0.0025", thin[0], "2.7")
    b.eq(w, "eq. 75 seeds' last rate, δ = 0.0025", thin[-1], "4.0")

    # P9: the seed operator's interior spectrum (§4.4).
    w = cited("stiff §2.3 P9", "§4.4")
    spectra = [r for m in ("matlab", "eq75") for r in f(ONE_D, f"seed_spectra/{m}")]
    b.eq(w, "largest |Im λ|, every grid", max(r["imag"] for r in spectra), "0")
    b.le(w, "largest Re λ, every grid", max(r["max_real"] for r in spectra), "0")
    b.eq(
        w,
        "min Re λ h² off FD4's −16/3, %, MATLAB 50",
        max(
            100 * abs(r["extreme"] / (-16 / 3) - 1)
            for r in f(ONE_D, "seed_spectra/matlab")
            if r["n"] == 50
        ),
        "0.6",
    )
    b.span(
        w,
        "largest Re λ, eq. 75 at 49 and 53, δ = 0",
        [
            r["max_real"]
            for r in f(ONE_D, "seed_spectra/eq75")
            if r["n"] in (49, 53) and r["delta"] == 0.0
        ],
        "-2.05",
        "-2.05",
    )

    # P10: the treatments (§4.5).
    w = cited("stiff §2.4 P10", "§4.5")
    disorder, spread = 0, []
    for d in (0.0,) + widths:
        for r in comparators_1d(f, "ramp", "matlab", d):
            if r["h"] < 4 * d * (1 - 1e-9):
                continue
            middle = [r[k] for k in MIDDLE_1D]
            ranked = (
                r["seeds"] < r["T1-FV"] < r["T1 harmonic 2c"] < min(middle)
                and max(middle) < r["T0 widened m=1"] < r["T0 widened m=2"]
            )
            disorder += not ranked
            spread.append(max(middle) / min(middle))
    b.eq(w, "MATLAB ramp rows out of the ranking, h ≥ 4δ", disorder, "0")
    b.eq(w, "… the middle three's spread, largest", max(spread), "2.02")
    lead = {
        r["n"]: r["T1-FV"] / r["seeds"]
        for r in comparators_1d(f, "ramp", "matlab", 0.0)
    }
    b.eq(w, "T1-FV ÷ seeds at 50, MATLAB ramp, δ = 0", lead[50], "53")
    b.eq(w, "T1-FV ÷ seeds at 400, MATLAB ramp, δ = 0", lead[400], "2.1e4")
    eq75 = comparators_1d(f, "ramp", "eq75", 0.0)
    b.span(
        w,
        "eq. 75 naive ÷ T1 (one cell), ramp, δ = 0",
        [r["naive"] / r["T1 harmonic 1c"] for r in eq75],
        "9",
        "35",
    )
    r = at(comparators_1d(f, "ramp", "eq75", 0.01), 101)
    b.eq(w, "eq. 75 T1-FV ÷ seeds at 101, δ = 0.01", r["T1-FV"] / r["seeds"], "440")
    fv = {
        d: column(comparators_1d(f, "ramp", "matlab", d), "T1-FV")
        for d in (0.0,) + widths
    }
    b.eq(
        w,
        "T1-FV at δ = 0.01, 0.0025 off δ = 0's, %, largest",
        max(
            100 * abs(fv[d][n] / fv[0.0][n] - 1) for d in (0.01, 0.0025) for n in fv[d]
        ),
        "1.2",
    )
    b.span(
        w,
        "T1-FV at δ = 0.04 above δ = 0's, %",
        [100 * (fv[0.04][n] / fv[0.0][n] - 1) for n in fv[0.04]],
        "7",
        "7",
    )
    capped = []
    for d in widths:
        rows = [r for r in comparators_1d(f, "ramp", "matlab", d) if r["h"] <= 0.26 * d]
        for k in NODAL_1D:
            capped += rates_per_doubling([r[k] for r in rows])
    b.span(
        w, "nodal treatments' rates for h ≤ δ/4, MATLAB ramp", capped, "1.95", "2.00"
    )
    r = at(comparators_1d(f, "ramp", "matlab", 0.04), 1600)
    for label, quoted in (
        ("naive", "2.05e-10"),
        ("T1 harmonic 1c", "1.62e-7"),
        ("T2 arithmetic 1c", "5.76e-7"),
        ("T1 harmonic 2c", "6.50e-7"),
        ("T1-FV", "1.35e-7"),
    ):
        b.eq(w, f"{label} at 1600, δ = 0.04, MATLAB ramp", r[label], quoted)
    b.span(
        w,
        "… the nodal treatments over T1-FV",
        [r[k] / r["T1-FV"] for k in NODAL_1D],
        "1.2",
        "4.8",
    )
    # T0 (m = 2) on its own floor, the widened medium's exact equilibrium.
    off = []
    for medium in ("matlab", "eq75"):
        for d in widths:
            built = {r["n"]: r for r in comparators_1d(f, "equilibrium", medium, d)}
            off += [
                abs(built[r["n"]]["T0 widened m=2"] / r["2"] - 1)
                for r in f(ONE_D, f"widened_floors/{medium}/{key(d)}")
                if r["h"] >= d * (1 - 1e-9)
            ]
    b.le(w, "T0 (m = 2) ÷ its floor − 1 wherever h ≥ δ", max(off), "0.001")
    jump = comparators_1d(f, "ramp", "matlab", 0.0)
    for label, low, high in (
        ("T0 widened m=1", "2.9", "2.9"),
        ("T0 widened m=2", "5.7", "5.8"),
    ):
        b.span(
            w,
            f"{label} ÷ naive, MATLAB ramp, δ = 0",
            [r[label] / r["naive"] for r in jump],
            low,
            high,
        )
    # The one-cell window ends on the jump, so the treated A is the naive one; the
    # errors differ by the solves' rounding, 1e-13 on errors of 1e-4 at most.
    b.le(
        w,
        "T1 (one cell) and T2 off naive at δ = 0 mid-cell, both problems",
        max(
            abs(r[k] / r["naive"] - 1)
            for p in PROBLEMS_1D
            for r in comparators_1d(f, p, "matlab", 0.0)
            for k in ("T1 harmonic 1c", "T2 arithmetic 1c")
        ),
        "1e-9",
    )
    b.span(
        w,
        "T1 (two cells)' rates, MATLAB ramp, δ = 0",
        rates_per_doubling([r["T1 harmonic 2c"] for r in jump]),
        "1.44",
        "1.50",
    )

    # The snapshot (§4.6): the ramp problem at h = 8δ.
    w = cited("stiff §2.5 snapshot", "§4.6")
    snap = f(ONE_D, "snapshot")
    rows = {r["label"]: r for r in snap["rows"]}
    b.eq(w, "h/δ", snap["h"] / snap["delta"], "8.08")
    order = ("naive", CONSTRUCTION_1D, "T1-FV", "seeds")
    for label, quoted in zip(
        order, ("2.04e-3", "7.12e-4", "3.20e-5", "2.95e-8"), strict=True
    ):
        b.eq(w, f"{label}'s error", rows[label]["error"], quoted)
    b.each(
        w,
        "each over the next",
        [
            rows[a]["error"] / rows[c]["error"]
            for a, c in zip(order, order[1:], strict=False)
        ],
        ["2.9", "22", "1100"],
    )
    b.eq(
        w,
        "naive's largest error, in h from the edge",
        rows["naive"]["at"] / snap["h"],
        "-1.5",
    )
    b.eq(
        cited("stiff §2.5 snapshot", "§4.6", "§6.8"),
        "naive's ‖e‖² beyond 2h, %",
        100 * (1 - rows["naive"]["local"]),
        "84",
    )
    b.eq(
        w,
        "the construction's largest error, in h from the edge",
        rows[CONSTRUCTION_1D]["at"] / snap["h"],
        "-0.5",
    )
    b.eq(w, "T1-FV's largest error at x", rows["T1-FV"]["at"], "-0.62")


# --- stiff note §5.1 and §5.3: what the 2-D study states ---------------------------

SEEDS = "heat2d_stiff_seeds.json"
JUMP = "heat2d_stiff_seeds_jump.json"
NAIVE = "heat2d_stiff_naive.json"
CURVED = "heat2d_stiff_seeds_a0.02_sine.json"
TANGENTIAL = "heat2d_stiff_seeds_tangential_a0.02_sine.json"
STENCILS = "heat2d_stiff_stencils.json"
SPLIT = {
    "A": "heat2d_stiff_seeds_tangential_a0.02_constant.json",
    "B": "heat2d_stiff_seeds_tangential_a0_sine.json",
}
RING = "heat2d_ring_results.json"
TREATMENTS = {
    "case 1": "heat2d_stiff_treatments.json",
    "case 2": "heat2d_stiff_treatments_a0.02_sine.json",
}
NAIVE_SEEDS = ("heat2d_stiff_naive_seed1.json", "heat2d_stiff_naive_seed2.json")
CURVED_WIDTHS = (0.0, 0.01, 0.005, 0.0025)
COUNTS_40K = (1250, 2500, 5000, 10000, 20000, 40000)
COUNTS_160K = (*COUNTS_40K, 80000, 160000)
WIDTHS = (0.04, 0.01, 0.005, 0.0025)
PROBLEMS = ("elliptic", "parabolic")


def sweep(f: Files, name: str, problem: str, delta: float) -> list[dict]:
    """A line set of a seed or treatment sweep: ``sweep[problem][δ]``."""
    return f(name, "sweep")[problem][key(delta)]


def knee_2d(f: Files, problem: str, delta: float) -> list[dict]:
    return f(NAIVE, "knee")[problem][key(delta)]


def near(rows: Sequence[dict], ratio: float, tol: float = 0.08) -> list[dict]:
    """The rows whose ``h/δ`` is within ``tol`` (relative) of ``ratio``."""
    return [
        r
        for r in rows
        if r.get("h_over_delta") and abs(r["h_over_delta"] / ratio - 1) <= tol
    ]


def snapshot_2d(f: Files, b: Book) -> None:
    """§5.1's table and bullets, and statement (11): four regimes on one node set."""
    name = "heat2d_stiff_snapshot.json"
    rows = {r["operator"]: r for r in f(name, "snapshot")}
    field = f(name, "snapshot/field")
    table = (
        ("naive", "naive/rms", ("2.63e-3", "1.82e-2", "0.708", "0.710", "0.25")),
        (
            "construction",
            "construction/rms",
            ("6.97e-4", "4.05e-3", "0.750", "0.790", "0.65"),
        ),
        ("direct", "direct/rms", ("3.71e-2", "1.52e-1", "0.250", "0.790", "0.58")),
        ("seeds", "seeds/rms", ("3.33e-6", "2.27e-5", "0.259", "0.980", "0.12")),
    )
    w = cited("stiff §5.1 snapshot", "§6.8")
    swept = at(sweep(f, SEEDS, "parabolic", field["delta"]), field["n"])
    for label, rms, quoted in table:
        for column_, q in zip(
            ("rms", "max", "x", "y", "near_share"), quoted, strict=True
        ):
            b.eq(w, f"{label}: {column_}", rows[label][column_], q)
        # "Their RMS errors are the sweep's to every digit printed."
        b.eq(w, f"{label}: the sweep's RMS at (2500, 0.0025)", swept[rms], quoted[0])
    y, h = np.asarray(field["y"]), field["h"]
    edge = (np.abs(y - 0.6) < 2 * h) | (np.abs(y - 0.8) < 2 * h)
    b.eq(w, "the nodes within 2h of a curve, %", 100 * edge.mean(), "15")
    b.eq(
        w, "naive leans on the edges", rows["naive"]["near_share"] / edge.mean(), "1.6"
    )
    b.eq(
        w,
        "the construction's share over the nodes'",
        rows["construction"]["near_share"] / edge.mean(),
        "4",
    )
    b.eq(w, "h/δ", h / field["delta"], "8.3")
    b.eq(w, "direct ÷ naive", rows["direct"]["rms"] / rows["naive"]["rms"], "14")
    b.eq(w, "naive ÷ seeds", rows["naive"]["rms"] / rows["seeds"]["rms"], "790")
    b.eq(
        w,
        "construction ÷ seeds",
        rows["construction"]["rms"] / rows["seeds"]["rms"],
        "210",
    )
    b.eq(
        w,
        "naive ÷ construction",
        rows["naive"]["rms"] / rows["construction"]["rms"],
        "3.8",
    )
    b.eq(
        w,
        "construction ÷ floor, parabolic",
        swept["construction/rms"] / swept["floor"],
        "0.98",
    )
    for n, quoted in ((5000, "1.3"),):
        row = at(knee_2d(f, "parabolic", field["delta"]), n)
        b.eq(
            w,
            f"naive ÷ construction at {n}",
            row["naive/rms"] / row["construction/rms"],
            quoted,
        )
    row = at(knee_2d(f, "parabolic", field["delta"]), 10000)
    b.le(
        w,
        "naive ÷ construction at 10,000 (they cross)",
        row["naive/rms"] / row["construction/rms"],
        "1",
    )

    w = cited("stiff §5.3 (11)", "§6.8")
    b.eq(w, "h/δ", h / field["delta"], "8.3")
    b.eq(w, "naive RMS", rows["naive"]["rms"], "2.63e-3")
    b.eq(
        w, "naive's share away from the edges", 1 - rows["naive"]["near_share"], "0.75"
    )
    b.eq(w, "construction RMS", rows["construction"]["rms"], "6.97e-4")
    b.eq(w, "direct RMS", rows["direct"]["rms"], "3.71e-2")
    b.eq(w, "seeds RMS", rows["seeds"]["rms"], "3.33e-6")


def references_2d(f: Files, b: Book) -> None:
    """Statement (1): the references (§4.1, §4.6). Skipped: the ring's reference
    error (E2.9's runs, port notes §2.9) and the product grid's two seconds."""
    w = "stiff §5.3 (1)"
    w6 = cited(w, "§6.1")
    flat = f("heat2d_stiff_references.json", "references")
    elliptic = [r for r in flat if r["growth"] == 0.0]
    # §4.1's table, the equilibrium references (growth 0); the parabolic ones
    # agree to 4e-13 … 2.4e-11.
    b.span(
        w6,
        "case 1's agreement with a finer reference",
        [r["agreement"] for r in elliptic],
        "7e-13",
        "2e-11",
    )
    b.eq(
        w6,
        "case 1's parabolic references against a finer one, worst",
        max(r["agreement"] for r in flat if r["growth"] == 1.0),
        "2.4e-11",
    )
    (wide,) = [r for r in elliptic if r["delta"] == 0.04]
    b.eq(w6, "α on the band's midline at δ = 0.04", 0.2 + wide["plateau"], "0.2107")
    (zero,) = [r for r in elliptic if r["delta"] == 0.0]
    b.eq(
        w6, "case 1 at δ = 0 against the analytic solution", zero["distance"], "3.8e-13"
    )
    b.eq(
        w6,
        "sup |v_δ − v₀| / δ, largest (5e-4)",
        max(r["distance"] / r["delta"] for r in elliptic if r["delta"] > 0),
        "1.6",
    )
    curved = f("heat2d_stiff_references_a0.02_sine.json", "references")
    b.le(
        w6,
        "case 2's grids against finer ones, both directions",
        max(max(r["n_x"], r["elements_check"]) for r in curved),
        "1e-11",
    )
    (jump,) = [r for r in curved if r["delta"] == 0.0 and "e26_rms" in r]
    b.eq(
        w6,
        "E2.6's 160,000-node run against the product grid, RMS",
        jump["e26_rms"],
        "4.3e-9",
    )


def naive_knee(f: Files, b: Book) -> None:
    """Statement (2): the naive product's knee (§4.2, H10)."""
    w = "stiff §5.3 (2)"
    w6 = cited(w, "§6.2")
    ell = {d: knee_2d(f, "elliptic", d) for d in (0.0, *WIDTHS)}
    par = {d: knee_2d(f, "parabolic", d) for d in (0.0, *WIDTHS)}
    b.eq(w6, "the jump's fit, 1250–160,000", fit(ell[0.0], "naive/rms"), "1.23")
    for problem, lines, low, high, where in (
        ("elliptic", ell, "0.86", "0.94", cited(w, "§1", "§6.2")),
        ("parabolic", par, "0.78", "0.97", w6),
    ):
        b.span(
            where,
            f"naive ÷ jump while h ≳ 6δ, {problem}",
            [
                r["naive/vs_jump"]
                for d in WIDTHS
                for r in lines[d]
                if r["h_over_delta"] > 5.5
            ],
            low,
            high,
        )
    b.span(
        cited(w, "§1", "§6.2"),
        "naive ÷ jump at h ≈ δ, elliptic",
        [r["naive/vs_jump"] for d in WIDTHS for r in near(ell[d], 1.05)],
        "0.081",
        "0.084",
    )
    b.eq(
        w6,
        "naive ÷ jump at 0.53δ, δ = 0.01",
        near(ell[0.01], 0.526, 0.01)[0]["naive/vs_jump"],
        "0.007",
    )
    b.eq(
        w6,
        "naive ÷ jump at 0.53δ, δ = 0.005",
        near(ell[0.005], 0.526, 0.01)[0]["naive/vs_jump"],
        "0.006",
    )
    b.eq(
        w6,
        "naive ÷ jump at 0.52δ, δ = 0.04",
        near(ell[0.04], 0.521, 0.01)[0]["naive/vs_jump"],
        "0.0026",
    )
    b.eq(
        cited(w, "§1", "§6.2"),
        "naive's fit at δ = 0.04, 1250–160,000 (§1: 'the fifth order')",
        fit(ell[0.04], "naive/rms"),
        "5.31",
    )
    for delta, first, last, quoted in (
        (0.01, 2500, 40000, "38"),
        (0.005, 10000, 160000, "34"),
    ):
        naive = column(ell[delta], "naive/rms")
        jump = column(ell[0.0], "naive/rms")
        b.eq(
            w6,
            f"the knee's depth in the jump's units, δ = {delta:g}",
            (naive[first] / naive[last]) / (jump[first] / jump[last]),
            quoted,
        )
    matched = [r["naive/flux"] for r in f(NAIVE, "matched/elliptic")]
    b.eq(
        w6,
        "the flux as a function of h/δ alone, matched pairs",
        max(max(matched), 1 / min(matched)),
        "1.7",
    )
    b.span(
        w6,
        "flux error while h ≥ 4δ",
        [r["naive/flux"] for d in WIDTHS for r in ell[d] if r["h_over_delta"] >= 4],
        "0.31",
        "0.86",
    )
    b.span(
        w6,
        "flux error on the jump, 10,000–160,000",
        [r["naive/flux"] for r in ell[0.0] if r["n"] >= 10000],
        "0.34",
        "0.37",
    )
    jump = column(ell[0.0], "naive/rms")
    # E5.10: the flux stalls over 10,000–160,000, and the RMS falls 6× there (the
    # notes' 20× was 1250 → 160,000).
    b.eq(w6, "the jump's RMS falls, 10,000 → 160,000", jump[10000] / jump[160000], "6")
    # Corrected by E5.3: the notes had "4–5 %"; δ = 0.0025 at 160,000 is 3.5 %.
    b.span(
        w6,
        "flux error at h ≈ δ, %",
        [100 * r["naive/flux"] for d in WIDTHS for r in near(ell[d], 1.05)],
        "3.5",
        "5.0",
    )
    b.span(
        w6,
        "flux error at 0.75δ, %",
        [100 * r["naive/flux"] for d in WIDTHS for r in near(ell[d], 0.745, 0.01)],
        "1",
        "1",
    )
    b.span(
        w6,
        "parabolic ÷ elliptic, naive, from 2500",
        [
            p["naive/rms"] / e["naive/rms"]
            for d in ell
            for e, p in zip(ell[d], par[d], strict=True)
            if e["n"] >= 2500
        ],
        "0.67",
        "1.09",
    )
    b.span(
        w6,
        "the 1250-node growing mode, max Re λ",
        [
            r["max_re"]
            for r in f(NAIVE, "spectra")
            if r["n"] == 1250 and r["operator"] == "naive"
        ],
        "14.8",
        "25.1",
    )


def construction_floor(f: Files, b: Book) -> None:
    """Statement (3): the δ = 0 construction on its floor (§4.2, H10)."""
    w = "stiff §5.3 (3)"
    w6 = cited(w, "§6.2")
    ell = {d: knee_2d(f, "elliptic", d) for d in WIDTHS}
    # The floor in δ, at the finest count (the floor moves a few per cent with n).
    b.span(
        w6,
        "the floor ÷ δ at 160,000",
        [at(ell[d], 160000)["floor"] / d for d in WIDTHS],
        "0.12",
        "0.25",
    )
    on = [
        r["construction/rms"] / r["floor"]
        for d in WIDTHS
        for r in ell[d]
        if r["h_over_delta"] >= 2.1
    ]
    b.span(w6, "construction ÷ floor while h ≥ 2.1δ (three digits)", on, "1.00", "1.00")
    dips = [
        100 * (1 - r["construction/rms"] / r["floor"])
        for d in WIDTHS
        for r in near(ell[d], 1.05)
    ]
    b.span(w6, "the dip at h ≈ δ, %", dips, "5", "10")
    b.eq(
        w6,
        "construction ÷ floor, δ = 0.01 at 160,000",
        at(ell[0.01], 160000)["construction/rms"] / at(ell[0.01], 160000)["floor"],
        "5.0",
    )
    b.eq(
        w6,
        "construction ÷ floor, δ = 0.005 at 160,000",
        at(ell[0.005], 160000)["construction/rms"] / at(ell[0.005], 160000)["floor"],
        "3",
    )
    wide = column(ell[0.04], "construction/rms")
    b.eq(w6, "construction, δ = 0.04, 1250", wide[1250], "4.7e-3")
    b.eq(w6, "construction, δ = 0.04, 160,000", wide[160000], "2.2e-2")
    b.le(
        w6,
        "construction's fit at δ = 0.04 (negative)",
        fit(ell[0.04], "construction/rms"),
        "0",
    )
    b.eq(
        cited(w, "§1", "§6.2"),
        "its best lead over naive",
        max(r["naive/rms"] / r["construction/rms"] for d in WIDTHS for r in ell[d]),
        "6",
    )
    b.eq(
        cited(w, "§1", "§6.2"),
        "the least h/δ at which it beats naive",
        min(
            r["h_over_delta"]
            for d in WIDTHS
            for r in ell[d]
            if r["construction/rms"] < r["naive/rms"]
        ),
        "3",
    )


def seeds_flat(f: Files, b: Book) -> None:
    """Statement (4): the seeds are the jump's rows at δ = 0 and need no switch.

    Skipped: the rows' 7.6e-12 warped and 1.9e-10 on the thin band (E4.5's
    operator comparison, §4.4, not a results table; the stencils table holds the
    plain rows' 1.6e-12 and 6.0e-13), and the naive product on the seeds' own
    30 / 4 groups (1.30e-6, 8.40e-6, 1.76e-7: §5.6's scratch).
    """
    w = "stiff §5.3 (4)"
    w6 = cited(w, "§6.3")
    stencils = f(STENCILS, "stencils")
    (case1,) = [r for r in stencils["jump_limit"] if r["material"] == "case 1"]
    b.eq(
        cited(w, "§5.2"),
        "the seed span against E2.3's basis at δ = 0",
        case1["span"],
        "4.9e-14",
    )
    b.eq(cited(w, "§5.2"), "case 1's crossing stencils", case1["stencils"], "576")
    jump = f(JUMP, "sweep")["elliptic"]["0"]
    seeds = column(jump, "seeds/rms")
    b.eq(w6, "the seed line at δ = 0, 1250", seeds[1250], "1.598e-5")
    b.eq(w6, "the seed line at δ = 0, 160,000", seeds[160000], "3.58e-10")
    b.eq(w6, "its fit over eight counts", fit(jump, "seeds/rms"), "4.54")
    b.eq(
        w6,
        "its fit over the port's six (to 40,000)",
        fit(jump, "seeds/rms", COUNTS_40K),
        "4.77",
    )
    for problem, quoted, where in (
        ("elliptic", ("4.23", "4.31", "4.23", "4.48"), cited(w, "§1", "§6.3")),
        ("parabolic", ("4.22", "4.28", "4.18", "4.44"), w6),
    ):
        b.each(
            where,
            f"the seeds' fits at δ = 0.04 … 0.0025, {problem}",
            [fit(sweep(f, SEEDS, problem, d), "seeds/rms") for d in WIDTHS],
            list(quoted),
        )
    windows = [
        fit(sweep(f, SEEDS, problem, d), "seeds/rms", COUNTS_40K[k:])
        for problem in PROBLEMS
        for d in WIDTHS
        for k in (2, 3)
    ]
    # Corrected by E5.3: the notes had 3.8–4.8 at δ > 0; the 4.8 is δ = 0's
    # (4.78 and 4.80 from 10,000), checked on its own below.
    b.span(
        w6, "fits over the windows from 5000 or 10,000, δ > 0", windows, "3.8", "4.3"
    )
    b.span(
        w6,
        "… and at the jump",
        [
            fit(sweep(f, SEEDS, problem, 0.0), "seeds/rms", COUNTS_40K[k:])
            for problem in PROBLEMS
            for k in (2, 3)
        ],
        "4.6",
        "4.8",
    )
    spreads = {}
    for problem in PROBLEMS:
        lines = [
            column(sweep(f, SEEDS, problem, d), "seeds/rms") for d in (0.0, *WIDTHS)
        ]
        spreads[problem] = {
            n: max(c[n] for c in lines) / min(c[n] for c in lines) for n in COUNTS_40K
        }
    b.span(
        cited(w, "§6.3"),
        "the five widths' spread at every count, elliptic",
        spreads["elliptic"].values(),
        "1.2",
        "2.4",
    )
    for n, quoted in ((1250, "1.16"), (20000, "2.38"), (40000, "1.67")):
        b.eq(w6, f"the spread at {n}", spreads["elliptic"][n], quoted)
    b.eq(
        w6, "the spread, parabolic, largest", max(spreads["parabolic"].values()), "2.53"
    )
    row = at(sweep(f, SEEDS, "elliptic", 0.0025), 40000)
    b.eq(
        cited(w, "§6.3"),
        "seeds at 40,000, δ = 0.0025",
        row["seeds/rms"],
        "7.99e-9",
    )
    b.eq(
        cited(w, "§6.3"),
        "naive at 40,000, δ = 0.0025",
        row["naive/rms"],
        "8.36e-5",
    )
    b.eq(w6, "construction at 40,000, δ = 0.0025", row["construction/rms"], "6.18e-4")
    # "Where the grid resolves the edge": δ = 0.04, resolved at every count.
    wide = sweep(f, SEEDS, "elliptic", 0.04)
    b.span(
        w6,
        "seeds ÷ direct, resolved (δ = 0.04)",
        [r["seeds/rms"] / r["direct/rms"] for r in wide],
        "0.10",
        "0.22",
    )
    b.span(
        w6,
        "seeds ÷ direct-reach, resolved (δ = 0.04)",
        [r["seeds/rms"] / r["direct-reach/rms"] for r in wide],
        "0.086",
        "0.22",
    )
    for problem, n, quoted in (
        ("elliptic", 20000, "0.98"),
        ("elliptic", 40000, "1.07"),
        ("parabolic", 40000, "1.15"),
    ):
        row = at(sweep(f, SEEDS, problem, 0.04), n)
        b.eq(
            cited(w, "§5.2", "§6.3") if quoted == "1.15" else w6,
            f"seeds ÷ naive at δ = 0.04, {n}, {problem}",
            row["seeds/rms"] / row["naive/rms"],
            quoted,
        )
    b.eq(w6, "naive's fit at δ = 0.04, 1250–40,000", fit(wide, "naive/rms"), "5.2")
    b.eq(w6, "the seeds' fit at δ = 0.04", fit(wide, "seeds/rms"), "4.2")
    b.within(
        w,
        "direct ÷ seeds at δ = 0.04, 20,000–40,000",
        [at(wide, n)["direct/rms"] / at(wide, n)["seeds/rms"] for n in (20000, 40000)],
        "5",
        "10",
    )
    ladder = [r for r in stencils["ladder"] if r["ratio"] > 0]
    b.within(
        cited(w, "§5.2"),
        "the seed block over the monomial one, column-scaled",
        [r["scaled"] / r["monomial"] for r in ladder],
        "0.46",
        "2.5",
    )


def seeds_2d(f: Files, b: Book) -> None:
    """The construction's own measurements that §5 of the manuscript quotes.

    Stiff §4.3's H1–H3 tables on the 2500-node case-1 set, and the stencil of the
    seed figure (``stencils/seed_functions``, E5.7). E5.7 (#48) added them; the
    δ = 0 span, the crossing count, the column-scaled conditioning and the
    tangential chain's 7.18e-9 are statement (4)'s and (8)'s checks, tagged
    there. Skipped, and traced to the notes: E4.5's warped rows (7.6e-12) and
    assembled operators (5.3e-13), the tangential series' 3.6e-10 and the first
    level cutoff's 3.55e-8 (scratch runs), the ring's width to ulp(1)/(w/h_s),
    the fold's share of the contact resistance, the level-0 warp's 3.3e-6 and
    2.3e-7, the diagonal shares (§4.10's one-offs), and every cost.
    """
    stencils = f(STENCILS, "stencils")
    w = "stiff §4.3 H1"
    chain = stencils["chain"]
    b.eq(
        cited(w, "§5.1"),
        "the seeds on a constant α against the monomials",
        max(r["monomials"] for r in chain),
        "1.4e-15",
    )
    b.eq(
        cited(w, "§5.1"),
        "the shift identity, every δ",
        max(r["shift"] for r in chain),
        "7e-16",
    )
    b.le(
        cited(w, "§5.1"),
        "the chain's residual by twelfth-order differences",
        max(r["residual"] for r in chain if r["residual"] is not None),
        "1e-10",
    )
    w = "stiff §4.3 H2"
    case1, thin = stencils["jump_limit"]
    b.eq(cited(w, "§5.2"), "φ₀₁ against the jump-aware warp", case1["warp"], "1.8e-15")
    b.eq(cited(w, "§5.2"), "the weights, plain Gaussians", case1["weights"], "1.6e-12")
    b.eq(cited(w, "§5.2"), "the thin band's crossing stencils", thin["stencils"], "333")
    b.eq(cited(w, "§5.2"), "… of them three-region", thin["three_region"], "240")
    b.eq(cited(w, "§5.2"), "the thin band's span distance", thin["span"], "1.7e-14")
    b.eq(cited(w, "§5.2"), "the thin band's weights", thin["weights"], "6.0e-13")
    b.span(
        cited(w, "§5.2"),
        "the span distance over δ/h at 1e-5 h, four anchors",
        [r["span"] / r["ratio"] for r in stencils["ladder"] if r["ratio"] == 1e-5],
        "0.46",
        "0.74",
    )
    lowest = min(r["anchor"] for r in stencils["ladder"])
    below = [r for r in stencils["ladder"] if r["anchor"] == lowest]
    by_ratio = {r["ratio"]: r for r in below}
    w = "stiff §4.3 H3"
    b.eq(cited(w, "§5.2"), "raw cond, resolved (8h), below", by_ratio[8.0]["raw"], "55")
    b.eq(cited(w, "§5.2"), "raw cond, δ = 0, below", by_ratio[0.0]["raw"], "212")
    # The seed figure's stencil (fig:seeds2d, stiff §4.3's E5.7 paragraph).
    w = "stiff §4.3 H2"
    figure = f(STENCILS, "stencils/seed_functions")
    b.eq(cited(w, "§5.2"), "the figure's edge in η", figure["eta_edge"], "0.47")
    b.eq(
        cited(w, "§5.2"),
        "the figure's nodes across the edge",
        sum(e > figure["eta_edge"] for e in figure["eta_nodes"]),
        "7",
    )
    b.le(
        cited(w, "§5.2"),
        "the figure's 1 − α_e at δ = h/2",
        1.0 - figure["alpha_e"]["0.5"],
        "0.004",
    )


def elliptic_ranks(f: Files, b: Book) -> None:
    """Statement (5): the 2-D elliptic problem ranks the methods. Skipped: the
    3–19 BD4 steps (the problem's t_end over h)."""
    w = "stiff §5.3 (5)"
    for delta, quoted in ((0.0, ("9.7e-5", "1.2e-8")),):
        eq75 = column(knee_1d(f, "equilibrium", "eq75", delta), "seeds")
        b.eq(w, "1-D eq. 75 seeds at equilibrium, 101, δ = 0", eq75[101], quoted[0])
        b.eq(w, "1-D eq. 75 seeds at equilibrium, 1601, δ = 0", eq75[1601], quoted[1])
    fv = column(comparators_1d(f, "equilibrium", "eq75", 0.0), "T1-FV")
    b.eq(w, "1-D eq. 75 T1-FV, 101", fv[101], "3e-14")
    b.eq(w, "1-D eq. 75 T1-FV, 1601", fv[1601], "1.4e-11")
    seeds = column(sweep(f, SEEDS, "elliptic", 0.0), "seeds/rms")
    b.eq(w, "2-D seeds at equilibrium, 1250", seeds[1250], "1.6e-5")
    b.eq(w, "2-D seeds at equilibrium, 40,000", seeds[40000], "5.3e-9")
    b.span(
        cited(w, "§6.3"),
        "parabolic ÷ elliptic at 40,000, every δ",
        [
            at(sweep(f, SEEDS, "parabolic", d), 40000)["seeds/rms"]
            / at(sweep(f, SEEDS, "elliptic", d), 40000)["seeds/rms"]
            for d in (0.0, *WIDTHS)
        ],
        "1.04",
        "1.17",
    )
    for delta, quoted in ((0.0, "2000"), (0.0025, "670")):
        flux = column(sweep(f, SEEDS, "elliptic", delta), "seeds/flux")
        b.eq(
            cited(w, "§6.3"),
            f"the seeds' flux reading falls, 1250 → 40,000, δ = {delta:g}",
            flux[1250] / flux[40000],
            quoted,
        )


def solvability(f: Files, b: Book) -> None:
    """Statement (6): solvability and spectra (§4.4; H5, H6). Skipped: SuperLU's
    0.04–0.06 s (timings) and 'every solve converged' (not a number)."""
    w = "stiff §5.3 (6)"
    w6 = cited(w, "§6.4")
    rows = {
        n: f(name, "rows")
        for n, name in (
            (2500, "heat2d_stiff_eigenvalues.json"),
            (10000, "heat2d_stiff_eigenvalues_rows_n10000.json"),
        )
    }
    for n, quoted in ((2500, ("0.409", "0.239")), (10000, ("0.404", "0.211"))):
        seeds = {r["ratio"]: r for r in rows[n] if r["label"] == "seeds"}
        b.eq(w6, f"seeds' least DDR at δ = 8h, {n}", seeds[8.0]["least"], quoted[0])
        b.eq(w6, f"seeds' least DDR at δ = 0, {n}", seeds[0.0]["least"], quoted[1])
        b.ge(
            w6,
            f"seeds' least DDR at every δ, never below the jump's, {n}",
            min(r["least"] for r in seeds.values()),
            quoted[1],
        )
    seeds = {r["ratio"]: r for r in rows[2500] if r["label"] == "seeds"}
    ratios = (8.0, 1.0, 0.125, 0.015625, 0.0)
    b.eq(w6, "seeds' condition estimate at 8h", seeds[8.0]["cond"], "7.1e3")
    b.eq(
        w6,
        "seeds' condition estimate, largest",
        max(r["cond"] for r in seeds.values()),
        "1.7e4",
    )
    b.eq(w6, "seeds' condition estimate at δ = 0", seeds[0.0]["cond"], "1.5e4")
    b.each(
        w6,
        "seeds' gmres iterations, 8h → 0",
        [seeds[r]["gmres/none"]["iterations"] for r in ratios],
        ["134", "155", "147", "160", "161"],
    )
    built = [
        r["gmres/none"]["iterations"]
        for r in rows[2500]
        if r["label"] == "construction"
    ]
    b.span(w6, "the construction's gmres iterations", built, "161", "208")
    b.le(
        w6,
        "the direct solves' residual, 2500",
        max(r["residual"] for r in rows[2500]),
        "1e-14",
    )
    spectra = {
        (r["delta"], r["label"]): r
        for r in f("heat2d_stiff_eigenvalues.json", "spectra")
    }
    seed_rows = [r for (d, label), r in spectra.items() if label == "seeds"]
    b.eq(
        w6,
        "the seeds' eigenvalues right of the axis, 1600",
        max(r["positive"] for r in seed_rows),
        "0",
    )
    b.eq(w6, "seeds' max Re λ at δ = 0", spectra[(0.0, "seeds")]["max_re"], "-7.27")
    b.eq(w6, "seeds' max Re λ at δ = 0.04", spectra[(0.04, "seeds")]["max_re"], "-7.73")
    b.le(
        w6, "seeds' h² max |Im λ|, 1600", max(r["max_im_h2"] for r in seed_rows), "0.23"
    )
    b.eq(
        w6,
        "plain seeds' BD4 root at δ = 0, 1600",
        spectra[(0.0, "seeds-plain")]["bd4"],
        "0.826",
    )
    at4900 = {
        (r["delta"], r["label"]): r
        for r in f("heat2d_stiff_eigenvalues_spectra_n4900.json", "spectra")
    }
    seeds = at4900[(0.0, "seeds")]
    for field, quoted in (
        ("max_re", "-7.27"),
        ("min_re_h2", "-13.19"),
        ("max_im_h2", "0.385"),
        ("bd4", "0.897"),
    ):
        b.eq(w6, f"seeds at 4900, δ = 0: {field}", seeds[field], quoted)
    b.eq(
        w6,
        "plain seeds at 4900, δ = 0: h² max |Im λ|",
        at4900[(0.0, "seeds-plain")]["max_im_h2"],
        "1.49",
    )
    case2 = f("heat2d_stiff_tangential.json", "tangential")["spectra"]
    b.eq(
        w6,
        "case 2: eigenvalues right of the axis",
        max(r["positive"] for r in case2),
        "0",
    )


def warp(f: Files, b: Book) -> None:
    """Statement (7): the warp (§4.5–§4.7; H7)."""
    w = "stiff §5.3 (7)"
    w6 = cited(w, "§6.3")
    ell = {d: sweep(f, SEEDS, "elliptic", d) for d in (0.0, *WIDTHS)}
    jump = column(ell[0.0], "seeds-plain/rms")
    seeds = column(ell[0.0], "seeds/rms")
    b.eq(w6, "plain ÷ warped at δ = 0, 1250", jump[1250] / seeds[1250], "2.3")
    b.eq(w6, "plain ÷ warped at δ = 0, 40,000", jump[40000] / seeds[40000], "6.9")
    curved = sweep(f, TANGENTIAL, "elliptic", 0.0)
    b.span(
        w6,
        "plain ÷ warped at δ = 0 on case 2, from 5000 (the chain)",
        [
            r["tangential-plain/rms"] / r["tangential/rms"]
            for r in curved
            if r["n"] >= 5000
        ],
        "5.5",
        "11.7",
    )
    # "While the edge is unresolved": h ≥ 3δ, the rows before the turn at 2δ.
    b.span(
        w6,
        "plain ÷ warped at δ = 0.0025, h ≥ 3δ",
        [
            r["seeds-plain/rms"] / r["seeds/rms"]
            for r in ell[0.0025]
            if r["h_over_delta"] > 2.5
        ],
        "1.7",
        "7.9",
    )
    b.le(
        cited(w, "§5.2", "§6.3"),
        "warped ÷ plain where the warp loses, both problems",
        max(
            r["seeds/rms"] / r["seeds-plain/rms"]
            for p in PROBLEMS
            for d in WIDTHS
            for r in sweep(f, SEEDS, p, d)
        ),
        "2.6",
    )
    b.span(
        w6,
        "the chain's plain ÷ warped on case 2 once h ≲ 2δ",
        [
            r["tangential-plain/rms"] / r["tangential/rms"]
            for d in (0.01, 0.005, 0.0025)
            for r in sweep(f, TANGENTIAL, "elliptic", d)
            if r["h_over_delta"] <= 2
        ],
        "0.55",
        "1.1",
    )
    row = at(ell[0.01], 5000)
    b.eq(w6, "δ = 0.01 at 5000, warped", row["seeds/rms"], "7.95e-7")
    b.eq(w6, "δ = 0.01 at 5000, plain", row["seeds-plain/rms"], "4.39e-7")


def flat_ratio(
    rows: Sequence[dict], label: str, flat_label: str | None = None
) -> list[float]:
    """``curved/<label>`` over ``flat/<flat_label>`` per row of a ``flat`` table."""
    other = flat_label or label
    return [r[f"curved/{label}"] / r[f"flat/{other}"] for r in rows]


def curved_feature(f: Files, b: Book) -> None:
    """Statement (8): route (a) and the tangential chain (§4.6–§4.7; H9, H13–H17).
    Skipped: the cost of a row (2.5–3.8× route (a)'s, 4.2–7.3× case 1's; timings)."""
    w = "stiff §5.3 (8)"
    w6 = cited(w, "§6.5")
    route = sweep(f, CURVED, "elliptic", 0.0)
    b.eq(
        w6,
        "route (a)'s crossing rows at δ = 0, fit",
        fit(route, "seeds/probe_crossing"),
        "0.38",
    )
    flat = f(CURVED, "flat")
    # Corrected by E5.3 (scope): §5.3 (8) had put 17–6742× at δ = 0; it is §4.6's
    # range over every δ and count, elliptic. At δ = 0 alone: 24–6742.
    elliptic = [r for r in flat if r["problem"] == "elliptic"]
    b.span(
        w6,
        "route (a) over the flat seeds, every δ and count",
        flat_ratio(elliptic, "seeds"),
        "17",
        "6742",
    )
    b.span(
        w6,
        "… at δ = 0",
        flat_ratio([r for r in elliptic if r["delta"] == 0.0], "seeds"),
        "24",
        "6742",
    )
    line_b = f("heat2d_stiff_seeds_tangential_a0_sine_jump.json", "sweep")["elliptic"][
        "0"
    ]
    b.eq(
        w6,
        "route (a) on flat lines with α along them (B), fit",
        fit(line_b, "seeds/rms"),
        "1.00",
    )
    mid = sweep(f, CURVED, "elliptic", 0.01)
    rows = [at(mid, n) for n in (10000, 20000, 40000)]
    b.each(
        w6,
        "route (a) at δ = 0.01, the last two rates",
        [
            rate_in_h(rows[0], rows[1], "seeds/rms"),
            rate_in_h(rows[1], rows[2], "seeds/rms"),
        ],
        ["4.87", "3.66"],
    )
    (last,) = [
        r
        for r in flat
        if r["delta"] == 0.01 and r["n"] == 40000 and r["problem"] == "elliptic"
    ]
    b.eq(
        w6,
        "route (a) over the flat seeds, δ = 0.01, 40,000",
        last["curved/seeds"] / last["flat/seeds"],
        "130",
    )
    chain = [
        r
        for r in f(TANGENTIAL, "flat")
        if r["n"] >= 5000 and r["problem"] == "elliptic"
    ]
    b.span(
        cited(w, "§1", "§6.5"),
        "the chain over the flat seeds from 5000, RMS",
        flat_ratio(chain, "tangential"),
        "1.4",
        "3.9",
    )
    flat_max = {}
    for d in (0.0, *WIDTHS):
        for r in sweep(f, SEEDS, "elliptic", d):
            flat_max[(d, r["n"])] = r["seeds/max"]
    chain_max = [
        r["tangential/max"] / flat_max[(d, r["n"])]
        for d in (0.0, 0.01, 0.005, 0.0025)
        for r in sweep(f, TANGENTIAL, "elliptic", d)
        if r["n"] >= 5000
    ]
    b.span(
        w6, "the chain over the flat seeds from 5000, max norm", chain_max, "1.9", "4.8"
    )
    split = [
        v
        for name in SPLIT.values()
        for v in flat_ratio(
            [
                r
                for r in f(name, "flat")
                if r["n"] >= 5000 and r["problem"] == "elliptic"
            ],
            "tangential",
        )
    ]
    b.span(w6, "the chain over the flat seeds on A and B", split, "1.1", "4.4")
    b.eq(
        cited(w, "§5.3", "§6.5"),
        "the chain at 40,000 on case 2, δ = 0",
        at(sweep(f, TANGENTIAL, "elliptic", 0.0), 40000)["tangential/rms"],
        "7.18e-9",
    )
    b.each(
        w6,
        "the chain at 40,000 on case 2, δ = 0.01, 0.005, 0.0025",
        [
            at(sweep(f, TANGENTIAL, "elliptic", d), 40000)["tangential/rms"]
            for d in (0.01, 0.005, 0.0025)
        ],
        ["3.01e-8", "3.00e-8", "2.69e-8"],
    )
    zero = sweep(f, TANGENTIAL, "elliptic", 0.0)
    b.span(
        w6,
        "the chain over E2.3's curved construction at δ = 0, from 5000",
        [r["tangential/rms"] / r["construction/rms"] for r in zero if r["n"] >= 5000],
        "0.14",
        "0.45",
    )
    crossing = [
        fit(sweep(f, name, "elliptic", d), "tangential/probe_crossing")
        for name, widths in (
            (TANGENTIAL, (0.0, 0.01, 0.005, 0.0025)),
            (SPLIT["A"], (0.0, 0.0025)),
            (SPLIT["B"], (0.0, 0.0025)),
        )
        for d in widths
    ]
    b.span(
        w6,
        "the chain's crossing rows, fits on every geometry and width",
        crossing,
        "2.7",
        "3.6",
    )
    coarse = [
        r
        for r in f(TANGENTIAL, "flat")
        if r["n"] <= 2500 and r["problem"] == "elliptic"
    ]
    # Corrected by E5.3: the notes had 2.8–8.1; the largest is 8.047 (δ = 0.0025
    # at 1250).
    b.span(
        cited(w, "§6.5", "§7.1"),
        "the chain over the flat seeds at the coarsest two counts",
        flat_ratio(coarse, "tangential"),
        "2.8",
        "8.0",
    )
    circles = f("heat2d_stiff_tangential.json", "tangential")["circles"]
    b.eq(
        w6,
        "the chain's rows on concentric circles, fit",
        fit(circles, "tangential"),
        "3.45",
    )
    b.span(
        w6,
        "E2.3 over the chain on the circles",
        [r["construction"] / r["tangential"] for r in circles],
        "4.6",
        "7.3",
    )


def ring(f: Files, b: Book) -> None:
    """Statement (9): EABE eq. 40's ring (§4.8, §4.10; H11). Skipped: the
    differences between fine-run rebuilds (1.3–3.9e-6, §4.8's scratch) and the
    global matrix's conditioning (not measured)."""
    w = "stiff §5.3 (9)"
    w6 = cited(w, "§6.6")
    zero = [r for r in f(RING, "conditioning") if r["delta"] == 0.0]
    b.span(
        cited(w, "§6.6"),
        "the seeds' worst residual on the matched profile, every s",
        [r["seeds-worst"] for r in zero],
        "1.3e-15",
        "2.1e-15",
    )
    (top,) = [r for r in zero if r["s"] == 1e11]
    b.eq(w6, "E2.3's worst residual at s = 1e11", top["e23-worst"], "1.36e-7")
    upper = [r for r in zero if r["s"] >= 1e5]
    # Corrected by E5.3: §5.3 (9) had single values 7.1e3 / 5.0e3 and 1.0e6; §4.8's
    # table has 7.2e3 / 5.1e3 and 1.1e6 at 10⁵ (5.1e3 to 10⁷).
    b.span(
        w6,
        "the seed block's mean condition number, raw, 1e5–1e11",
        [r["block-raw-mean"] for r in upper],
        "7.1e3",
        "7.2e3",
    )
    b.span(
        w6,
        "the seed block's, column-scaled",
        [r["block-scaled-mean"] for r in upper],
        "5.0e3",
        "5.1e3",
    )
    b.span(w6, "the seed system's", [r["system-mean"] for r in upper], "1.0e6", "1.1e6")
    b.eq(w6, "E2.3's block at s = 1e11", top["e23-block-mean"], "4.6e10")
    b.eq(w6, "E2.3's system at s = 1e11", top["e23-system-mean"], "2.7e13")
    convergence = {float(s): rows for s, rows in f(RING, "convergence").items()}
    b.span(
        w6,
        "seeds ÷ E2.3, 5000–40,000, every s",
        [
            r["seeds"]["full"] / r["construction"]["full"]
            for rows in convergence.values()
            for r in rows
            if 5000 <= r["n"] <= 40000
        ],
        "0.52",
        "0.97",
    )
    for label, low, high, where in (
        ("seeds", "4.43", "4.71", cited(w, "§1", "§6.6")),
        ("construction", "4.08", "4.13", w6),
    ):
        b.span(
            where,
            f"{label}' fits over 1250–40,000",
            [
                fit([r[label] for r in rows], "full", COUNTS_40K)
                for rows in convergence.values()
            ],
            low,
            high,
        )
    # Corrected by E5.3: the notes had 1.4e-7 at s ≥ 10⁸; it is 1.4–1.6e-7.
    b.span(
        w6,
        "the seeds at 80,000, s ≥ 1e8",
        [
            at(rows, 80000)["seeds"]["full"]
            for s, rows in convergence.items()
            if s >= 1e8
        ],
        "1.4e-7",
        "1.6e-7",
    )
    for s, quoted in ((1e3, ("8.2", "11.8")), (1e11, ("2.2", "3.2"))):
        for n, q in zip((40000, 80000), quoted, strict=True):
            row = at(convergence[s], n)
            b.eq(
                w6,
                f"without the flux seeds ÷ E2.3, s = {s:g}, {n}",
                row["seeds15"]["full"] / row["construction"]["full"],
                q,
            )

    smooth = {}
    for name, rows in f(RING, "smooth").items():
        s, d = name.split("|")
        smooth[(float(s), float(d))] = rows
    widths = {k: v for k, v in smooth.items() if k[1] > 0}
    b.span(
        cited(w, "§6.6", "§7.1"),
        "the far field's share of the nodes, %",
        [100 * r["error-seeds"]["far-share"] for rows in widths.values() for r in rows],
        "74",
        "96",
    )
    b.span(
        w6,
        "the seeds' far-field fits, 2500–20,000, every (s, δ)",
        [
            fit([r["error-seeds"] for r in rows], "far", (2500, 5000, 10000, 20000))
            for rows in widths.values()
        ],
        "4.0",
        "4.8",
    )
    to_40k = sorted(
        fit([r["error-seeds"] for r in rows], "far") for rows in widths.values()
    )
    # The one line that floors is the lowest fit (s = 1e11, δ = 0.001, §4.10).
    b.span(
        w6,
        "the seeds' far-field fits to 40,000, the five that do not floor",
        to_40k[1:],
        "4.4",
        "4.8",
    )
    b.span(
        w6,
        "E2.3's far-field floor",
        [r["error-construction"]["far"] for rows in widths.values() for r in rows],
        "1e-4",
        "7e-4",
    )
    b.span(
        w6,
        "naive and direct at δ = 0.00025, far field",
        [
            r[f"error-{lab}"]["far"]
            for (s, d), rows in widths.items()
            if d == 0.00025
            for r in rows
            for lab in ("naive", "direct")
        ],
        "5e-3",
        "5e-3",
    )
    # s = 1e11, δ = 0.001 is the rule's discrepant fine run, read apart below.
    b.span(
        w6,
        "the seeds at 40,000, δ ≤ 0.001 (the discrepant pair aside)",
        [
            at_error(rows, 40000)
            for (s, d), rows in widths.items()
            if d <= 0.001 and (s, d) != (1e11, 0.001)
        ],
        "7.1e-7",
        "1.0e-6",
    )
    b.each(
        w6,
        "the steep last rates at s = 1e3, δ = 0.0025 and 0.001",
        [last_rate(widths[(1e3, d)]) for d in (0.0025, 0.001)],
        ["6.7", "6.9"],
    )
    b.eq(
        w6,
        "the warp on every row, δ = 0.001 at 20,000 (the outlier)",
        max(
            at(rows, 20000)["error-seeds-warp"]["far"]
            for (s, d), rows in widths.items()
            if d == 0.001
        ),
        "6.2e-5",
    )
    fired = [
        r["error-seeds"]["far"]
        / min(r["error-seeds-warp"]["far"], r["error-seeds-plain"]["far"])
        for rows in widths.values()
        for r in rows
        if 0 < r["error-seeds"]["plain"] < r["error-seeds"]["rows"]
    ]
    b.span(
        w6,
        "the rule over the better uniform choice, where it fires",
        fired,
        "0.91",
        "1.25",
    )
    gap = f(RING, "fine-gap")["1e11|0.001"]
    b.eq(
        cited(w, "§6.6", "§7.1"),
        "the rule's fine run against the plain-built one, s = 1e11, δ = 0.001",
        gap["rms"],
        "3.55e-6",
    )
    plain = [r["error-seeds|seeds-plain"] for r in smooth[(1e11, 0.001)]]
    b.eq(
        w6,
        "the seeds against the plain-built run at 40,000",
        at(plain, 40000)["far"],
        "8.0e-7",
    )
    b.eq(w6, "their fit", fit(plain, "far"), "4.77")


def at_error(rows: Sequence[dict], n: int) -> float:
    return at(rows, n)["error-seeds"]["far"]


def last_rate(rows: Sequence[dict]) -> float:
    errors = [r["error-seeds"] for r in rows]
    return rate_in_h(errors[-2], errors[-1], "far")


TREATMENT_LABELS = (
    "harmonic-0.5h",
    "harmonic-1h",
    "arithmetic-0.5h",
    "arithmetic-1h",
    "widened-1h",
    "widened-2h",
)


def treatments(f: Files, b: Book) -> None:
    """Statement (10): the six coefficient treatments (§4.9; H12)."""
    w = "stiff §5.3 (10)"
    w6 = cited(w, "§6.7")
    gains = []
    for case, name in TREATMENTS.items():
        for problem in PROBLEMS:
            for r in f(name, f"ratios/{problem}"):
                # Case 1's parabolic 1250-node set carries the growing mode (§4.9).
                if case == "case 1" and problem == "parabolic" and r["n"] == 1250:
                    continue
                gains.extend(1 / r[label] for label in TREATMENT_LABELS)
    b.le(
        cited(w, "§1", "§1.1", "§6.7", "§7.3"),
        "the most a treatment beats sampling by, RMS, anywhere",
        max(gains),
        "1.75",
    )

    def max_gain(name: str, keep: Callable[[str, int], bool]) -> float:
        out = []
        for problem in PROBLEMS:
            for rows in f(name, "sweep")[problem].values():
                for r in rows:
                    if keep(problem, r["n"]):
                        out.extend(
                            r["naive/max"] / r[f"{label}/max"]
                            for label in TREATMENT_LABELS
                        )
        return max(out)

    # The parabolic 1250-node set of case 1 aside again (its growing mode).
    b.eq(
        w6,
        "the max-norm gain on case 1 at 1250",
        max_gain(TREATMENTS["case 1"], lambda p, n: n == 1250 and p == "elliptic"),
        "3.1",
    )
    b.eq(
        w6,
        "the max-norm gain on case 2 at 2500",
        max_gain(TREATMENTS["case 2"], lambda p, n: n == 2500),
        "2.4",
    )
    b.le(
        w6,
        "the most a treatment beats sampling from 5000, max norm",
        max(max_gain(name, lambda p, n: n >= 5000) for name in TREATMENTS.values()),
        "1.55",
    )
    # Corrected by E5.3: §5.3 (10) had 1.55× "in either norm" from 5000; in the RMS
    # it is 1.60 (case 1, arithmetic h/2, δ = 0.0025, 10,000 nodes: §4.9's 0.62).
    b.eq(
        w6,
        "the most a treatment beats sampling from 5000, RMS",
        max(
            1 / r[label]
            for name in TREATMENTS.values()
            for p in PROBLEMS
            for r in f(name, f"ratios/{p}")
            if r["n"] >= 5000
            for label in TREATMENT_LABELS
        ),
        "1.60",
    )
    at_jump = [
        r
        for name in TREATMENTS.values()
        for p in PROBLEMS
        for r in f(name, f"ratios/{p}")
        if r["delta"] == 0.0 and r["n"] >= 5000
    ]
    b.span(
        w6,
        "the radius-h harmonic mean over sampling at the jump, from 5000",
        [r["harmonic-1h"] for r in at_jump],
        "1.6",
        "3.4",
    )
    # Corrected by E5.3: §5.3 (10) had "the radius-h means are 1.6–3.4×", the
    # harmonic one's; the arithmetic radius-h mean is 1.28–3.00× (§4.9's table).
    b.span(
        w6,
        "the radius-h arithmetic mean over sampling at the jump, from 5000",
        [r["arithmetic-1h"] for r in at_jump],
        "1.3",
        "3.0",
    )

    resolved = at(sweep(f, TREATMENTS["case 1"], "elliptic", 0.04), 40000)
    b.span(
        w6,
        "the disc means over naive at 40,000, δ = 0.04",
        [
            resolved[f"{label}/rms"] / resolved["naive/rms"]
            for label in TREATMENT_LABELS[:4]
        ],
        "584",
        "3100",
    )
    crossings = [
        x
        for name in TREATMENTS.values()
        for p in PROBLEMS
        for label in ("harmonic-0.5h", "arithmetic-0.5h")
        for xs in f(name, f"crossovers/{p}")[label].values()
        for x in xs
    ]
    b.span(w6, "the half-spacing discs' crossovers, h/δ", crossings, "1.8", "3.5")
    orders = {
        r["n"]: r["orders"]
        for r in f(TREATMENTS["case 1"], "h12/parabolic/rms")
        if r["delta"] == 0.0
    }
    b.eq(
        w6,
        "the seeds' lead over the best treatment at 1250, orders",
        orders[1250],
        "2.3",
    )
    b.eq(w6, "… at 40,000", orders[40000], "4.7")
    # E5.10 (#51): the caveat quoted case 1's parabolic line at the jump as the
    # whole lead. Over both cases, both problems and every width H12 measures
    # (the jump; δ = 0.005 and 0.0025 while h ≥ 4δ) it is 1.6 to 4.9.
    leads = [
        r["orders"]
        for name in TREATMENTS.values()
        for p in PROBLEMS
        for r in f(name, f"h12/{p}/rms")
    ]
    b.span(
        cited(w, "§1", "§6.7", "§7.3"),
        "the seeds' lead over the best treatment, both cases and problems, orders",
        leads,
        "1.6",
        "4.9",
    )


def results_2d(f: Files, b: Book) -> None:
    """The 2-D study's own tables that §6 of the manuscript quotes.

    Keyed by the hypothesis of the notes' §3.7 or §3.10 each table answers: H10,
    the naive knee and the construction (stiff §4.2); H5 and H6, the matrices
    (§4.4); H4 and H8, the flat sweep (§4.5); H9 and H16, the curved feature
    (§4.6, §4.7); H11, the ring (§4.8, §4.10); H12, the treatments (§4.9). E5.8
    (#49) added them; the statements' own numbers are tagged in their functions.
    Skipped, and traced to the notes: the timings (the marches' 167 s against
    2.6 s, a row's milliseconds, SuperLU's), the naive product on the seeds' own
    30 / 4 groups (§5.6's scratch), the frozen profile against the flat
    construction on A (that line is not in A's results file), the differences
    between rebuilds of the ring's fine runs (§4.8), and where the naive
    operator's growing mode sits (§4.2's scratch).
    """
    # H10: the naive knee and the δ = 0 construction (§6.2).
    w = cited("stiff §4.2 H10", "§6.2")
    ell = {d: knee_2d(f, "elliptic", d) for d in (0.0, *WIDTHS)}
    par = {d: knee_2d(f, "parabolic", d) for d in (0.0, *WIDTHS)}
    jump = column(ell[0.0], "naive/rms")
    b.eq(w, "the jump's naive line at 1250", jump[1250], "4.14e-3")
    b.eq(w, "the jump's naive line at 160,000", jump[160000], "2.07e-4")
    b.span(
        w,
        "the jump's naive rates between successive counts",
        [
            rate_in_h(a, c, "naive/rms")
            for a, c in zip(ell[0.0], ell[0.0][1:], strict=False)
        ],
        "-0.5",
        "3.1",
    )
    spread = []
    for d in (0.0, *WIDTHS):
        sets = [
            column(f(name, "knee")["elliptic"][key(d)], "naive/rms")
            for name in (NAIVE, *NAIVE_SEEDS)
        ]
        for n in set.intersection(*(set(c) for c in sets)):
            values = [c[n] for c in sets]
            spread.append(max(values) / min(values))
    b.span(
        cited("stiff §4.2 H10", "§6", "§6.2", "§7.1"),
        "the naive error's spread over three node sets at one count and width",
        spread,
        "1.03",
        "2.0",
    )
    b.span(
        w,
        "naive ÷ jump at h ≈ 2δ",
        [r["naive/vs_jump"] for d in WIDTHS for r in near(ell[d], 2.1, 0.02)],
        "0.19",
        "0.27",
    )
    for delta, first, last, drop, jump_drop in (
        (0.01, 2500, 40000, "311", "8.1"),
        (0.005, 10000, 160000, "211", "6.2"),
    ):
        naive = column(ell[delta], "naive/rms")
        b.eq(
            w,
            f"naive's drop from h ≈ 2δ to δ/2, δ = {delta:g}",
            naive[first] / naive[last],
            drop,
        )
        b.eq(
            w,
            f"the jump's over the same counts, δ = {delta:g}",
            jump[first] / jump[last],
            jump_drop,
        )
    b.eq(
        w,
        "naive with α ≡ 1 on the same nodes, fit from 2500",
        fit([r for r in ell[0.0] if r["n"] >= 2500], "uniform"),
        "4.98",
    )
    b.span(
        w,
        "naive parabolic ÷ elliptic at 1250, the jump and δ ≤ 0.01",
        [
            at(par[d], 1250)["naive/rms"] / at(ell[d], 1250)["naive/rms"]
            for d in (0.0, 0.01, 0.005, 0.0025)
        ],
        "1.7",
        "2.0",
    )
    b.le(
        w,
        "the naive operators' rightmost eigenvalue at 2500 (none growing)",
        max(
            r["max_re"]
            for r in f(NAIVE, "spectra")
            if r["n"] == 2500 and r["operator"] == "naive"
        ),
        "-7.3",
    )
    b.span(
        w,
        "the construction's flux reading while h ≥ 4δ, δ = 0.0025",
        [r["construction/flux"] for r in ell[0.0025] if r["h_over_delta"] >= 4],
        "0.018",
        "0.020",
    )
    b.span(
        w,
        "the construction's flux reading at h ≈ δ",
        [r["construction/flux"] for d in WIDTHS for r in near(ell[d], 1.05)],
        "0.81",
        "0.89",
    )

    # H5 and H6: the matrices (§6.4).
    rows = f("heat2d_stiff_eigenvalues.json", "rows")
    w = cited("stiff §4.4 H5", "§6.4")
    for label, low, high in (
        ("construction", "1.3e4", "1.7e4"),
        ("naive", "2.0e4", "5.5e4"),
    ):
        b.span(
            w,
            f"the {label}'s condition estimates at 2500",
            [r["cond"] for r in rows if r["label"] == label],
            low,
            high,
        )
    for label, low, high in (("seeds", "95", "101"), ("construction", "100", "147")):
        b.span(
            w,
            f"BiCGSTAB's iterations for the {label}, 2500",
            [r["bicgstab/none"]["iterations"] for r in rows if r["label"] == label],
            low,
            high,
        )
    b.eq(
        w,
        "solves that did not converge, every method, preconditioner and width",
        sum(
            r[f"{method}/{pre}"]["info"] != 0
            for name in (
                "heat2d_stiff_eigenvalues.json",
                "heat2d_stiff_eigenvalues_rows_n10000.json",
            )
            for r in f(name, "rows")
            for method in ("gmres", "bicgstab")
            for pre in ("none", "appendix-b", "spilu")
        ),
        "0",
    )
    w = cited("stiff §4.4 H6", "§6.4")
    (naive,) = [
        r
        for r in f("heat2d_stiff_eigenvalues.json", "spectra")
        if r["label"] == "naive" and r["delta"] == 0.0
    ]
    b.eq(
        w, "the naive operator's rightmost eigenvalue at 1600", naive["max_re"], "1003"
    )
    b.eq(w, "… and BD4's largest root there", naive["bd4"], "1.038")

    # H4 and H8: the flat sweep (§6.3).
    w = cited("stiff §4.5 H4", "§6.3")
    lines = {d: sweep(f, SEEDS, "elliptic", d) for d in (0.0, *WIDTHS)}
    b.span(
        w,
        "the seeds' rates between successive counts",
        [
            rate_in_h(a, c, "seeds/rms")
            for rows_ in lines.values()
            for a, c in zip(rows_, rows_[1:], strict=False)
        ],
        "2.6",
        "5.9",
    )
    for label, fits in (
        ("naive", ((0.0, "1.50"), (0.04, "5.18"))),
        ("direct", ((0.0025, "1.87"), (0.04, "4.24"))),
    ):
        for delta, quoted in fits:
            b.eq(
                w,
                f"{label}'s fit over 1250–40,000, δ = {delta:g}",
                fit(lines[delta], f"{label}/rms"),
                quoted,
            )
    w = cited("stiff §4.5 H8", "§6.3")
    for delta, quoted in ((0.0, "6"), (0.0025, "25"), (0.01, "61"), (0.04, "100")):
        b.eq(
            w,
            f"the seeded share of the rows at 40,000, δ = {delta:g}, %",
            100 * at(lines[delta], 40000)["seeds/rows"] / 40000,
            quoted,
        )
    b.span(
        w,
        "the seeds over naive at δ = 0.01, 20,000 and 40,000",
        [
            at(lines[0.01], n)["seeds/rms"] / at(lines[0.01], n)["naive/rms"]
            for n in (20000, 40000)
        ],
        "0.003",
        "0.004",
    )

    # H9 and H16: the curved feature (§6.5).
    w = cited("stiff §4.6 H9", "§6.5")
    b.each(
        w,
        "the frozen profile's fits, δ = 0, 0.01, 0.005, 0.0025",
        [fit(sweep(f, CURVED, "elliptic", d), "seeds/rms") for d in CURVED_WIDTHS],
        ["1.33", "3.07", "2.40", "1.71"],
    )
    smooth = [r for d in CURVED_WIDTHS[1:] for r in sweep(f, CURVED, "elliptic", d)]
    b.span(
        w,
        "the frozen profile over the naive operator, δ > 0",
        [r["seeds/rms"] / r["naive/rms"] for r in smooth],
        "0.04",
        "0.39",
    )
    for label, quoted in (("direct", "0.05"), ("construction", "0.46")):
        b.le(
            w,
            f"the frozen profile over the {label}, δ > 0, largest",
            max(r["seeds/rms"] / r[f"{label}/rms"] for r in smooth),
            quoted,
        )
    b.eq(
        w,
        "the jump-aware rows' probe fit on case 2",
        fit(sweep(f, TANGENTIAL, "elliptic", 0.0), "construction/probe_crossing"),
        "2.86",
    )
    for geometry, frozen in (("A", "0.36"), ("B", "0.18")):
        rows = sweep(f, SPLIT[geometry], "elliptic", 0.0)
        b.eq(
            w,
            f"the jump-aware rows' probe fit on {geometry}",
            fit(rows, "construction/probe_crossing"),
            "3.0",
        )
        b.eq(
            w,
            f"the frozen rows' probe fit on {geometry}",
            fit(rows, "seeds/probe_crossing"),
            frozen,
        )
    w = cited("stiff §4.7 H16", "§6.5")
    chain = {d: sweep(f, TANGENTIAL, "elliptic", d) for d in CURVED_WIDTHS}
    b.each(
        w,
        "the chain's fits from 5000, δ = 0, 0.01, 0.005, 0.0025",
        [fit(chain[d], "tangential/rms", COUNTS_40K[2:]) for d in CURVED_WIDTHS],
        ["4.96", "4.23", "3.86", "4.13"],
    )
    b.span(
        w,
        "the chain's fits over 1250–40,000",
        [fit(chain[d], "tangential/rms") for d in CURVED_WIDTHS],
        "4.24",
        "5.72",
    )
    line_b = f("heat2d_stiff_seeds_tangential_a0_sine_jump.json", "sweep")["elliptic"]
    line_b = line_b["0"]
    b.eq(
        w,
        "the chain on B at the jump to 160,000",
        fit(line_b, "tangential/rms"),
        "4.49",
    )
    b.eq(
        w,
        "the jump-aware operator on B to 160,000",
        fit(line_b, "construction/rms"),
        "4.21",
    )
    probe, error = [], []
    for name in (TANGENTIAL, SPLIT["A"], SPLIT["B"]):
        rows = sweep(f, name, "elliptic", 0.0025)
        first, last = at(rows, 20000), at(rows, 40000)
        probe.append(rate_in_h(first, last, "tangential/probe_crossing"))
        error.append(rate_in_h(first, last, "tangential/rms"))
    b.span(w, "the chain's probe, 20,000 → 40,000, δ = 0.0025", probe, "1.7", "2.2")
    b.span(w, "… and its errors over the same step", error, "2.5", "3.0")

    # H11: the ring (§6.6).
    w = cited("stiff §4.8 H11", "§6.6")
    (top,) = [
        r for r in f(RING, "conditioning") if r["delta"] == 0.0 and r["s"] == 1e11
    ]
    b.eq(w, "the seeds on the stored radii at s = 1e11", top["stored-worst"], "1.2e-10")
    convergence = f(RING, "convergence")
    b.span(
        w,
        "the seeds over the jump-aware operator at 1250 and 2500",
        [
            r["seeds"]["full"] / r["construction"]["full"]
            for rows in convergence.values()
            for r in rows
            if r["n"] <= 2500
        ],
        "1.03",
        "1.46",
    )
    probes = f(RING, "smooth")
    b.span(
        w,
        "the jump-aware rows over the seeds' on the probe at δ = 0",
        [
            r["probe-construction"]["seeded"] / r["probe-seeds"]["seeded"]
            for name in ("1e3|0", "1e11|0")
            for r in probes[name]
        ],
        "43",
        "144",
    )
    spectrum = [r for r in f(RING, "spectrum") if r["label"] == "seeds"]
    b.eq(
        w,
        "the seed operator's eigenvalues right of the axis, 5000, every (s, δ)",
        max(r["positive"] for r in spectrum),
        "0",
    )
    b.span(w, "its slowest mode", [r["max_re"] for r in spectrum], "-14.3", "-14.1")
    b.eq(
        w,
        "the seeds against the rule's fine run, s = 1e11, δ = 0.001, 40,000",
        at(probes["1e11|0.001"], 40000)["error-seeds"]["far"],
        "3.6e-6",
    )

    # H12: the treatments (§6.7).
    w = cited("stiff §4.9 H12", "§6.7")
    b.span(
        w,
        "T0 over sampling at the jump, from 5000",
        [
            r[label]
            for name in TREATMENTS.values()
            for p in PROBLEMS
            for r in f(name, f"ratios/{p}")
            if r["delta"] == 0.0 and r["n"] >= 5000
            for label in ("widened-1h", "widened-2h")
        ],
        "1.7",
        "8.2",
    )
    fits = []
    for p in PROBLEMS:
        rows = sweep(f, TREATMENTS["case 1"], p, 0.04)
        if p == "parabolic":  # the 1250-node growing mode (§4.9's GROWING_BELOW)
            rows = [r for r in rows if r["n"] >= 2500]
        fits += [
            fit(rows, f"{label}/{norm}")
            for label in TREATMENT_LABELS[:4]
            for norm in ("rms", "max")
        ]
    b.span(
        w,
        "the disc means' fits at δ = 0.04, both norms and problems",
        fits,
        "1.95",
        "2.05",
    )
    resolved = at(sweep(f, TREATMENTS["case 1"], "elliptic", 0.04), 40000)
    b.span(
        w,
        "the radius-h mean over the half-spacing one, δ = 0.04, 40,000",
        [
            resolved[f"{mean}-1h/rms"] / resolved[f"{mean}-0.5h/rms"]
            for mean in ("harmonic", "arithmetic")
        ],
        "4.00",
        "4.00",
    )
    b.span(
        w,
        "T0 (m = 2) over its own floor, case 1",
        [
            r["widened-2h/rms"] / r["widened-2h/own_floor"]
            for d in (0.0, *WIDTHS)
            for r in sweep(f, TREATMENTS["case 1"], "elliptic", d)
            if r.get("widened-2h/own_floor")
        ],
        "0.996",
        "1.000",
    )
    jump_row = at(sweep(f, TREATMENTS["case 1"], "elliptic", 0.0), 40000)
    b.span(
        w,
        "the flux on the first rows at the jump, 40,000, the treatments moving α",
        [
            jump_row[f"{label}/flux"]
            for label in ("harmonic-1h", "arithmetic-1h", "widened-1h", "widened-2h")
        ],
        "0.30",
        "0.59",
    )
    b.eq(w, "… and sampling's", jump_row["naive/flux"], "0.34")


STATEMENTS = (
    one_d,
    seeds_1d,
    results_1d,
    snapshot_2d,
    references_2d,
    naive_knee,
    construction_floor,
    seeds_flat,
    seeds_2d,
    elliptic_ranks,
    solvability,
    warp,
    curved_feature,
    ring,
    treatments,
    results_2d,
)


def build(data_dir: Path) -> list[Check]:
    """Every check, from the files in ``data_dir``."""
    f, b = Files(data_dir), Book()
    for statement in STATEMENTS:
        statement(f, b)
    return b.checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quiet", action="store_true", help="failures and the summary")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args(argv)
    problems = verify(args.data_dir)
    if problems:
        for p in problems:
            print("  " + p)
        print(f"{args.data_dir}: {len(problems)} problems; no number checked")
        return 1
    for name, changed in stale(args.data_dir).items():
        print(
            f"note: {name}: its code changed since its commit: {', '.join(changed)}",
            file=sys.stderr,
        )
    checks = build(args.data_dir)
    failed = [c for c in checks if not c.ok]
    for c in failed if args.quiet else checks:
        print(c.line())
    print(f"{len(checks) - len(failed)} of {len(checks)} numbers hold")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
