"""Regenerate ``paper/data/`` from the documented runs (E5.3, #44).

The manuscript's data is one results file per documented run: the 1-D study's
(stiff note §2.5), the 2-D study's (§5.1's table, the commands in
``heat2d_stiff.py``'s and ``heat2d_stiff_eigenvalues.py``'s docstrings) and the
ring's (§4.8, ``heat2d_ring.py``'s docstring). ``RUNS`` lists each command and
the file it writes. This script reruns them with ``--data-dir paper/data``
from the working caches under ``outputs/``: about 13 minutes, 8.7 of them
``--mode tangential`` and a minute each for ``--mode references --amplitude
0.02`` and the 1-D driver, none of which is served from a cache whole. Cold,
the caches cost what the notes record (hours for the sweeps, some 20
CPU-hours for the ring). It refuses a tree with uncommitted changes outside
``paper/data`` (the files would record ``dirty: true``), and writes each
run's screen output to ``outputs/paper_data/<file>.log``.

``verify`` is the gate ``scripts/paper_numbers.py`` runs before any number:
the data directory holds exactly these files, each written by its driver
from its command (``argv``, ``WHERE`` aside), with its tables matching their
checksum, from a clean tree, at a commit in the history of HEAD, and naming
no absolute path (the runs are given ``paper/data`` relative to the root, so
the files are the same from any checkout and carry no machine's home). A file
holds the last run of its name, so a near-miss of a documented command (fewer
counts, another seed) run with ``--data-dir paper/data`` is refused here, not
quoted.

    uv run python scripts/paper_data.py              # every run, ~13 min cached
    uv run python scripts/paper_data.py --only heat2d_stiff_snapshot.json
    uv run python scripts/paper_data.py --list       # the commands
    uv run python scripts/paper_data.py --verify     # the files against RUNS
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from heat_interfaces.results_cache import (
    WHERE,
    WHERE_FLAGS,
    changed_since,
    command,
    git_state,
    provenance,
    read_results,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data"
LOGS = ROOT / "outputs" / "paper_data"

COUNTS_40K = ("1250", "2500", "5000", "10000", "20000", "40000")
COUNTS_160K = (*COUNTS_40K, "80000", "160000")
SEED_COUNTS = ("1250", "2500", "5000", "10000", "20000")

IMPORTS = {"heat2d_ring.py": ("heat2d_extremes.py",)}
"""The drivers a driver imports (``Run.code``)."""


@dataclass(frozen=True)
class Run:
    """One documented command: ``uv run python scripts/<script> <argv>``."""

    script: str
    argv: tuple[str, ...]
    results: str
    note: str = ""

    @property
    def driver(self) -> str:
        return Path(self.script).stem

    @property
    def code(self) -> tuple[str, ...]:
        """What the file's numbers are computed by: the package and its driver's."""
        scripts = (self.script, *IMPORTS.get(self.script, ()))
        return ("src/heat_interfaces", *(f"scripts/{s}" for s in scripts))

    def line(self) -> str:
        return " ".join(["scripts/" + self.script, *self.argv])


def _stiff(*argv: str, results: str, note: str = "") -> Run:
    return Run("heat2d_stiff.py", argv, results, note)


def _eig(*argv: str, results: str, note: str = "") -> Run:
    return Run("heat2d_stiff_eigenvalues.py", argv, results, note)


_TANGENTIAL = ("naive", "construction", "seeds", "tangential", "tangential-plain")

RUNS: tuple[Run, ...] = (
    Run("heat1d_stiff.py", (), "heat1d_stiff.json", "stiff note §2.1–§2.5"),
    _stiff(
        *("--mode", "references", "--deltas"),
        *("0", "0.04", "0.01", "0.005", "0.0025", "0.002", "0.001", "0.0005"),
        results="heat2d_stiff_references.json",
        note="§4.1",
    ),
    _stiff(
        *("--mode", "naive", "--counts", *COUNTS_160K),
        results="heat2d_stiff_naive.json",
        note="§4.2, the knee to 160,000",
    ),
    _stiff(
        *("--mode", "naive", "--seed", "1", "--counts", *SEED_COUNTS),
        "--spectrum-counts",
        results="heat2d_stiff_naive_seed1.json",
        note="§4.2, the scatter",
    ),
    _stiff(
        *("--mode", "naive", "--seed", "2", "--counts", *SEED_COUNTS),
        "--spectrum-counts",
        results="heat2d_stiff_naive_seed2.json",
        note="§4.2, the scatter",
    ),
    _stiff("--mode", "stencils", results="heat2d_stiff_stencils.json", note="§4.3"),
    _stiff("--mode", "snapshot", results="heat2d_stiff_snapshot.json", note="§5.1"),
    _stiff(
        *("--mode", "seeds", "--operators", "naive", "construction", "direct"),
        *("direct-reach", "seeds", "seeds-plain", "seeds-edge"),
        *("--counts", *COUNTS_40K),
        results="heat2d_stiff_seeds.json",
        note="§4.5, §4.10",
    ),
    _stiff(
        *("--mode", "seeds", "--deltas", "0"),
        *("--operators", "naive", "construction", "seeds"),
        *("--counts", *COUNTS_160K),
        results="heat2d_stiff_seeds_jump.json",
        note="§4.5, H4 to 160,000",
    ),
    _stiff(
        *("--mode", "references", "--amplitude", "0.02"),
        results="heat2d_stiff_references_a0.02_sine.json",
        note="§4.6, about a minute (the product grids do not cache)",
    ),
    _stiff(
        *("--mode", "seeds", "--amplitude", "0.02", "--operators", "naive"),
        *("construction", "construction-flat", "direct", "direct-reach", "seeds"),
        *("seeds-plain", "--counts", *COUNTS_40K),
        results="heat2d_stiff_seeds_a0.02_sine.json",
        note="§4.6, route (a)",
    ),
    _stiff(
        *("--mode", "seeds", "--amplitude", "0.02", "--operators", "naive"),
        *("construction", "direct", "direct-reach", "seeds", "seeds-plain"),
        *("tangential", "tangential-plain", "tangential-edge"),
        *("--counts", *COUNTS_40K),
        results="heat2d_stiff_seeds_tangential_a0.02_sine.json",
        note="§4.7, §4.10, the tangential chain on case 2",
    ),
    _stiff(
        *("--mode", "seeds", "--amplitude", "0.02", "--inside", "constant"),
        *("--deltas", "0", "0.0025", "--operators", *_TANGENTIAL),
        *("--counts", *COUNTS_40K),
        results="heat2d_stiff_seeds_tangential_a0.02_constant.json",
        note="§4.7, geometry A",
    ),
    _stiff(
        *("--mode", "seeds", "--inside", "sine", "--deltas", "0", "0.0025"),
        *("--operators", *_TANGENTIAL, "--counts", *COUNTS_40K),
        results="heat2d_stiff_seeds_tangential_a0_sine.json",
        note="§4.7, geometry B",
    ),
    _stiff(
        *("--mode", "seeds", "--inside", "sine", "--deltas", "0"),
        *("--operators", *_TANGENTIAL, "--counts", *COUNTS_160K),
        results="heat2d_stiff_seeds_tangential_a0_sine_jump.json",
        note="§4.7, geometry B's δ = 0 to 160,000",
    ),
    _stiff(
        *("--mode", "tangential", "--counts", *COUNTS_40K),
        results="heat2d_stiff_tangential.json",
        note="§4.7, H14–H15; 8.7 min, not cached",
    ),
    _stiff(
        *("--mode", "treatments", "--counts", *COUNTS_40K),
        results="heat2d_stiff_treatments.json",
        note="§4.9, case 1",
    ),
    _stiff(
        *("--mode", "treatments", "--amplitude", "0.02", "--counts", *COUNTS_40K),
        results="heat2d_stiff_treatments_a0.02_sine.json",
        note="§4.9, case 2",
    ),
    _eig(results="heat2d_stiff_eigenvalues.json", note="§4.4, 1600 and 2500 nodes"),
    _eig(
        *("--mode", "rows", "--n", "10000"),
        results="heat2d_stiff_eigenvalues_rows_n10000.json",
        note="§4.4",
    ),
    _eig(
        *("--mode", "spectra", "--spectrum-n", "4900"),
        *("--deltas", "0", "0.005", "--figure-delta", "0.005"),
        results="heat2d_stiff_eigenvalues_spectra_n4900.json",
        note="§4.4",
    ),
    Run(
        "heat2d_ring.py",
        (
            *("--s", "1e3", "1e8", "1e9", "1e10", "1e11"),
            *("--counts", "1250", "2500", "5000", "10000", "20000", "40000", "80000"),
            *("--reference-n", "160000", "--conditioning-s", "1e3", "1e4", "1e5"),
            *("1e6", "1e7", "1e8", "1e9", "1e10", "1e11"),
            *("--conditioning-smooth-s", "1e3", "1e7", "1e11"),
            *("--conditioning-n", "10000", "--deltas", "0.0025", "0.001", "0.00025"),
            *("--smooth-s", "1e3", "1e11"),
            *("--probe-counts", "2500", "5000", "10000", "20000", "40000"),
            *("--fine-n", "160000", "--plain-fine", "1e11:0.001"),
            *("--spectrum-n", "5000"),
        ),
        "heat2d_ring_results.json",
        "§4.8, §4.10",
    ),
)

BY_NAME = {run.results: run for run in RUNS}


def absolute_paths(data: dict) -> list[str]:
    """The absolute paths a results file records where a run wrote (``WHERE``)."""
    argv = list(data.get("argv") or [])
    values = [b for a, b in zip(argv, argv[1:], strict=False) if a in WHERE_FLAGS]
    joined = tuple(f"{flag}=" for flag in WHERE_FLAGS)
    values += [t.split("=", 1)[1] for t in argv if t.startswith(joined)]
    values += [str(v) for k, v in (data.get("args") or {}).items() if k in WHERE and v]
    return sorted({v for v in values if Path(v).is_absolute()})


def relative(path: Path, root: Path = ROOT) -> str:
    """``path`` relative to ``root`` when inside it (what a run is told)."""
    path = Path(path).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def verify(data_dir: Path = DATA, root: Path = ROOT) -> list[str]:
    """Every reason ``data_dir`` is not the documented runs' files; empty if none."""
    data_dir = Path(data_dir)
    problems = []
    present = {p.name for p in data_dir.glob("*.json")}
    for name in sorted(present - set(BY_NAME)):
        problems.append(f"{name}: not a documented run's file")
    for run in RUNS:
        path = data_dir / run.results
        if not path.exists():
            problems.append(f"{run.results}: missing (run {run.line()})")
            continue
        try:
            data = read_results(path)
        except (ValueError, OSError) as err:
            problems.append(f"{run.results}: {err}")
            continue
        if data.get("driver") != run.driver:
            problems.append(f"{run.results}: written by {data.get('driver')}")
        if command(data.get("argv") or []) != list(run.argv):
            problems.append(
                f"{run.results}: made by {' '.join(data.get('argv') or ['?'])!r},"
                f" not the documented {' '.join(run.argv)!r}"
            )
        problems.extend(f"{run.results}: {p}" for p in provenance(data, root))
        for path in absolute_paths(data):
            problems.append(f"{run.results}: records the absolute path {path}")
    return problems


def stale(data_dir: Path = DATA, root: Path = ROOT) -> dict[str, list[str]]:
    """``{file: its run's code changed since the file's commit}``, where any did.

    Informational (``changed_since``): the working caches key on labels, so a
    rerun after such a change may reprint the same numbers; the reader judges.
    """
    out: dict[str, list[str]] = {}
    for run in RUNS:
        path = Path(data_dir) / run.results
        if not path.exists():
            continue
        sha = (read_results(path).get("git") or {}).get("sha")
        changed = changed_since(sha, run.code, root) if sha else []
        if changed:
            out[run.results] = changed
    return out


def run(runs: list[Run], data_dir: Path) -> None:
    """Each run, one after another, writing into ``data_dir``; the log per run."""
    LOGS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for k, r in enumerate(runs, 1):
        t0 = time.perf_counter()
        log = LOGS / (Path(r.results).stem + ".log")
        print(f"[{k}/{len(runs)}] {r.line()}", flush=True)
        with log.open("w") as out:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / r.script), *r.argv]
                + ["--data-dir", relative(data_dir)],
                cwd=ROOT,
                stdout=out,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"    {r.results} in {time.perf_counter() - t0:.1f} s", flush=True)
    print(f"{len(runs)} runs in {time.perf_counter() - start:.0f} s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="+", metavar="FILE", choices=sorted(BY_NAME))
    parser.add_argument("--list", action="store_true", help="print the commands")
    parser.add_argument("--verify", action="store_true", help="check, do not run")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run on a tree with uncommitted changes (the files will not verify)",
    )
    args = parser.parse_args(argv)
    if args.list:
        for r in RUNS:
            print(f"{r.results:48s} {r.note}\n    uv run python {r.line()}")
        return 0
    if args.verify:
        problems = verify(args.data_dir)
        for p in problems:
            print("  " + p)
        print(f"{len(RUNS)} documented runs; {len(problems)} problems")
        return 1 if problems else 0
    if git_state(ROOT)["dirty"] and not args.allow_dirty:
        parser.error(
            "the tree has uncommitted changes outside paper/data: commit them first,"
            " or the results files record dirty: true"
        )
    runs = [BY_NAME[name] for name in args.only] if args.only else list(RUNS)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    run(runs, args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
