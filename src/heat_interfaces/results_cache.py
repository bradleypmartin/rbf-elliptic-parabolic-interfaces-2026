"""One JSON per driver run: the tables a driver printed, with their provenance.

Plan D1 and E5.3 (#44): every number the manuscript quotes traces to a
driver's tables, and the drivers' ``--data-dir`` writes those tables here as
one file per run, ``<driver>.json``, beside the screen output. The file
carries the driver's name, the command line it was run with and its parsed
arguments, the date, the git commit the run was made at (and whether the
tree was clean), the phases' run times and the tables themselves, so
``scripts/paper_numbers.py`` can assert each quoted number against a
committed copy under ``paper/data/``, say which commit produced it, and
tell a documented run's file from a near-miss's. The schema is this repo's
own (plan D2).

    {"schema": 2, "driver": "heat1d_stiff", "date": "2026-09-22T18:04:11Z",
     "git": {"sha": "…", "dirty": false}, "argv": ["--counts", …],
     "args": {…}, "timings": {…}, "tables": {name: rows}}

A table is whatever the driver keeps: a list of row dicts, or a dict of
them keyed by medium and by δ. ``jsonable`` makes it JSON: float keys are
written with ``%g`` (``0.0025``, not ``0.0025000000000000001``), numpy
scalars and arrays become Python numbers and lists, paths become strings.
The arrays a figure draws (``…/curves``, ``…/field`` tables) are kept to six
figures with ``rounded``, so that ``scripts/paper_figures.py`` draws every
manuscript figure from these files alone. The knee cache
(``heat1d_stiff_knee.json``) is a different thing: a working cache the
driver reads back to skip marches; this file is never read by a driver, only
by the manuscript's scripts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 2
"""Schema 2 (E5.3) is schema 1 (E3.6) with ``argv``, the command line."""

READABLE = frozenset({1, 2})
"""The schemas ``read_results`` accepts; the number check asks for ``SCHEMA``."""

WHERE = frozenset({"outputs", "data_dir"})
"""Arguments that say where a run writes, not what it computed: a rerun that
differs only in them is the same run (``write``'s overwrite warning)."""

WHERE_FLAGS = ("--outputs", "--data-dir")
"""``WHERE`` on the command line, which ``command`` drops."""

ROOT = Path(__file__).resolve().parents[2]

RUN_OUTPUTS = ("paper/data",)
"""Where the documented runs write their own results files (E5.3). The dirty flag
ignores them: rerunning the documented commands one after another, the second
would otherwise record the first one's file as an uncommitted change."""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True
    )


def git_state(root: Path | None = None) -> dict[str, Any]:
    """``{"sha": HEAD, "dirty": uncommitted changes?}``; ``sha`` is None outside git.

    ``dirty`` counts every change but under ``RUN_OUTPUTS``.
    """
    root = ROOT if root is None else Path(root)
    try:
        head = _git(root, "rev-parse", "HEAD")
        excluded = [f":(exclude){path}" for path in RUN_OUTPUTS]
        status = _git(root, "status", "--porcelain", "--", ".", *excluded)
    except OSError:
        return {"sha": None, "dirty": None}
    if head.returncode or status.returncode:
        return {"sha": None, "dirty": None}
    return {"sha": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


def command(argv: Sequence[str]) -> list[str]:
    """``argv`` without ``WHERE_FLAGS`` and their values: what a run computed."""
    out, skip = [], False
    for token in argv:
        if skip:
            skip = False
        elif token in WHERE_FLAGS:
            skip = True
        elif not token.startswith(tuple(f"{flag}=" for flag in WHERE_FLAGS)):
            out.append(token)
    return out


def provenance(data: dict[str, Any], root: Path | None = None) -> list[str]:
    """What stops a results file from backing a quoted number; empty when nothing.

    The file must be of this ``SCHEMA`` (so it names its command line), made in
    a git checkout from a clean tree (``git_state``), at a commit in the
    history of this checkout's HEAD: a run made on a branch that was later
    rewritten, or on another clone's unpushed commit, cannot be traced.
    """
    root = ROOT if root is None else Path(root)
    problems = []
    if data.get("schema") != SCHEMA:
        problems.append(f"schema {data.get('schema')}, not {SCHEMA}")
    git = data.get("git") or {}
    sha = git.get("sha")
    if sha is None:
        problems.append("made outside a git checkout")
    else:
        if git.get("dirty") is not False:
            problems.append(f"made from a tree with uncommitted changes (at {sha:.9})")
        if _git(root, "merge-base", "--is-ancestor", sha, "HEAD").returncode:
            problems.append(f"commit {sha:.9} is not in the history of HEAD")
    return problems


def changed_since(
    sha: str, paths: Iterable[str], root: Path | None = None
) -> list[str]:
    """The files under ``paths`` that differ between ``sha`` and the working tree.

    Informational: a changed file may or may not move a number (a docstring
    does not), and the working caches key on labels, not on code, so rerunning
    a documented command after such a change reprints the cached numbers
    anyway (stiff note §5.1's caveat). ``scripts/paper_numbers.py`` lists
    them; it is for the reader to judge whether a cache version needs a bump.
    """
    root = ROOT if root is None else Path(root)
    diff = _git(root, "diff", "--name-only", sha, "--", *paths)
    return diff.stdout.split() if diff.returncode == 0 else []


def source_hash(paths: Iterable[Path]) -> str:
    """SHA-256 over the files' names and bytes, in sorted order.

    A working cache stores this for the code its entries depend on and
    discards itself when it changes (E5.3, the 1-D knee cache).
    """
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def rounded(values: Any, digits: int = 6) -> Any:
    """``values`` (an array or nested lists) as lists of floats to ``digits`` figures.

    For the arrays a figure draws: six figures are below what a line or a
    colour shows, and they keep the committed files a few hundred kilobytes.
    """
    array = np.asarray(values, dtype=float)
    flat = [float(f"{v:.{digits}g}") for v in array.ravel()]
    return np.array(flat).reshape(array.shape).tolist()


def float_keys(table: dict[str, Any]) -> dict[Any, Any]:
    """One level of ``table`` with its numeric keys read back as floats.

    ``jsonable`` writes a float key as ``%g``; a figure or a check that looks
    rows up by δ or by s wants the number back. Other keys stay strings.
    """
    out: dict[Any, Any] = {}
    for key, value in table.items():
        try:
            out[float(key)] = value
        except ValueError:
            out[key] = value
    return out


def jsonable(obj: Any) -> Any:
    """``obj`` with numpy, paths and float keys turned into what ``json`` writes."""
    if isinstance(obj, dict):
        return {_key(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def finite(table: Any, placeholders: Collection[str], key: str | None = None) -> Any:
    """``table`` with a driver's non-finite placeholders as None; any other raises.

    A driver's printed tables may use ``inf`` or ``nan`` for "not applicable"
    (the jump's h/δ, a rate with no earlier count), which strict JSON cannot
    hold and ``write`` refuses. The keys a driver names in ``placeholders``
    become null; a non-finite value under any other key is a failed solve or
    march, and it raises here, naming the key, rather than being recorded. A
    value in a list takes the key of the dict it sits under, and a dict keyed
    by numbers (δ, counts) keeps its parent's.
    """
    if isinstance(table, dict):
        return {
            k: finite(v, placeholders, k if isinstance(k, str) else key)
            for k, v in table.items()
        }
    if isinstance(table, (list, tuple)):
        return [finite(v, placeholders, key) for v in table]
    if isinstance(table, (float, np.floating)) and not np.isfinite(table):
        if key in placeholders:
            return None
        raise ValueError(f"{key!r} is {table}, and it is not a placeholder")
    return table


def _key(k: Any) -> str:
    if isinstance(k, (float, np.floating)):
        return f"{float(k):g}"
    if isinstance(k, (int, np.integer)):
        return str(int(k))
    return str(k)


@dataclass
class ResultsCache:
    """The tables of one driver run, written by ``write``.

    ``add`` stores a table under a name (a later ``add`` with the same name
    replaces it); ``time`` records a phase's seconds. ``args`` is the parsed
    namespace as a dict, ``vars(args)``; ``argv`` the command line the driver's
    ``main`` was given, ``sys.argv[1:]`` when it was given none.
    """

    driver: str
    args: dict[str, Any] = field(default_factory=dict)
    argv: Sequence[str] | None = None
    tables: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, table: Any) -> None:
        self.tables[name] = jsonable(table)

    def time(self, phase: str, seconds: float) -> None:
        self.timings[phase] = float(seconds)

    def payload(self) -> dict[str, Any]:
        argv = sys.argv[1:] if self.argv is None else self.argv
        return {
            "schema": SCHEMA,
            "driver": self.driver,
            "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git": git_state(),
            "argv": [str(a) for a in argv],
            "args": jsonable(self.args),
            "timings": dict(self.timings),
            "tables": self.tables,
        }

    def write(self, *paths: Path) -> list[Path]:
        """The same payload to every path: ``outputs/`` and, given, ``--data-dir``.

        A file already there from a run with other arguments (``WHERE`` aside)
        is overwritten with a warning naming them: a results file holds the
        last run of its name, and a near-miss of a documented command must not
        replace the manuscript's tables in silence (E4.10, the /spar review).
        """
        payload = self.payload()
        text = dumps(payload) + "\n"
        written = []
        for path in paths:
            path = Path(path)
            changed = changed_args(path, payload["args"])
            if changed:
                warnings.warn(
                    f"{path}: overwriting a run with other {', '.join(changed)}",
                    stacklevel=2,
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            written.append(path)
        return written


def dumps(obj: Any, level: int = 0) -> str:
    """``json.dumps(obj, indent=1)``, but a list of scalars on one line.

    The figures' arrays (thousands of numbers) and the command line read as
    one line each, and the rows and tables keep one key per line. A NaN or an
    infinity would be written as a token no strict parser reads: it raises
    here instead (``allow_nan=False``), so the number check never sees one.
    """
    pad = " " * (level + 1)
    if isinstance(obj, dict) and obj:
        items = [f"{pad}{json.dumps(k)}: {dumps(v, level + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * level + "}"
    if isinstance(obj, list) and any(isinstance(v, (dict, list)) for v in obj):
        items = [pad + dumps(v, level + 1) for v in obj]
        return "[\n" + ",\n".join(items) + "\n" + " " * level + "]"
    return json.dumps(obj, allow_nan=False)


def changed_args(path: Path, args: dict[str, Any]) -> list[str]:
    """The arguments (``WHERE`` aside) in which ``path``'s run differs from ``args``.

    Empty when there is no file; ``["the file"]`` when it is not a results file.
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        old = read_results(path)["args"]
    except (ValueError, KeyError, TypeError):
        return ["the file"]
    keys = (set(old) | set(args)) - WHERE
    return sorted(k for k in keys if old.get(k) != args.get(k))


def read_results(path: Path) -> dict[str, Any]:
    """A results file, checked to be one (a ``READABLE`` schema, ``tables`` present)."""
    data = json.loads(Path(path).read_text())
    if data.get("schema") not in READABLE or "tables" not in data:
        raise ValueError(f"{path} is not a results file of schema {SCHEMA}")
    return data
