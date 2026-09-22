"""One JSON per driver run: the tables a driver printed, with their provenance.

Plan D1 and E5.3 (#44): every number the manuscript quotes traces to a
driver's tables, and the drivers' ``--data-dir`` writes those tables here as
one file per run, ``<driver>.json``, beside the screen output. The file
carries the driver's name, its parsed arguments, the date, the git commit
the run was made at (and whether the tree was clean), the phases' run
times and the tables themselves, so ``scripts/paper_numbers.py`` can assert
each quoted number against a committed copy under ``paper/data/`` and say
which commit produced it. The schema is this repo's own (plan D2).

    {"schema": 1, "driver": "heat1d_stiff", "date": "2026-09-22T18:04:11Z",
     "git": {"sha": "…", "dirty": false}, "args": {…}, "timings": {…},
     "tables": {name: rows}}

A table is whatever the driver keeps: a list of row dicts, or a dict of
them keyed by medium and by δ. ``jsonable`` makes it JSON: float keys are
written with ``%g`` (``0.0025``, not ``0.0025000000000000001``), numpy
scalars and arrays become Python numbers and lists, paths become strings.
The knee cache (``heat1d_stiff_knee.json``) is a different thing: a working
cache the driver reads back to skip marches; this file is never read by a
driver, only by the manuscript's scripts.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 1


def git_state(root: Path | None = None) -> dict[str, Any]:
    """``{"sha": HEAD, "dirty": uncommitted changes?}``; ``sha`` is None outside git."""
    root = Path(__file__).resolve().parents[2] if root is None else Path(root)
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}
    return {"sha": sha, "dirty": bool(status.strip())}


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
    namespace as a dict, ``vars(args)``.
    """

    driver: str
    args: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, table: Any) -> None:
        self.tables[name] = jsonable(table)

    def time(self, phase: str, seconds: float) -> None:
        self.timings[phase] = float(seconds)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "driver": self.driver,
            "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git": git_state(),
            "args": jsonable(self.args),
            "timings": dict(self.timings),
            "tables": self.tables,
        }

    def write(self, *paths: Path) -> list[Path]:
        """The same payload to every path: ``outputs/`` and, given, ``--data-dir``."""
        # A NaN or an infinity would be written as a token no strict parser
        # reads; the number check must see it fail here instead.
        text = json.dumps(self.payload(), indent=1, allow_nan=False) + "\n"
        written = []
        for path in paths:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            written.append(path)
        return written


def read_results(path: Path) -> dict[str, Any]:
    """A results file, checked to be one (``schema`` and ``tables`` present)."""
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA or "tables" not in data:
        raise ValueError(f"{path} is not a results file of schema {SCHEMA}")
    return data
