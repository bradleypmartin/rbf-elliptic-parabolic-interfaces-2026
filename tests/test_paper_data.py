"""``scripts/paper_data.py`` (E5.3, #44): the documented runs and their gate."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat_interfaces.results_cache import SCHEMA, ResultsCache, git_state  # noqa: E402
from paper_data import (  # noqa: E402
    BY_NAME,
    ROOT,
    RUNS,
    absolute_paths,
    main,
    relative,
    stale,
    verify,
)


def test_the_runs_are_the_notes_documented_commands():
    # Stiff note §2.5 (1-D), §5.1's table (2-D, 21 files) and the ring's.
    assert len(RUNS) == len(BY_NAME) == 22
    for run in RUNS:
        assert (ROOT / "scripts" / run.script).exists()
        assert "--data-dir" not in run.argv and "--outputs" not in run.argv
        assert run.results.startswith(run.driver.removesuffix("_results"))
    assert {run.driver for run in RUNS} == {
        "heat1d_stiff",
        "heat2d_stiff",
        "heat2d_stiff_eigenvalues",
        "heat2d_ring",
    }


def _file(data_dir: Path, run, argv=None, git=None) -> Path:
    cache = ResultsCache(run.driver, {}, [*(run.argv if argv is None else argv)])
    payload = cache.payload() | ({"git": git} if git else {})
    path = data_dir / run.results
    path.write_text(json.dumps(payload))
    return path


def test_verify_passes_the_documented_files_and_names_every_other(tmp_path):
    head = git_state(ROOT)["sha"]
    clean = {"sha": head, "dirty": False}
    for run in RUNS:
        _file(tmp_path, run, [*run.argv, "--data-dir", "paper/data"], clean)
    assert verify(tmp_path) == []

    naive = BY_NAME["heat2d_stiff_naive.json"]
    _file(tmp_path, naive, ["--mode", "naive", "--counts", "1250"], clean)
    (tmp_path / "heat2d_stiff.json").write_text("{}")
    _file(tmp_path, RUNS[0], [], {"sha": head, "dirty": True})
    (tmp_path / "heat2d_ring_results.json").unlink()
    problems = verify(tmp_path)
    assert len(problems) == 4
    assert problems[0] == "heat2d_stiff.json: not a documented run's file"
    assert problems[1].startswith("heat1d_stiff.json: made from a tree with")
    assert problems[2].startswith(
        "heat2d_stiff_naive.json: made by '--mode naive --counts 1250'"
    )
    assert problems[3].startswith("heat2d_ring_results.json: missing")


def test_verify_refuses_schema_1_and_another_driver(tmp_path):
    head = git_state(ROOT)["sha"]
    for run in RUNS:
        _file(tmp_path, run, git={"sha": head, "dirty": False})
    path = tmp_path / RUNS[0].results
    old = json.loads(path.read_text()) | {"schema": 1, "driver": "heat2d_stiff"}
    path.write_text(json.dumps(old))
    assert verify(tmp_path) == [
        "heat1d_stiff.json: written by heat2d_stiff",
        f"heat1d_stiff.json: schema 1, not {SCHEMA}",
    ]


def test_main_lists_verifies_and_refuses_a_dirty_tree(tmp_path, capsys, monkeypatch):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert out.count("uv run python scripts/") == len(RUNS)
    assert main(["--verify", "--data-dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"{len(RUNS)} documented runs; {len(RUNS)} problems" in out
    monkeypatch.setattr(
        "paper_data.git_state", lambda root: {"sha": "0" * 40, "dirty": True}
    )
    with pytest.raises(SystemExit):
        main(["--only", "heat2d_stiff_snapshot.json", "--data-dir", str(tmp_path)])
    assert "uncommitted changes" in capsys.readouterr().err


def test_the_list_runs_from_the_command_line():
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "paper_data.py"), "--list"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "heat2d_ring_results.json" in out


def test_stale_names_each_files_changed_code(tmp_path):
    # A file's run is computed by the package and its own driver (the ring's
    # and heat2d_extremes'); another driver's change does not concern it.
    ring = BY_NAME["heat2d_ring_results.json"]
    assert ring.code == (
        "src/heat_interfaces",
        "scripts/heat2d_ring.py",
        "scripts/heat2d_extremes.py",
    )
    head = git_state(ROOT)["sha"]
    _file(tmp_path, ring, git={"sha": head, "dirty": False})
    # At HEAD only uncommitted edits to the ring's own code could show.
    assert all(f.startswith(ring.code) for f in stale(tmp_path).get(ring.results, []))
    first = subprocess.run(
        ["git", "-C", str(ROOT), "rev-list", "--max-parents=0", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()[0]
    _file(tmp_path, ring, git={"sha": first, "dirty": False})
    assert "scripts/heat2d_ring.py" in stale(tmp_path)[ring.results]


def test_the_files_name_no_absolute_path(tmp_path, monkeypatch):
    # The /spar review of E5.3: every file had recorded the checkout's absolute
    # --data-dir (a home directory, in a public repository, and bytes that
    # differ from checkout to checkout). The runs are told paper/data.
    head = git_state(ROOT)["sha"]
    clean = {"sha": head, "dirty": False}
    for run in RUNS:
        _file(tmp_path, run, [*run.argv, "--data-dir", "paper/data"], clean)
    assert verify(tmp_path) == []
    run = RUNS[0]
    _file(tmp_path, run, ["--data-dir", "/home/someone/paper/data"], clean)
    assert verify(tmp_path) == [
        "heat1d_stiff.json: records the absolute path /home/someone/paper/data"
    ]
    data = {"argv": ["--data-dir=/a/b"], "args": {"outputs": "/c", "counts": "/d"}}
    assert absolute_paths(data) == ["/a/b", "/c"]
    assert relative(ROOT / "paper" / "data") == "paper/data"
    assert relative(tmp_path) == str(tmp_path.resolve())
    calls = []
    monkeypatch.setattr(
        "paper_data.subprocess.run", lambda cmd, **kw: calls.append(cmd)
    )
    monkeypatch.setattr("paper_data.LOGS", tmp_path / "logs")
    from paper_data import run as run_all

    run_all([BY_NAME["heat2d_stiff_snapshot.json"]], ROOT / "paper" / "data")
    assert calls[0][-2:] == ["--data-dir", "paper/data"]
