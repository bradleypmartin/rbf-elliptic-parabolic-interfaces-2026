"""``scripts/paper_data.py`` (E5.3, #44): the documented runs and their gate."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat_interfaces.results_cache import SCHEMA, ResultsCache, git_state  # noqa: E402
from paper_data import BY_NAME, ROOT, RUNS, main, verify  # noqa: E402


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
