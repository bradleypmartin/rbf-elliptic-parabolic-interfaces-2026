"""``results_cache`` (plan D1; E5.3, #44): the schema of a driver's results file."""

import json
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from heat_interfaces.results_cache import (
    SCHEMA,
    ResultsCache,
    changed_since,
    command,
    dumps,
    finite,
    float_keys,
    git_state,
    jsonable,
    provenance,
    read_results,
    rounded,
    source_hash,
)


def test_jsonable_turns_numpy_paths_and_float_keys_into_json():
    obj = {
        0.0025: [{"n": np.int64(3), "e": np.float64(1e-3), "x": np.arange(2.0)}],
        1: Path("a/b"),
        "t": (1, 2),
        np.float64(0.1): None,
    }
    out = jsonable(obj)
    assert out == {
        "0.0025": [{"n": 3, "e": 1e-3, "x": [0.0, 1.0]}],
        "1": "a/b",
        "t": [1, 2],
        "0.1": None,
    }
    json.dumps(out)


def test_write_and_read_round_trip_with_the_provenance(tmp_path):
    argv = ["--deltas", "0", "0.01", "--data-dir", "paper/data"]
    cache = ResultsCache(
        "demo", {"outputs": Path("outputs"), "deltas": [0.0, 0.01]}, argv
    )
    cache.add("knee", {"matlab": {0.01: [{"n": 50, "naive": 1e-3}]}})
    cache.add("knee", {"matlab": {0.01: [{"n": 50, "naive": 2e-3}]}})
    cache.time("knee", 1.5)
    paths = cache.write(tmp_path / "a" / "demo.json", tmp_path / "b" / "demo.json")
    assert all(p.exists() for p in paths)
    assert paths[0].read_text() == paths[1].read_text()
    data = read_results(paths[0])
    # Schema 2, every key in its place (E5.3's test of the cache schema).
    assert list(data) == [
        "schema",
        "driver",
        "date",
        "git",
        "argv",
        "args",
        "timings",
        "tables",
    ]
    assert data["schema"] == SCHEMA == 2 and data["driver"] == "demo"
    datetime.strptime(data["date"], "%Y-%m-%dT%H:%M:%SZ")
    assert data["argv"] == argv
    assert data["args"] == {"outputs": "outputs", "deltas": [0.0, 0.01]}
    assert data["timings"] == {"knee": 1.5}
    assert data["tables"] == {"knee": {"matlab": {"0.01": [{"n": 50, "naive": 2e-3}]}}}
    assert set(data["git"]) == {"sha", "dirty"}
    # A driver's main given no argv records the process's own.
    no_argv = ResultsCache("demo").write(tmp_path / "c.json")[0]
    assert read_results(no_argv)["argv"] == sys.argv[1:]
    (tmp_path / "other.json").write_text("{}")
    with pytest.raises(ValueError, match="not a results file"):
        read_results(tmp_path / "other.json")
    # Schema 1 (E3.6–E4.10's files) still reads; it has no command line.
    old = {k: v for k, v in data.items() if k != "argv"} | {"schema": 1}
    (tmp_path / "old.json").write_text(json.dumps(old))
    assert read_results(tmp_path / "old.json")["schema"] == 1


def test_write_refuses_a_nan_or_an_infinity(tmp_path):
    cache = ResultsCache("demo")
    cache.add("bad", [{"local": np.float64("nan")}])
    with pytest.raises(ValueError):
        cache.write(tmp_path / "demo.json")
    cache.add("bad", [{"rate": float("inf")}])
    with pytest.raises(ValueError):
        cache.write(tmp_path / "demo.json")
    assert not (tmp_path / "demo.json").exists()


def test_finite_nulls_the_named_placeholders_and_refuses_any_other(tmp_path):
    # E4.10: the 2-D drivers print inf and nan for "not applicable" (the
    # jump's h/δ, an empty row group); those keys become null, and a
    # non-finite value anywhere else is a failed run that must not be written.
    table = {
        0.0: [{"h_over_delta": float("inf"), "rms": 1e-5}],
        0.01: [{"h_over_delta": 2.1, "least": np.float64("nan"), "rms": 2e-6}],
    }
    clean = finite(table, {"h_over_delta", "least"}, "knee")
    assert clean[0.0] == [{"h_over_delta": None, "rms": 1e-5}]
    assert clean[0.01] == [{"h_over_delta": 2.1, "least": None, "rms": 2e-6}]
    cache = ResultsCache("demo")
    cache.add("knee", clean)
    cache.write(tmp_path / "demo.json")
    assert read_results(tmp_path / "demo.json")["tables"]["knee"]["0"][0] == {
        "h_over_delta": None,
        "rms": 1e-5,
    }
    with pytest.raises(ValueError, match="'rms'"):
        finite({0.0: [{"rms": float("nan")}]}, {"h_over_delta"}, "knee")
    # A table that is itself a list of numbers takes the table's name.
    assert finite([1.0, float("inf")], {"rates"}, "rates") == [1.0, None]
    with pytest.raises(ValueError, match="'knee'"):
        finite([float("inf")], set(), "knee")


def test_write_warns_when_it_overwrites_another_runs_file(tmp_path):
    # E4.10's /spar review: a near-miss of a documented command must not
    # replace its results file in silence; where it writes does not count.
    path = tmp_path / "demo.json"
    ResultsCache("demo", {"counts": [1250, 2500], "outputs": "a"}).write(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ResultsCache("demo", {"counts": [1250, 2500], "outputs": "b"}).write(path)
        ResultsCache(
            "demo", {"counts": [1250, 2500], "outputs": "a", "data_dir": "d"}
        ).write(path)
    with pytest.warns(UserWarning, match="other counts"):
        ResultsCache("demo", {"counts": [1250], "outputs": "a"}).write(path)
    assert read_results(path)["args"]["counts"] == [1250]
    path.write_text("{}")
    with pytest.warns(UserWarning, match="the file"):
        ResultsCache("demo", {}).write(path)


def test_git_state_names_this_checkout_and_nothing_outside_one(tmp_path):
    state = git_state()
    assert isinstance(state["sha"], str) and len(state["sha"]) == 40
    assert isinstance(state["dirty"], bool)
    assert git_state(tmp_path) == {"sha": None, "dirty": None}


def test_dumps_is_json_with_the_scalar_lists_on_one_line():
    obj = {
        "argv": ["--mode", "naive"],
        "rows": [{"n": 1, "e": [1.5, None]}, {"n": 2, "e": []}],
        "curves": {"x": [0.1, 0.2, 0.3]},
        "empty": {},
    }
    text = dumps(obj)
    assert json.loads(text) == obj
    assert '"argv": ["--mode", "naive"]' in text
    assert '"x": [0.1, 0.2, 0.3]' in text and '"e": [1.5, null]' in text
    assert text.startswith('{\n "argv"') and '\n  {\n   "n": 1,' in text
    with pytest.raises(ValueError):
        dumps({"x": [float("nan")]})


def test_command_drops_where_a_run_writes():
    argv = ["--mode", "seeds", "--outputs", "o", "--data-dir=paper/data", "--seed", "1"]
    assert command(argv) == ["--mode", "seeds", "--seed", "1"]
    assert command(["--data-dir", "d"]) == []


def _repo(path: Path) -> Path:
    """A one-commit git repository at ``path`` (the provenance tests)."""

    def git(*args):
        who = ("-c", "user.name=t", "-c", "user.email=t@t")
        subprocess.run(
            ["git", "-C", str(path), *who, *args],
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (path / "src").mkdir()
    (path / "src" / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "one")
    return path


def test_git_state_ignores_the_runs_own_results_files(tmp_path):
    # E5.3: rerunning the documented commands into paper/data one after the
    # other, the second must not record the first one's file as a change.
    repo = _repo(tmp_path)
    assert git_state(repo)["dirty"] is False
    (repo / "paper" / "data").mkdir(parents=True)
    (repo / "paper" / "data" / "run.json").write_text("{}")
    assert git_state(repo)["dirty"] is False
    (repo / "src" / "a.py").write_text("x = 2\n")
    assert git_state(repo)["dirty"] is True


def test_provenance_refuses_a_dirty_tree_a_foreign_commit_and_schema_1(tmp_path):
    repo = _repo(tmp_path)
    head = git_state(repo)["sha"]
    good = {"schema": SCHEMA, "git": {"sha": head, "dirty": False}}
    assert provenance(good, repo) == []
    dirty = good | {"git": {"sha": head, "dirty": True}}
    assert provenance(dirty, repo) == [
        f"made from a tree with uncommitted changes (at {head[:9]})"
    ]
    foreign = good | {"git": {"sha": "f" * 40, "dirty": False}}
    assert provenance(foreign, repo) == [
        "commit fffffffff is not in the history of HEAD"
    ]
    assert provenance(good | {"git": {"sha": None, "dirty": None}}, repo) == [
        "made outside a git checkout"
    ]
    assert provenance(good | {"schema": 1}, repo) == [f"schema 1, not {SCHEMA}"]
    # This repository's own HEAD is in its history.
    here = git_state()
    assert provenance({"schema": SCHEMA, "git": here | {"dirty": False}}) == []


def test_changed_since_lists_the_files_that_moved(tmp_path):
    repo = _repo(tmp_path)
    head = git_state(repo)["sha"]
    assert changed_since(head, ["src"], repo) == []
    (repo / "src" / "a.py").write_text("x = 2\n")
    assert changed_since(head, ["src"], repo) == ["src/a.py"]
    assert changed_since(head, ["docs"], repo) == []


def test_source_hash_follows_the_bytes_not_the_order(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    first = source_hash([a, b])
    assert source_hash([b, a]) == first and len(first) == 64
    b.write_text("y = 3\n")
    assert source_hash([a, b]) != first


def test_rounded_and_float_keys():
    assert rounded(np.array([[1 / 3, 2e-10 / 3], [1.0, -np.pi]])) == [
        [0.333333, 6.66667e-11],
        [1.0, -3.14159],
    ]
    assert rounded([]) == []
    table = jsonable({0.0025: 1, 0.0: 2, "matlab": 3, 1e11: 4})
    assert float_keys(table) == {0.0025: 1, 0.0: 2, "matlab": 3, 1e11: 4}
