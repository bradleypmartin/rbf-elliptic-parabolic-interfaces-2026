"""``results_cache`` (plan D1; E5.3, #44): the schema of a driver's results file."""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from heat_interfaces.results_cache import (
    SCHEMA,
    ResultsCache,
    finite,
    git_state,
    jsonable,
    read_results,
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
    cache = ResultsCache("demo", {"outputs": Path("outputs"), "deltas": [0.0, 0.01]})
    cache.add("knee", {"matlab": {0.01: [{"n": 50, "naive": 1e-3}]}})
    cache.add("knee", {"matlab": {0.01: [{"n": 50, "naive": 2e-3}]}})
    cache.time("knee", 1.5)
    paths = cache.write(tmp_path / "a" / "demo.json", tmp_path / "b" / "demo.json")
    assert all(p.exists() for p in paths)
    assert paths[0].read_text() == paths[1].read_text()
    data = read_results(paths[0])
    assert data["schema"] == SCHEMA and data["driver"] == "demo"
    datetime.strptime(data["date"], "%Y-%m-%dT%H:%M:%SZ")
    assert data["args"] == {"outputs": "outputs", "deltas": [0.0, 0.01]}
    assert data["timings"] == {"knee": 1.5}
    assert data["tables"] == {"knee": {"matlab": {"0.01": [{"n": 50, "naive": 2e-3}]}}}
    assert set(data["git"]) == {"sha", "dirty"}
    (tmp_path / "other.json").write_text("{}")
    with pytest.raises(ValueError, match="not a results file"):
        read_results(tmp_path / "other.json")


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
