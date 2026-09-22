"""``results_cache`` (plan D1; E5.3, #44): the schema of a driver's results file."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from heat_interfaces.results_cache import (
    SCHEMA,
    ResultsCache,
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


def test_git_state_names_this_checkout_and_nothing_outside_one(tmp_path):
    state = git_state()
    assert isinstance(state["sha"], str) and len(state["sha"]) == 40
    assert isinstance(state["dirty"], bool)
    assert git_state(tmp_path) == {"sha": None, "dirty": None}
