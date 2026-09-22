"""The E3 driver: E3.2's references are built, checked, cached and reused."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat1d_stiff import (  # noqa: E402
    CHECK_MAX_WIDTH,
    CHECK_N_CHEB,
    MAX_WIDTH,
    N_CHEB,
    check_references,
    main,
    reference_path,
)


def test_main_builds_checks_and_then_reuses_the_references(tmp_path, capsys):
    argv = ["--outputs", str(tmp_path), "--deltas", "0", "0.0025", "--media", "matlab"]
    main(argv)
    out = capsys.readouterr().out
    assert "within the 1e-10" in out and out.count("solved") == 2
    for delta in (0.0, 0.0025):
        for n, w in ((N_CHEB, MAX_WIDTH), (CHECK_N_CHEB, CHECK_MAX_WIDTH)):
            stem = reference_path(tmp_path, "matlab", delta, n, w)
            for suffix in (".npz", ".json"):
                assert (tmp_path / (stem.name + suffix)).exists(), stem.name + suffix
    main(argv)
    out = capsys.readouterr().out
    assert out.count("cached") == 2 and "solved" not in out


def test_rows_carry_the_run_and_both_checks(tmp_path):
    (row,) = check_references(["eq75"], [0.01], tmp_path, (16, 0.25), (20, 0.25))
    assert row["medium"] == "eq75" and row["delta"] == 0.01
    assert row["elements"] > 3 and row["steps"] > 100 and not row["reused"]
    assert row["agreement"] < 1e-8 and row["elliptic"] < 1e-8
