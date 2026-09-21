"""The E2.8 driver: the 2016 (seconds, error) markers, both references, the figures."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_iterative import (  # noqa: E402
    FIG2016,
    METHODS,
    PRECONDITIONERS,
    PROBLEMS,
    main,
    reference_path,
)


def test_2016_markers_are_five_per_line_and_bicgstab_is_absent_from_fig_5_17():
    assert set(FIG2016) == {"control", "case3-none", "case3-appendix-b"}
    assert set(FIG2016["control"]) == {"direct", "gmres", "bicgstab"}
    assert set(FIG2016["case3-none"]) == {"direct", "gmres"}
    for panel, lines in FIG2016.items():
        expected = 4 if panel == "case3-appendix-b" else 5
        for pairs in lines.values():
            assert len(pairs) == expected
            assert all(seconds > 0 and error > 0 for seconds, error in pairs)
    # The three control lines share their error levels, which is what supports
    # reading the unstated counts as 1250 … 20,000.
    direct = [e for _, e in FIG2016["control"]["direct"]]
    for solver in ("gmres", "bicgstab"):
        for (_, e), d in zip(FIG2016["control"][solver], direct, strict=True):
            assert e == pytest.approx(d, rel=0.25)
    assert [label for label, _, _ in METHODS] == ["gmres", "gmres(20)", "bicgstab"]
    assert PRECONDITIONERS == ("none", "appendix-b", "spilu")
    assert PROBLEMS == ("control", "case3")


def test_main_writes_the_four_figures_and_both_references(tmp_path, capsys):
    argv = [
        "--counts",
        "900",
        "1800",
        "--reference-n",
        "3600",
        "--ddr-n",
        "900",
        "--iterations",
        "20",
        "--outputs",
        str(tmp_path),
    ]
    main(argv)
    for name in ("performance", "iterations", "ddr", "control"):
        assert (tmp_path / f"heat2d_iterative_{name}.png").stat().st_size > 0
    for problem in PROBLEMS:
        path = reference_path(tmp_path, problem, 3600, 0)
        assert path.exists() and path.with_suffix(".json").exists()
    out = capsys.readouterr().out
    assert "control reference: 3600 nodes" in out and "case3 reference:" in out
    assert "solved now" in out and "iteration ratios" in out
    assert "DDR of other operators" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--counts", "900"],
        ["--counts", "100", "900"],
        ["--counts", "900", "1800", "--reference-n", "1800"],
    ],
)
def test_main_refuses_thin_sweeps(tmp_path, argv):
    with pytest.raises(SystemExit):
        main([*argv, "--outputs", str(tmp_path)])
    assert not list(tmp_path.iterdir())
