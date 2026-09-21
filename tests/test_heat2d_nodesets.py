"""The E2.1 driver: three node-set figures and the spacing table."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_nodesets import main, report  # noqa: E402
from heat_interfaces.heat2d import build_node_set, case3  # noqa: E402


def test_report_counts_every_kind_of_node():
    nodes = build_node_set(case3(), 900, iterations=5)
    row = report(3, nodes, 0.1)
    assert row["n"] == 900 and row["per_row"] == 64
    assert row["straddle"] + row["dirichlet"] + row["free"] == 900
    assert row["dirichlet"] == 29 + 29 + 9
    assert 0.0 < row["min"] <= row["median"] <= row["max"]  # 5 steps: unrelaxed


def test_main_writes_the_three_figures(tmp_path, capsys):
    main(["--n", "400", "--iterations", "5", "--outputs", str(tmp_path)])
    for case in (1, 2, 3):
        assert (tmp_path / f"heat2d_nodes_case{case}.png").stat().st_size > 0
    out = capsys.readouterr().out
    assert "NN/h min" in out and out.count("\n") == 4


def test_main_refuses_unknown_cases(tmp_path):
    with pytest.raises(SystemExit):
        main(["--cases", "4", "--outputs", str(tmp_path)])
    assert not list(tmp_path.iterdir())
