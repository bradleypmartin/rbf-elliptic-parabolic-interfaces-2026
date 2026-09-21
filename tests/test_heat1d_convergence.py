"""The E1.4 driver: the sweep reproduces Fig. 4-7 and the three figures are written."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat1d_convergence import main, sweep  # noqa: E402
from heat_interfaces.heat1d import dissertation_alpha  # noqa: E402


def test_sweep_reports_both_norms_for_every_operator():
    errs = sweep((101, 201), dissertation_alpha())
    assert set(errs) == {"jump-aware", "naive", "direct"}
    assert errs["jump-aware"]["rms"][0] == pytest.approx(2.73e-3, rel=0.02)
    assert errs["naive"]["rms"][0] == pytest.approx(1.32e-2, rel=0.02)
    assert errs["direct"]["relative"][1] > 0.2
    for d in errs.values():
        assert d["rms"].shape == d["relative"].shape == (2,)


def test_main_writes_the_three_figures(tmp_path, capsys):
    main(["--counts", "101", "201", "--outputs", str(tmp_path)])
    for name in ("heat1d_solutions", "heat1d_errors", "heat1d_convergence"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
    out = capsys.readouterr().out
    assert "101-node solutions" in out and "RMS rates" in out


@pytest.mark.parametrize("argv", [["--n", "100"], ["--counts", "101", "200"]])
def test_main_refuses_counts_that_move_an_interface_off_a_node(tmp_path, argv):
    # 100 nodes put x = 0 mid-cell and x = 0.5 on neither a node nor a
    # midpoint; the figure's placement is part of what is reproduced.
    with pytest.raises(SystemExit):
        main([*argv, "--outputs", str(tmp_path)])
    assert not list(tmp_path.iterdir())
