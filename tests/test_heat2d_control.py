"""The E2.2 driver: the control and case-1 sweep, the spectrum, and the two figures."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_control import (  # noqa: E402
    LINES,
    main,
    rates,
    spectrum,
    spectrum_summary,
    sweep,
)


def test_sweep_reports_every_line_and_the_rates_follow_h():
    rows = sweep((900, 1800))
    assert [r["n"] for r in rows] == [900, 1800]
    for name in LINES:
        assert all(r[name] > 0 and r[name + "-seconds"] > 0 for r in rows)
    r = rates(rows, "control-lap")
    assert np.isnan(r[0]) and r[1] > 3.0
    assert rows[0]["case1-direct"] == pytest.approx(rows[1]["case1-direct"], rel=0.15)


def test_spectrum_summary_counts_complex_eigenvalues():
    spec = spectrum(900)
    for row in spectrum_summary(spec):
        assert row["count"] == 900 - 2 * 29  # two Dirichlet rows of round(0.95 * 30)
        assert 0 < row["complex"] < row["count"]
        assert row["min_re_h2"] < -5 and row["max_im_h2"] > 0


def test_main_writes_the_two_figures(tmp_path, capsys):
    main(
        [
            "--counts",
            "900",
            "1250",
            "--spectrum-n",
            "900",
            "--outputs",
            str(tmp_path),
        ]
    )
    for name in ("heat2d_control_convergence", "heat2d_control_spectrum"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
    out = capsys.readouterr().out
    assert "control-lap" in out and "BD4 max" in out


@pytest.mark.parametrize(
    "argv", [["--counts", "900"], ["--counts", "100", "900"], ["--spectrum-n", "50"]]
)
def test_main_refuses_thin_sweeps(tmp_path, argv):
    with pytest.raises(SystemExit):
        main([*argv, "--outputs", str(tmp_path)])
    assert not list(tmp_path.iterdir())
