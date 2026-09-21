"""The E2.5 driver: the case-1 sweep, the spectrum, and the two figures."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_case1 import (  # noqa: E402
    FIG7,
    FIG55_PARABOLIC,
    SPECTRA,
    main,
    rates,
    spectrum,
    spectrum_summary,
    sweep,
)


def test_sweep_reports_both_lines_with_steps_and_times():
    rows = sweep((900, 1800))
    assert [r["n"] for r in rows] == [900, 1800]
    for r in rows:
        assert 0 < r["parabolic"] < 1e-3 and 0 < r["elliptic"] < 1e-3
        assert r["parabolic-half-steps"] == 2 * r["parabolic-steps"]
        assert r["parabolic-half"] == pytest.approx(r["parabolic"], rel=0.05)
        assert r["elliptic-seconds"] > 0 and r["parabolic-seconds"] > 0
    assert rows[0]["parabolic-steps"] == 3  # h = 1/29 at 900 nodes
    assert np.isnan(rates(rows, "elliptic")[0]) and rates(rows, "parabolic")[1] > 3.0


def test_fig_5_5_has_six_points_per_line():
    assert (
        sorted(FIG7)
        == sorted(FIG55_PARABOLIC)
        == [1250, 2500, 5000, 10000, 20000, 40000]
    )
    for n in FIG7:
        assert 0.8 < FIG55_PARABOLIC[n] / FIG7[n] < 2.5


def test_spectrum_summary_covers_the_three_operators():
    # At 900 nodes the naive product operator carries the spurious growing
    # mode of port notes §2.2 (a real eigenvalue near +850 here), so only the
    # interface-aware operators are asked to be damped.
    spec = spectrum(900)
    rows = spectrum_summary(spec)
    assert [r["name"] for r in rows] == [name for name, _, _ in SPECTRA]
    for r in rows:
        assert r["count"] == 900 - 2 * 29
        assert 0 < r["complex"] < r["count"]
        assert r["min_re_h2"] < -5
        if r["name"] != "naive":
            assert r["max_re"] < 0 and r["bd4_h"] < 1 and r["bd4_dt"] < 1
    assert rows[2]["max_re"] > 0


def test_main_writes_the_two_figures(tmp_path, capsys):
    main(["--counts", "900", "1250", "--spectrum-n", "900", "--outputs", str(tmp_path)])
    for name in ("heat2d_case1_convergence", "heat2d_case1_spectrum"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
    out = capsys.readouterr().out
    assert "parabolic" in out and "BD4 max" in out and "ratio to 2016" in out


@pytest.mark.parametrize(
    "argv", [["--counts", "900"], ["--counts", "100", "900"], ["--spectrum-n", "50"]]
)
def test_main_refuses_thin_sweeps(tmp_path, argv):
    with pytest.raises(SystemExit):
        main([*argv, "--outputs", str(tmp_path)])
    assert not list(tmp_path.iterdir())
