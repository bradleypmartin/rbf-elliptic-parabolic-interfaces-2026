"""The E2.3 driver: the case-1 sweep, the continuity table, the conditioning table
and the figure."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_interface import (  # noqa: E402
    FIG7,
    OFFSETS,
    conditioning,
    continuity_table,
    main,
    rates,
    sweep,
)


def test_sweep_reports_both_lines_and_the_aware_line_is_fourth_order():
    rows = sweep((900, 1800), iterations=20)
    assert [r["n"] for r in rows] == [900, 1800]
    for r in rows:
        assert 0 < r["aware"] < r["naive"] and r["flat-diff"] == 0.0
        assert 0 < r["crossing"] <= r["group"]
    assert rates(rows, "aware")[1] > 3.0
    assert np.isnan(rates(rows, "aware")[0])


def test_continuity_table_shows_the_orders_of_both_variants():
    table = continuity_table()
    assert len(OFFSETS) == 4
    for variants in table.values():
        curved, flat = variants["curved"], variants["flat"]
        assert curved.shape == flat.shape == (4, 2)
        # u jumps fall at least like ξ^5 for the curved variant, like ξ² flat.
        assert np.all(curved[:-1, 0] / curved[1:, 0] > 30)
        assert np.all(flat[:-1, 0] / flat[1:, 0] == pytest.approx(4.0, rel=0.1))
        assert np.all(curved[:-1, 1] / curved[1:, 1] == pytest.approx(16.0, rel=0.1))


def test_conditioning_has_no_trend_with_the_node_count():
    rows = conditioning((900, 3600))
    for r in rows:
        assert r["stencils"] > 100
        assert r["outside-max"] < 1e3 and r["translation-max"] < 1e4
    for name in ("outside", "inside", "translation"):
        assert rows[1][name + "-median"] == pytest.approx(
            rows[0][name + "-median"], rel=0.2
        )


def test_fig7_has_six_points():
    assert sorted(FIG7) == [1250, 2500, 5000, 10000, 20000, 40000]


def test_main_writes_the_figure(tmp_path, capsys):
    main(
        [
            "--counts",
            "900",
            "1250",
            "--conditioning-counts",
            "900",
            "--iterations",
            "20",
            "--outputs",
            str(tmp_path),
        ]
    )
    assert (tmp_path / "heat2d_interface_convergence.png").stat().st_size > 0
    out = capsys.readouterr().out
    assert "aware" in out and "ratios per halving" in out and "C outside" in out


@pytest.mark.parametrize(
    "argv",
    [["--counts", "900"], ["--counts", "100", "900"], ["--conditioning-counts", "50"]],
)
def test_main_refuses_thin_sweeps(tmp_path, argv):
    with pytest.raises(SystemExit):
        main([*argv, "--outputs", str(tmp_path)])
    assert not list(tmp_path.iterdir())
