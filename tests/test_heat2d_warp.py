"""The E2.4 driver: the four combinations on case 1, the case-2 run and the figures."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_warp import FIG7, VARIANTS, main, rates  # noqa: E402


def test_fig7_has_six_markers_and_the_variants_are_fig_11s_corners():
    assert sorted(FIG7) == [1250, 2500, 5000, 10000, 20000, 40000]
    assert [name for name, _, _ in VARIANTS] == [
        "warp+rows",
        "plain+rows",
        "warp,none",
        "plain,none",
    ]
    assert VARIANTS[0][1:] == (True, True) and VARIANTS[-1][1:] == (False, False)


def test_rates_are_orders_per_halving_of_h():
    rows = [{"h": 0.02, "e": 1.6e-3}, {"h": 0.01, "e": 1.0e-4}]
    order = rates(rows, "e")
    assert np.isnan(order[0]) and order[1] == pytest.approx(4.0)


def test_main_writes_the_two_figures(tmp_path, capsys):
    argv = [
        "--counts",
        "300",
        "600",
        "--case2-counts",
        "300",
        "--iterations",
        "20",
        "--outputs",
        str(tmp_path),
    ]
    main(argv)
    for name in ("warp_case1", "warped_rbf"):
        assert (tmp_path / f"heat2d_{name}.png").stat().st_size > 0
    out = capsys.readouterr().out
    assert "case 1: warped vs plain Gaussians" in out
    assert "case 2: the four combinations" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--counts", "300"],
        ["--counts", "100", "300"],
        ["--counts", "300", "600", "--case2-counts", "100"],
    ],
)
def test_main_refuses_thin_sweeps(tmp_path, argv):
    with pytest.raises(SystemExit):
        main([*argv, "--outputs", str(tmp_path)])
    assert not list(tmp_path.iterdir())
