"""The E2.6 driver: the reference cache, the three sweeps, the check and the figures."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_case2 import (  # noqa: E402
    FIG10,
    FIG11_NONE,
    FIG511,
    VARIANTS,
    main,
    nearest_marker,
    rates,
    reference_error_estimate,
    reference_path,
    total_seconds,
)


def test_2016_tables_have_seven_markers_and_the_hybrid_is_absent():
    counts = [1250, 2500, 5000, 10000, 20000, 40000, 80000]
    for table in (*FIG10.values(), FIG11_NONE):
        assert sorted(table) == counts
    for n in counts:
        assert FIG10["fd4"][n] > FIG10["flat"][n] > FIG10["curved"][n]
        assert FIG11_NONE[n] > FIG10["curved"][n]
    assert set(FIG511) == {"fd4", "rbf"}
    assert [name for name, _, _, _ in VARIANTS] == ["flat", "curved", "none"]
    assert nearest_marker(1260) == 1250 and nearest_marker(322_400) == 80000


def test_reference_error_estimate_is_a_third_at_half_the_count():
    rows = [{"n": 40000, "curved": 3e-8}, {"n": 80000, "curved": 9e-9}]
    estimate, n = reference_error_estimate(rows, 160000)
    assert n == 80000 and estimate == pytest.approx(3e-9)
    assert np.isnan(rates(rows, "curved")[0]) if "h" in rows[0] else True


def test_total_seconds_sums_the_methods_stages():
    row = {
        "curved-nodes": 1.0,
        "curved-build": 2.0,
        "curved-solve": 3.0,
        "fd4-build": 0.5,
        "fd4-solve": 0.25,
    }
    assert total_seconds(row, "curved") == 6.0 and total_seconds(row, "fd4") == 0.75


def test_main_writes_the_figures_and_caches_the_reference(tmp_path, capsys):
    argv = [
        "--counts",
        "900",
        "1800",
        "--fd4-counts",
        "900",
        "1800",
        "3600",
        "--reference-n",
        "3600",
        "--check-n",
        "3600",
        "--iterations",
        "20",
        "--outputs",
        str(tmp_path),
    ]
    main(argv)
    for name in ("convergence", "ablation", "performance"):
        assert (tmp_path / f"heat2d_case2_{name}.png").stat().st_size > 0
    path = reference_path(tmp_path, 3600, 0)
    assert path.exists() and path.with_suffix(".json").exists()
    out = capsys.readouterr().out
    assert "solved now" in out and "resampling check" in out and "Cartesian FD4" in out
    main(argv)
    assert "cached" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["--counts", "900"],
        ["--counts", "100", "900"],
        ["--counts", "900", "1800", "--reference-n", "1800"],
        ["--counts", "900", "1800", "--fd4-counts", "50"],
    ],
)
def test_main_refuses_thin_sweeps(tmp_path, argv):
    with pytest.raises(SystemExit):
        main([*argv, "--outputs", str(tmp_path)])
    assert not list(tmp_path.iterdir())
