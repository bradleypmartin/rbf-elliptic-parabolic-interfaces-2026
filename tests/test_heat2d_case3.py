"""The E2.7 driver: Fig. 14's tables, the reference cache, the mesh plot, the check."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_case3 import (  # noqa: E402
    FIG14,
    VARIANTS,
    main,
    nearest_marker,
    reference_path,
)


def test_fig14_has_seven_rbf_markers_and_twelve_fd4_doublings():
    rbf_counts = [1250, 2500, 5000, 10000, 20000, 40000, 80000]
    assert sorted(FIG14["flat"]) == rbf_counts
    assert sorted(FIG14["curved"]) == rbf_counts
    assert sorted(FIG14["fd4"]) == [1250 * 2**k for k in range(12)]
    for n in rbf_counts:
        assert FIG14["fd4"][n] > FIG14["flat"][n] >= FIG14["curved"][n]
    assert [name for name, _, _ in VARIANTS] == ["flat", "curved", "plain"]
    assert nearest_marker(1260) == 1250 and nearest_marker(1280292) == 1280000


def test_main_writes_the_figures_and_caches_the_reference(tmp_path, capsys):
    argv = [
        "--counts",
        "900",
        "1800",
        "--fd4-counts",
        "900",
        "--reference-n",
        "3600",
        "--check-n",
        "3600",
        "--mesh-n",
        "900",
        "--iterations",
        "20",
        "--outputs",
        str(tmp_path),
    ]
    main(argv)
    for name in ("convergence", "solution"):
        assert (tmp_path / f"heat2d_case3_{name}.png").stat().st_size > 0
    path = reference_path(tmp_path, 3600, 0)
    assert path.exists() and path.with_suffix(".json").exists()
    out = capsys.readouterr().out
    assert "solved now" in out and "Cartesian FD4" in out
    assert "mesh plot: 900 nodes" in out and "resampling check on the ring" in out


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
