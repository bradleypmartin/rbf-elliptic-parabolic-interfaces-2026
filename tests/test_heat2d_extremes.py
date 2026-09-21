"""The E2.9 driver: Fig. 19 and 20's markers, the per-s references, the figures."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_extremes import (  # noqa: E402
    CONDITIONING_S,
    FIG19,
    FIG20,
    MARKERS,
    S_VALUES,
    main,
    reference_path,
    tag,
)


def test_fig19_has_seven_markers_per_s_and_the_last_line_is_flat():
    assert S_VALUES == (1e3, 1e8, 1e9, 1e10, 1e11)
    assert set(FIG19) == set(S_VALUES) == set(MARKERS)
    counts = [1250, 2500, 5000, 10000, 20000, 40000, 80000]
    for table in FIG19.values():
        assert sorted(table) == counts
    line = [FIG19[1e3][n] for n in counts]
    assert all(a > b for a, b in zip(line[:-1], line[1:], strict=True))
    assert FIG19[1e11][20000] == FIG19[1e11][40000] == FIG19[1e11][80000]


def test_fig20_grows_like_s_squared_over_nine_decades():
    assert set(FIG20) == set(CONDITIONING_S) and len(FIG20) == 9
    s = np.array(sorted(FIG20))
    slope = np.polyfit(np.log(s), np.log([FIG20[k] for k in s]), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.05)


def test_reference_paths_share_case_3s_file_at_s_1000():
    out = Path("outputs")
    assert reference_path(out, 1e3, 160000, 0).name == (
        "heat2d_case3_reference_n160000_seed0.npz"
    )
    assert reference_path(out, 1e8, 20000, 0).name == (
        "heat2d_extremes_reference_s1e8_n20000_seed0.npz"
    )
    assert tag(1e3) == "1e3" and tag(1e11) == "1e11"


def test_main_writes_the_two_figures_and_caches_the_reference(tmp_path, capsys):
    argv = [
        "--s",
        "1e3",
        "--counts",
        "900",
        "1800",
        "--reference-n",
        "3600",
        "--conditioning-s",
        "1e3",
        "1e6",
        "--conditioning-n",
        "900",
        "--spectrum-n",
        "900",
        "--check-n",
        "3600",
        "--iterations",
        "20",
        "--outputs",
        str(tmp_path),
    ]
    main(argv)
    for name in ("convergence", "conditioning"):
        assert (tmp_path / f"heat2d_extremes_{name}.png").stat().st_size > 0
    path = reference_path(tmp_path, 1e3, 3600, 0)
    assert path.exists() and path.with_suffix(".json").exists()
    out = capsys.readouterr().out
    assert "references per s" in out and "eq. 40 per s" in out
    assert "continuity matrices and stencil systems at 900 nodes" in out
    assert "interior spectrum" in out and "resampling check per s" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--counts", "900"],
        ["--counts", "900", "1800", "--reference-n", "1800"],
        ["--counts", "900", "1800", "--s", "0"],
        ["--counts", "900", "1800", "--conditioning-s", "-1"],
    ],
)
def test_main_refuses_thin_sweeps(tmp_path, argv):
    with pytest.raises(SystemExit):
        main([*argv, "--outputs", str(tmp_path)])
    assert not list(tmp_path.iterdir())
