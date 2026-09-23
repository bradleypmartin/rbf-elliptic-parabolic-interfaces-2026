"""The E4.8 driver: the ring's sweeps, their cache, the far read, the refusals."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_ring import (  # noqa: E402
    CACHE,
    RESULTS,
    exact_mode,
    far_read,
    main,
    reach_stencils,
    ring_domain,
    without_gap,
)
from heat_interfaces.heat2d import (  # noqa: E402
    RingMode,
    SmoothBand,
    SmoothRingMode,
    build_node_set,
    case3,
)


def test_ring_domains_carry_the_gap_and_the_composition():
    jump = ring_domain(1e11, 0.0)
    assert jump.material.gap == 1e-11 and not isinstance(jump.material, SmoothBand)
    smooth = ring_domain(1e11, 1e-3, constant=True)
    assert smooth.material.composition == "resistance"
    assert smooth.material.inside.value == pytest.approx(1.0 / 1.5e11)
    assert isinstance(exact_mode(ring_domain(1e3, 0.0, True)), RingMode)
    assert isinstance(exact_mode(smooth), SmoothRingMode)
    assert without_gap(smooth.material).gap is None
    assert without_gap(jump.material).gap is None


def test_the_far_read_skips_the_nodes_the_fine_stencils_seed():
    # Away from the ring the fine set's standard interpolant reads a smooth
    # function to its order; the nodes whose nearest fine node is seeded are
    # left out.
    domain = ring_domain(1e3, 1e-3)
    fine = build_node_set(domain, 5000, seed=0, iterations=20)
    coarse = build_node_set(domain, 1250, seed=1, iterations=20)
    stencils = reach_stencils(fine, domain)
    u = np.sin(2 * np.pi * fine.x) * fine.y
    got, keep = far_read(u, fine, stencils, domain.material, coarse)
    assert 0.5 < keep.mean() < 0.95
    want = np.sin(2 * np.pi * coarse.x[keep]) * coarse.y[keep]
    assert np.abs(got - want).max() < 1e-4
    r = np.hypot(coarse.x - 0.5, coarse.y - 0.5)
    assert not keep[np.abs(r - 0.3495) < 0.02].any()


def test_main_runs_the_four_parts_and_reuses_its_cache(tmp_path, capsys):
    argv = [
        "--s",
        "1e3",
        "--counts",
        "1250",
        "2500",
        "--reference-n",
        "3600",
        "--conditioning-s",
        "1e11",
        "--conditioning-n",
        "1250",
        "--deltas",
        "--smooth-s",
        "1e3",
        "--probe-counts",
        "1250",
        "--iterations",
        "20",
        "--outputs",
        str(tmp_path),
    ]
    tables = main(argv)
    out = capsys.readouterr().out
    assert "Fig. 19's twin" in out and "Fig. 20's twin" in out
    assert "the smooth ring" in out
    for name in ("convergence", "conditioning", "smooth"):
        assert (tmp_path / f"heat2d_ring_{name}.png").stat().st_size > 0
    assert (tmp_path / CACHE).exists() and (tmp_path / RESULTS).exists()
    (row,) = tables["conditioning"]
    assert row["seeds-worst"] < 1e-14 < row["stored-worst"] < row["e23-worst"]
    first = tables["convergence"][1e3]
    assert first[1]["seeds"]["full"] < first[0]["seeds"]["full"]
    probe = tables["smooth"]["1e3|0"][0]
    assert probe["probe-seeds"]["seeded"] < probe["probe-naive"]["seeded"]
    # A second run reads everything back from the cache, to the bit.
    again = main(argv)
    assert again["conditioning"] == tables["conditioning"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--counts", "900", "1800"],
        ["--counts", "1250", "2500", "--reference-n", "2500"],
        ["--counts", "--conditioning-s", "3"],
        ["--counts", "--deltas", "0"],
        ["--counts", "--probe-counts", "2500", "--fine-n", "2500"],
        ["--counts", "--conditioning-n", "900"],
        ["--counts", "--spectrum-n", "900"],
    ],
)
def test_main_refuses_what_it_cannot_run(tmp_path, argv):
    with pytest.raises(SystemExit):
        main([*argv, "--outputs", str(tmp_path)])
    assert not list(tmp_path.iterdir())


def test_the_driver_uses_case_3_bit_for_bit_at_s_1000():
    a, b = ring_domain(1e3, 0.0).material, case3().material
    assert a == b
