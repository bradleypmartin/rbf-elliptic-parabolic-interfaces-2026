"""The E4.5 driver (#36): the seed rows' dominance and solvers, and the spectra.

The numbers the notes quote are the default run's (2500 nodes for the rows,
1600 for the spectra); here the smallest sets that still show each effect.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_stiff_eigenvalues import (  # noqa: E402
    LABELS,
    ROW_CACHE,
    ROW_CACHE_META,
    condition_estimate,
    dominance,
    load_cache,
    main,
    row_key,
    save_cache,
    spectra_path,
)
from heat_interfaces.results_cache import read_results  # noqa: E402


def test_condition_estimate_is_a_lower_bound_on_the_one_norm_condition_number():
    # Higham and Tisseur's estimator underestimates by construction; on these
    # matrices it lands within 15 % of the dense number.
    rng = np.random.default_rng(0)
    for scale in (0.1, 0.4):
        a = np.eye(40) * 4.0 + rng.standard_normal((40, 40)) * scale
        exact = np.linalg.cond(a, 1)
        estimate = condition_estimate(sp.csr_array(a))
        assert 0.5 * exact <= estimate <= exact * (1.0 + 1e-9)


def test_dominance_reads_the_group_and_every_row():
    ddr = np.array([0.5, 2.0, 4.0, 8.0])
    group = np.array([True, True, False, False])
    out = dominance(ddr, group)
    assert (out["least"], out["median"], out["below1"]) == (0.5, 1.25, 0.5)
    assert (out["all-least"], out["all-median"], out["all-below1"]) == (0.5, 3.0, 0.25)
    empty = dominance(ddr, np.zeros(4, dtype=bool))
    assert np.isnan(empty["least"]) and empty["all-median"] == 3.0


class Args:
    """The defaults ``row_key`` reads."""

    seed, iterations, rtol, maxiter, neighbours, sweeps = 0, 100, 1e-8, 3000, 37, 3


def test_the_row_cache_round_trips_and_rejects_another_studys_file(tmp_path):
    key = row_key("seeds", 0.125, 1250, Args())
    assert key == "rows r0.125 n1250 s0 i100 rtol1e-08 m3000 k37 w3 seeds"
    cache = {key: {"least": 0.28}}
    save_cache(tmp_path, cache)
    assert load_cache(tmp_path) == cache
    data = json.loads((tmp_path / ROW_CACHE).read_text())
    data["meta"] = {**ROW_CACHE_META, "version": -1}
    (tmp_path / ROW_CACHE).write_text(json.dumps(data))
    assert load_cache(tmp_path) == {}


def test_the_keys_carry_every_argument_the_numbers_depend_on(tmp_path):
    # The knee cache's trap (stiff note §3.8): a flag that moves the numbers
    # and not the key gives the earlier run's answer back in silence.
    base = row_key("seeds", 0.125, 1250, Args())
    for field, value in (
        ("seed", 1),
        ("iterations", 50),
        ("rtol", 1e-10),
        ("maxiter", 500),
        ("neighbours", 50),
        ("sweeps", 5),
    ):
        other = type("Other", (Args,), {field: value})
        assert row_key("seeds", 0.125, 1250, other()) != base, field
    path = spectra_path(tmp_path, 900, 0.005, 0, 100)
    assert path.name == "heat2d_stiff_spectra_n900_d0.005_i100_s0.npz"
    assert spectra_path(tmp_path, 900, 0.005, 0, 50) != path
    assert spectra_path(tmp_path, 900, 0.005, 1, 100) != path


def test_another_preconditioner_setting_is_not_read_from_the_cache(tmp_path):
    # End to end on the cheapest line: three sweeps, then one, same node set.
    argv = ["--mode", "rows", "--n", "1250", "--ratios", "0", "--labels", "naive"]
    argv += ["--outputs", str(tmp_path)]
    three = main(argv)["rows"][0]
    # The same results file from a run with other --sweeps: said, not silent.
    with pytest.warns(UserWarning, match="other sweeps"):
        one = main([*argv, "--sweeps", "1"])["rows"][0]
    assert one["b-median"] != three["b-median"]
    assert one["median"] == three["median"]
    assert len(load_cache(tmp_path)) == 2


def test_the_row_table_at_1250_nodes(tmp_path, capsys):
    # H5: at δ = 0 the seed rows *are* the construction's rows, so every
    # reading coincides; at δ/h = 1/8 they are different rows whose dominance
    # and iteration counts stay between the construction's and better, with
    # no breakdown, while the seeds' own error is orders below.
    argv = ["--mode", "rows", "--n", "1250", "--ratios", "0.125", "0"]
    argv += ["--outputs", str(tmp_path)]
    rows = main(argv)["rows"]
    out = capsys.readouterr().out
    assert "diagonal dominance" not in out and "DDR least" in out
    assert (tmp_path / "heat2d_stiff_dominance_n1250.png").exists()
    # E4.10: the run's results file, named by the mode and the count.
    written = read_results(tmp_path / "heat2d_stiff_eigenvalues_rows_n1250.json")
    assert written["driver"] == "heat2d_stiff_eigenvalues"
    assert len(written["tables"]["rows"]) == len(rows)
    assert set(written["timings"]) == {"rows", "total"}
    table = {(r["ratio"], r["label"]): r for r in rows}
    assert set(table) == {(r, label) for r in (0.125, 0.0) for label in LABELS}

    jump_seeds, jump_built = table[(0.0, "seeds")], table[(0.0, "construction")]
    assert jump_seeds["rows"] == jump_built["rows"]
    for key in ("least", "median", "all-least", "all-median"):
        assert jump_seeds[key] == pytest.approx(jump_built[key], rel=1e-9)
    assert jump_seeds["error"] == pytest.approx(jump_built["error"], rel=1e-6)

    for ratio in (0.0, 0.125):
        seeds = table[(ratio, "seeds")]
        built = table[(ratio, "construction")]
        naive = table[(ratio, "naive")]
        assert seeds["rows"] >= built["rows"]
        assert seeds["least"] > 0.2 and seeds["median"] > 0.5
        assert naive["least"] < seeds["least"] and naive["nnz"] > 3 * seeds["nnz"]
        # The seeds' matrix is no worse conditioned than the construction's
        # and an order below the naive product's.
        assert seeds["cond"] < 2 * built["cond"] < naive["cond"]
        for method in ("gmres", "bicgstab"):
            for pc in ("none", "appendix-b", "spilu"):
                cell = seeds[f"{method}/{pc}"]
                assert cell["info"] == 0 and cell["residual"] < 1e-7
                assert cell["distance"] < 1e-5
                assert cell["iterations"] <= built[f"{method}/{pc}"]["iterations"] + 5
        assert seeds["b-median"] >= built["b-median"] - 1e-9

    # A marginally resolved edge is where the seeds part from the jump's rows.
    assert (
        table[(0.125, "seeds")]["error"] < table[(0.125, "construction")]["error"] / 20
    )
    assert table[(0.125, "construction")]["error"] < table[(0.125, "naive")]["error"]
    # The warp is worth a factor on the same rows (H7).
    assert table[(0.125, "seeds-plain")]["error"] > 2 * table[(0.125, "seeds")]["error"]

    t0 = time.perf_counter()
    again = main(argv)["rows"]
    assert time.perf_counter() - t0 < 10
    assert [r["error"] for r in again] == [r["error"] for r in rows]


def test_the_spectra_at_900_nodes(tmp_path, capsys):
    # H6: the seed operator's spectrum is the construction's at δ = 0 and
    # stays in the left half-plane at δ > 0, where the naive operator keeps
    # the coarse set's growing mode (port notes §2.2's +847 at 900 nodes);
    # with plain Gaussians the crossing rows' complex loop returns.
    argv = ["--mode", "spectra", "--spectrum-n", "900", "--deltas", "0", "0.005"]
    argv += ["--figure-delta", "0.005", "--outputs", str(tmp_path)]
    rows = main(argv)["spectra"]
    assert "h² max |Im|" in capsys.readouterr().out
    assert (tmp_path / "heat2d_stiff_spectra_n900.png").exists()
    assert (tmp_path / "heat2d_stiff_eigenvalues_spectra_n900.json").exists()
    assert (tmp_path / "heat2d_stiff_spectra_n900_d0.005_i100_s0.npz").exists()
    # E5.3: the command line, and the eigenvalues the manuscript's figure draws.
    written = read_results(tmp_path / "heat2d_stiff_eigenvalues_spectra_n900.json")
    assert written["argv"] == argv
    figure = written["tables"]["spectra/figure"]
    assert figure["delta"] == 0.005 and figure["n"] == 900
    assert set(figure["eigenvalues"]) == set(LABELS)
    table = {(r["delta"], r["label"]): r for r in rows}
    for label, lam in figure["eigenvalues"].items():
        row = table[(0.005, label)]
        assert len(lam["re"]) == len(lam["im"]) == row["count"]
        assert max(lam["re"]) == pytest.approx(row["max_re"], rel=1e-5)
    for delta in (0.0, 0.005):
        naive, seeds = table[(delta, "naive")], table[(delta, "seeds")]
        assert naive["positive"] >= 1 and naive["max_re"] > 100
        assert seeds["positive"] == 0 and seeds["max_re"] < -7 and seeds["bd4"] < 1
        assert seeds["max_im_h2"] < 0.5
    for key in ("max_re", "min_re_h2", "max_im_h2", "bd4"):
        assert table[(0.0, "seeds")][key] == pytest.approx(
            table[(0.0, "construction")][key], rel=1e-9
        )
    # Plain Gaussians bring back port notes §2.5's complex loop at δ = 0.
    assert (
        table[(0.0, "seeds-plain")]["max_im_h2"]
        > 5 * table[(0.0, "seeds")]["max_im_h2"]
    )

    t0 = time.perf_counter()
    again = main(argv)["spectra"]
    assert time.perf_counter() - t0 < 10
    assert [r["max_re"] for r in again] == [r["max_re"] for r in rows]
