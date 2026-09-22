"""The E3 driver: E3.2's references, built, checked, cached, reused; E3.3's knee;
E3.4's seed line, weights and spectra; E3.5's comparator table; E3.6's seed
functions, snapshot and results file."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat1d_stiff import (  # noqa: E402
    CHECK_MAX_WIDTH,
    CHECK_N_CHEB,
    COMPARATORS,
    KNEE_CACHE,
    KNEE_CACHE_META,
    MAX_WIDTH,
    N_CHEB,
    OPERATORS,
    RESULTS,
    SEED_RATIOS,
    SNAPSHOT_OPERATORS,
    TREATMENTS,
    WIDENINGS,
    check_references,
    elliptic_sweep,
    floor_constants,
    knee_grids,
    load_knee_cache,
    main,
    parabolic_sweep,
    reference_path,
    row_residuals,
    save_knee_cache,
    seed_functions,
    seed_spectra,
    snapshot,
    weights_vs_jump,
    widened_floors,
)
from heat_interfaces.results_cache import read_results  # noqa: E402


def test_main_builds_checks_and_then_reuses_the_references(tmp_path, capsys):
    argv = [
        *("--outputs", str(tmp_path)),
        *("--deltas", "0", "0.0025"),
        *("--media", "matlab"),
        *("--counts", "50", "100"),
        *("--data-dir", str(tmp_path / "data")),
    ]
    main(argv)
    out = capsys.readouterr().out
    # Two references, four knee rows, four comparator rows.
    assert "within the 1e-10" in out and out.count("solved") == 2 + 4 + 4
    # E3.6: the two figures and the results file, written to both directories.
    assert (tmp_path / "heat1d_stiff_seeds.png").exists()
    assert (tmp_path / "heat1d_stiff_snapshot.png").exists()
    results = read_results(tmp_path / RESULTS)
    assert (tmp_path / "data" / RESULTS).read_text() == (tmp_path / RESULTS).read_text()
    assert results["driver"] == "heat1d_stiff" and results["args"]["media"] == [
        "matlab"
    ]
    assert results["args"]["data_dir"] == str(tmp_path / "data")
    assert set(results["timings"]) == {
        "references",
        "knee",
        "comparators",
        "figures",
        "total",
    }
    ramp = results["tables"]["knee/ramp"]["matlab"]
    assert set(ramp) == {"0", "0.0025"} and [r["n"] for r in ramp["0"]] == [50, 100]
    assert set(ramp["0"][0]) >= {"n", "h", "floor", *OPERATORS}
    assert set(results["tables"]["comparators/ramp"]["matlab"]["0.0025"][0]) >= set(
        COMPARATORS
    )
    assert [r["label"] for r in results["tables"]["snapshot"]["rows"]] == list(
        SNAPSHOT_OPERATORS
    )
    assert [r["ratio"] for r in results["tables"]["seed_functions"]["rows"]] == [
        0.0,
        *SEED_RATIOS,
    ]
    assert len(results["tables"]["references"]) == 2
    for delta in (0.0, 0.0025):
        for n, w in ((N_CHEB, MAX_WIDTH), (CHECK_N_CHEB, CHECK_MAX_WIDTH)):
            stem = reference_path(tmp_path, "matlab", delta, n, w)
            for suffix in (".npz", ".json"):
                assert (tmp_path / (stem.name + suffix)).exists(), stem.name + suffix
    assert (tmp_path / "heat1d_stiff_knee.png").exists()
    assert (tmp_path / "heat1d_stiff_treatments.png").exists()
    cache = json.loads((tmp_path / KNEE_CACHE).read_text())
    assert cache["meta"] == KNEE_CACHE_META
    errors = cache["errors"]
    # Two δ, two counts, the three knee lines and the six treatments (the
    # naive and seed columns of the comparators share the knee's keys).
    labels = set(OPERATORS) | set(TREATMENTS)
    assert len(errors) == 2 * 2 * len(labels)
    assert all(0 < v < 1 for v in errors.values())
    main(argv)
    out = capsys.readouterr().out
    assert out.count("cached") == 2 + 4 + 4 and "solved" not in out
    # A cache from another problem is ignored and the knee runs are redone.
    cache["meta"]["t_end"] = 3.0
    (tmp_path / KNEE_CACHE).write_text(json.dumps(cache))
    main(argv)
    out = capsys.readouterr().out
    assert out.count("cached") == 2 and out.count("solved") == 4 + 4
    assert json.loads((tmp_path / KNEE_CACHE).read_text())["meta"] == KNEE_CACHE_META


def test_knee_cache_round_trips_and_rejects_another_problems_file(tmp_path):
    save_knee_cache(tmp_path, {"b": 2.0, "a": 1.0})
    assert load_knee_cache(tmp_path) == {"a": 1.0, "b": 2.0}
    data = json.loads((tmp_path / KNEE_CACHE).read_text())
    assert list(data["errors"]) == ["a", "b"]
    data["meta"]["bc"] = [1.0, 0.5]
    (tmp_path / KNEE_CACHE).write_text(json.dumps(data))
    assert load_knee_cache(tmp_path) == {}
    assert load_knee_cache(tmp_path / "missing") == {}


def test_rows_carry_the_run_and_both_checks(tmp_path):
    (row,) = check_references(["eq75"], [0.01], tmp_path, (16, 0.25), (20, 0.25))
    assert row["medium"] == "eq75" and row["delta"] == 0.01
    assert row["elements"] > 3 and row["steps"] > 100 and not row["reused"]
    assert row["agreement"] < 1e-8 and row["elliptic"] < 1e-8


# --- E3.3 (#28) --------------------------------------------------------------


def test_knee_grids_follow_each_medium_placement_and_floor():
    # 101 resolves to the even neighbour (ties go up); a repeat is dropped.
    matlab = knee_grids("matlab", [50, 100, 101, 200, 200])
    assert [g.n for g in matlab] == [50, 100, 102, 200]
    assert all(g.placement(0.0) == "cell" for g in matlab)
    eq75 = knee_grids("eq75", [50, 100, 200, 400])
    assert [g.n for g in eq75] == [101, 201, 401]
    assert all(g.placement(xi) == "node" for g in eq75 for xi in (0.0, 0.5))


def _rates(errors):
    e = np.asarray(errors)
    return np.log2(e[:-1] / e[1:])


@pytest.mark.parametrize("name", ["matlab", "eq75"])
def test_the_naive_elliptic_knee_sits_at_h_equal_delta(name):
    # P2 (stiff note §1.7): about first order while h ≳ δ, fourth once h ≲ δ,
    # a drop of two orders across h = δ; the pre-knee rates are noisy, so
    # they are averaged over two doublings.
    delta = 0.0025
    grids = knee_grids(name, [50, 100, 200, 400, 800, 1600, 3200])
    rows = elliptic_sweep(name, delta, grids)
    ratio = np.array([r["h"] / delta for r in rows])
    naive = np.array([r["naive"] for r in rows])
    pre = ratio > 3.9  # three points on the MATLAB medium, two on eq. 75
    post = ratio < 0.6
    assert pre.sum() >= 2 and post.sum() == 2
    assert np.mean(_rates(naive[pre])) < 2.0
    assert _rates(naive[post])[0] > 3.8
    at_2 = naive[np.argmin(np.abs(ratio - 2.0))]
    at_half = naive[np.argmin(np.abs(ratio - 0.5))]
    assert at_2 / at_half > 100
    # The jump-aware line at δ = 0 is E1's: exact on the two constants,
    # fourth order on eq. 75.
    jump = [r["δ = 0 construction"] for r in elliptic_sweep(name, 0.0, grids[:4])]
    if name == "matlab":
        assert np.all(np.array(jump) < 1e-11)
    else:
        assert np.all(_rates(jump) > 3.8)


def test_the_delta_zero_construction_sits_on_the_floor_only_while_h_exceeds_delta():
    # P3 as measured: on the floor ‖u₀ − u_δ‖/‖u_δ‖ (about 0.75 δ) to 1 % for
    # h ≥ 4δ, and an order of magnitude above it at h = δ/2, where the
    # rebuilt rows enforce a jump's kink that the resolved solution lacks.
    grids = knee_grids("matlab", [50, 100, 200, 400, 800, 1600])
    for delta in (0.01, 0.0025):
        rows = elliptic_sweep("matlab", delta, grids)
        for r in rows:
            if r["h"] >= 4 * delta:
                assert r["δ = 0 construction"] == pytest.approx(r["floor"], rel=0.01)
                assert r["floor"] / delta == pytest.approx(0.75, rel=0.05)
            if r["h"] <= 0.5 * delta:
                assert r["δ = 0 construction"] > 10 * r["floor"]


def test_the_parabolic_knee_and_floor_on_the_matlab_medium(tmp_path):
    delta = 0.01
    grids = knee_grids("matlab", [100, 200, 400, 800])
    cache = {}
    rows = parabolic_sweep("matlab", delta, grids, tmp_path, cache, (16, 0.25))
    assert len(cache) == 12
    naive = _rates([r["naive"] for r in rows])
    assert naive[0] < 1.5 and naive[1] > 4.0 and naive[2] > 3.8
    # The floor at t = 2 is O(δ) (about 0.28 δ); the construction is on it at
    # h = 2δ and far above it at h = δ/8.
    assert rows[0]["floor"] / delta == pytest.approx(0.28, rel=0.1)
    assert rows[0]["δ = 0 construction"] == pytest.approx(rows[0]["floor"], rel=0.02)
    assert rows[-1]["δ = 0 construction"] > 10 * rows[-1]["floor"]
    # The cache is read back: the second sweep computes nothing.
    again = parabolic_sweep("matlab", delta, grids, tmp_path, cache, (16, 0.25))
    assert all("solved" not in r for r in again) and len(cache) == 12


def test_floor_constants_match_the_closed_form():
    rows = floor_constants("matlab", (0.0, 0.04, 0.01, 0.0025))
    assert len(rows) == 3 and rows[0]["closed"] == pytest.approx(8.788898, abs=1e-6)
    for r in rows:
        assert r["measured"] == pytest.approx(r["closed"], rel=1e-12)
    # Two edges at contrast ten; the sinusoid's variation across the edge
    # keeps the study's δ short of the limit and the approach is first order.
    rows = floor_constants("eq75", (0.01, 0.001, 0.0001))
    assert rows[0]["closed"] == pytest.approx(2 * 10.361633, abs=1e-5)
    gaps = np.array([r["closed"] - r["measured"] for r in rows])
    assert np.all(gaps > 0) and np.all(gaps[:-1] / gaps[1:] > 6)


@pytest.mark.parametrize("name", ["matlab", "eq75"])
def test_the_delta_zero_rows_are_first_order_in_delta_over_h(name):
    rows = row_residuals(name, 200, (1.0, 0.1, 0.01, 0.001))
    scaled = np.array([r["scaled"] for r in rows])
    # h² max|L_h (u_δ − u₀)| / δ settles to a constant as δ/h → 0 ...
    assert scaled[-1] == pytest.approx(scaled[-2], rel=0.05)
    # ... and is well below it at δ = h, where the edge is partly resolved.
    assert scaled[0] < 0.5 * scaled[-1]
    on_jump = np.array([r["on_jump"] for r in rows])
    if name == "matlab":
        assert np.all(on_jump < 1e-12)
    else:
        assert np.all(on_jump > 1e-8)


# --- E3.4 (#29) --------------------------------------------------------------


def test_the_seed_line_is_on_the_floor_of_the_reference_at_every_delta(tmp_path):
    # P7 on the ramp problem: the δ = 0.01 and δ = 0 seed lines agree to 2 %
    # and are fourth order (the study's reference resolution: the coarse one
    # of the tests above floors the 400-node point at 3e-10); the construction
    # is 1e4 above them at h = δ/8.
    grids = knee_grids("matlab", [100, 200, 400])
    cache = {}
    res = (N_CHEB, MAX_WIDTH)
    lines = {}
    for d in (0.0, 0.01):
        rows = parabolic_sweep("matlab", d, grids, tmp_path, cache, res)
        lines[d] = [r["seeds"] for r in rows]
    for line in lines.values():
        assert np.all(_rates(line) > 3.8), line
    np.testing.assert_allclose(lines[0.01], lines[0.0], rtol=2e-2)
    rows = elliptic_sweep("matlab", 0.01, grids)
    assert all(r["seeds"] < 1e-12 for r in rows)
    assert rows[-1]["δ = 0 construction"] > 1e4 * rows[-1]["seeds"]


def test_weights_vs_jump_reproduces_the_scratch_numbers_and_the_conditioning():
    rows = weights_vs_jump("matlab", 200, (0.0, 1.0, 0.1, 0.01, 0.001))
    assert rows[0]["difference"] < 1e-12
    np.testing.assert_allclose(
        [r["difference"] for r in rows[1:]], [0.843, 0.113, 0.0107, 0.00107], rtol=0.02
    )
    assert all(80 < r["cond"] < 150 for r in rows)
    # On eq. 75 the δ = 0 difference is the O(h alpha'/alpha) floor, not zero.
    rows = weights_vs_jump("eq75", 200, (0.0, 0.001))
    assert rows[0]["difference"] > 0.1
    assert rows[1]["difference"] == pytest.approx(rows[0]["difference"], rel=0.05)


def test_seed_spectra_are_stable_where_the_construction_is_not():
    rows = seed_spectra("eq75", (49, 101), (0.0, 0.0025))
    assert [r["n"] for r in rows] == [49, 49, 101, 101]
    for r in rows:
        assert r["max_real"] < -2 and r["imag"] == 0.0 and r["bd4"] < 1
        assert r["extreme"] == pytest.approx(-16 / 3, rel=0.01)


# --- E3.5 (#30) --------------------------------------------------------------


def test_the_comparator_table_has_every_treatment_at_every_delta(tmp_path):
    # P10 on the MATLAB medium at a coarse reference: T1-FV second order with
    # a δ-independent constant, the seeds fourth order, and the ranking
    # seeds < T1-FV < T1 (two cells) < naive < T0 (m = 1) < T0 (m = 2) at
    # δ = 0 and wherever the naive operator is not in its dip at h ≈ 2δ
    # (§2.2), i.e. at h ≥ 4δ; T0 with m = 1 is the naive operator once h ≤ δ.
    grids = knee_grids("matlab", [50, 100, 200])
    cache = {}
    tables = {}
    for delta in (0.0, 0.01):
        rows = parabolic_sweep(
            "matlab", delta, grids, tmp_path, cache, (16, 0.25), COMPARATORS
        )
        assert all(set(COMPARATORS) <= set(r) for r in rows)
        tables[delta] = rows
    assert len(cache) == 2 * 3 * len(COMPARATORS)
    for delta, rows in tables.items():
        fv = _rates([r["T1-FV"] for r in rows])
        assert np.all(np.abs(fv - 2) < 0.05), (delta, fv)
        for r in rows:
            assert r["seeds"] < r["T1-FV"] < r["T1 harmonic 2c"]
            if delta == 0.0 or r["h"] >= 4 * delta:
                assert r["T1 harmonic 2c"] < r["naive"] < r["T0 widened m=1"]
            if delta == 0.0 or r["h"] >= 2 * delta:
                assert r["T0 widened m=1"] < r["T0 widened m=2"]
    np.testing.assert_allclose(
        [r["T1-FV"] for r in tables[0.01]], [r["T1-FV"] for r in tables[0.0]], rtol=0.02
    )
    # With the edge mid-cell the one-cell means change nothing at δ = 0.
    for r in tables[0.0]:
        assert r["T1 harmonic 1c"] == pytest.approx(r["naive"], rel=1e-10)
        assert r["T2 arithmetic 1c"] == pytest.approx(r["naive"], rel=1e-10)
    # The elliptic table: T1-FV and the seeds exact, the rest not.
    rows = elliptic_sweep("matlab", 0.01, grids, COMPARATORS)
    for r in rows:
        assert r["T1-FV"] < 1e-12 and r["seeds"] < 1e-12
        assert min(r[t] for t in TREATMENTS if t != "T1-FV") > 1e-5


def test_the_widened_edge_sits_on_its_own_floor_until_the_edge_is_resolved():
    # T0's floor ‖u_{max(δ, m h)} − u_δ‖/‖u_δ‖ is about 0.7 (m h − δ) on the
    # MATLAB medium (§2.2's 0.73–0.75 δ, drifting to 0.66 by a width of 0.08
    # as the widened tails reach the boundary), and T0's error is that floor
    # to 1 % while the naive operator resolves the widened edge (m = 2,
    # h = width / 2); with m = 1 the operator sits at its own knee, at or
    # above the floor, and is the naive operator itself once h ≤ δ.
    delta = 0.01
    grids = knee_grids("matlab", [50, 100, 200, 400])
    floors = widened_floors("matlab", delta, grids)
    errors = elliptic_sweep("matlab", delta, grids, COMPARATORS)
    assert WIDENINGS == (1, 2)
    for f, e in zip(floors, errors, strict=True):
        for m in WIDENINGS:
            width = max(delta, m * f["h"])
            if width > delta:
                assert 0.65 < f[m] / (width - delta) < 0.76, (f["n"], m)
            else:
                assert f[m] < 1e-13
        assert e["T0 widened m=2"] == pytest.approx(f[2], rel=1e-2)
        if f["h"] > delta:
            assert e["T0 widened m=1"] >= 0.99 * f[1]
        else:
            assert e["T0 widened m=1"] == pytest.approx(e["naive"], rel=0.02)


# --- E3.6 (#31) --------------------------------------------------------------


def test_the_seed_functions_meet_the_translated_basis_at_first_order_in_delta_over_h():
    # At δ = 0 the march is E1.2's algebra function by function (P4 checked the
    # weights only); away from it the sup distance of each seed from the
    # translated basis is first order in δ/h (5.1e-2, 5.1e-3, 5.1e-4 for φ₁),
    # while the monomials stay O(1) away: the kink of the 1/9 | 1 jump.
    data = seed_functions("matlab", 200, ratios=(0.1, 0.01, 0.001))
    rows = {r["ratio"]: r for r in data["rows"]}
    assert max(rows[0.0]["vs_jump"]) < 1e-13
    assert min(rows[0.0]["vs_monomials"]) > 0.5
    for k in range(4):
        d = np.array([rows[r]["vs_jump"][k] for r in (0.1, 0.01, 0.001)])
        assert np.all(np.abs(np.log10(d[:-1] / d[1:]) - 1) < 0.1), (k, d)
    assert rows[0.001]["vs_jump"][0] == pytest.approx(5.12e-4, rel=0.02)
    # Stencil units: the nodes at 0, ±½, ±1, the edge a quarter of the way right;
    # on the node's side of it the δ = 0 seeds are the monomials themselves.
    np.testing.assert_allclose(data["xi_nodes"], [-1.0, -0.5, 0.0, 0.5, 1.0])
    assert data["xi_edges"] == pytest.approx([0.25])
    left = data["xi"] < 0.0
    for k in range(5):
        np.testing.assert_allclose(
            data["seeds"][0.0][k][left], data["monomials"][k][left], atol=1e-13
        )
    assert data["h_s"] == pytest.approx(2 * data["h"])


def test_the_snapshot_repeats_the_sweeps_numbers_and_says_where_the_error_sits(
    tmp_path,
):
    # The same four marches as the sweep's row at 100 nodes and δ = 0.0025
    # (h = 8δ): naive on its first-order line, the construction on its floor,
    # T1-FV second order, the seeds on the δ = 0 line, four orders below naive.
    # The naive and construction errors are not confined to the edge (a wrong
    # effective resistance shifts the whole profile), though the largest
    # nodal error of the naive operator sits beside it.
    grids = knee_grids("matlab", [100])
    cache: dict[str, float] = {}
    rows = parabolic_sweep(
        "matlab", 0.0025, grids, tmp_path, cache, (16, 0.25), SNAPSHOT_OPERATORS
    )
    pic = snapshot("matlab", 0.0025, 100, tmp_path, (16, 0.25))
    got = {r["label"]: r for r in pic["rows"]}
    for label in SNAPSHOT_OPERATORS:
        assert got[label]["error"] == pytest.approx(rows[0][label], rel=1e-12)
    assert (
        got["naive"]["error"]
        > got["δ = 0 construction"]["error"]
        > got["T1-FV"]["error"]
        > got["seeds"]["error"]
    )
    assert got["naive"]["error"] > 1e4 * got["seeds"]["error"]
    assert got["naive"]["local"] < 0.3 and got["δ = 0 construction"]["local"] < 0.3
    assert abs(got["naive"]["at"]) <= 2 * pic["h"]
    assert pic["n"] == 100 and pic["x"].shape == pic["u"].shape == (100,)
    assert pic["u_fine"].shape == pic["alpha_fine"].shape == (2001,)
