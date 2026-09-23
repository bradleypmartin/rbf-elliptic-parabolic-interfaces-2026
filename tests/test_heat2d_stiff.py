"""The E4 driver: E4.2's separable references through the smooth flat band,
E4.3's naive baseline with its straddling-row diagnostics, and E4.6's δ sweep."""

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_stiff import (  # noqa: E402
    CASE1,
    CURVED_CACHE,
    CURVED_CACHE_META,
    KNEE_CACHE,
    KNEE_CACHE_META,
    OPERATORS,
    QUANTITIES,
    STUDY_DELTAS,
    Geometry,
    curve_level,
    edge_diagnostics,
    flat_twin,
    knee_key,
    load_knee_cache,
    main,
    matched_ratios,
    pair_fluxes,
    row_profile,
    save_knee_cache,
    seed_label,
    seed_operators,
)
from heat_interfaces.heat1d.domain import TANH_REACH  # noqa: E402
from heat_interfaces.heat2d import (  # noqa: E402
    BOUNDARY,
    ProductGridReference,
    SmoothBand,
    build_node_set,
    build_stencils,
    case1,
    case1_reference,
    case2,
    seed_operator,
)

_nodes = {}


def nodes_2500():
    if not _nodes:
        _nodes["set"] = build_node_set(case1(), 2500)
    return _nodes["set"]


def test_the_reference_table_at_the_study_deltas(capsys):
    tables = main(["--mode", "references", "--deltas", *map(str, STUDY_DELTAS)])
    rows = tables["references"]
    out = capsys.readouterr().out
    assert "separable reference" in out and "distance / δ" in out
    assert len(rows) == 2 * len(STUDY_DELTAS)
    for r in rows:
        assert r["agreement"] < 5e-11
        if r["delta"] == 0.0:
            assert r["distance"] < 1e-12 and r["plateau"] == 0.0
        else:
            # The O(δ) gap to the jump solution (H10's floor).
            assert 0.6 < r["distance"] / r["delta"] < 1.7
    wide = [r for r in rows if r["delta"] == 0.04]
    assert all(abs(r["plateau"] - 1.07e-2) < 1e-4 for r in wide)


def test_row_profile_is_the_separable_modes_profile_exactly():
    # On every fixed row (straddling and Dirichlet, the staggered ones too)
    # the sin 2πx coefficient of e^{ct} sin 2πx v(y) is e^{ct} v(y_row); a
    # cos 2πx or sin 4πx field has none.
    nodes = nodes_2500()
    ref = case1_reference(0.01, 1.0)
    u = ref(nodes.x, nodes.y, 0.1)
    other = np.cos(2 * np.pi * nodes.x) + np.sin(4 * np.pi * nodes.x) * nodes.y
    for row in nodes.straddle_rows + nodes.dirichlet_rows:
        y = nodes.y[row.index[0]]
        assert row_profile(nodes, row, u) == pytest.approx(
            np.exp(0.1) * ref.v(y), abs=1e-14
        )
        assert abs(row_profile(nodes, row, other)) < 1e-14


def test_curve_level_is_the_straddled_line():
    nodes = nodes_2500()
    assert curve_level(nodes, 0) == pytest.approx(0.6, abs=1e-15)
    assert curve_level(nodes, 1) == pytest.approx(0.8, abs=1e-15)


@pytest.mark.parametrize("curve, level", [(0, 0.6), (1, 0.8)])
def test_pair_fluxes_are_one_sided_and_exact_on_quadratic_profiles(curve, level):
    # A profile quadratic on each side with a kink at the curve: the three
    # rows of a side see only their own quadratic, so the flux at ±h/2 is
    # α p′ there to rounding, whatever the other side does.
    nodes = nodes_2500()
    medium = SmoothBand(case1().material, 0.0)
    s = nodes.y - level
    below = 1.0 + 2.0 * s + 3.0 * s**2
    above = 1.0 + 7.0 * s - 5.0 * s**2
    u = np.sin(2 * np.pi * nodes.x) * np.where(s < 0, below, above)
    q_below, q_above = pair_fluxes(nodes, medium, u, curve)
    y_lo, y_hi = level - nodes.h / 2, level + nodes.h / 2
    alpha_lo = float(medium.alpha(0.0, y_lo))
    alpha_hi = float(medium.alpha(0.0, y_hi))
    # Outside the band below 0.6 and above 0.8, inside it between.
    assert {alpha_lo, alpha_hi} == {1.0, 0.2}
    assert q_below == pytest.approx(alpha_lo * (2.0 + 6.0 * (y_lo - level)), rel=1e-11)
    assert q_above == pytest.approx(alpha_hi * (7.0 - 10.0 * (y_hi - level)), rel=1e-11)


def test_edge_diagnostics_vanish_on_the_reference_and_read_a_planted_error():
    nodes = nodes_2500()
    delta, t = 0.005, 0.1
    medium = SmoothBand(case1().material, delta)
    ref = case1_reference(delta, 1.0)
    exact = ref(nodes.x, nodes.y, t)
    clean = edge_diagnostics(nodes, medium, ref, exact, t)
    assert set(clean) == set(QUANTITIES)
    assert max(clean.values()) < 1e-13
    # A planted error sin 2πx · ε (y − 0.6) below the lower curve only: the
    # profile reads ε (−h/2) there, the flux ε α / |α v′(0.6)| on that side,
    # and the jump the same, since the other side sees nothing.
    eps = 1e-3
    planted = eps * np.sin(2 * np.pi * nodes.x) * np.minimum(nodes.y - 0.6, 0.0)
    d = edge_diagnostics(nodes, medium, ref, exact + planted, t)
    scale = abs(float(ref.flux_y(0.25, 0.6, t)))
    alpha = float(medium.alpha(0.0, 0.6 - nodes.h / 2))
    assert d["profile"] == pytest.approx(eps * nodes.h / 2, rel=1e-9)
    assert d["flux"] == pytest.approx(eps * alpha / scale, rel=1e-9)
    assert d["jump"] == pytest.approx(eps * alpha / scale, rel=1e-9)
    assert d["max"] == pytest.approx(np.abs(planted).max(), rel=1e-12)


def test_matched_ratios_pair_each_delta_and_count_with_half_and_four_times():
    def row(n, delta, value):
        h = 1.0 / round(0.95 * np.sqrt(n))
        return {"n": n, "h": h, "h_over_delta": h / delta, "q": value}

    results = {
        0.0: [row(1250, 1.0, 9.0)],
        0.01: [row(1250, 0.01, 8.0), row(2500, 0.01, 6.0)],
        0.005: [row(5000, 0.005, 4.0), row(10000, 0.005, 3.0)],
    }
    rows = matched_ratios(results, ["q"])
    assert [(r["delta"], r["n"], r["q"]) for r in rows] == [
        (0.01, 1250, 2.0),
        (0.01, 2500, 2.0),
    ]
    for r in rows:
        assert r["h_over_delta"] == pytest.approx(r["h_over_delta_finer"], rel=0.02)


def test_knee_cache_round_trips_and_rejects_another_studys_file(tmp_path):
    key = knee_key("parabolic", 0.0025, 1250, "naive", 0, 100, 0.1)
    assert key == "parabolic t0.1 d0.0025 n1250 s0 i100 naive"
    assert knee_key("elliptic", None, 1250, "uniform", 0, 100, 0.1) == (
        "elliptic d- n1250 s0 i100 uniform"
    )
    cache = {key: {"rms": 1.5e-3}}
    save_knee_cache(tmp_path, cache)
    assert load_knee_cache(tmp_path) == cache
    data = json.loads((tmp_path / KNEE_CACHE).read_text())
    data["meta"] = {**KNEE_CACHE_META, "version": -1}
    (tmp_path / KNEE_CACHE).write_text(json.dumps(data))
    assert load_knee_cache(tmp_path) == {}


def test_the_naive_baseline_at_the_two_smallest_counts(tmp_path, capsys):
    argv = [
        "--mode",
        "naive",
        "--counts",
        "1250",
        "2500",
        "--spectrum-counts",
        "1250",
        "--outputs",
        str(tmp_path),
    ]
    tables = main(argv)
    out = capsys.readouterr().out
    assert "growing-mode check" in out and "matched h/δ" not in out
    assert (tmp_path / "heat2d_stiff_knee.png").exists()
    ell, par = tables["knee"]["elliptic"], tables["knee"]["parabolic"]
    # δ = 0: port notes §2.2's naive and uniform (control) lines and §2.4's
    # warped aware line, on the same node sets.
    jump = ell[0.0]
    assert [r["naive/rms"] for r in jump] == pytest.approx([4.14e-3, 2.78e-3], rel=0.01)
    assert [r["uniform"] for r in jump] == pytest.approx([9.68e-5, 2.23e-6], rel=0.01)
    assert [r["construction/rms"] for r in jump] == pytest.approx(
        [1.598e-5, 3.795e-6], rel=1e-3
    )
    for problem in (ell, par):
        for r in problem[0.0]:
            assert r["floor"] == 0.0 and r["h_over_delta"] == float("inf")
            assert set(f"{op}/{q}" for op in OPERATORS for q in QUANTITIES) <= set(r)
        # The construction sits on its O(δ) floor while h >= 4δ (H10).
        for delta in (0.005, 0.0025):
            for r in problem[delta]:
                assert r["construction/rms"] / r["floor"] == pytest.approx(1, abs=0.02)
    for r in ell[0.0025] + ell[0.0]:
        # Unresolved: the flux on the innermost pair is off by its own size.
        assert r["naive/flux"] > 0.3 and r["naive/jump"] > 0.3
    for r in ell[0.04]:
        # Resolved: two orders better, and the construction off the floor.
        assert r["naive/flux"] < 0.02 and r["naive/rms"] < 1e-4
        assert r["construction/rms"] > r["floor"] and r["construction/flux"] > 0.5
    spectra = {(r["delta"], r["operator"]): r for r in tables["spectra"]}
    for delta in STUDY_DELTAS:
        # The naive operator's coarse-set growing mode is there at every δ,
        # BD4 at dt = h amplifies it; the construction is damped.
        naive, built = spectra[(delta, "naive")], spectra[(delta, "construction")]
        assert naive["n"] == 1250 and naive["positive"] >= 1 and naive["bd4"] > 1
        assert built["max_re"] < 0 and built["bd4"] < 1
        assert 1.0 < built["parabolic_over_elliptic"] < 1.15

    # Everything is cached: the second run builds nothing and agrees.
    t0 = time.perf_counter()
    again = main(argv)
    assert time.perf_counter() - t0 < 10
    assert again["knee"] == tables["knee"]


def test_the_stencil_study_at_1250_nodes(capsys):
    # E4.4 (#35), stiff note §4.3: H1–H3 and the march's cost on the smallest
    # set; the 2500-node numbers the notes quote are the default run's.
    tables = main(["--mode", "stencils", "--stencil-n", "1250"])["stencils"]
    out = capsys.readouterr().out
    assert "H1, the march is the chain" in out and "stencil study" in out
    for r in tables["chain"]:
        assert r["monomials"] < 5e-15 and r["shift"] < 5e-15 and r["warp"] == 0.0
        assert r["one_d"] < 5e-13
        assert r["ratio"] == 0.0 or r["residual"] < 2e-10
    case, thin = tables["jump_limit"]
    assert case["stencils"] > 300 and case["three_region"] == 0
    assert thin["three_region"] > 100
    for r in (case, thin):
        assert r["span"] < 1e-13 and r["weights"] < 1e-11 and r["warp"] < 1e-14
    for anchor in {r["anchor"] for r in tables["ladder"]}:
        rows = [r for r in tables["ladder"] if r["anchor"] == anchor]
        spans = np.array([r["span"] for r in rows])  # δ/h from 8 down, then 0
        assert np.all(np.diff(spans[:-1]) < 0) and spans[-1] < 1e-13
        assert spans[-4] / spans[-2] == pytest.approx(100.0, rel=0.02)
        scaled = np.array([r["scaled"] for r in rows])
        monomial = rows[0]["monomial"]
        assert np.all((scaled > monomial / 3) & (scaled < 3 * monomial))
    for r in tables["timing"]:
        assert r["median_ms"] < 30.0


def test_seed_label_carries_the_warp_and_a_non_default_reach():
    assert seed_label(True) == "seeds" and seed_label(False) == "seeds-plain"
    assert seed_label(True, 5.0) == "seeds-r5"
    assert seed_label(False, 5.0) == "seeds-plain-r5"
    # E4.11's chain rides in the label too, and is read over case 1's seeds.
    assert seed_label(True, tangential=True) == "tangential"
    assert seed_label(False, 5.0, True) == "tangential-plain-r5"
    assert flat_twin("tangential-plain-r5") == "seeds-plain-r5"
    assert flat_twin("construction") == "construction"


def test_seed_operators_build_both_warps_from_one_march():
    # E4.6's ablation is a second weight solve on the marched basis, so the
    # two operators must be ``seed_operator``'s own, row for row.
    nodes = build_node_set(case1(), 900)
    medium = SmoothBand(case1().material, 0.005)
    domain = replace(case1(), material=medium)
    stencils = build_stencils(nodes, domain, interface=BOUNDARY, reach=TANH_REACH)
    ops, seeded = seed_operators(nodes, medium, stencils, (True, False))
    assert 0 < seeded < nodes.n
    for warp in (True, False):
        one = seed_operator(nodes, medium, stencils, warp=warp)
        difference = abs(ops[warp] - one).max()
        assert difference <= 1e-9 * abs(one).max()


def test_the_seed_sweep_at_the_two_smallest_counts(tmp_path, capsys):
    # E4.6 (#37), stiff note §4.5: H4, H7 and H8 at 900 and 1250 nodes; the
    # documented sweep runs the same lines to 40,000.
    argv = [
        "--mode",
        "seeds",
        "--counts",
        "900",
        "1250",
        "--deltas",
        "0",
        "0.04",
        "0.005",
        "--outputs",
        str(tmp_path),
    ]
    tables = main(argv)
    out = capsys.readouterr().out
    assert "H8, the resolved-edge penalty" in out and "H4 at δ = 0" in out
    assert (tmp_path / "heat2d_stiff_seeds.png").exists()
    results = tables["sweep"]
    for problem, lines in results.items():
        # H4: at δ = 0 the seed rows are E2.3's, so the line is the
        # construction's — port notes §2.4–2.5's, 1.598e-5 at 1250 nodes.
        for r in lines[0.0]:
            assert r["seeds/rms"] == pytest.approx(r["construction/rms"], rel=1e-7)
        assert [r["seeds/rms"] for r in lines[0.0]] == pytest.approx(
            [4.937e-5, 1.598e-5] if problem == "elliptic" else [5.329e-5, 1.764e-5],
            rel=1e-3,
        )
        for delta, rows in lines.items():
            for i, r in enumerate(rows):
                # Flat in δ: the seeds are within a factor 2 of their δ = 0
                # value at every width, where naive and construction move by
                # orders (the comparators' own knee, E4.3).
                assert 0.5 < r["seeds/rms"] / lines[0.0][i]["seeds/rms"] < 2.0
                # H7: the warp is worth a factor, and never costs one.
                assert r["seeds-plain/rms"] > 0.95 * r["seeds/rms"]
                # The rule: every row at δ = 0.04 (20 δ covers the strip),
                # the crossing rows at δ = 0, in between elsewhere.
                if delta == 0.04:
                    assert r["seeds/rows"] == r["n"]
                else:
                    assert 0 < r["seeds/rows"] < r["n"]
    for problem in results:
        # H8: every row seeded at δ = 0.04 and still ahead of the direct
        # operator, which is the resolved edge's own method.
        resolved = tables[f"resolved/{problem}"]
        assert resolved and all(r["delta"] == 0.04 for r in resolved)
        for r in resolved:
            assert r["fraction"] == 1.0 and r["over/direct"] < 1.2
            # The control with the seeds' own stencils: 30 / 4 on every row
            # here, so it isolates the seed rows from the smaller stencil.
            assert r["direct-reach/rms"] > 0.0
        ratios = {(r["delta"], r["n"]): r for r in tables[f"ratios/{problem}"]}
        # The rule needs no δ: the seeds beat the naive operator at every
        # unresolved width and the construction at every resolved one.
        for (delta, _), r in ratios.items():
            assert r["over/construction"] <= 1.0 + 1e-6
            if delta in (0.0, 0.005):
                assert r["over/naive"] < 0.1
        assert ratios[(0.0, 1250)]["fraction"] < 0.4

    # Everything is cached: the second run builds nothing and agrees.
    t0 = time.perf_counter()
    again = main(argv)
    assert time.perf_counter() - t0 < 15
    assert again["sweep"] == results


def test_the_seed_sweep_needs_its_own_line():
    with pytest.raises(SystemExit):
        main(["--mode", "seeds", "--operators", "naive", "construction"])


def test_the_delta_zero_regression_run(tmp_path, capsys):
    # The H4 regression the notes send E4.7 to (`--deltas 0`, no ablation):
    # no positive width and no `seeds-plain` line, so the figure keeps only
    # the panels with something in them and the run still finishes.
    tables = main(
        [
            "--mode",
            "seeds",
            "--deltas",
            "0",
            "--operators",
            "naive",
            "construction",
            "seeds",
            "--counts",
            "900",
            "--outputs",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert "H4 at δ = 0" in out and "seed sweep" in out
    assert (tmp_path / "heat2d_stiff_seeds.png").exists()
    for lines in tables["sweep"].values():
        (row,) = lines[0.0]
        assert row["seeds/rms"] == pytest.approx(row["construction/rms"], rel=1e-7)
        assert "seeds-plain/rms" not in row


# --- E4.7: the curved feature ---------------------------------------------------------


def test_the_geometries_keep_case_ones_keys_and_cache():
    assert CASE1.is_case1 and CASE1.tag == "" and CASE1.domain() == case1()
    assert CASE1.cache() == (KNEE_CACHE, KNEE_CACHE_META)
    two = Geometry(0.02, "sine")
    assert two.name == "case 2" and two.domain() == case2()
    assert two.cache() == (CURVED_CACHE, CURVED_CACHE_META)
    key = knee_key("elliptic", 0.0, 1250, "seeds", 0, 100, 0.1)
    assert knee_key("elliptic", 0.0, 1250, "seeds", 0, 100, 0.1, two.tag) == (
        f"a0.02 sine {key}"
    )
    assert isinstance(two.reference(0.0, 0.0), ProductGridReference)
    assert two.reference(0.0, 0.0) is two.reference(0.0, 0.0)
    with pytest.raises(ValueError, match="inside"):
        Geometry(0.02, "linear")


def test_the_curved_sweep_at_the_two_smallest_counts(tmp_path, capsys):
    # E4.7 (#38), stiff note §4.6: route (a) on case 2. At δ = 0 the
    # construction is E2.6's curved line (3.06e-5 at 1250 nodes, now against
    # the product grid) and route (a) is not: its seed rows are O(1)-wrong
    # where alpha varies along the edge, 3.80e-4 at 1250 and the probe's
    # seeded rows barely converging while E2.3's crossing rows do.
    argv = [
        "--mode",
        "seeds",
        "--amplitude",
        "0.02",
        "--counts",
        "900",
        "1250",
        "--deltas",
        "0",
        "0.0025",
        "--operators",
        "construction",
        "seeds",
        "--outputs",
        str(tmp_path),
    ]
    tables = main(argv)
    out = capsys.readouterr().out
    assert "route (a) at δ = 0 against the curved construction" in out
    assert "H9's probe" in out and "case 2 over case 1" in out
    assert (tmp_path / CURVED_CACHE).exists() and not (tmp_path / KNEE_CACHE).exists()
    assert (tmp_path / "heat2d_stiff_seeds_a0.02_sine.png").exists()
    elliptic = tables["sweep"]["elliptic"]
    jump = {r["n"]: r for r in elliptic[0.0]}
    assert jump[1250]["construction/rms"] == pytest.approx(3.059e-5, rel=1e-3)
    assert jump[1250]["seeds/rms"] == pytest.approx(3.804e-4, rel=1e-3)
    probe = [r["seeds/probe_crossing"] for r in elliptic[0.0]]
    built = [r["construction/probe_crossing"] for r in elliptic[0.0]]
    assert probe[1] / probe[0] > 0.8 and built[1] / built[0] < 0.6
    for lines in tables["sweep"].values():
        # Both problems: 12× the construction at the jump, and still below the
        # δ = 0 construction on a 0.0025 edge, which it reads as a jump.
        assert lines[0.0][1]["seeds/rms"] > 5.0 * lines[0.0][1]["construction/rms"]
        for r in lines[0.0025]:
            assert r["seeds/rms"] < r["construction/rms"]
    again = main(argv)
    assert again["sweep"] == tables["sweep"]


def test_the_tangential_line_on_the_curved_sweep(tmp_path, capsys):
    # E4.11 (#81), stiff note §3.10 and §4.7: the tangential chain as the
    # sweep's main line, beside route (a). At 1250 nodes it is the coarsest
    # point of its line (1.25e-4 at δ = 0, 4× E2.3's); what the test pins is
    # the probe: its crossing rows converge between 900 and 1250 nodes where
    # route (a)'s stall, and its figure and labels are its own, E4.7's
    # untouched.
    argv = [
        "--mode",
        "seeds",
        "--amplitude",
        "0.02",
        "--counts",
        "900",
        "1250",
        "--deltas",
        "0",
        "0.0025",
        "--operators",
        "construction",
        "seeds",
        "tangential",
        "tangential-plain",
        "--outputs",
        str(tmp_path),
    ]
    tables = main(argv)
    out = capsys.readouterr().out
    assert "the tanh edges of case 2 (the tangential chain)" in out
    assert "the tangential chain at δ = 0 against the curved construction" in out
    assert (tmp_path / "heat2d_stiff_tangential_a0.02_sine.png").exists()
    assert not (tmp_path / "heat2d_stiff_seeds_a0.02_sine.png").exists()
    elliptic = tables["sweep"]["elliptic"]
    jump = {r["n"]: r for r in elliptic[0.0]}
    assert jump[1250]["tangential/rms"] == pytest.approx(1.253e-4, rel=1e-3)
    assert jump[1250]["seeds/rms"] == pytest.approx(3.804e-4, rel=1e-3)
    for delta in (0.0, 0.0025):
        rows = elliptic[delta]
        chain = [r["tangential/probe_crossing"] for r in rows]
        frozen = [r["seeds/probe_crossing"] for r in rows]
        assert chain[1] / chain[0] < 0.65 and frozen[1] / frozen[0] > 0.75
        assert chain[1] < 0.6 * frozen[1]
    ratios = tables["ratios/elliptic"]
    assert all("over/tangential-plain" in r for r in ratios)
    again = main(argv)
    assert again["sweep"] == tables["sweep"]


def test_the_tangential_tables(capsys, monkeypatch):
    # E4.11's own tables: H14 on the concentric circles (from 2500 nodes, the
    # focal-distance guard), H15's span distance on case 2, and H6's twin on
    # case 2 (here on 900 nodes at two widths; the documented run is 1600 at
    # the four).
    import heat2d_stiff

    monkeypatch.setattr(heat2d_stiff, "TANGENTIAL_SPECTRUM_N", 900)
    monkeypatch.setattr(heat2d_stiff, "TANGENTIAL_TIMING", (1250, 10))
    monkeypatch.setattr(heat2d_stiff, "CURVED_DELTAS", (0.0, 0.0025))
    tables = main(["--mode", "tangential", "--counts", "1250", "2500"])["tangential"]
    out = capsys.readouterr().out
    assert "H14, the concentric circles" in out and "H15, case 2" in out
    (circle,) = tables["circles"]
    assert circle["n"] == 2500
    assert circle["tangential"] == pytest.approx(3.81e-3, rel=0.01)
    assert circle["tangential"] < 0.3 * circle["construction"] < circle["seeds"]
    assert [r["n"] for r in tables["span"]] == [1250, 2500]
    assert tables["span"][1]["median"] < 0.8 * tables["span"][0]["median"]
    # H6's twin: no seed operator on case 2 has an eigenvalue right of the axis,
    # and the warped rows' spectrum is E2.3's at δ = 0 (max Re −7.18, 2026-09-22).
    spectra = tables["spectra"]
    assert all(r["positive"] == 0 for r in spectra)
    at_zero = {r["operator"]: r for r in spectra if r["delta"] == 0.0}
    for label in ("seeds", "tangential"):
        assert at_zero[label]["max_re"] == pytest.approx(
            at_zero["construction"]["max_re"], abs=0.01
        )
    assert [r["rows"] for r in tables["timing"]] == [10, 10]
    assert all(r["tangential_ms"] > 0 for r in tables["timing"])


def test_the_curved_reference_table(tmp_path, capsys):
    rows = main(
        [
            "--mode",
            "references",
            "--amplitude",
            "0.02",
            "--deltas",
            "0",
            "0.0025",
            "--growth",
            "0",
            "--outputs",
            str(tmp_path),
        ]
    )["references"]
    assert "product-grid reference" in capsys.readouterr().out
    assert [r["delta"] for r in rows] == [0.0, 0.0025]
    for r in rows:
        assert r["n_x"] < 1e-11 and r["elements_check"] < 1e-11
        assert "e26_rms" not in r  # E2.6's run is not under tmp_path
    assert 3e-3 < rows[1]["distance"] < 3e-2


def test_the_curved_geometries_run_the_seed_sweep_and_the_references_only():
    for mode in ("naive", "stencils", "all"):
        with pytest.raises(SystemExit):
            main(["--mode", mode, "--amplitude", "0.02"])
