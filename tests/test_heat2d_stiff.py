"""The E4 driver: E4.2's separable references through the smooth flat band, and
E4.3's naive baseline with its straddling-row diagnostics."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_stiff import (  # noqa: E402
    KNEE_CACHE,
    KNEE_CACHE_META,
    OPERATORS,
    QUANTITIES,
    STUDY_DELTAS,
    curve_level,
    edge_diagnostics,
    knee_key,
    load_knee_cache,
    main,
    matched_ratios,
    pair_fluxes,
    row_profile,
    save_knee_cache,
)
from heat_interfaces.heat2d import (  # noqa: E402
    SmoothBand,
    build_node_set,
    case1,
    case1_reference,
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
