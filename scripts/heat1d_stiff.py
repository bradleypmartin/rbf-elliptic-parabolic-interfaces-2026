"""E3 (#5): the 1-D stiff-edge study. E3.2: references; E3.3: the knee; E3.4: seeds.

The media are ``SmoothEdges(matlab_alpha(), δ)``, the ``1/9 | 1`` jump at 0
with a tanh edge of width δ, and the same for eq. 75's layer; the parabolic
problem is the MATLAB one of port notes §1.4 (zero start, ``u(-1, t)`` ramped
to 1 over [0, 1], the solution at t = 2). The references (plan D4) are the
quadrature for the equilibrium problem and the Chebyshev-element solution
with Radau in time for the parabolic one, cut at ``EDGE_CUTS`` about every
edge, wide elements split, and cached under ``outputs/`` as
``heat1d_stiff_reference_<medium>_d<δ>_n<n_cheb>_w<max_width>.npz`` + ``.json``.

The driver first builds (or loads) the parabolic references for the study's δ
values and reports, per medium and δ, the run (elements, unknowns, Radau
steps, seconds) and the two checks the stiff note's P1 asks for: the
agreement with a finer resolution (cached too) and the Chebyshev equilibrium
against the quadrature (E3.2).

It then runs the knee study (E3.3; stiff note §1.7 predicts, §2.2 records):
naive ``Dx A Dx`` and the δ = 0 construction (``jump_aware_operator`` on the
smooth medium, E1.2's rows at the edge centre with the pieces' data) against
the true-δ references, elliptic and parabolic, over a doubling sequence of
node counts at every δ, each δ > 0 line printed with the floor it is read
against (the two equilibria's, or the two parabolic references', difference
at the nodes); the constant of that floor against its closed form
(``edge_resistance_deficit``); and the δ = 0 rows' residual on the true-δ
equilibrium at fixed h as δ/h shrinks. Figure: ``heat1d_stiff_knee.png``.
The parabolic errors are cached in ``heat1d_stiff_knee.json`` so a sweep
extended with ``--counts`` reruns only the new counts.

E3.4 (#29; stiff note §1.9's P4–P9, §2.3 records) puts the seed operator
(``seed_operator``, ``heat1d/stiff.py``) on the same sweep as a third line and
adds: the seed weights against E1.2's as δ/h shrinks (P4) with the stencil
solve's condition number at each δ/h (P5), the seed rows' residual on the
true-δ equilibrium next to the δ = 0 rows' (P6), and the seed operator's
interior spectrum at the study's counts and at the coarse eq. 75 counts the
δ = 0 construction cannot run (P9).

E3.5 (#30; P10, §2.4 records) runs the coefficient treatments of plan §3.4
(``heat1d/treatments.py``) on the same grids and problems as a *comparator
table* per medium and δ, elliptic and parabolic: naive ``Dx A Dx`` (the
untreated end), T1 the harmonic cell mean over one and two cells, T2 the
arithmetic mean, T0 the edge widened to ``max(δ, m h)`` for m = 1, 2 (all
four a changed medium under the naive operator), T1-FV the three-point
conservative scheme with exact face conductances (stiff note §1.8 item 3,
its own operator), and the seeds (the target). T0 is also read against its
own floor, the widened medium's exact equilibrium against the true one.
Figure: ``heat1d_stiff_treatments.png``; the parabolic errors share the knee
cache, keyed by label.

E3.6 (#31; §2.5 records) adds the two figures the manuscript draws its
construction and its headline from, and the results file. The *seed
functions* (``heat1d_stiff_seeds.png``): the five seeds across P4's window
in stencil units at δ/h = ½ and 0.1, against the monomials (constant α)
and E1.2's translated basis (δ = 0, re-centred at the evaluation point), with
the sup distances per seed; the march at δ = 0 is checked against the
algebra function by function. The *parabolic snapshot*
(``heat1d_stiff_snapshot.png``): the ramp solution at ``T_END`` on a coarse
grid with the edge unresolved (``--snapshot-n``, ``--snapshot-delta``; 100
nodes and δ = 0.0025, h = 8δ), the reference against naive, the δ = 0
construction, T1-FV and the seeds, with the pointwise errors and where they
sit. Every table the driver prints goes to ``outputs/heat1d_stiff.json``
(``results_cache.ResultsCache``: command line, args, date, git SHA, timings,
tables), with the two figures' arrays to six figures (``seed_functions/curves``,
``snapshot/curves``), and, with ``--data-dir``, to that directory too, for the
manuscript's figures and number check (E5.3, #44).

    uv run python scripts/heat1d_stiff.py     # 2 min cold, 6 s with everything cached
    uv run python scripts/heat1d_stiff.py --counts 50 100 200 400 800 1600 3200 6400
                                              # the two extra counts: + 3 min, once
    uv run python scripts/heat1d_stiff.py --deltas 0 0.0025 --media matlab
    uv run python scripts/heat1d_stiff.py --data-dir paper/data   # the documented run
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import heat_interfaces  # noqa: E402
from heat_interfaces.heat1d import (  # noqa: E402
    RADAU_ATOL,
    RADAU_RTOL,
    Grid1D,
    Jump,
    ParabolicReference,
    SmoothEdges,
    arithmetic_cells,
    bd4_amplification,
    bd4_march,
    chebyshev_equilibrium,
    dissertation_alpha,
    edge_resistance_deficit,
    equilibrium_exact,
    face_conductance_operator,
    grid_for,
    harmonic_cells,
    interior_operator,
    inverse_alpha_integral,
    jump_aware_operator,
    matlab_alpha,
    naive_operator,
    normalized_l2,
    parabolic_reference,
    ramp_boundary,
    seed_basis,
    seed_operator,
    seed_profiles,
    seed_weights,
    seeded_windows,
    shift_matrix,
    solve_equilibrium,
    stencil_weights,
    straddling_windows,
    translated_basis,
    widened_edge,
)
from heat_interfaces.heat1d.domain import X_MAX, X_MIN  # noqa: E402
from heat_interfaces.heat1d.interface import region_index  # noqa: E402
from heat_interfaces.plotting import AWARE, CONSTRUCTION, NAIVE, REFERENCE  # noqa: E402
from heat_interfaces.results_cache import (  # noqa: E402
    ResultsCache,
    rounded,
    source_hash,
)

STUDY_DELTAS = (0.0, 0.04, 0.01, 0.0025)
"""The edge widths of the study (E3.3, #28): the jump, and three sub-grid ones."""

MEDIA = {"matlab": matlab_alpha, "eq75": dissertation_alpha}
"""The jump media the smooth edges are built on, by the name used in file names."""

RAMP, T_END = 1.0, 2.0
"""The ramp problem: ``u(-1, t)`` reaches 1 at ``RAMP``, read at ``T_END``."""

BC = (1.0, 0.0)
"""``u(-1), u(1)`` of the equilibrium problem, and of the ramp once it is done."""

N_CHEB, MAX_WIDTH = 20, 0.1
"""The reference's resolution: nodes per element, and the element width split above.

Twenty nodes resolve a tanh element (Bernstein parameter ≥ 3.4, see
``EDGE_CUTS``) to 1e-11 and a 0.1-wide piece of eq. 75's sinusoid to 1e-12,
and keep the round-off floor of the collocation system, which grows with the
node count on the δ-wide elements, at 1e-12 (MATLAB medium) to 1e-11
(eq. 75). Thirty-two nodes on unsplit elements left eq. 75's half-unit
layer at 7e-9, and forty-eight raised the floor and the Radau step count on
sub-grid edges (E3.2, 2026-09-21).
"""

CHECK_N_CHEB, CHECK_MAX_WIDTH = 24, 0.05
"""The finer resolution the reference is checked against."""

QUAD_PANELS = 4
"""Gauss panels per cut interval in the elliptic check (a margin over the default 1)."""

PROBLEM = f"matlab-ramp{RAMP:g}"

KNEE_COUNTS = (50, 100, 200, 400, 800, 1600)
"""Target node counts of the knee sweep; ``knee_grids`` picks the admissible ones.

Six doublings show the knee of δ = 0.04 and 0.01 with rates on both sides
and the δ = 0.0025 knee itself (h/δ = 1 at 800, ½ at 1600); ``--counts``
extends to 3200 and 6400 (8 s per 6400-node BD4 march, cached afterwards).
"""

PLACEMENT = {"matlab": "cell", "eq75": "node"}
"""Where the edge centres sit relative to the nodes: E1's conventions.

Mid-cell for the MATLAB edge at 0 (even counts, four rebuilt rows), on nodes
for eq. 75's two edges (``4k + 1`` counts, which is the only common
placement they have; ``node_counts``).
"""

MIN_COUNT = {"matlab": 5, "eq75": 101}
"""Node counts the sweep does not go below.

Eq. 75 needs 101: with ``h α′/α ≈ 1`` where the sinusoid meets the layer's
edges (49 nodes) the translated basis is under-resolved and the jump-aware
operator's spectrum crosses into the right half-plane (largest real part
248 at 49 nodes, 8.8 at 53, −2.1 at 101 and −2.06 from 201 to 801, with no
eigenvalue in the right half-plane from 101 on), so E1 never ran it coarser
and neither does this sweep. Measured on the *interior* operator, the
Dirichlet rows removed (``interior_operator``, what BD4 steps and what port
notes §1.5 plot); the full matrix keeps two one-sided end rows in place of
the boundary condition and its spectrum says nothing about the march.
"""

OPERATORS = {
    "naive": naive_operator,
    "δ = 0 construction": jump_aware_operator,
    "seeds": seed_operator,
}
"""The three lines of the knee study.

``jump_aware_operator`` on a ``SmoothEdges`` medium is §1.7's δ = 0
construction: E1.2's rows across the edge centre with the pieces' Taylor
data, as if the edge were a jump, and the direct rows on the smooth alpha
everywhere else. ``seed_operator`` (E3.4) marches the seeds through the true
edge on every row within ``TANH_REACH`` δ of a centre.
"""

COLOURS = {"naive": NAIVE, "δ = 0 construction": CONSTRUCTION, "seeds": AWARE}
"""Orange, purple, blue (``plotting``); the floors are dotted in the construction's."""


def _treated(transform, **kwargs):
    """``naive_operator`` on the medium ``transform(grid, medium, **kwargs)`` builds."""

    def build(grid: Grid1D, medium: SmoothEdges):
        return naive_operator(grid, transform(grid, medium, **kwargs))

    return build


WIDENINGS = (1, 2)
"""T0's factors m: the edge widened to ``max(δ, m h)``."""

TREATMENTS = {
    "T1 harmonic 1c": _treated(harmonic_cells, cells=1),
    "T1 harmonic 2c": _treated(harmonic_cells, cells=2),
    "T2 arithmetic 1c": _treated(arithmetic_cells, cells=1),
    **{f"T0 widened m={m}": _treated(widened_edge, m=m) for m in WIDENINGS},
    "T1-FV": face_conductance_operator,
}
"""E3.5's coefficient treatments (plan §3.4; ``heat1d/treatments.py``).

Each is a changed medium sampled by the naive operator, except T1-FV, the
conservative three-point scheme with exact face conductances, which is an
operator of its own (stiff note §1.8). T3, the band-limited alpha, is
dropped (E5.2, #43, ``LITERATURE.md`` §1d).
"""

COMPARATORS = {"naive": naive_operator, **TREATMENTS, "seeds": seed_operator}
"""The comparator table's columns: P10's ranking with its two ends.

The naive and seed columns share their cache keys with ``OPERATORS``, so the
knee sweep's marches are not repeated.
"""

COMPARATOR_STYLE = {
    "naive": (NAIVE, "-"),
    "T1 harmonic 1c": ("#8c564b", "-"),
    "T1 harmonic 2c": ("#8c564b", "--"),
    "T2 arithmetic 1c": ("#d62728", "-"),
    "T0 widened m=1": ("#7f7f7f", "-"),
    "T0 widened m=2": ("#7f7f7f", "--"),
    "T1-FV": ("#2ca02c", "-"),
    "seeds": (AWARE, "-"),
}
"""Colour and line style per comparator; the knee's orange and blue keep their sense."""

SPECTRUM_COUNTS = {"matlab": (50, 100, 400), "eq75": (49, 53, 101, 401)}
"""Node counts of the spectrum check (P9); eq. 75's go below ``MIN_COUNT``."""

RESIDUAL_RATIOS = (1.0, 0.5, 0.1, 0.01, 0.001)
"""δ/h at which the δ = 0 rows are tested at fixed h (§1.4's list for the weights)."""

KNEE_CACHE = "heat1d_stiff_knee.json"
"""Where the parabolic knee errors are kept between runs.

``{"meta": KNEE_CACHE_META, "errors": {knee_key: error}}``. The header names
the problem (its label, ``T_END``, ``BC``, Radau's tolerances) and the code
the errors come from (``KNEE_SOURCES``' hash, E5.3), and a file whose header
differs is ignored and overwritten, as ``ParabolicReference`` does for the
references. The hash is of the bytes, so a docstring edit in those files
also empties the cache: a two-minute cold run, the price of not trusting a
label to notice a change to the marcher or to an operator (§2.2's caveat).
"""

KNEE_SOURCES = (
    *sorted((Path(heat_interfaces.__file__).parent / "heat1d").glob("*.py")),
    Path(heat_interfaces.__file__).parent / "fd_weights.py",
    Path(__file__).resolve(),
)
"""What the knee errors depend on: the 1-D package, the FD weights and this driver
(its operator tables, ``COMPARATORS`` and ``SNAPSHOT_OPERATORS``)."""

KNEE_CACHE_META = {
    "problem": PROBLEM,
    "t_end": T_END,
    "bc": list(BC),
    "rtol": RADAU_RTOL,
    "atol": RADAU_ATOL,
    "source": source_hash(KNEE_SOURCES),
}

MARKERS = ("o", "s", "^", "D", "v")

SEED_RATIOS = (0.5, 0.1)
"""δ/h of the seed-function figure (E3.6): the edge at half a cell, and at a tenth.

Half a cell is where the seeds differ most from both limits (P4's 47 % in
the weights, P5's condition-number peak); a tenth is where they are within
11 % of E1.2's translated basis. The window is P4's (``study_window`` at
``--residual-n`` nodes).
"""

SNAPSHOT_N, SNAPSHOT_DELTA = 100, 0.0025
"""The parabolic snapshot's grid: 100 nodes (h = 0.02) and δ = 0.0025, so h = 8δ.

The edge is unresolved by a factor eight and the naive operator sits on its
first-order line (2.0e-3 at this point of §2.2's table), the δ = 0
construction on its floor (7.1e-4, the resistance deficit ``c δ``), T1-FV
on its second-order line and the seeds on the δ = 0 jump-aware line
(3.0e-8): the four regimes of §2 in one picture.
"""

SNAPSHOT_OPERATORS = {
    "naive": naive_operator,
    "δ = 0 construction": jump_aware_operator,
    "T1-FV": face_conductance_operator,
    "seeds": seed_operator,
}
"""The snapshot's solutions: the knee's baselines, the best treatment, the seeds."""

SNAPSHOT_STYLE = {
    "naive": (NAIVE, "o"),
    "δ = 0 construction": (CONSTRUCTION, "s"),
    "T1-FV": (COMPARATOR_STYLE["T1-FV"][0], "^"),
    "seeds": (AWARE, "D"),
}
"""Colour and marker per snapshot solution; the comparators' colours kept."""

RESULTS = "heat1d_stiff.json"
"""The run's results file (``results_cache``) under ``--outputs`` and ``--data-dir``."""


def study_medium(name: str, delta: float) -> SmoothEdges:
    return SmoothEdges(MEDIA[name](), delta)


def reference_path(
    outputs: Path, name: str, delta: float, n_cheb: int, max_width: float
) -> Path:
    """The cache files' stem, ``heat1d_stiff_reference_matlab_d0.0025_n20_w0.1`` say."""
    stem = f"heat1d_stiff_reference_{name}_d{delta:g}_n{n_cheb}_w{max_width:g}"
    return outputs / stem


def study_reference(
    name: str,
    delta: float,
    outputs: Path,
    n_cheb: int = N_CHEB,
    max_width: float = MAX_WIDTH,
) -> ParabolicReference:
    """The cached parabolic reference of the ramp problem on ``name`` at ``delta``."""
    return parabolic_reference(
        study_medium(name, delta),
        np.zeros_like,
        ramp_boundary(RAMP, *BC),
        T_END,
        n_cheb=n_cheb,
        problem=PROBLEM,
        cache=reference_path(outputs, name, delta, n_cheb, max_width),
        max_width=max_width,
    )


def check_references(
    names: Sequence[str],
    deltas: Sequence[float],
    outputs: Path,
    resolution: tuple[int, float] = (N_CHEB, MAX_WIDTH),
    check: tuple[int, float] = (CHECK_N_CHEB, CHECK_MAX_WIDTH),
) -> list[dict]:
    """Build (or load) the references and both checks, one row per medium and δ.

    ``agreement`` is the max difference between the reference and the
    ``check`` resolution on 2001 points; ``elliptic`` the max difference
    between the Chebyshev equilibrium at the reference's resolution and the
    quadrature.
    """
    x = np.linspace(-1.0, 1.0, 2001)
    rows = []
    for name in names:
        for delta in deltas:
            ref = study_reference(name, delta, outputs, *resolution)
            finer = study_reference(name, delta, outputs, *check)
            medium = study_medium(name, delta)
            n_cheb, max_width = resolution
            cheb = chebyshev_equilibrium(medium, *BC, x, n_cheb, 0.0, max_width)
            quad = equilibrium_exact(medium, *BC, x, n_panels=QUAD_PANELS)
            rows.append(
                {
                    "medium": name,
                    "delta": delta,
                    "elements": ref.meta["elements"],
                    "unknowns": ref.meta["unknowns"],
                    "steps": ref.meta["steps"],
                    "seconds": ref.meta["seconds"],
                    "check_seconds": finer.meta["seconds"],
                    "reused": ref.reused,
                    "agreement": float(
                        np.max(np.abs(ref.evaluate(x) - finer.evaluate(x)))
                    ),
                    "elliptic": float(np.max(np.abs(cheb - quad))),
                }
            )
    return rows


def print_references(
    rows: list[dict], resolution: tuple[int, float], check: tuple[int, float]
) -> None:
    print(
        f"parabolic references: {PROBLEM}, t = {T_END:g}, Radau rtol {RADAU_RTOL:g},"
        f" atol {RADAU_ATOL:g}; {resolution[0]} Chebyshev nodes per element, elements"
        f" split above {resolution[1]:g}; checked against {check[0]} nodes and"
        f" {check[1]:g}"
    )
    print(
        f"  {'medium':8s}{'δ':>8s}{'elem':>6s}{'unkn':>6s}{'steps':>7s}{'seconds':>9s}"
        f"{'check s':>9s}{'agreement':>11s}{'ell. vs quad':>14s}  source"
    )
    for r in rows:
        print(
            f"  {r['medium']:8s}{r['delta']:8g}{r['elements']:6d}{r['unknowns']:6d}"
            f"{r['steps']:7d}{r['seconds']:9.2f}{r['check_seconds']:9.2f}"
            f"{r['agreement']:11.1e}{r['elliptic']:14.1e}"
            f"  {'cached' if r['reused'] else 'solved'}"
        )


# --- E3.3 (#28): the knee, and the δ = 0 construction on a smooth edge ----------


def knee_grids(name: str, counts: Sequence[int]) -> list[Grid1D]:
    """The sweep's grids: the admissible count nearest each target, in order.

    Admissible means every edge centre has the medium's ``PLACEMENT``;
    counts that resolve below ``MIN_COUNT[name]`` or repeat one are dropped.
    """
    interfaces = MEDIA[name]().interfaces
    grids: list[Grid1D] = []
    for n in counts:
        g = grid_for(interfaces, PLACEMENT[name], n)
        if g.n >= MIN_COUNT[name] and all(g.n != kept.n for kept in grids):
            grids.append(g)
    return grids


def elliptic_sweep(
    name: str, delta: float, grids: Sequence[Grid1D], operators=OPERATORS
) -> list[dict]:
    """Each operator on the equilibrium problem at ``delta``, one row per grid.

    Errors are ``‖e‖₂/‖u‖₂`` against the quadrature on the smooth medium.
    ``floor`` is ``‖u₀ − u_δ‖₂/‖u_δ‖₂`` at the nodes, the difference of the
    jump's and the smooth medium's exact equilibria: what the δ = 0
    construction converges to while the grid does not resolve the edge
    (§1.7), first order in δ. ``operators`` maps a column label to
    ``op(grid, medium)``: the knee's ``OPERATORS`` or the ``COMPARATORS``.
    """
    medium, jump = study_medium(name, delta), study_medium(name, 0.0)
    rows = []
    for g in grids:
        ref = equilibrium_exact(medium, *BC, g.x)
        row = {
            "n": g.n,
            "h": g.h,
            "floor": normalized_l2(equilibrium_exact(jump, *BC, g.x), ref),
        }
        for label, op in operators.items():
            row[label] = normalized_l2(solve_equilibrium(op(g, medium), *BC), ref)
        rows.append(row)
    return rows


def knee_key(name: str, delta: float, n: int, label: str, resolution) -> str:
    """The cache key of one parabolic error; the reference's resolution is in it."""
    return f"{name} d{delta:g} n{n} {label} ref{resolution[0]}/{resolution[1]:g}"


def parabolic_sweep(
    name: str,
    delta: float,
    grids: Sequence[Grid1D],
    outputs: Path,
    cache: dict[str, float],
    resolution: tuple[int, float] = (N_CHEB, MAX_WIDTH),
    operators=OPERATORS,
) -> list[dict]:
    """BD4 at ``dt = h`` on the ramp problem against the cached reference, per grid.

    Errors are ``‖e‖₂/‖u‖₂`` at ``t = T_END``. ``floor`` is the δ = 0
    reference against the δ one at the nodes. ``cache`` holds errors from
    earlier runs (``knee_key``); the ones missing are computed and added,
    the caller saves it. ``operators`` as in ``elliptic_sweep``; the cache
    is keyed by label, so a column shared by two maps is marched once.
    """
    medium = study_medium(name, delta)
    ref = study_reference(name, delta, outputs, *resolution)
    jump_ref = study_reference(name, 0.0, outputs, *resolution)
    boundary = ramp_boundary(RAMP, *BC)
    rows = []
    for g in grids:
        at_nodes = ref.evaluate(g.x)
        row = {
            "n": g.n,
            "h": g.h,
            "floor": normalized_l2(jump_ref.evaluate(g.x), at_nodes),
        }
        for label, op in operators.items():
            key = knee_key(name, delta, g.n, label, resolution)
            if key not in cache:
                u = bd4_march(op(g, medium), np.zeros(g.n), T_END, g.h, boundary)
                cache[key] = normalized_l2(u, at_nodes)
                row.setdefault("solved", []).append(label)
            row[label] = cache[key]
        rows.append(row)
    return rows


def load_knee_cache(outputs: Path) -> dict[str, float]:
    """The cached errors; empty when the file is missing or from another problem."""
    path = outputs / KNEE_CACHE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if data.get("meta") != KNEE_CACHE_META:
        return {}
    return dict(data["errors"])


def save_knee_cache(outputs: Path, cache: dict[str, float]) -> None:
    data = {"meta": KNEE_CACHE_META, "errors": dict(sorted(cache.items()))}
    (outputs / KNEE_CACHE).write_text(json.dumps(data, indent=1) + "\n")


def floor_constants(name: str, deltas: Sequence[float]) -> list[dict]:
    """``(F₀(1) − F_δ(1)) / δ`` at each δ > 0 against the closed form over the edges.

    ``F(1) = ∫_{−1}^{1} dξ/α`` is the total resistance; the smooth edges lower
    it by ``c δ`` per edge with ``c = edge_resistance_deficit`` of the two
    pieces' values at the centre (§1.7). Exact to rounding on constant pieces;
    a first-order approach on eq. 75, whose sinusoid varies across the edge.
    """
    jump = MEDIA[name]()
    f0 = float(inverse_alpha_integral(jump, np.array(X_MAX)))
    closed = sum(
        edge_resistance_deficit(
            float(jump.taylor(k, "left", 0)[0]), float(jump.taylor(k, "right", 0)[0])
        )
        for k in range(len(jump.interfaces))
    )
    rows = []
    for delta in deltas:
        if delta == 0.0:
            continue
        smooth = SmoothEdges(jump, delta)
        f_delta = float(inverse_alpha_integral(smooth, np.array(X_MAX)))
        rows.append(
            {
                "delta": delta,
                "measured": (f0 - f_delta) / delta,
                "closed": closed,
                "f0": f0,
            }
        )
    return rows


def row_residuals(
    name: str, n: int, ratios: Sequence[float] = RESIDUAL_RATIOS
) -> list[dict]:
    """The δ = 0 rows on the true-δ equilibrium at fixed h, for δ = ratio · h.

    ``residual`` is ``max |L_h (u_δ − u₀)|`` over the rows whose windows
    straddle an edge centre: the rows' response to what the smooth edge
    changes in the solution (on constant pieces ``L_h u₀`` is rounding there,
    E1.2's exactness; on eq. 75 it is the sinusoid's own third-order
    truncation, reported as ``on_jump`` so the two are not confused).
    ``scaled`` is ``h² residual / δ``, which tends to a constant as δ/h → 0:
    the rows are off at first order in δ/h, the residual twin of §1.4's
    weight comparison, which needs no seeds. ``naive`` is ``h · max |L_h
    u_δ|`` over the naive operator's interior rows, the O(1) scale of a row
    that misses the slope jump outright. ``seeds`` (E3.4) is ``h² max |L_h
    u_δ|`` over the seed operator's seeded rows on the true-δ equilibrium
    itself: rounding on constant pieces (P6), the sinusoid's own truncation
    on eq. 75, next to ``on_jump``.
    """
    (g,) = knee_grids(name, [n])
    u0 = equilibrium_exact(study_medium(name, 0.0), *BC, g.x)
    rows = []
    for ratio in ratios:
        delta = ratio * g.h
        medium = study_medium(name, delta)
        u = equilibrium_exact(medium, *BC, g.x)
        straddle = [i for i, _, _ in straddling_windows(g, medium)]
        aware = jump_aware_operator(g, medium)
        residual = float(np.max(np.abs((aware @ (u - u0))[straddle])))
        on_jump = float(np.max(np.abs((aware @ u0)[straddle])))
        naive = float(np.max(np.abs((naive_operator(g, medium) @ u)[2:-2])))
        seeded = [i for i, _, _ in seeded_windows(g, medium)]
        seeds = float(np.max(np.abs((seed_operator(g, medium) @ u)[seeded])))
        rows.append(
            {
                "n": g.n,
                "h": g.h,
                "ratio": ratio,
                "delta": delta,
                "residual": residual,
                "scaled": residual * g.h**2 / delta,
                "on_jump": on_jump * g.h**2,
                "naive": naive * g.h,
                "seeds": seeds * g.h**2,
            }
        )
    return rows


def widened_floors(
    name: str, delta: float, grids: Sequence[Grid1D], factors: Sequence[int] = WIDENINGS
) -> list[dict]:
    """T0's floor per grid: the widened medium's exact equilibrium against the true one.

    ``‖u_{max(δ, m h)} − u_δ‖₂/‖u_δ‖₂`` at the nodes for each factor ``m``,
    quadrature against quadrature: what T0 converges to if the naive
    operator resolves the widened edge, ``c (m h − δ)`` in resistance on
    constant pieces (§2.2's floor constant), first order in h.
    """
    medium = study_medium(name, delta)
    rows = []
    for g in grids:
        ref = equilibrium_exact(medium, *BC, g.x)
        row = {"n": g.n, "h": g.h}
        for m in factors:
            wide = widened_edge(g, medium, m)
            row[m] = normalized_l2(equilibrium_exact(wide, *BC, g.x), ref)
        rows.append(row)
    return rows


def study_window(name: str, n: int) -> tuple[Grid1D, np.ndarray, float, list[Jump]]:
    """The grid, the five nodes and the centre of P4's stencil, and E1.2's jumps.

    The window across the first edge centre: for the MATLAB medium (mid-cell)
    the row whose node is half a cell left of the centre, §1.4's layout; for
    eq. 75 (on nodes) the row on the centre itself, whose seeds carry the
    owner's ``alpha_e`` (§1.5).
    """
    (g,) = knee_grids(name, [n])
    jump = MEDIA[name]()
    x = g.snapped(jump.interfaces)
    i = int(np.searchsorted(x, jump.interfaces[0], side="right")) - 1
    jumps = [
        Jump(xi, jump.taylor(k, "left", 4), jump.taylor(k, "right", 4))
        for k, xi in enumerate(jump.interfaces)
    ]
    return g, x[i - 2 : i + 3], float(x[i]), jumps


def weights_vs_jump(
    name: str, n: int, ratios: Sequence[float] = (0.0, *RESIDUAL_RATIOS)
) -> list[dict]:
    """P4 and P5: the seed weights against E1.2's, and ``cond A``, per δ/h.

    ``difference`` is ``max |w_seed − w_jump| / max |w_jump|`` on the window
    of ``study_window``; first order in δ/h on constant pieces (§1.4's
    84 % … 0.11 %), and on eq. 75 saturating at the O(h alpha'/alpha) floor at
    which E1.2's degree-4-truncated smooth-piece operator and the exact chain
    part company. ``cond`` is the stencil solve's condition number in the
    stencil coordinate (23.5 at constant alpha for the centred window) and
    ``cond_scaled`` the same with every row of A scaled to unit max norm: the
    seeds carry a normalisation ``alpha_e^{⌈k/2⌉}`` (§1.2) that the weights
    never see, and on a centre node alpha_e is the owner's value at δ = 0
    and the blend's at δ > 0.
    """
    g, nodes, centre, jumps = study_window(name, n)
    jump = MEDIA[name]()
    seen = [k for k, xi in enumerate(jump.interfaces) if nodes[0] < xi < nodes[-1]]
    w_jump = stencil_weights([jumps[k] for k in seen], nodes, centre)
    rows = []
    for ratio in ratios:
        medium = SmoothEdges(jump, ratio * g.h)
        w = seed_weights(nodes, centre, medium)
        a, _ = seed_basis(nodes, centre, medium)
        rows.append(
            {
                "n": g.n,
                "h": g.h,
                "ratio": ratio,
                "difference": float(
                    np.max(np.abs(w - w_jump)) / np.max(np.abs(w_jump))
                ),
                "cond": float(np.linalg.cond(a)),
                "cond_scaled": float(
                    np.linalg.cond(a / np.max(np.abs(a), axis=1, keepdims=True))
                ),
            }
        )
    return rows


def seed_spectra(
    name: str, counts: Sequence[int], deltas: Sequence[float]
) -> list[dict]:
    """P9: the seed operator's interior spectrum per count and δ.

    ``max_real`` and the largest ``|Im λ| / |λ|``, the extreme ``min Re λ · h²``
    (−16/3 for FD4's second derivative in the α = 1 material), and BD4's
    largest amplification at ``dt = h``. The grids keep the medium's
    placement but not ``MIN_COUNT``, so eq. 75's 49 and 53 nodes are
    included: the counts at which the δ = 0 construction is unstable.
    """
    interfaces = MEDIA[name]().interfaces
    rows = []
    for n in counts:
        g = grid_for(interfaces, PLACEMENT[name], n)
        for delta in deltas:
            op = seed_operator(g, study_medium(name, delta))
            lam = np.linalg.eigvals(interior_operator(op).toarray())
            rows.append(
                {
                    "n": g.n,
                    "delta": delta,
                    "max_real": float(np.max(lam.real)),
                    "imag": float(np.max(np.abs(lam.imag)) / np.max(np.abs(lam))),
                    "extreme": float(np.min(lam.real) * g.h**2),
                    "bd4": float(np.max(bd4_amplification(g.h * lam))),
                }
            )
    return rows


def _rates(values: Sequence[float]) -> list[str]:
    out = ["     -"]
    for a, b in zip(values[:-1], values[1:], strict=True):
        out.append(f"{np.log2(a / b):6.2f}" if a > 0 and b > 0 else "     -")
    return out


def print_sweep(
    title: str, results: dict[float, list[dict]], cached: bool = False
) -> None:
    """One table per δ; with ``cached`` a source column says which rows were rerun."""
    labels = _labels(results)
    print(f"\n{title}")
    for delta, rows in results.items():
        print(f"  δ = {delta:g}")
        head = f"    {'n':>6s}{'h/δ':>9s}"
        for label in labels:
            head += f"{label:>20s}{'rate':>7s}"
        print(head + f"{'floor':>11s}" + ("  source" if cached else ""))
        rates = {label: _rates([r[label] for r in rows]) for label in labels}
        for i, r in enumerate(rows):
            line = f"    {r['n']:6d}"
            line += f"{r['h'] / delta:9.3f}" if delta else f"{'∞':>9s}"
            for label in labels:
                line += f"{r[label]:20.3e}{rates[label][i]:>7s}"
            line += f"{r['floor']:11.3e}" if delta else f"{'-':>11s}"
            if cached:
                line += "  solved" if r.get("solved") else "  cached"
            print(line)


def _labels(results: dict[float, list[dict]]) -> list[str]:
    """The operator columns of a sweep's rows, in order."""
    rows = next(iter(results.values()))
    return [k for k in rows[0] if k not in ("n", "h", "floor", "solved")]


def print_comparators(
    title: str, results: dict[float, list[dict]], cached: bool = False
) -> None:
    """The comparator table per δ: every column as ``error (rate)``, compactly."""
    labels = _labels(results)
    print(f"\n{title}")
    print("  columns: " + ", ".join(labels))
    for delta, rows in results.items():
        print(f"  δ = {delta:g}")
        head = f"    {'n':>6s}{'h/δ':>8s}"
        for label in labels:
            short = label.replace(" harmonic", "").replace(" arithmetic", "")
            head += f"{short.replace(' widened', ''):>17s}"
        print(head + f"{'floor':>10s}" + ("  source" if cached else ""))
        rates = {label: _rates([r[label] for r in rows]) for label in labels}
        for i, r in enumerate(rows):
            line = f"    {r['n']:6d}"
            line += f"{r['h'] / delta:8.2f}" if delta else f"{'∞':>8s}"
            for label in labels:
                rate = rates[label][i].strip()
                line += f"{r[label]:10.2e} ({rate:>4s})"
            line += f"{r['floor']:10.2e}" if delta else f"{'-':>10s}"
            if cached:
                line += "  solved" if r.get("solved") else "  cached"
            print(line)


def print_widened_floors(
    name: str, delta: float, elliptic: list[dict], floors: list[dict]
) -> None:
    """T0's elliptic error next to its own floor, and the ratio, per factor m."""
    print(
        f"\nT0 against its floor, {name}, δ = {delta:g}: the naive operator on the"
        f" edge widened to max(δ, m h), and ‖u_{{max(δ, m h)}} − u_δ‖/‖u_δ‖ (quadrature"
        f" against quadrature); ratio → 1 once the naive operator sees the widened"
        f" edge fully, 'naive' where m h ≤ δ leaves the medium as it is"
    )
    head = f"    {'n':>6s}{'h/δ':>8s}"
    for m in WIDENINGS:
        head += f"{f'T0 m={m}':>11s}{'floor':>10s}{'ratio':>7s}"
    print(head)
    for e, f in zip(elliptic, floors, strict=True):
        assert e["n"] == f["n"]
        line = f"    {e['n']:6d}{e['h'] / delta:8.2f}"
        for m in WIDENINGS:
            error = e[f"T0 widened m={m}"]
            # A zero floor is the medium itself (m h ≤ δ, up to an ulp of h):
            # T0 is then the naive operator.
            ratio = f"{error / f[m]:7.3f}" if f[m] > 1e-13 else f"{'naive':>7s}"
            line += f"{error:11.2e}{f[m]:10.2e}{ratio}"
        print(line)


def print_floors(name: str, rows: list[dict]) -> None:
    print(
        f"\nfloor constant, {name}: (F₀(1) − F_δ(1)) / δ against the closed form"
        f" Σ_edges (a − b) ln(a/b) / (2ab) = {rows[0]['closed']:.6f}"
        f"  [F₀(1) = {rows[0]['f0']:.4f}]"
    )
    for r in rows:
        print(
            f"    δ = {r['delta']:<8g} measured {r['measured']:.6f}"
            f"   ratio to closed form {r['measured'] / r['closed']:.4f}"
        )


def print_residuals(name: str, rows: list[dict]) -> None:
    r0 = rows[0]
    print(
        f"\nδ = 0 rows on the true-δ equilibrium, {name}, {r0['n']} nodes"
        f" (h = {r0['h']:.4g}), δ = (δ/h) h: max |L_h (u_δ − u₀)| on the straddling"
        f" rows, and h² of it over δ (→ constant: first order in δ/h);"
        f" h² |L_h u₀| there (rounding on constants); the naive rows' h |L_h u_δ|;"
        f" the seed rows' h² |L_h u_δ| (rounding on constants)"
    )
    print(
        f"    {'δ/h':>8s}{'residual':>12s}{'h² res / δ':>13s}{'h² |L_h u₀|':>14s}"
        f"{'naive h |L_h u_δ|':>19s}{'seeds h² |L_h u_δ|':>20s}"
    )
    for r in rows:
        print(
            f"    {r['ratio']:8g}{r['residual']:12.3e}{r['scaled']:13.4f}"
            f"{r['on_jump']:14.2e}{r['naive']:19.3e}{r['seeds']:20.2e}"
        )


def print_weights(name: str, rows: list[dict]) -> None:
    r0 = rows[0]
    print(
        f"\nseed weights against E1.2's, {name}, {r0['n']} nodes (h = {r0['h']:.4g}),"
        f" the window across the first edge: max |w_seed − w_jump| / max |w_jump|,"
        f" and cond A of the stencil solve in the stencil coordinate"
    )
    print(f"    {'δ/h':>8s}{'difference':>12s}{'cond A':>9s}{'rows scaled':>13s}")
    for r in rows:
        print(
            f"    {r['ratio']:8g}{r['difference']:12.2e}{r['cond']:9.1f}"
            f"{r['cond_scaled']:13.1f}"
        )


def print_spectra(name: str, rows: list[dict]) -> None:
    print(
        f"\nseed operator's interior spectrum, {name}: largest real part, largest"
        f" |Im λ| / max |λ|, min Re λ · h² (FD4's −16/3 in the α = 1 material),"
        f" BD4's largest amplification at dt = h"
    )
    print(
        f"    {'n':>6s}{'δ':>8s}{'max Re λ':>11s}{'|Im|/|λ|':>10s}"
        f"{'min Re λ h²':>13s}{'BD4':>8s}"
    )
    for r in rows:
        print(
            f"    {r['n']:6d}{r['delta']:8g}{r['max_real']:11.3g}{r['imag']:10.1e}"
            f"{r['extreme']:13.4f}{r['bd4']:8.4f}"
        )


def plot_knee(
    elliptic: dict[str, dict[float, list[dict]]],
    parabolic: dict[str, dict[float, list[dict]]],
    path: Path,
) -> None:
    """Error against node count, one row of panels per medium: equilibrium, ramp.

    ``COLOURS``: orange naive, purple the δ = 0 construction, blue the seeds;
    the δ = 0 lines are thin and unmarked (E1's lines), each δ > 0 has its
    marker, dotted purple is the floor of that δ, and the dashed verticals
    mark ``h = δ``.
    """
    names = list(elliptic)
    fig, axes = plt.subplots(
        len(names), 2, figsize=(9.6, 3.8 * len(names)), squeeze=False, sharey="row"
    )
    titles = {"matlab": "MATLAB medium $1/9\\,|\\,1$", "eq75": "eq. 75 medium"}
    for i, name in enumerate(names):
        panels = (
            ("equilibrium", elliptic[name]),
            (f"ramp problem, $t = {T_END:g}$, BD4 $dt = h$", parabolic[name]),
        )
        for ax, (problem, results) in zip(axes[i], panels, strict=True):
            positive = sorted(d for d in results if d > 0)
            for delta, rows in results.items():
                n = np.array([r["n"] for r in rows], dtype=float)
                if delta == 0.0:
                    style = dict(lw=0.8, alpha=0.7)
                else:
                    style = dict(
                        marker=MARKERS[positive.index(delta) % len(MARKERS)],
                        ms=4,
                        lw=1.0,
                    )
                    floor = [r["floor"] for r in rows]
                    ax.loglog(n, floor, ":", color=CONSTRUCTION, lw=0.8, alpha=0.6)
                    ax.axvline(
                        (X_MAX - X_MIN) / delta + 1, color=REFERENCE, lw=0.6, ls="--"
                    )
                for label, colour in COLOURS.items():
                    ax.loglog(n, [r[label] for r in rows], color=colour, **style)
            ax.set_title(f"{titles.get(name, name)}, {problem}", fontsize=10)
            ax.set_xlabel("nodes")
            ax.grid(True, which="both", alpha=0.3)
        axes[i, 0].set_ylabel("$\\|e\\|_2 / \\|u\\|_2$")
    handles = [
        Line2D([], [], color=NAIVE, label="naive $D_x A D_x$"),
        Line2D([], [], color=CONSTRUCTION, label="δ = 0 construction"),
        Line2D([], [], color=AWARE, label="seeds"),
        Line2D(
            [],
            [],
            color=CONSTRUCTION,
            ls=":",
            label="floor $\\|u_0 - u_\\delta\\| / \\|u_\\delta\\|$",
        ),
        Line2D([], [], color=REFERENCE, ls="--", lw=0.6, label="$h = \\delta$"),
    ]
    positive = sorted(d for d in next(iter(elliptic.values())) if d > 0)
    for k, delta in enumerate(positive):
        handles.append(
            Line2D(
                [],
                [],
                color="k",
                marker=MARKERS[k % len(MARKERS)],
                ls="",
                ms=4,
                label=f"δ = {delta:g}",
            )
        )
    handles.append(Line2D([], [], color="k", lw=0.8, alpha=0.7, label="δ = 0 (jump)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparators(
    elliptic: dict[str, dict[float, list[dict]]],
    parabolic: dict[str, dict[float, list[dict]]],
    path: Path,
) -> None:
    """The comparators against node count: a row per medium and problem, a column per δ.

    ``COMPARATOR_STYLE`` colours; the dashed verticals mark ``h = δ``.
    """
    names = list(elliptic)
    deltas = sorted(d for d in next(iter(elliptic.values())) if d > 0)
    panels = []
    for name in names:
        panels.append((name, "equilibrium", elliptic[name]))
        panels.append((name, f"ramp, $t = {T_END:g}$", parabolic[name]))
    fig, axes = plt.subplots(
        len(panels),
        len(deltas),
        figsize=(3.3 * len(deltas), 2.9 * len(panels)),
        squeeze=False,
        sharey="row",
    )
    titles = {"matlab": "MATLAB $1/9\\,|\\,1$", "eq75": "eq. 75"}
    for i, (name, problem, results) in enumerate(panels):
        labels = _labels(results)
        for j, delta in enumerate(deltas):
            ax = axes[i, j]
            rows = results[delta]
            n = np.array([r["n"] for r in rows], dtype=float)
            for label in labels:
                colour, ls = COMPARATOR_STYLE.get(label, ("k", ":"))
                ax.loglog(n, [r[label] for r in rows], color=colour, ls=ls, lw=1.0)
            ax.axvline((X_MAX - X_MIN) / delta + 1, color=REFERENCE, lw=0.6, ls="--")
            ax.set_title(
                f"{titles.get(name, name)}, {problem}, δ = {delta:g}", fontsize=8
            )
            ax.grid(True, which="both", alpha=0.3)
            if i == len(panels) - 1:
                ax.set_xlabel("nodes")
        axes[i, 0].set_ylabel("$\\|e\\|_2 / \\|u\\|_2$")
    handles = [
        Line2D([], [], color=colour, ls=ls, label=label)
        for label, (colour, ls) in COMPARATOR_STYLE.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def seed_functions(
    name: str,
    n: int,
    ratios: Sequence[float] = SEED_RATIOS,
    points: int = 201,
) -> dict:
    """The seeds across P4's window against the monomials and the translated basis.

    Everything in stencil units: ``xi = (x - x_e) / h_s`` on ``[-1, 1]``
    (``points`` of them plus the nodes), in which the march works and
    constant α gives ``xi^k`` exactly. ``jump`` is E1.2's translated basis
    (``translated_basis`` at δ = 0) re-centred at the evaluation point with
    ``shift_matrix`` and scaled by ``h_s^k``, which is what the seeds'
    initial conditions ``phi_k(x_e) = phi_k'(x_e) = 0`` pick out of its
    span. On a jump between constant pieces (the MATLAB medium) the two
    agree to rounding at δ = 0 (``rows[0]``), function by function, the
    check P4 made on the weights only; on eq. 75 they differ by the
    ``O(h alpha'/alpha)`` gap between E1.2's degree-4-truncated pieces and
    the exact chain (0.19–0.40 at h = 0.01, §2.3's P4), which no δ closes,
    so the eq. 75 seeds are read against the reference solution, never
    against E1.2 (§2.5). Seed ``k`` is
    multiplied by ``(alpha_0 / alpha_e)^ceil(k/2)``, ``alpha_0`` the δ = 0
    medium's value at the evaluation point and ``alpha_e`` the blend's, so
    the lines are comparable across δ: the seeds carry the normalisation
    ``alpha_e^ceil(k/2)`` (§1.2), each moment condition is homogeneous in
    it, and the weights never see it. ``rows`` holds, per δ/h, the sup
    distance of each seed from the translated basis and from the monomials.
    """
    g, nodes, centre, jumps = study_window(name, n)
    jump_medium = MEDIA[name]()
    h_s = float(np.max(np.abs(nodes - centre)))
    xi_nodes = (nodes - centre) / h_s
    xi = np.union1d(np.linspace(-1.0, 1.0, points), xi_nodes)
    x = centre + h_s * xi
    count = nodes.size
    k = np.arange(count)
    scale = h_s ** k[:, None]
    monomials = xi[None, :] ** k[:, None]
    anchor = int(region_index(jumps, centre)[0])
    regions = translated_basis(jumps, anchor)
    recentre = shift_matrix(jumps[regions[anchor].jump].position - centre, count - 1)
    which = region_index(jumps, x)
    jump = np.empty((count, xi.size))
    for r, region in enumerate(regions):
        mask = which == r
        if mask.any():
            jump[:, mask] = (region.evaluate(jumps, x[mask]) @ recentre).T
    jump /= scale
    alpha_0 = float(jump_medium.alpha(np.array([centre]))[0])
    seeds, alpha, rows = {}, {}, []
    for ratio in (0.0, *ratios):
        medium = SmoothEdges(jump_medium, ratio * g.h)
        alpha_e = float(medium.alpha(np.array([centre]))[0])
        phi = seed_profiles([centre], [h_s], xi, medium, count)[0]
        phi *= (alpha_0 / alpha_e) ** np.ceil(k / 2)[:, None]
        seeds[ratio] = phi
        alpha[ratio] = medium.alpha(x)
        rows.append(
            {
                "ratio": ratio,
                "vs_jump": np.max(np.abs(phi - jump), axis=1)[1:].tolist(),
                "vs_monomials": np.max(np.abs(phi - monomials), axis=1)[1:].tolist(),
            }
        )
    return {
        "medium": name,
        "n": g.n,
        "h": g.h,
        "h_s": h_s,
        "xi": xi,
        "xi_nodes": xi_nodes,
        "xi_edges": [(xc - centre) / h_s for xc in jump_medium.interfaces],
        "monomials": monomials,
        "jump": jump,
        "seeds": seeds,
        "alpha": alpha,
        "rows": rows,
    }


def seed_curves(data: dict) -> dict:
    """``seed_functions``' arrays to six figures: what the manuscript's figure draws.

    ``seeds`` and ``alpha`` are keyed by δ/h (0 the δ = 0 medium), each seed
    array ``[k][point]`` on ``xi`` as ``monomials`` and ``jump`` are (E5.3).
    """
    return {
        "xi": rounded(data["xi"]),
        "xi_nodes": rounded(data["xi_nodes"]),
        "xi_edges": rounded(data["xi_edges"]),
        "monomials": rounded(data["monomials"]),
        "jump": rounded(data["jump"]),
        "seeds": {r: rounded(phi) for r, phi in data["seeds"].items()},
        "alpha": {r: rounded(a) for r, a in data["alpha"].items()},
    }


def print_seed_functions(data: dict) -> None:
    print(
        f"\nseed functions on P4's window, {data['medium']}, {data['n']} nodes"
        f" (h = {data['h']:.4g}, h_s = {data['h_s']:.4g}), in stencil units:"
        f" max |φ_k^seed − φ_k^jump| against E1.2's translated basis re-centred"
        f" at the node, and max |φ_k^seed − ξ^k| against the monomials, k = 1 … 4"
    )
    print(
        f"    {'δ/h':>6s}"
        + "".join(f"{'jump k=' + str(k):>12s}" for k in range(1, 5))
        + "".join(f"{'mono k=' + str(k):>12s}" for k in range(1, 5))
    )
    for r in data["rows"]:
        print(
            f"    {r['ratio']:6g}"
            + "".join(f"{v:12.2e}" for v in r["vs_jump"])
            + "".join(f"{v:12.2e}" for v in r["vs_monomials"])
        )


def plot_seeds(data: dict, path: Path) -> None:
    """α across the window on top; the seeds k = 1 … 4 below, three lines each.

    Grey dashed the monomial ``xi^k``, purple E1.2's translated basis (the
    δ = 0 limit), blue the seeds at each δ/h of ``SEED_RATIOS`` (solid the
    first, dashed the rest), the nodes marked on the first; the dotted
    verticals are the edge centres.
    """
    xi, ratios = data["xi"], [r for r in data["seeds"] if r > 0]
    fig = plt.figure(figsize=(8.0, 7.2))
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 2.2, 2.2))
    top = fig.add_subplot(grid[0, :])
    styles = ["-", "--", "-.", ":"]
    for j, ratio in enumerate(ratios):
        top.plot(
            xi,
            data["alpha"][ratio],
            color=AWARE,
            ls=styles[j % 4],
            lw=1.0,
            label=f"δ/h = {ratio:g}",
        )
    top.plot(xi, data["alpha"][0.0], color=CONSTRUCTION, lw=0.8, label="δ = 0")
    top.set_ylabel("α")
    top.set_title(
        f"P4's window on the {data['medium']} medium, {data['n']} nodes"
        f" (h = {data['h']:.3g}), ξ = (x − x_e) / h_s",
        fontsize=10,
    )
    top.legend(fontsize=8, loc="upper left", ncol=3)
    for k in range(1, 5):
        ax = fig.add_subplot(grid[1 + (k - 1) // 2, (k - 1) % 2])
        ax.plot(xi, data["monomials"][k], color=REFERENCE, ls="--", lw=0.9)
        ax.plot(xi, data["jump"][k], color=CONSTRUCTION, lw=1.0)
        for j, ratio in enumerate(ratios):
            ax.plot(xi, data["seeds"][ratio][k], color=AWARE, ls=styles[j % 4], lw=1.1)
        node_values = np.interp(data["xi_nodes"], xi, data["seeds"][ratios[0]][k])
        ax.plot(data["xi_nodes"], node_values, "o", color=AWARE, ms=4)
        for xc in data["xi_edges"]:
            if -1 <= xc <= 1:
                ax.axvline(xc, color=REFERENCE, lw=0.6, ls=":")
        ax.set_title(f"$\\varphi_{k}$", fontsize=10)
        ax.grid(True, alpha=0.3)
        if k >= 3:
            ax.set_xlabel("ξ")
    handles = [
        Line2D(
            [], [], color=REFERENCE, ls="--", label="monomial $\\xi^k$ (constant α)"
        ),
        Line2D([], [], color=CONSTRUCTION, label="translated basis (δ = 0, E1.2)"),
        *[
            Line2D([], [], color=AWARE, ls=styles[j % 4], label=f"seeds, δ/h = {r:g}")
            for j, r in enumerate(ratios)
        ],
        Line2D([], [], color=AWARE, marker="o", ls="", ms=4, label="nodes"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def snapshot(
    name: str,
    delta: float,
    n: int,
    outputs: Path,
    resolution: tuple[int, float] = (N_CHEB, MAX_WIDTH),
    operators=SNAPSHOT_OPERATORS,
    points: int = 2001,
) -> dict:
    """The ramp solution at ``T_END`` on one coarse grid: reference, solutions, errors.

    ``rows`` per operator: ``‖e‖₂/‖u‖₂`` (the sweep's number for this grid),
    ``max`` the largest nodal error and ``at`` its node, and ``local`` the
    share of ``‖e‖₂²`` on the nodes within ``2h`` of an edge centre: near 1
    would say the error is confined to the edge, near 0 that a wrong
    effective resistance has shifted the whole profile.
    """
    (g,) = knee_grids(name, [n])
    medium = study_medium(name, delta)
    ref = study_reference(name, delta, outputs, *resolution)
    boundary = ramp_boundary(RAMP, *BC)
    x_fine = np.linspace(X_MIN, X_MAX, points)
    at_nodes = ref.evaluate(g.x)
    centres = np.asarray(medium.interfaces, dtype=float)
    near = np.min(np.abs(g.x[:, None] - centres[None, :]), axis=1) <= 2 * g.h
    solutions, rows = {}, []
    for label, op in operators.items():
        u = bd4_march(op(g, medium), np.zeros(g.n), T_END, g.h, boundary)
        e = u - at_nodes
        solutions[label] = u
        rows.append(
            {
                "label": label,
                "error": normalized_l2(u, at_nodes),
                "max": float(np.max(np.abs(e))),
                "at": float(g.x[int(np.argmax(np.abs(e)))]),
                "local": float(np.sum(e[near] ** 2) / np.sum(e**2)),
            }
        )
    return {
        "medium": name,
        "delta": delta,
        "n": g.n,
        "h": g.h,
        "x": g.x,
        "u": at_nodes,
        "x_fine": x_fine,
        "u_fine": ref.evaluate(x_fine),
        "alpha_fine": medium.alpha(x_fine),
        "centres": centres,
        "solutions": solutions,
        "rows": rows,
    }


def snapshot_curves(data: dict) -> dict:
    """``snapshot``'s arrays to six figures: the manuscript's figure's (E5.3)."""
    return {
        "x": rounded(data["x"]),
        "u": rounded(data["u"]),
        "x_fine": rounded(data["x_fine"]),
        "u_fine": rounded(data["u_fine"]),
        "alpha_fine": rounded(data["alpha_fine"]),
        "centres": rounded(data["centres"]),
        "solutions": {k: rounded(u) for k, u in data["solutions"].items()},
    }


def print_snapshot(data: dict) -> None:
    print(
        f"\nsnapshot: ramp problem at t = {T_END:g}, {data['medium']}, δ ="
        f" {data['delta']:g}, {data['n']} nodes (h = {data['h']:.4g}, h/δ ="
        f" {data['h'] / data['delta']:.3g}): ‖e‖₂/‖u‖₂, max |e| and its node, and"
        f" the share of ‖e‖₂² within 2h of an edge centre"
    )
    print(
        f"    {'operator':>20s}{'‖e‖₂/‖u‖₂':>12s}{'max |e|':>11s}"
        f"{'at x':>9s}{'local':>8s}"
    )
    for r in data["rows"]:
        print(
            f"    {r['label']:>20s}{r['error']:12.3e}{r['max']:11.3e}{r['at']:9.3f}"
            f"{r['local']:8.3f}"
        )


def plot_snapshot(data: dict, path: Path) -> None:
    """Left the solution (reference, nodal solutions, a zoom at the edge); right the
    interior nodes' errors on a log scale; the dashed verticals are the edge centres."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    left.plot(
        data["x_fine"], data["u_fine"], color=REFERENCE, lw=1.0, label="reference"
    )
    for label, (colour, marker) in SNAPSHOT_STYLE.items():
        if label in data["solutions"]:
            left.plot(
                data["x"],
                data["solutions"][label],
                marker,
                color=colour,
                ms=3,
                mfc="none",
                lw=0,
                label=label,
            )
    for xc in data["centres"]:
        left.axvline(xc, color=REFERENCE, lw=0.6, ls="--")
        right.axvline(xc, color=REFERENCE, lw=0.6, ls="--")
    left.set_xlabel("x")
    left.set_ylabel("u")
    left.set_title(
        f"ramp problem at $t = {T_END:g}$, {data['n']} nodes, δ = {data['delta']:g}"
        f" (h = {data['h'] / data['delta']:.3g} δ)",
        fontsize=10,
    )
    left.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.98, 0.5))
    xc = float(data["centres"][0])
    span = 4 * data["h"]
    inset = left.inset_axes((0.56, 0.52, 0.4, 0.38))
    window = np.abs(data["x_fine"] - xc) <= span
    inset.plot(data["x_fine"][window], data["u_fine"][window], color=REFERENCE, lw=1.0)
    nodes = np.abs(data["x"] - xc) <= span
    for label, (colour, marker) in SNAPSHOT_STYLE.items():
        if label in data["solutions"]:
            inset.plot(
                data["x"][nodes],
                data["solutions"][label][nodes],
                marker,
                color=colour,
                ms=3,
                mfc="none",
                lw=0,
            )
    inset.axvline(xc, color=REFERENCE, lw=0.6, ls="--")
    inset.set_title("the edge, ±4h", fontsize=7)
    inset.tick_params(labelsize=6)
    interior = slice(1, -1)  # the Dirichlet ends are exact by construction
    for label, (colour, _marker) in SNAPSHOT_STYLE.items():
        if label in data["solutions"]:
            e = np.abs(data["solutions"][label] - data["u"])[interior]
            right.semilogy(data["x"][interior], e, color=colour, lw=0.9, label=label)
    right.set_xlabel("x")
    right.set_ylabel("$|u_h - u|$ at the interior nodes")
    right.set_title("pointwise error", fontsize=10)
    right.set_ylim(bottom=1e-12)
    right.grid(True, which="both", alpha=0.3)
    right.legend(fontsize=8, loc="lower center", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deltas", type=float, nargs="+", default=list(STUDY_DELTAS))
    parser.add_argument("--media", nargs="+", default=list(MEDIA), choices=list(MEDIA))
    parser.add_argument("--n-cheb", type=int, default=N_CHEB)
    parser.add_argument("--max-width", type=float, default=MAX_WIDTH)
    parser.add_argument("--check-n-cheb", type=int, default=CHECK_N_CHEB)
    parser.add_argument("--check-max-width", type=float, default=CHECK_MAX_WIDTH)
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=list(KNEE_COUNTS),
        help="target node counts of the knee sweep (placement per medium)",
    )
    parser.add_argument(
        "--residual-n",
        type=int,
        default=200,
        help="target node count of the δ/h residual table",
    )
    parser.add_argument("--snapshot-n", type=int, default=SNAPSHOT_N)
    parser.add_argument("--snapshot-delta", type=float, default=SNAPSHOT_DELTA)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"also write the results file {RESULTS} here (paper/data for E5.3)",
    )
    args = parser.parse_args(argv)
    args.outputs.mkdir(parents=True, exist_ok=True)
    results = ResultsCache("heat1d_stiff", vars(args), argv)

    t0 = time.perf_counter()
    resolution = (args.n_cheb, args.max_width)
    check = (args.check_n_cheb, args.check_max_width)
    rows = check_references(args.media, args.deltas, args.outputs, resolution, check)
    print_references(rows, resolution, check)
    worst = max(r["agreement"] for r in rows)
    results.add("references", rows)
    results.time("references", time.perf_counter() - t0)
    print(
        f"\nworst agreement between resolutions {worst:.1e}"
        f" ({'within' if worst < 1e-10 else 'OUTSIDE'} the 1e-10 of #27);"
        f" {time.perf_counter() - t0:.1f} s; references in {args.outputs}/"
    )

    t1 = time.perf_counter()
    cache = load_knee_cache(args.outputs)
    elliptic: dict[str, dict[float, list[dict]]] = {}
    parabolic: dict[str, dict[float, list[dict]]] = {}
    compared_e: dict[str, dict[float, list[dict]]] = {}
    compared_p: dict[str, dict[float, list[dict]]] = {}
    for name in args.media:
        grids = knee_grids(name, args.counts)
        elliptic[name] = {d: elliptic_sweep(name, d, grids) for d in args.deltas}
        print_sweep(
            f"equilibrium problem, {name} (edges {PLACEMENT[name]}), ‖e‖₂/‖u‖₂"
            f" against the quadrature at each δ",
            elliptic[name],
        )
        parabolic[name] = {
            d: parabolic_sweep(name, d, grids, args.outputs, cache, resolution)
            for d in args.deltas
        }
        save_knee_cache(args.outputs, cache)
        print_sweep(
            f"ramp problem at t = {T_END:g}, BD4 dt = h, {name} (edges"
            f" {PLACEMENT[name]}), ‖e‖₂/‖u‖₂ against the Chebyshev reference at each δ",
            parabolic[name],
            cached=True,
        )
        floors = floor_constants(name, args.deltas)
        if floors:
            print_floors(name, floors)
            results.add(f"floor_constants/{name}", floors)
        residuals = row_residuals(name, args.residual_n)
        print_residuals(name, residuals)
        results.add(f"row_residuals/{name}", residuals)
        weights = weights_vs_jump(name, args.residual_n)
        print_weights(name, weights)
        results.add(f"weights_vs_jump/{name}", weights)
        deltas = sorted({0.0, *(d for d in args.deltas if d > 0)}, reverse=True)[-2:]
        spectra = seed_spectra(name, SPECTRUM_COUNTS[name], deltas)
        print_spectra(name, spectra)
        results.add(f"seed_spectra/{name}", spectra)
    plot_knee(elliptic, parabolic, args.outputs / "heat1d_stiff_knee.png")
    results.add("knee/equilibrium", elliptic)
    results.add("knee/ramp", parabolic)
    results.time("knee", time.perf_counter() - t1)
    print(
        f"\nknee study {time.perf_counter() - t1:.1f} s; figure and"
        f" {KNEE_CACHE} in {args.outputs}/"
    )

    t2 = time.perf_counter()
    for name in args.media:
        grids = knee_grids(name, args.counts)
        compared_e[name] = {
            d: elliptic_sweep(name, d, grids, COMPARATORS) for d in args.deltas
        }
        print_comparators(
            f"comparators (E3.5), equilibrium problem, {name}: ‖e‖₂/‖u‖₂ (rate)"
            f" against the quadrature at each δ",
            compared_e[name],
        )
        compared_p[name] = {
            d: parabolic_sweep(
                name, d, grids, args.outputs, cache, resolution, COMPARATORS
            )
            for d in args.deltas
        }
        save_knee_cache(args.outputs, cache)
        print_comparators(
            f"comparators (E3.5), ramp problem at t = {T_END:g}, BD4 dt = h, {name}:"
            f" ‖e‖₂/‖u‖₂ (rate) against the Chebyshev reference at each δ",
            compared_p[name],
            cached=True,
        )
        for d in args.deltas:
            if d > 0:
                floors = widened_floors(name, d, grids)
                print_widened_floors(name, d, compared_e[name][d], floors)
                results.add(f"widened_floors/{name}/{d:g}", floors)
    plot_comparators(
        compared_e, compared_p, args.outputs / "heat1d_stiff_treatments.png"
    )
    results.add("comparators/equilibrium", compared_e)
    results.add("comparators/ramp", compared_p)
    results.time("comparators", time.perf_counter() - t2)
    print(
        f"\ncomparators {time.perf_counter() - t2:.1f} s; figure"
        f" heat1d_stiff_treatments.png in {args.outputs}/"
    )

    t3 = time.perf_counter()
    name = args.media[0]
    functions = seed_functions(name, args.residual_n)
    print_seed_functions(functions)
    plot_seeds(functions, args.outputs / "heat1d_stiff_seeds.png")
    results.add(
        "seed_functions",
        {k: functions[k] for k in ("medium", "n", "h", "h_s", "rows")},
    )
    results.add("seed_functions/curves", seed_curves(functions))
    picture = snapshot(
        name, args.snapshot_delta, args.snapshot_n, args.outputs, resolution
    )
    print_snapshot(picture)
    plot_snapshot(picture, args.outputs / "heat1d_stiff_snapshot.png")
    results.add(
        "snapshot", {k: picture[k] for k in ("medium", "delta", "n", "h", "rows")}
    )
    results.add("snapshot/curves", snapshot_curves(picture))
    results.time("figures", time.perf_counter() - t3)
    results.time("total", time.perf_counter() - t0)
    paths = [args.outputs / RESULTS]
    if args.data_dir is not None:
        paths.append(args.data_dir / RESULTS)
    results.write(*paths)
    print(
        f"\nseed functions and snapshot {time.perf_counter() - t3:.1f} s; figures"
        f" heat1d_stiff_seeds.png and heat1d_stiff_snapshot.png in {args.outputs}/;"
        f" {time.perf_counter() - t0:.1f} s in all; results in"
        f" {', '.join(str(p) for p in paths)}"
    )


if __name__ == "__main__":
    main()
