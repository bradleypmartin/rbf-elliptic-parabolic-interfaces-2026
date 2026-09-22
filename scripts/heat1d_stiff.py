"""E3 (#5): the 1-D stiff-edge study. E3.2: the references; E3.3: the naive knee.

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
extended with ``--counts`` reruns only the new counts. The seeds (E3.4), the
treatments (E3.5) and the manuscript figures (E3.6) are added by their
tickets.

    uv run python scripts/heat1d_stiff.py     # 45 s cold, 4 s with everything cached
    uv run python scripts/heat1d_stiff.py --counts 50 100 200 400 800 1600 3200 6400
                                              # the two extra counts: + 3 min, once
    uv run python scripts/heat1d_stiff.py --deltas 0 0.0025 --media matlab
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

from heat_interfaces.heat1d import (  # noqa: E402
    RADAU_ATOL,
    RADAU_RTOL,
    Grid1D,
    ParabolicReference,
    SmoothEdges,
    bd4_march,
    chebyshev_equilibrium,
    dissertation_alpha,
    edge_resistance_deficit,
    equilibrium_exact,
    grid_for,
    inverse_alpha_integral,
    jump_aware_operator,
    matlab_alpha,
    naive_operator,
    normalized_l2,
    parabolic_reference,
    ramp_boundary,
    solve_equilibrium,
    straddling_windows,
)
from heat_interfaces.heat1d.domain import X_MAX, X_MIN  # noqa: E402
from heat_interfaces.plotting import AWARE, NAIVE, REFERENCE  # noqa: E402

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

OPERATORS = {"naive": naive_operator, "δ = 0 construction": jump_aware_operator}
"""The two lines of the knee study.

``jump_aware_operator`` on a ``SmoothEdges`` medium is §1.7's δ = 0
construction: E1.2's rows across the edge centre with the pieces' Taylor
data, as if the edge were a jump, and the direct rows on the smooth alpha
everywhere else.
"""

RESIDUAL_RATIOS = (1.0, 0.5, 0.1, 0.01, 0.001)
"""δ/h at which the δ = 0 rows are tested at fixed h (§1.4's list for the weights)."""

KNEE_CACHE = "heat1d_stiff_knee.json"
"""Where the parabolic knee errors are kept between runs.

``{"meta": KNEE_CACHE_META, "errors": {knee_key: error}}``. The header names
the problem (its label, ``T_END``, ``BC``, Radau's tolerances) and a file
whose header differs is ignored and overwritten, as ``ParabolicReference``
does for the references. A change to the BD4 marcher or to an operator's
construction is not detectable this way: delete the file after one.
"""

KNEE_CACHE_META = {
    "problem": PROBLEM,
    "t_end": T_END,
    "bc": list(BC),
    "rtol": RADAU_RTOL,
    "atol": RADAU_ATOL,
}

MARKERS = ("o", "s", "^", "D", "v")


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


def elliptic_sweep(name: str, delta: float, grids: Sequence[Grid1D]) -> list[dict]:
    """Both operators on the equilibrium problem at ``delta``, one row per grid.

    Errors are ``‖e‖₂/‖u‖₂`` against the quadrature on the smooth medium.
    ``floor`` is ``‖u₀ − u_δ‖₂/‖u_δ‖₂`` at the nodes, the difference of the
    jump's and the smooth medium's exact equilibria: what the δ = 0
    construction converges to while the grid does not resolve the edge
    (§1.7), first order in δ.
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
        for label, op in OPERATORS.items():
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
) -> list[dict]:
    """BD4 at ``dt = h`` on the ramp problem against the cached reference, per grid.

    Errors are ``‖e‖₂/‖u‖₂`` at ``t = T_END``. ``floor`` is the δ = 0
    reference against the δ one at the nodes. ``cache`` holds errors from
    earlier runs (``knee_key``); the ones missing are computed and added,
    the caller saves it.
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
        for label, op in OPERATORS.items():
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
    that misses the slope jump outright.
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
    labels = list(OPERATORS)
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
        f" h² |L_h u₀| there (rounding on constants); the naive rows' h |L_h u_δ|"
    )
    print(
        f"    {'δ/h':>8s}{'residual':>12s}{'h² res / δ':>13s}{'h² |L_h u₀|':>14s}"
        f"{'naive h |L_h u_δ|':>19s}"
    )
    for r in rows:
        print(
            f"    {r['ratio']:8g}{r['residual']:12.3e}{r['scaled']:13.4f}"
            f"{r['on_jump']:14.2e}{r['naive']:19.3e}"
        )


def plot_knee(
    elliptic: dict[str, dict[float, list[dict]]],
    parabolic: dict[str, dict[float, list[dict]]],
    path: Path,
) -> None:
    """Error against node count, one row of panels per medium: equilibrium, ramp.

    Orange is naive, blue the δ = 0 construction, as everywhere; the δ = 0
    lines are thin and unmarked (E1's lines), each δ > 0 has its marker,
    dotted blue is the floor of that δ, and the dashed verticals mark
    ``h = δ``.
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
                    ax.loglog(n, floor, ":", color=AWARE, lw=0.8, alpha=0.6)
                    ax.axvline(
                        (X_MAX - X_MIN) / delta + 1, color=REFERENCE, lw=0.6, ls="--"
                    )
                ax.loglog(n, [r["naive"] for r in rows], color=NAIVE, **style)
                ax.loglog(
                    n, [r["δ = 0 construction"] for r in rows], color=AWARE, **style
                )
            ax.set_title(f"{titles.get(name, name)}, {problem}", fontsize=10)
            ax.set_xlabel("nodes")
            ax.grid(True, which="both", alpha=0.3)
        axes[i, 0].set_ylabel("$\\|e\\|_2 / \\|u\\|_2$")
    handles = [
        Line2D([], [], color=NAIVE, label="naive $D_x A D_x$"),
        Line2D([], [], color=AWARE, label="δ = 0 construction"),
        Line2D(
            [],
            [],
            color=AWARE,
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
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
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
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    args.outputs.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    resolution = (args.n_cheb, args.max_width)
    check = (args.check_n_cheb, args.check_max_width)
    rows = check_references(args.media, args.deltas, args.outputs, resolution, check)
    print_references(rows, resolution, check)
    worst = max(r["agreement"] for r in rows)
    print(
        f"\nworst agreement between resolutions {worst:.1e}"
        f" ({'within' if worst < 1e-10 else 'OUTSIDE'} the 1e-10 of #27);"
        f" {time.perf_counter() - t0:.1f} s; references in {args.outputs}/"
    )

    t1 = time.perf_counter()
    cache = load_knee_cache(args.outputs)
    elliptic: dict[str, dict[float, list[dict]]] = {}
    parabolic: dict[str, dict[float, list[dict]]] = {}
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
        print_residuals(name, row_residuals(name, args.residual_n))
    plot_knee(elliptic, parabolic, args.outputs / "heat1d_stiff_knee.png")
    print(
        f"\nknee study {time.perf_counter() - t1:.1f} s; figure and"
        f" {KNEE_CACHE} in {args.outputs}/"
    )


if __name__ == "__main__":
    main()
