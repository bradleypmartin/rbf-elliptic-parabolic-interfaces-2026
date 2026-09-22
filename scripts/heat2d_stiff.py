"""E4 (#6), the 2-D stiff-edge study: the references, the naive knee, the seeds.

The medium is case 1's band with tanh edges of width δ in the signed normal
distance (``SmoothBand(case1().material, δ)``, stiff note §3.1), which on
case 1 is E3.2's 1-D medium in ``y`` bit for bit. The reference is the
separable ``u = e^{ct} sin 2πx v(y)`` with ``(α v′)′ − (4π² α + c) v = 0``,
``v(0) = 0``, ``v(1) = 1``, by E3.2's Chebyshev elements in ``y``
(``case1_reference``): the elliptic problem at ``c = 0`` and the parabolic
one at E2.5's ``c_t = 1``, exact in time.

``--mode references`` (E4.2, #33; stiff note §4.1) reports, per δ and ``c``:
the reference's elements, interior unknowns and build time; its agreement
with a finer resolution (24 nodes on 0.05-wide elements, ``CHECK_N_CHEB``,
``CHECK_MAX_WIDTH``); its distance from the δ = 0 reference,
``sup |v_δ − v_0|`` and that over δ (the O(δ) floor the δ = 0 construction
sits on, H10), and at δ = 0 its distance from the analytic ``case1_exact``;
and how far the band's midline is from the plateau, ``α(0.7) − 0.2``.

``--mode naive`` (E4.3, #34; stiff note §3.7 H10 predicts, §4.2 records) is
the knee study on scattered nodes. On case 1's node sets (seed 0, E2.1's
straddling rows; a smooth domain's node set is the jump's) two operators run
on the smooth medium at every δ and count: *naive* ``Dx A Dx + Dy A Dy`` on
port notes §2.2's stencils with α sampled at the nodes, and *the δ = 0
construction*, ``interface_aware_operator`` (warped, E2.4's settings) whose
crossing rows read the pieces as if δ were 0. Both problems: the equilibrium
by SuperLU and the parabolic problem by BD4 at ``dt = h`` from the
reference's analytic history to ``T_END`` (E2.5's). Errors are the RMS over
all nodes, Dirichlet rows included (port notes §2.10). Beside them: the
*floor*, the δ and δ = 0 references' difference at the nodes (what the
construction converges to while the grid cannot see the edge), and the
*uniform* line, the naive operator on the same nodes with α ≡ 1 against
``control_exact`` (the companion's "resolution floor": the same problem
without the feature). The naive systems are factored with
``PRODUCT_ORDERING`` (the same solution to 5e-12, five times faster at
40,000 nodes); the construction keeps SuperLU's default, so its δ = 0 line
is port notes §2.4–2.5's bit for bit.

Then what separates a resolved edge from an unresolved one, read off each
discrete solution on the straddling rows (``edge_diagnostics``): the *y-profile
error* at the innermost pair (``±h/2`` off each curve), each row's
``sin 2πx`` coefficient (``row_profile``, exact for the separable mode)
against the reference's; the *flux* ``α ∂_y u`` on each side of each curve,
from the one-sided quadratic through that side's three rows' profiles
(``pair_fluxes``), and the *flux jump* across the pair, both against the same
functional of the reference and relative to the reference's flux at the
curve. The *matched-h/δ* table sets each quantity at ``(δ, n)`` against
``(δ/2, 4n)``, where ``h/δ`` is the same to 2 % and ``h`` is halved: a
quantity that is a function of ``h/δ`` alone has ratio 1. The *growing-mode
check* (the E2.5 breadcrumb on #34) takes the interior spectrum of both
operators at ``--spectrum-n`` nodes per δ, with BD4's largest root modulus
at ``dt = h``. Figure: ``heat2d_stiff_knee.png``. The errors and diagnostics
are cached in ``heat2d_stiff_knee.json``, keyed by δ, count, seed, operator
and problem, so an extended ``--counts`` reruns only the new counts.

``--mode stencils`` (E4.4, #35; stiff note §3.7 H1–H3, §4.3) is the scalar
seeds on real stencils of the ``--stencil-n`` case-1 set (2500 nodes, seed
0). H1 on one stencil below the band at δ = 0, h/8, h and 8h: the seeds on
a band of equal pieces against the monomials, the shift identity
``g_j^{(a,b)} = C(a, j) g_0^{(a−j,b)}``, the residual ``L φ_e − α_e Σ C φ_e′``
on a 25 × 801 grid by twelfth-order differences with alpha read from the
medium (nothing shared with the march), the seeds of ``ηᵇ`` against E3.4's
1-D march on the profile in ``y``, and ``ψ₀₁ ≡ α_e``, the warp's
cancellation. H2 at δ = 0 over every crossing stencil of case 1 and of a
band one spacing thick (three regions, translated twice by E2.3): the
principal angle between the seed span and the translated basis's, the seed
weights with E2.3's plain Gaussians against ``stencil_weights(warp=False)``,
and ``φ₀₁`` against E2.4's warped normal coordinate. Then per δ/h from 8 to
1e-5 on the four innermost-row anchors: the same distances (H2's δ > 0
half) and the seed block's condition number raw and column-scaled, beside
the monomial and translated blocks' (H3). Last, ``seed_basis``'s cost over
every crossing stencil per δ, with E2.3's for scale (#35's acceptance
line). About 20 s.

    uv run python scripts/heat2d_stiff.py              # 2.5 min cold, 21 s cached
    uv run python scripts/heat2d_stiff.py --mode naive \
        --counts 1250 2500 5000 10000 20000 40000 80000 160000   # 58 min once
    uv run python scripts/heat2d_stiff.py --mode naive --seed 1 \
        --counts 1250 2500 5000 10000 20000 --spectrum-counts   # the scatter
    uv run python scripts/heat2d_stiff.py --mode references --deltas 0 0.001 0.0005
    uv run python scripts/heat2d_stiff.py --mode stencils       # 21 s
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import replace
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.linalg import subspace_angles  # noqa: E402

from heat_interfaces.fd_weights import fornberg_weights  # noqa: E402
from heat_interfaces.heat1d.march import bd4_amplification  # noqa: E402
from heat_interfaces.heat1d.stiff import seed_profiles as seed_profiles_1d  # noqa: E402
from heat_interfaces.heat2d import (  # noqa: E402
    BOUNDARY,
    INTERFACE_KIND,
    PRODUCT_ORDERING,
    REFERENCE_MAX_WIDTH,
    REFERENCE_N_CHEB,
    ROW_OFFSETS,
    Band,
    Constant2D,
    FlatLine,
    NodeSet,
    Row,
    SeedBasis,
    SmoothBand,
    augmented_solve,
    block_condition,
    build_node_set,
    build_stencils,
    case1,
    case1_exact,
    case1_reference,
    control_exact,
    gaussian_derivative,
    interface_aware_operator,
    interface_crossings,
    interface_stencil,
    interior_eigenvalues,
    knn,
    march_parabolic,
    naive_operator,
    polynomial_block,
    polynomial_exponents,
    profile_medium,
    rms_error,
    seed_basis,
    seed_profiles,
    solve_equilibrium,
    stencil_weights,
)
from heat_interfaces.plotting import CONSTRUCTION, NAIVE, REFERENCE  # noqa: E402

STUDY_DELTAS = (0.0, 0.04, 0.01, 0.005, 0.0025)
"""The jump and E4.3's four edge widths (#34)."""

GROWTH = (0.0, 1.0)
"""``c`` of the elliptic problem and of E2.5's parabolic one (``c_t = 1``)."""

CHECK_N_CHEB, CHECK_MAX_WIDTH = 24, 0.05
"""The finer resolution the reference is checked against (E3.2's)."""

Y = np.linspace(0.0, 1.0, 4001)
"""Where the reference is compared: 4001 points on ``[0, 1]``."""

MIDLINE = 0.7
"""The band's midline, where a wide edge keeps α furthest from the plateau."""

KNEE_COUNTS = (1250, 2500, 5000, 10000)
"""The default counts; the documented sweep adds 20,000, 40,000 and 80,000."""

T_END = 0.1
"""Where E2.5 (and dissertation Fig. 5-5) reads the parabolic error."""

PROBLEMS = {"elliptic": 0.0, "parabolic": 1.0}
"""Problem name to ``c``: the equilibrium, and E2.5's ``e^{t}`` mode."""

OPERATORS = ("naive", "construction")
"""The two lines of the knee table; ``uniform`` is the α ≡ 1 run beside them."""

QUANTITIES = ("rms", "max", "profile", "flux", "jump")
"""What ``edge_diagnostics`` returns, in the order the tables print it."""

NAIVE_COLUMNS = ("rms", "vs_jump", "max", "profile", "flux", "jump")
"""The naive line's printed readings: ``vs_jump`` is its RMS over the jump's RMS
on the same node set, which cancels the node set's own scatter."""

SPECTRUM_COUNTS = (1250, 2500)
"""The naive operator's coarse-set growing mode: at 1250, none from 1800 on the jump."""

KNEE_CACHE = "heat2d_stiff_knee.json"
KNEE_CACHE_META = {
    "study": "E4.3 case 1 with tanh edges against the separable references",
    "version": 1,
    "reference": [REFERENCE_N_CHEB, REFERENCE_MAX_WIDTH],
    "naive_ordering": PRODUCT_ORDERING,
}
"""Bump ``version`` after any change to an operator, a diagnostic or the march."""

MARKERS = ("o", "s", "^", "D")
"""One marker per δ > 0, in ``STUDY_DELTAS`` order."""

STENCIL_N = 2500
"""The node set of the one-stencil study (E4.4): ``h = 1/48``, 576 crossing stencils."""

STENCIL_ANCHORS = (0.5896, 0.6104, 0.7896, 0.8104)
"""The innermost straddling rows at 2500 nodes, at ``x = 0.5``: below the band,
inside it at either line, above it."""

CHAIN_RATIOS = (1.0 / 8.0, 1.0, 8.0)
"""δ/h of #35's checks: an unresolved, a marginal and a resolved edge."""

LADDER_RATIOS = (8.0, 1.0, 0.5, 1.0 / 8.0, 1.0 / 64.0, 1e-3, 1e-4, 1e-5)
"""δ/h of the jump-limit and conditioning ladders (P4 and P5's in 1-D)."""

RESIDUAL_GRID = (25, 801)
"""``(ξ, η)`` points of H1's residual grid on ``[−1, 1]²``."""

RESIDUAL_HALF = 6
"""Half-width of the residual's centred differences: twelfth order."""

THIN = Band(FlatLine(0.6), FlatLine(0.62), Constant2D(0.2), Constant2D(1.0))
"""A band one spacing thick at 2500 nodes: its stencils reach all three regions."""


# --- E4.2: the references ---------------------------------------------------------


def reference_rows(
    deltas: Sequence[float],
    growth: Sequence[float] = GROWTH,
    resolution: tuple[int, float] = (REFERENCE_N_CHEB, REFERENCE_MAX_WIDTH),
    check: tuple[int, float] = (CHECK_N_CHEB, CHECK_MAX_WIDTH),
) -> list[dict[str, float]]:
    """One row per (δ, c): the reference's size, cost, agreement and distances."""
    rows = []
    for c in growth:
        base = case1_reference(0.0, c, *resolution).v(Y)
        for delta in deltas:
            t0 = time.perf_counter()
            ref = case1_reference(delta, c, *resolution)
            seconds = time.perf_counter() - t0
            fine = case1_reference(delta, c, *check)
            v = ref.v(Y)
            if delta == 0.0:
                distance = np.abs(v - case1_exact(c).v(Y)).max()
            else:
                distance = np.abs(v - base).max()
            band = SmoothBand(case1().material, delta)
            rows.append(
                {
                    "delta": delta,
                    "growth": c,
                    "elements": ref.elements,
                    "unknowns": ref.unknowns,
                    "ms": 1e3 * seconds,
                    "agreement": float(np.abs(v - fine.v(Y)).max()),
                    "distance": float(distance),
                    "plateau": float(band.alpha(0.0, MIDLINE)) - 0.2,
                }
            )
    return rows


def print_references(rows: list[dict[str, float]]) -> None:
    print(
        "case 1 with tanh edges: the separable reference v(y), "
        f"{REFERENCE_N_CHEB} nodes per element, elements <= {REFERENCE_MAX_WIDTH}"
    )
    print(
        "  agreement: against 24 nodes on 0.05-wide elements; distance: "
        "sup |v − v at δ = 0|, at δ = 0 sup |v − case1_exact|"
    )
    print(
        "        δ    c  elements  unknowns     ms  agreement   distance  "
        "distance / δ  α(0.7) − 0.2"
    )
    for r in rows:
        ratio = "-" if r["delta"] == 0.0 else f"{r['distance'] / r['delta']:.3f}"
        print(
            f"  {r['delta']:7.4f}  {r['growth']:3.1f}  {r['elements']:8d}  "
            f"{r['unknowns']:8d}  {r['ms']:5.1f}  {r['agreement']:9.1e}  "
            f"{r['distance']:9.3e}  {ratio:>12}  {r['plateau']:12.2e}"
        )


# --- E4.3: the diagnostics on the straddling rows ------------------------------------


def row_profile(nodes: NodeSet, row: Row, u: np.ndarray) -> float:
    """``(2/m) Σ u sin 2πx`` over one straddling row of ``m`` nodes.

    The rows are equispaced in ``x`` over the period (E2.1), so for ``m >= 3``
    ``Σ sin² 2πx_i = m/2`` and the separable ``e^{ct} sin 2πx v(y)`` returns
    ``e^{ct} v(y_row)`` exactly: the row's value of the y-profile, with the
    x-scatter of a discrete solution averaged out.
    """
    x = nodes.x[row.index]
    return float(2.0 / x.size * np.sum(u[row.index] * np.sin(2.0 * np.pi * x)))


def curve_level(nodes: NodeSet, curve: int) -> float:
    """The ``y`` of straddled curve ``curve``, between between its innermost pair."""
    lo, hi = nodes.rows_of(curve, ROW_OFFSETS[0])
    return 0.5 * float(nodes.y[lo.index[0]] + nodes.y[hi.index[0]])


def pair_fluxes(
    nodes: NodeSet, medium, u: np.ndarray, curve: int
) -> tuple[float, float]:
    """``α ∂_y`` of the y-profile at ``−h/2`` and ``+h/2`` off ``curve``, one-sided.

    Each side's three rows (``ROW_OFFSETS``, the middle one staggered in
    ``x``, which ``row_profile`` does not see) give three samples of the
    profile; the quadratic through them is differentiated at the innermost
    row (Fornberg weights), and α is the medium's there. Nothing is read
    across the curve, so the value is the discrete solution's own flux on
    that side. Linear in ``u``.
    """
    fluxes = []
    for side in (-1, 1):
        rows = [nodes.rows_of(curve, o)[(side + 1) // 2] for o in ROW_OFFSETS]
        y = np.array([nodes.y[r.index[0]] for r in rows])
        p = np.array([row_profile(nodes, r, u) for r in rows])
        slope = fornberg_weights(y[0], y, 1)[1] @ p
        fluxes.append(float(medium.alpha(0.0, y[0])) * slope)
    return fluxes[0], fluxes[1]


def edge_diagnostics(
    nodes: NodeSet, medium, ref, u: np.ndarray, t: float = 0.0
) -> dict[str, float]:
    """The RMS and max errors and three straddling-row readings of ``u``, vs ``ref``.

    ``profile``: the largest ``|row_profile|`` of the error on the innermost
    pairs (absolute, like the RMS). ``flux``: the largest one-sided
    ``pair_fluxes`` error over both curves and sides; ``jump``: the largest
    error in the flux jump ``q₊ − q₋`` across a pair; both relative to the
    reference's flux ``α v′`` at that curve (continuous across it). The
    functionals are linear, so they are applied to the error: their own
    truncation cancels against the reference's.
    """
    exact = ref(nodes.x, nodes.y, t)
    e = u - exact
    out = {
        "rms": rms_error(u, exact),
        "max": float(np.abs(e).max()),
        "profile": 0.0,
        "flux": 0.0,
        "jump": 0.0,
    }
    for curve in range(len(nodes.straddle_rows) // (2 * len(ROW_OFFSETS))):
        scale = abs(float(ref.flux_y(0.25, curve_level(nodes, curve), t)))
        lo, hi = nodes.rows_of(curve, ROW_OFFSETS[0])
        for row in (lo, hi):
            out["profile"] = max(out["profile"], abs(row_profile(nodes, row, e)))
        below, above = pair_fluxes(nodes, medium, e, curve)
        out["flux"] = max(out["flux"], abs(below) / scale, abs(above) / scale)
        out["jump"] = max(out["jump"], abs(above - below) / scale)
    return out


# --- E4.3: the sweep -----------------------------------------------------------------


def knee_key(
    problem: str,
    delta: float | None,
    n: int,
    label: str,
    seed: int,
    iterations: int,
    t_end: float,
) -> str:
    """The cache key of one line's numbers at one (problem, δ, n)."""
    d = "-" if delta is None else f"{delta:g}"
    t = f" t{t_end:g}" if PROBLEMS[problem] else ""
    return f"{problem}{t} d{d} n{n} s{seed} i{iterations} {label}"


def load_knee_cache(outputs: Path) -> dict[str, dict[str, float]]:
    """The cached numbers; empty when the file is missing or from another study."""
    path = outputs / KNEE_CACHE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if data.get("meta") != KNEE_CACHE_META:
        return {}
    return dict(data["entries"])


def save_knee_cache(outputs: Path, cache: dict[str, dict[str, float]]) -> None:
    data = {"meta": KNEE_CACHE_META, "entries": dict(sorted(cache.items()))}
    (outputs / KNEE_CACHE).write_text(json.dumps(data, indent=1) + "\n")


def top_row(growth: float):
    """The Dirichlet row at ``y = 1`` of every problem here: ``e^{ct} sin 2πx``."""

    def top(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        return np.exp(growth * t) * np.sin(2.0 * np.pi * x)

    return top


def _solve(problem, op, nodes, ref, t_end, permc_spec=None) -> tuple[np.ndarray, float]:
    """``(u, t)``: the equilibrium, or BD4 at ``dt = h`` from the analytic history.

    ``ref`` is a ``SeparableReference`` or ``control_exact``; both are 0 on
    ``y = 0`` and ``top_row`` on ``y = 1``.
    """
    values = (0.0, top_row(PROBLEMS[problem]))
    if PROBLEMS[problem] == 0.0:
        return solve_equilibrium(op, nodes, values, None, permc_spec), 0.0
    u0 = ref(nodes.x, nodes.y, 0.0)
    u = march_parabolic(
        op, nodes, u0, t_end, nodes.h, values, solution=ref, permc_spec=permc_spec
    )
    return u, t_end


def knee_sweep(
    counts: Sequence[int],
    deltas: Sequence[float],
    cache: dict[str, dict[str, float]],
    seed: int = 0,
    iterations: int = 100,
    t_end: float = T_END,
    problems: Sequence[str] = tuple(PROBLEMS),
) -> dict[str, dict[float, list[dict]]]:
    """``results[problem][δ]``: one row per count, the two lines, floor and uniform.

    A row holds ``n``, ``h``, ``h_over_delta``, ``floor`` (the RMS of the δ and
    δ = 0 references' difference at the nodes), ``uniform`` (the naive α ≡ 1
    run's RMS error against ``control_exact``) and, per operator in
    ``OPERATORS``, the ``edge_diagnostics`` under ``"<operator>/<quantity>"``,
    and, when δ = 0 is swept, ``naive/vs_jump``: the naive RMS error over the
    jump's on the same node set.
    What ``cache`` lacks is computed and added (the caller saves it); a
    count whose every entry is cached builds no node set.
    """
    domain = case1()
    results: dict[str, dict[float, list[dict]]] = {
        p: {d: [] for d in deltas} for p in problems
    }
    for n in counts:

        def key(problem, delta, label, n=n):
            return knee_key(problem, delta, n, label, seed, iterations, t_end)

        needed = [key(p, None, "uniform") for p in problems] + [
            key(p, d, label)
            for p in problems
            for d in deltas
            for label in ("floor", *OPERATORS)
        ]
        if any(k not in cache for k in needed):
            t0 = time.perf_counter()
            nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
            plain = build_stencils(nodes, domain)
            crossing = build_stencils(nodes, domain, interface=BOUNDARY)
            for problem in problems:
                c = PROBLEMS[problem]
                k = key(problem, None, "uniform")
                if k not in cache:
                    op = naive_operator(nodes, Constant2D(1.0), plain)
                    ref = control_exact(c)
                    u, t = _solve(problem, op, nodes, ref, t_end, PRODUCT_ORDERING)
                    cache[k] = {"rms": rms_error(u, ref(nodes.x, nodes.y, t))}
                base = case1_reference(0.0, c)
                for delta in deltas:
                    medium = SmoothBand(domain.material, delta)
                    ref = case1_reference(delta, c)
                    t = t_end if c else 0.0
                    k = key(problem, delta, "floor")
                    if k not in cache:
                        cache[k] = {
                            "h": nodes.h,
                            "floor": rms_error(
                                base(nodes.x, nodes.y, t), ref(nodes.x, nodes.y, t)
                            ),
                        }
                    for label in OPERATORS:
                        k = key(problem, delta, label)
                        if k in cache:
                            continue
                        if label == "naive":
                            op = naive_operator(nodes, medium, plain)
                            permc = PRODUCT_ORDERING
                        else:
                            op = interface_aware_operator(nodes, medium, crossing)
                            permc = None
                        u, t = _solve(problem, op, nodes, ref, t_end, permc)
                        cache[k] = edge_diagnostics(nodes, medium, ref, u, t)
            print(f"  (n = {n}: {time.perf_counter() - t0:.1f} s)", flush=True)
        for problem in problems:
            uniform = cache[key(problem, None, "uniform")]["rms"]
            for delta in deltas:
                floor = cache[key(problem, delta, "floor")]
                row = {
                    "n": n,
                    "h": floor["h"],
                    "h_over_delta": floor["h"] / delta if delta else float("inf"),
                    "floor": floor["floor"],
                    "uniform": uniform,
                }
                for label in OPERATORS:
                    for q, value in cache[key(problem, delta, label)].items():
                        row[f"{label}/{q}"] = value
                if 0.0 in deltas:
                    jump = cache[key(problem, 0.0, "naive")]["rms"]
                    row["naive/vs_jump"] = row["naive/rms"] / jump
                results[problem][delta].append(row)
    return results


def growing_modes(
    n: int,
    deltas: Sequence[float],
    cache: dict[str, dict[str, float]],
    seed: int = 0,
    iterations: int = 100,
) -> list[dict]:
    """Per δ and operator at ``n`` nodes: the interior spectrum's right edge.

    ``max_re`` and ``positive`` (eigenvalues with a positive real part), and
    BD4's largest root modulus at ``dt = h`` over the whole spectrum (below
    one: every mode damped). Cached like the sweep.
    """
    domain = case1()
    nodes = plain = crossing = None
    rows = []
    for delta in deltas:
        for label in OPERATORS:
            key = f"spectrum d{delta:g} n{n} s{seed} i{iterations} {label}"
            if key not in cache:
                if nodes is None:
                    nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
                    plain = build_stencils(nodes, domain)
                    crossing = build_stencils(nodes, domain, interface=BOUNDARY)
                medium = SmoothBand(domain.material, delta)
                if label == "naive":
                    op = naive_operator(nodes, medium, plain)
                else:
                    op = interface_aware_operator(nodes, medium, crossing)
                lam = interior_eigenvalues(op, nodes)
                cache[key] = {
                    "h": nodes.h,
                    "max_re": float(lam.real.max()),
                    "positive": int(np.sum(lam.real > 0)),
                    "bd4": float(bd4_amplification(nodes.h * lam).max()),
                }
            rows.append({"delta": delta, "operator": label, "n": n, **cache[key]})
    return rows


def matched_ratios(
    results: dict[float, list[dict]], quantities: Sequence[str]
) -> list[dict]:
    """``Q(δ, n) / Q(δ/2, 4n)`` wherever both are in ``results``: same h/δ, h halved.

    ``h = 1/round(0.95 √n)`` halves to within 1.5 % when ``n`` quadruples, so
    the two runs share ``h/δ`` to about 2 %; a quantity that depends on
    ``h/δ`` alone has ratio 1, one that also scales like ``h^p`` has ``2^p``.
    """
    rows = []
    for delta, lines in results.items():
        half = delta / 2
        if delta == 0.0 or half not in results:
            continue
        finer = {r["n"]: r for r in results[half]}
        for r in lines:
            s = finer.get(4 * r["n"])
            if s is None:
                continue
            row = {
                "delta": delta,
                "n": r["n"],
                "h_over_delta": r["h_over_delta"],
                "h_over_delta_finer": s["h_over_delta"],
            }
            for q in quantities:
                row[q] = r[q] / s[q]
            rows.append(row)
    return rows


# --- E4.3: printing and the figure ----------------------------------------------


def _rate(a: dict, b: dict, name: str) -> float:
    return float(np.log(a[name] / b[name]) / np.log(a["h"] / b["h"]))


def _with_rates(rows: list[dict], name: str) -> list[str]:
    cells = [f"{rows[0][name]:.2e}       "]
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        cells.append(f"{b[name]:.2e} ({_rate(a, b, name):5.2f})")
    return cells


def print_knee(results: dict[float, list[dict]], title: str) -> None:
    """RMS error (order per halving of h) of the naive line per δ, uniform beside."""
    deltas = list(results)
    first = results[deltas[0]]
    print(f"\n{title}")
    header = "     n       h |  uniform α ≡ 1       |"
    for d in deltas:
        name = "δ = 0 (jump)" if d == 0.0 else f"δ = {d:g}"
        header += f" {name:<20} |"
    print(header)
    uniform = _with_rates(first, "uniform")
    columns = [_with_rates(results[d], "naive/rms") for d in deltas]
    for i, row in enumerate(first):
        line = f"{row['n']:6d}  {row['h']:.4f} | {uniform[i]:<20} |"
        for col in columns:
            line += f" {col[i]:<20} |"
        print(line)


def print_construction(results: dict[float, list[dict]], title: str) -> None:
    """The δ = 0 construction's RMS error per δ, with (error / floor) beside."""
    print(f"\n{title}")
    deltas = list(results)
    header = "     n |"
    for d in deltas:
        name = "δ = 0: E2.4's line" if d == 0.0 else f"δ = {d:g}: error (/ floor)"
        header += f" {name:<26} |"
    print(header)
    for i, row in enumerate(results[deltas[0]]):
        line = f"{row['n']:6d} |"
        for d in deltas:
            r = results[d][i]
            if d == 0.0:
                cell = f"{r['construction/rms']:.3e}"
            else:
                ratio = r["construction/rms"] / r["floor"]
                cell = f"{r['construction/rms']:.3e} ({ratio:5.3f})"
            line += f" {cell:<26} |"
        print(line)
    floors = ", ".join(
        f"{d:g}: {results[d][0]['floor']:.3e} … {results[d][-1]['floor']:.3e}"
        for d in deltas
        if d > 0
    )
    print(f"  floors (RMS of the δ and δ = 0 references' difference): {floors}")


def print_diagnostics(results: dict[float, list[dict]], title: str) -> None:
    """Per δ: h/δ, the naive solution's readings, and the construction's."""
    print(f"\n{title}")
    print(
        "      δ       n    h/δ |  naive: RMS    ÷ jump       max   profile      flux"
        "      jump |  construction: RMS      flux      jump"
    )
    for delta, rows in results.items():
        for r in rows:
            ratio = "   jump" if delta == 0.0 else f"{r['h_over_delta']:7.2f}"
            line = f"  {delta:6.4f}  {r['n']:6d} {ratio} |"
            line += "".join(f"  {r['naive/' + q]:8.2e}" for q in NAIVE_COLUMNS) + " |"
            for q in ("rms", "flux", "jump"):
                line += f"  {r['construction/' + q]:8.2e}"
            print(line)


def print_matched(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(
        "   δ →  δ/2   n → 4n   h/δ  (4n) |  naive: RMS    ÷ jump       max   profile"
        "      flux      jump"
    )
    for r in rows:
        line = (
            f"  {r['delta']:6.4f}  {r['n']:6d}  {r['h_over_delta']:5.2f}"
            f" ({r['h_over_delta_finer']:5.2f}) |"
        )
        line += "".join(f"  {r['naive/' + q]:8.2f}" for q in NAIVE_COLUMNS)
        print(line)


def print_spectra(rows: list[dict]) -> None:
    print(
        "\ngrowing-mode check: the interior spectrum's right edge, BD4's largest"
        " root modulus at dt = h, and the parabolic / elliptic error ratio at that"
        " count (from the sweep; - if the count is not in it)"
    )
    print("      n       δ  operator        max Re λ  positive  BD4 max |ζ|  par / ell")
    for r in rows:
        ratio = r.get("parabolic_over_elliptic")
        cell = "        -" if ratio is None else f"{ratio:9.3f}"
        print(
            f"  {r['n']:5d}  {r['delta']:6.4f}  {r['operator']:<13} {r['max_re']:10.3f}"
            f"  {r['positive']:8d}  {r['bd4']:11.3f}  {cell}"
        )


def plot_knee(results: dict[str, dict[float, list[dict]]], path: Path) -> None:
    """Top: RMS error against N, elliptic and parabolic; bottom: the collapse in h/δ.

    Orange is naive, purple the δ = 0 construction, one marker per δ; the
    jump's naive line is thin and unmarked, dotted purple the floors, grey
    the uniform (α ≡ 1) naive run, dashed verticals ``h = δ``. The bottom
    panels plot three readings of the naive elliptic solution against
    ``h/δ``: the relative flux error on the innermost pair, the RMS error
    over the jump's on the same node set, and the RMS error itself, with
    the jump's range as a band; a reading that depends on ``h/δ`` alone
    collapses to one curve.
    """
    fig = plt.figure(figsize=(11.0, 8.4))
    grid = fig.add_gridspec(2, 6)
    top = [fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3:])]
    bottom = [fig.add_subplot(grid[1, 2 * k : 2 * k + 2]) for k in range(3)]
    names = {"elliptic": "equilibrium", "parabolic": f"parabolic, t = {T_END:g}"}
    for ax, (problem, lines) in zip(top, results.items(), strict=True):
        positive = [d for d in lines if d > 0]
        first = next(iter(lines.values()))
        n = np.array([r["n"] for r in first], dtype=float)
        ax.loglog(n, [r["uniform"] for r in first], color=REFERENCE, lw=1.0)
        for delta, rows in lines.items():
            if delta == 0.0:
                ax.loglog(n, [r["naive/rms"] for r in rows], color=NAIVE, lw=0.8)
                continue
            marker = MARKERS[positive.index(delta) % len(MARKERS)]
            ax.loglog(
                n, [r["naive/rms"] for r in rows], color=NAIVE, marker=marker, ms=5
            )
            ax.loglog(
                n,
                [r["construction/rms"] for r in rows],
                color=CONSTRUCTION,
                marker=marker,
                ms=5,
                lw=0.9,
            )
            ax.loglog(n, [r["floor"] for r in rows], ":", color=CONSTRUCTION, lw=0.8)
            at = (1.0 / (0.95 * delta)) ** 2
            if n[0] <= at <= n[-1]:
                ax.axvline(at, color=REFERENCE, lw=0.6, ls="--")
        ax.set_xticks(n, [f"{int(k)}" for k in n], fontsize=7, rotation=45)
        ax.set_xticks([], minor=True)
        ax.set_title(f"case 1, tanh edges: {names[problem]}", fontsize=10)
        ax.set_xlabel("nodes N")
        ax.set_ylabel("RMS error in u")
        ax.grid(True, which="both", alpha=0.25)
    elliptic = results["elliptic"]
    for ax, quantity, label in (
        (bottom[0], "naive/flux", "naive: flux error on the pair / |α v′|"),
        (bottom[1], "naive/vs_jump", "naive: RMS ÷ the jump's, same nodes"),
        (bottom[2], "naive/rms", "naive: RMS error in u"),
    ):
        positive = [d for d in elliptic if d > 0]
        # The jump's own ratio is 1 by definition; no band for it.
        if 0.0 in elliptic and quantity != "naive/vs_jump":
            jump = [r[quantity] for r in elliptic[0.0]]
            ax.axhspan(min(jump), max(jump), color=NAIVE, alpha=0.15, lw=0)
        for delta in positive:
            rows = elliptic[delta]
            ax.loglog(
                [r["h_over_delta"] for r in rows],
                [r[quantity] for r in rows],
                color=NAIVE,
                marker=MARKERS[positive.index(delta) % len(MARKERS)],
                ms=5,
                lw=0.8,
            )
        ax.axvline(1.0, color=REFERENCE, lw=0.6, ls="--")
        ax.set_xlabel("h / δ")
        ax.set_title(label, fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
    handles = [
        Line2D([], [], color=NAIVE, label="naive $D_x A D_x + D_y A D_y$"),
        Line2D([], [], color=CONSTRUCTION, label="δ = 0 construction"),
        Line2D([], [], color=CONSTRUCTION, ls=":", label="floor: ref. δ − ref. 0"),
        Line2D([], [], color=REFERENCE, label="naive, uniform α ≡ 1"),
        Line2D([], [], color=NAIVE, lw=0.8, label="naive, δ = 0 (jump; band below)"),
        Line2D([], [], color=REFERENCE, ls="--", lw=0.6, label="h = δ"),
    ]
    positive = [d for d in elliptic if d > 0]
    for k, delta in enumerate(positive):
        handles.append(
            Line2D(
                [],
                [],
                color="k",
                marker=MARKERS[k % len(MARKERS)],
                ls="",
                ms=5,
                label=f"δ = {delta:g}",
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_naive(args) -> dict:
    """E4.3's tables and figure; returns every table by name."""
    t0 = time.perf_counter()
    cache = load_knee_cache(args.outputs)
    results = knee_sweep(
        args.counts, args.deltas, cache, args.seed, args.iterations, args.t_end
    )
    save_knee_cache(args.outputs, cache)
    tables: dict = {"knee": results}
    for problem, lines in results.items():
        what = (
            "equilibrium"
            if problem == "elliptic"
            else f"parabolic, BD4 dt = h from the analytic history, t = {args.t_end:g}"
        )
        print_knee(
            lines,
            f"naive Dx A Dx + Dy A Dy through case 1's tanh edges, {what}: RMS error"
            " (order per halving of h) against the separable reference at each δ",
        )
        print_construction(
            lines,
            f"the δ = 0 construction (interface_aware_operator on the smooth medium),"
            f" {what}: RMS error (error / floor)",
        )
        print_diagnostics(
            lines,
            f"what the discrete solution says on the straddling rows, {what}:"
            " profile = |error of the row's sin 2πx coefficient| at ±h/2 (absolute);"
            " flux, jump = one-sided α ∂_y at ±h/2 and its jump across the pair,"
            " error / |α v′ at the curve|",
        )
        matched = matched_ratios(lines, [f"naive/{q}" for q in NAIVE_COLUMNS])
        if matched:
            print_matched(
                matched,
                f"matched h/δ, {what}: Q(δ, n) / Q(δ/2, 4n), the same h/δ at half"
                " the h (1: a function of h/δ alone; 2^p: also ∝ h^p)",
            )
        tables[f"matched/{problem}"] = matched
    spectra = []
    for n in args.spectrum_counts:
        rows = growing_modes(n, args.deltas, cache, args.seed, args.iterations)
        save_knee_cache(args.outputs, cache)
        if n in args.counts:
            i = list(args.counts).index(n)
            for r in rows:
                d, label = r["delta"], r["operator"]
                par = results["parabolic"][d][i][f"{label}/rms"]
                r["parabolic_over_elliptic"] = (
                    par / results["elliptic"][d][i][f"{label}/rms"]
                )
        spectra.extend(rows)
    if spectra:
        print_spectra(spectra)
        tables["spectra"] = spectra
    args.outputs.mkdir(parents=True, exist_ok=True)
    plot_knee(results, args.outputs / "heat2d_stiff_knee.png")
    print(
        f"\nknee study {time.perf_counter() - t0:.1f} s; figure and {KNEE_CACHE}"
        f" in {args.outputs}/"
    )
    return tables


# --- E4.4: the scalar seeds on one stencil ----------------------------------------


EXPONENTS = [tuple(int(v) for v in e) for e in polynomial_exponents(4)]


def anchor_stencil(nodes: NodeSet, y: float, x: float = 0.5) -> np.ndarray:
    """The 30 nearest nodes of the node nearest ``(x, y)``, that node first."""
    target = int(np.argmin((nodes.x - x) ** 2 + (nodes.y - y) ** 2))
    idx, _ = knn(nodes.xy, 30, query=nodes.xy[target : target + 1])
    return nodes.xy[idx[0]]


def crossing_stencils(nodes: NodeSet, material: Band) -> np.ndarray:
    """``(m, 30)`` node indices of the interface group's stencils that cross (E2.3)."""
    domain = replace(case1(), material=material)
    stencils = build_stencils(nodes, domain, interface=BOUNDARY)
    (group,) = [g for g in stencils.groups if g.kind == INTERFACE_KIND]
    return group.index[interface_crossings(nodes, material, group.index)]


def span_distance(p: np.ndarray, s: np.ndarray) -> float:
    """The sine of the largest principal angle between two column spans."""
    return float(np.sin(subspace_angles(p, s).max()))


def seed_weights_with(sb: SeedBasis, st) -> np.ndarray:
    """The seed rows' weights with E2.3's plain Gaussian block of ``st``."""
    b_rbf = sb.alpha_e * gaussian_derivative(st.xi, st.eta, st.eps, "lap")
    w = augmented_solve(
        st.gaussian_block()[None],
        sb.block[None],
        b_rbf[None, :, None],
        sb.rhs[None, :, None],
    )
    return w[0, :, 0] / sb.scale**2


def _centred(f: np.ndarray, step: float, axis: int) -> np.ndarray:
    """Centred d/dx of order ``2 RESIDUAL_HALF`` along ``axis``; NaN at the ends."""
    half = RESIDUAL_HALF
    w = fornberg_weights(0.0, step * np.arange(-half, half + 1), 1)[1]
    f = np.moveaxis(f, axis, -1)
    m = f.shape[-1]
    out = np.full_like(f, np.nan)
    out[..., half : m - half] = sum(
        wk * f[..., k : m - 2 * half + k] for k, wk in enumerate(w)
    )
    return np.moveaxis(out, -1, axis)


def chain_residual(sb: SeedBasis, medium: SmoothBand) -> float:
    """H1: ``max |L φ_e − α_e Σ C φ_e′|`` on ``RESIDUAL_GRID``, relative per seed.

    Differences in both directions with alpha from the medium at the physical
    points: nothing but the seed values is shared with the march.
    """
    n_xi, n_eta = RESIDUAL_GRID
    xi, eta = np.linspace(-1.0, 1.0, n_xi), np.linspace(-1.0, 1.0, n_eta)
    phi = seed_profiles(sb.profile, eta, alpha_e=sb.alpha_e).values(xi[:, None])
    f = sb.frame
    c, s = np.cos(f.angle), np.sin(f.angle)
    x = f.x0 + f.scale * (c * xi[:, None] - s * eta[None, :])
    y = f.y0 + f.scale * (s * xi[:, None] + c * eta[None, :])
    alpha = medium.alpha(x, y)
    worst = 0.0
    for k, (a, b) in enumerate(EXPONENTS):
        lf = sum(
            _centred(alpha * _centred(phi[..., k], d[1] - d[0], ax), d[1] - d[0], ax)
            for ax, d in ((0, xi), (1, eta))
        )
        rhs = np.zeros_like(lf)
        if a >= 2:
            rhs += a * (a - 1) * phi[..., EXPONENTS.index((a - 2, b))]
        if b >= 2:
            rhs += b * (b - 1) * phi[..., EXPONENTS.index((a, b - 2))]
        rhs *= sb.alpha_e
        err = np.nanmax(np.abs(lf - rhs)) / max(np.abs(rhs).max(), 1.0)
        worst = max(worst, float(err))
    return worst


def chain_rows(nodes: NodeSet, h: float, xy: np.ndarray) -> list[dict]:
    """H1 on one stencil per δ/h: monomials, shift identity, residual, 1-D, warp."""
    band = case1().material
    level = Band(band.lower, band.upper, Constant2D(0.37), Constant2D(0.37))
    rows = []
    for ratio in (0.0, *CHAIN_RATIOS):
        medium = SmoothBand(band, ratio * h)
        sb = seed_basis(xy, medium)
        flat = seed_basis(xy, SmoothBand(level, ratio * h))
        monomials = np.abs(flat.block - polynomial_block(flat.xi, flat.eta, 4)).max()
        eta = np.union1d(sb.eta, np.linspace(-1.0, 1.0, 41))
        p = seed_profiles(sb.profile, eta, alpha_e=sb.alpha_e)
        ch = p.chain
        shift = (
            max(
                np.abs(
                    p.g[ch.index(a, b, j)] - comb(a, j) * p.g[ch.index(a - j, b, 0)]
                ).max()
                for a, b, j in ch.levels
            )
            / np.abs(p.g).max()
        )
        warp = np.abs(p.psi[ch.index(0, 1, 0)] - sb.alpha_e).max() / sb.alpha_e
        u = np.unique(sb.eta)
        one = seed_profiles_1d([sb.frame.y0], [sb.scale], u, profile_medium(medium), 5)
        two = seed_profiles(sb.profile, u, alpha_e=sb.alpha_e).values(np.zeros(u.size))
        cols = [EXPONENTS.index((0, b)) for b in range(5)]
        rows.append(
            {
                "ratio": ratio,
                "monomials": float(monomials),
                "shift": float(shift),
                "residual": chain_residual(sb, medium) if ratio > 0 else float("nan"),
                "one_d": float(
                    np.abs(one[0].T - two[:, cols]).max() / np.abs(one).max()
                ),
                "warp": float(warp),
                "largest": float(np.abs(p.g).max()),
            }
        )
    return rows


def jump_limit_rows(nodes: NodeSet) -> list[dict]:
    """H2 at δ = 0 over every crossing stencil, case 1 and the thin band."""
    rows = []
    for name, band in (("case 1", case1().material), ("thin band", THIN)):
        index = crossing_stencils(nodes, band)
        span, weights, warp, regions = 0.0, 0.0, 0.0, 0
        for idx in index:
            xy = nodes.xy[idx]
            sb = seed_basis(xy, band)
            plain = interface_stencil(xy, band, 4, warp=False)
            warped = interface_stencil(xy, band, 4, warp=True)
            span = max(span, span_distance(plain.polynomial_block(), sb.block))
            w_ref = stencil_weights(xy, band, 4, warp=False)
            w = seed_weights_with(sb, plain)
            weights = max(weights, float(np.abs(w - w_ref).max() / np.abs(w_ref).max()))
            warp = max(warp, float(np.abs(sb.warp - warped.eta).max()))
            regions += int(np.ptp(plain.region) == 2)
        rows.append(
            {
                "material": name,
                "stencils": len(index),
                "three_region": regions,
                "span": span,
                "weights": weights,
                "warp": warp,
            }
        )
    return rows


def ladder_rows(nodes: NodeSet, h: float) -> list[dict]:
    """H2's δ > 0 half and H3, per anchor and δ/h: distance to E2.3, conditioning."""
    band = case1().material
    rows = []
    for y in STENCIL_ANCHORS:
        xy = anchor_stencil(nodes, y)
        plain = interface_stencil(xy, band, 4, warp=False)
        p = plain.polynomial_block()
        w_ref = stencil_weights(xy, band, 4, warp=False)
        jump = seed_basis(xy, band)
        monomial = float(np.linalg.cond(polynomial_block(jump.xi, jump.eta, 4)))
        translated = float(np.linalg.cond(p))
        for ratio in (*LADDER_RATIOS, 0.0):
            sb = jump if ratio == 0.0 else seed_basis(xy, SmoothBand(band, ratio * h))
            w = seed_weights_with(sb, plain)
            raw, scaled = block_condition(sb.block)
            rows.append(
                {
                    "anchor": float(xy[0, 1]),
                    "ratio": ratio,
                    "span": span_distance(p, sb.block),
                    "weights": float(np.abs(w - w_ref).max() / np.abs(w_ref).max()),
                    "raw": raw,
                    "scaled": scaled,
                    "monomial": monomial,
                    "translated": translated,
                }
            )
    return rows


def timing_rows(nodes: NodeSet, h: float) -> list[dict]:
    """#35's acceptance line: one stencil's march over every crossing stencil."""
    band = case1().material
    index = crossing_stencils(nodes, band)
    rows = []
    t0 = time.perf_counter()
    for idx in index:
        stencil_weights(nodes.xy[idx], band, 4, warp=False)
    e23 = (time.perf_counter() - t0) / len(index)
    for ratio in (0.0, *CHAIN_RATIOS):
        medium = SmoothBand(band, ratio * h)
        times = []
        for idx in index:
            t = time.perf_counter()
            seed_basis(nodes.xy[idx], medium)
            times.append(time.perf_counter() - t)
        times = 1e3 * np.array(times)
        rows.append(
            {
                "ratio": ratio,
                "stencils": len(index),
                "median_ms": float(np.median(times)),
                "max_ms": float(times.max()),
                "total_s": float(times.sum() / 1e3),
                "e23_ms": 1e3 * e23,
            }
        )
    return rows


def print_stencil_study(tables: dict, n: int, h: float) -> None:
    print(
        f"\nthe scalar seeds on one stencil (E4.4, H1–H3): case 1, {n} nodes, "
        f"h = {h:.5f}, 30 / 4 stencils"
    )
    print(
        "\nH1, the march is the chain (anchor y = "
        f"{tables['chain_anchor']:.4f}): monomials = |S − ξᵃηᵇ| with α ≡ 0.37;"
        " shift = the identity g_j = C(a,j) g_0 of the lower seed, relative;"
        f" residual = L φ − α_e Σ C φ′ on {RESIDUAL_GRID[0]} × {RESIDUAL_GRID[1]},"
        f" order-{2 * RESIDUAL_HALF} differences; 1-D = the seeds of ηᵇ against"
        " E3.4's march on the y-profile; warp = |ψ₀₁ − α_e| / α_e"
    )
    print("    δ/h  monomials      shift   residual       1-D       warp  max |g|")
    for r in tables["chain"]:
        print(
            f"  {r['ratio']:5g}  {r['monomials']:9.1e}  {r['shift']:9.1e}"
            f"  {r['residual']:9.1e}  {r['one_d']:8.1e}  {r['warp']:9.1e}"
            f"  {r['largest']:7.3g}"
        )
    print(
        "\nH2 at δ = 0, every crossing stencil: span = sin of the largest principal"
        " angle between the seed block's span and E2.3's translated block's;"
        " weights = the seed solve with E2.3's plain Gaussians against"
        " stencil_weights(warp=False), max relative; warp ="
        " |φ₀₁ − E2.4's warped η| at the nodes"
    )
    print("  material    stencils  three-region      span   weights      warp")
    for r in tables["jump_limit"]:
        print(
            f"  {r['material']:<10}  {r['stencils']:8d}  {r['three_region']:12d}"
            f"  {r['span']:8.1e}  {r['weights']:8.1e}  {r['warp']:8.1e}"
        )
    print(
        "\nH2 for δ > 0 and H3: distance to E2.3 and the seed block's condition"
        " number (raw / columns scaled) per anchor; monomial and translated are the"
        " constant-α and E2.3 blocks on the same nodes"
    )
    by_anchor: dict[float, list[dict]] = {}
    for r in tables["ladder"]:
        by_anchor.setdefault(r["anchor"], []).append(r)
    for anchor, rows in by_anchor.items():
        print(
            f"  anchor y = {anchor:.4f}: cond monomial {rows[0]['monomial']:.1f},"
            f" translated {rows[0]['translated']:.1f}"
        )
        print("       δ/h      span   weights    cond raw  cond scaled")
        for r in rows:
            print(
                f"    {r['ratio']:6.3g}  {r['span']:8.2e}  {r['weights']:8.2e}"
                f"  {r['raw']:10.1f}  {r['scaled']:11.1f}"
            )
    print(
        "\none stencil's seed_basis (march, block, right-hand side) over every"
        " crossing stencil; E2.3's stencil_weights for scale"
    )
    print("    δ/h  stencils  median ms  max ms  total s  E2.3 ms")
    for r in tables["timing"]:
        print(
            f"  {r['ratio']:5g}  {r['stencils']:8d}  {r['median_ms']:9.2f}"
            f"  {r['max_ms']:6.2f}  {r['total_s']:7.2f}  {r['e23_ms']:7.2f}"
        )


def run_stencils(args) -> dict:
    """E4.4's tables: H1–H3 and the march's cost on the ``--stencil-n`` set."""
    t0 = time.perf_counter()
    nodes = build_node_set(
        case1(), args.stencil_n, seed=args.seed, iterations=args.iterations
    )
    h = 1.0 / round(0.95 * np.sqrt(args.stencil_n))
    xy = anchor_stencil(nodes, STENCIL_ANCHORS[0])
    tables = {
        "chain_anchor": float(xy[0, 1]),
        "chain": chain_rows(nodes, h, xy),
        "jump_limit": jump_limit_rows(nodes),
        "ladder": ladder_rows(nodes, h),
        "timing": timing_rows(nodes, h),
    }
    print_stencil_study(tables, args.stencil_n, h)
    print(f"\nstencil study {time.perf_counter() - t0:.1f} s")
    return {"stencils": tables}


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mode", choices=("all", "references", "naive", "stencils"), default="all"
    )
    parser.add_argument("--deltas", type=float, nargs="+", default=list(STUDY_DELTAS))
    parser.add_argument("--growth", type=float, nargs="+", default=list(GROWTH))
    parser.add_argument("--counts", type=int, nargs="+", default=list(KNEE_COUNTS))
    parser.add_argument("--t-end", type=float, default=T_END)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--spectrum-counts",
        type=int,
        nargs="*",
        default=list(SPECTRUM_COUNTS),
        help="counts of the growing-mode check (none skips it)",
    )
    parser.add_argument("--stencil-n", type=int, default=STENCIL_N)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    if args.mode in ("all", "naive"):
        if any(n < 300 for n in args.counts) or list(args.counts) != sorted(
            set(args.counts)
        ):
            parser.error("give increasing counts of 300 nodes or more")
        if 0.0 not in args.deltas:
            parser.error("the knee study needs δ = 0 (the jump line and the floors)")
    args.outputs.mkdir(parents=True, exist_ok=True)
    tables: dict = {}
    if args.mode in ("all", "references"):
        tables["references"] = reference_rows(args.deltas, args.growth)
        print_references(tables["references"])
    if args.mode in ("all", "naive"):
        tables.update(run_naive(args))
    if args.mode in ("all", "stencils"):
        tables.update(run_stencils(args))
    return tables


if __name__ == "__main__":
    main()
