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

``--mode seeds`` (E4.6, #37; stiff note §3.7 H4, H7 and H8, §4.5 records) is
the flat δ sweep: the seed operator against E4.3's two lines and the *direct*
operator (the blind smooth stencil, the one to beat where the grid resolves
the edge) at every (n, δ), elliptic and parabolic, against the same separable
references. ``--operators`` picks the lines (E4.3's two are reread from
``heat2d_stiff_knee.json``, not re-solved), ``--seed-reach`` the rule's reach,
which rides in the cache label with the warp so that E4.3's entries survive.
The tables: each line's RMS error per δ with its order per halving of h and
its own fit; the seeds over every other line at each (δ, n) with the seeded
rows, their share of N and the march per row; H8's penalty where ``h ≤ δ``
(the seeds against the direct operator, no threshold anywhere in the rule);
H7's warp as two lines of the sweep (``seeds-plain`` is the same march with
plain Gaussians, built from the same ``seed_basis`` so the ablation costs
weight solves and not marches); and H4's δ = 0 regression, where the seed
rows *are* E2.3's and the line must be port notes §2.4–2.5's.
Figure: ``heat2d_stiff_seeds.png``.

``--amplitude`` and ``--inside`` (E4.7, #38; stiff note §3.5 and H9, §4.6
records) move the same sweep onto another band: ``--amplitude 0.02`` is
case 2's sine pair with its ``0.2 + 0.1 sin 2πx sin 2πy`` inside (EABE eq.
35), and ``--inside constant`` on the curves or ``--inside sine`` on flat lines
split case 2's two departures from case 1, the curvature and the tangential
variation of alpha. The seeds are route (a) (the flat seeds along the foot
point's true normal), the widths default to ``CURVED_DELTAS``, the reference
is the sheared product grid (``ProductGridReference``) at every δ, and the
numbers go to their own ``CURVED_CACHE``. Beside E4.6's tables: H9's
truncation probe (each line's rows on the equilibrium reference, over the
seeded, crossing and bulk rows; case 1 prints it too for entries built since
E4.7) and each line's curved error over case 1's at equal (δ, n), from
``KNEE_CACHE``. ``--mode references --amplitude 0.02`` checks the product grid
against finer grids in both directions and, at δ = 0, E2.6's 160,000-node run
against it. Figure: ``heat2d_stiff_seeds_<tag>.png``.

``--operators … tangential tangential-plain`` (E4.11, #81; stiff note §3.10
H13–H17, §4.7 records) adds the tangential chain to the same sweep: the seeds
in the foot curve's own coordinates with alpha's and the metric's variation
along it carried by coupled levels. With it the tangential line is the sweep's
main one (the ratios, the warp, H8, the δ = 0 comparison with E2.3, the probe),
its curved errors are read over case 1's ``seeds`` at equal (δ, n) (the two
are the same rows on flat lines), and the figure is
``heat2d_stiff_tangential_<tag>.png``, route (a)'s left as it was.
``--mode tangential`` prints the chain's own tables: H14, the crossing rows'
truncation on ``RingMode`` through two concentric circles (E2.3, route (a),
tangential, from 2500 nodes), H15's distance between the tangential span and
E2.3's translated basis on case 2 at δ = 0, H6's twin on case 2 (the seed
operators' interior spectra at 1600 nodes per δ), and H17's timing (the median
``seed_basis`` of 400 seeded rows per δ at 10,000 nodes, both chains).

    uv run python scripts/heat2d_stiff.py              # 2.5 min cold, 21 s cached
    uv run python scripts/heat2d_stiff.py --mode naive \
        --counts 1250 2500 5000 10000 20000 40000 80000 160000   # 58 min once
    uv run python scripts/heat2d_stiff.py --mode naive --seed 1 \
        --counts 1250 2500 5000 10000 20000 --spectrum-counts   # the scatter
    uv run python scripts/heat2d_stiff.py --mode references --deltas 0 0.001 0.0005
    uv run python scripts/heat2d_stiff.py --mode stencils       # 21 s
    uv run python scripts/heat2d_stiff.py --mode seeds \
        --counts 1250 2500 5000 10000 20000 40000               # 32 min once
    uv run python scripts/heat2d_stiff.py --mode seeds --deltas 0 \
        --operators naive construction seeds \
        --counts 1250 2500 5000 10000 20000 40000 80000 160000  # H4 to the end
    uv run python scripts/heat2d_stiff.py --mode references --amplitude 0.02  # 67 s
    uv run python scripts/heat2d_stiff.py --mode seeds --amplitude 0.02 \
        --counts 1250 2500 5000 10000 20000 40000               # case 2, once
    uv run python scripts/heat2d_stiff.py --mode seeds --amplitude 0.02 \
        --inside constant --deltas 0 0.0025 --operators naive construction seeds \
        --counts 1250 2500 5000 10000 20000 40000               # the curvature alone
    uv run python scripts/heat2d_stiff.py --mode seeds --inside sine \
        --deltas 0 0.0025 --operators naive construction seeds \
        --counts 1250 2500 5000 10000 20000 40000               # alpha along the edge
    uv run python scripts/heat2d_stiff.py --mode seeds --amplitude 0.02 \
        --operators naive construction direct direct-reach seeds seeds-plain \
        tangential tangential-plain \
        --counts 1250 2500 5000 10000 20000 40000               # E4.11 on case 2
    uv run python scripts/heat2d_stiff.py --mode seeds --amplitude 0.02 \
        --inside constant --deltas 0 0.0025 --operators naive construction seeds \
        tangential tangential-plain --counts 1250 2500 5000 10000 20000 40000
    uv run python scripts/heat2d_stiff.py --mode seeds --inside sine \
        --deltas 0 0.0025 --operators naive construction seeds tangential \
        tangential-plain --counts 1250 2500 5000 10000 20000 40000   # E4.11's A, B
    uv run python scripts/heat2d_stiff.py --mode seeds --inside sine --deltas 0 \
        --operators naive construction seeds tangential tangential-plain \
        --counts 1250 2500 5000 10000 20000 40000 80000 160000  # B's δ = 0, 23 min
    uv run python scripts/heat2d_stiff.py --mode tangential \
        --counts 1250 2500 5000 10000 20000 40000               # H14, H15, spectra
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.linalg import subspace_angles  # noqa: E402

from heat_interfaces.fd_weights import fornberg_weights  # noqa: E402
from heat_interfaces.heat1d.domain import TANH_REACH  # noqa: E402
from heat_interfaces.heat1d.march import bd4_amplification  # noqa: E402
from heat_interfaces.heat1d.stiff import seed_profiles as seed_profiles_1d  # noqa: E402
from heat_interfaces.heat2d import (  # noqa: E402
    BOUNDARY,
    GA_SHAPE,
    INTERFACE_KIND,
    INTERIOR_KIND,
    PRODUCT_N_X,
    PRODUCT_ORDERING,
    REFERENCE_MAX_WIDTH,
    REFERENCE_N_CHEB,
    ROW_OFFSETS,
    STRIP,
    Band,
    Circle,
    Constant2D,
    Domain,
    FlatLine,
    NodeSet,
    ProductGridReference,
    Reference,
    RingMode,
    Row,
    SeedBasis,
    SineGraph,
    SineProduct,
    SmoothBand,
    Stencils,
    augmented_solve,
    block_condition,
    build_node_set,
    build_stencils,
    case1,
    case1_exact,
    case1_reference,
    case2,
    control_exact,
    direct_operator,
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
    seed_weights,
    seeded_rows,
    solve_equilibrium,
    stencil_weights,
    weights_of,
)
from heat_interfaces.plotting import AWARE, CONSTRUCTION, NAIVE, REFERENCE  # noqa: E402

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
"""The default counts of both sweeps; E4.3's documented run adds 20,000, 40,000
and 80,000, E4.6's 20,000 and 40,000."""

T_END = 0.1
"""Where E2.5 (and dissertation Fig. 5-5) reads the parabolic error."""

PROBLEMS = {"elliptic": 0.0, "parabolic": 1.0}
"""Problem name to ``c``: the equilibrium, and E2.5's ``e^{t}`` mode."""

OPERATORS = ("naive", "construction")
"""The two lines of the knee table; ``uniform`` is the α ≡ 1 run beside them."""

SWEEP_LABELS = (
    "naive",
    "construction",
    "direct",
    "direct-reach",
    "seeds",
    "seeds-plain",
)
"""E4.6's six lines, in the order the tables print them: E4.3's two, the blind
smooth operator H8 measures the seeds against, that operator on the *seeds' own*
stencil groups (30 / 4 wherever the rule seeds, 42 / 5 elsewhere: the same
matrix as the seeds with the marched rows taken out, which is what separates
the seed rows from their smaller stencil), and the seeds with the warp on and
off (H7). ``naive`` and ``construction`` are E4.3's cached entries, reread."""

EXTRA_LABELS = ("construction-flat", "tangential", "tangential-plain")
"""Lines ``--operators`` offers beyond the default six: E2.3 with its interface
expansion off (``curvature=False``, EABE Fig. 10's "linear interface"), which is
``construction`` bit for bit on flat lines and the δ = 0 limit route (a) is
measured against on a curved one (E4.7, §4.6); and E4.11's tangential chain
(§3.10), warped and plain, the seeds in the foot curve's own coordinates."""

SEEDS_PLAIN = "#7fb3d5"
"""The seeds' ablation: the same blue as the seeds, lighter (``plotting``'s key,
and ``heat2d_stiff_eigenvalues.py``'s)."""

DIRECT_REACH = "#bcbcbc"
"""The direct operator on the seeds' stencils: the same grey, lighter."""

CONSTRUCTION_FLAT = "#c2a5cf"
"""E2.3 with the flat interface: the construction's purple, lighter."""

TANGENTIAL = "#17becf"
"""The tangential seeds (E4.11): a cyan beside the seeds' blue, which the dataviz
validator separates from it, the naive orange and the construction's purple
(normal-vision ΔE ≥ 20, colour-blind ΔE ≥ 18); every line keeps its marker."""

TANGENTIAL_PLAIN = "#9edae5"
"""Their plain-Gaussian ablation, lighter, as ``SEEDS_PLAIN`` is the seeds'; the
tangential figure draws it only as the warp panel's ratio."""

STYLE = {
    "naive": (NAIVE, "o"),
    "construction": (CONSTRUCTION, "s"),
    "direct": (REFERENCE, "d"),
    "direct-reach": (DIRECT_REACH, "*"),
    "seeds": (AWARE, "^"),
    "seeds-plain": (SEEDS_PLAIN, "v"),
    "construction-flat": (CONSTRUCTION_FLAT, "x"),
    "tangential": (TANGENTIAL, "P"),
    "tangential-plain": (TANGENTIAL_PLAIN, "X"),
}
"""Colour and marker per line, E4.5's driver's."""

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

INSIDE = {"constant": Constant2D(0.2), "sine": SineProduct(0.2, 0.1)}
"""The band's piece: case 1's constant, or case 2's ``0.2 + 0.1 sin 2πx sin 2πy``."""

CASE2_AMPLITUDE = 0.02
"""Case 2's sine pair ``c + 0.02 sin 2πx`` (EABE eq. 35)."""

CURVED_DELTAS = (0.0, 0.01, 0.005, 0.0025)
"""The curved sweep's widths (plan E4.7): the jump and E4.3's three narrowest."""

CURVED_CACHE = "heat2d_stiff_curved.json"
CURVED_CACHE_META = {
    "study": "E4.7 sine bands with tanh edges against the product-grid references",
    "version": 2,
    "reference": ["product grid", PRODUCT_N_X, REFERENCE_N_CHEB, REFERENCE_MAX_WIDTH],
    "naive_ordering": PRODUCT_ORDERING,
}
"""Its own file, so that nothing here can move E4.3's and E4.6's case-1 entries.

Version 2 is E4.11's tangential chain with every seed to level 4 (#81): a file
from before it may hold the first level cutoff's ``tangential`` entries under
the same labels, and is refused whole. Bump it after any change to either
chain, the tangential series or their sampling, as ``KNEE_CACHE_META`` says."""

CASE2_REFERENCE = "heat2d_case2_reference_n160000_seed0.npz"
"""E2.6's cached 160,000-node jump-aware run, which the δ = 0 product grid checks."""

CHECK_N_X = 65
"""The finer Fourier grid the product-grid reference is checked against."""


@dataclass(frozen=True)
class Geometry:
    """The band a sweep runs on: flat or sine interfaces, and which piece inside.

    ``Geometry()`` is case 1 (flat lines, the constant 0.2) and keeps
    everything E4.3–E4.6 built: the separable references, ``KNEE_CACHE`` and
    its keys, the straddling-row diagnostics. ``Geometry(0.02, "sine")`` is
    case 2 (EABE eq. 35), and the two mixed ones split case 2's two
    departures from case 1 (stiff note §4.6): the sine pair with the constant
    piece (the curvature alone) and the flat lines with case 2's piece (the
    tangential variation of alpha alone). Every geometry but case 1 is read
    against the product grid and cached in ``CURVED_CACHE`` under its
    ``tag``.
    """

    amplitude: float = 0.0
    inside: str = "constant"

    def __post_init__(self) -> None:
        if self.inside not in INSIDE:
            raise ValueError(f"inside must be one of {sorted(INSIDE)}")
        if not self.amplitude >= 0.0:
            raise ValueError("the amplitude must be non-negative")

    @property
    def is_case1(self) -> bool:
        return self.amplitude == 0.0 and self.inside == "constant"

    @property
    def tag(self) -> str:
        return "" if self.is_case1 else f"a{self.amplitude:g} {self.inside}"

    @property
    def name(self) -> str:
        if self.is_case1:
            return "case 1"
        if self.amplitude == CASE2_AMPLITUDE and self.inside == "sine":
            return "case 2"
        curves = (
            "flat lines"
            if self.amplitude == 0.0
            else f"sine pair a = {self.amplitude:g}"
        )
        return f"{curves}, {self.inside} inside"

    def domain(self) -> Domain:
        if self.is_case1:
            return case1()
        if self.amplitude == 0.0:
            curves = (FlatLine(0.6), FlatLine(0.8))
        else:
            curves = (SineGraph(0.6, self.amplitude), SineGraph(0.8, self.amplitude))
        band = Band(*curves, INSIDE[self.inside], Constant2D(1.0))
        return Domain(band, band.interfaces, STRIP)

    def reference(self, delta: float, growth: float):
        """The separable reference on case 1, the product grid elsewhere; memoised."""
        key = (self, delta, growth)
        if key not in _REFERENCES:
            if self.is_case1:
                _REFERENCES[key] = case1_reference(delta, growth)
            else:
                medium = SmoothBand(self.domain().material, delta)
                _REFERENCES[key] = ProductGridReference(medium, growth)
        return _REFERENCES[key]

    def cache(self) -> tuple[str, dict]:
        return (
            (KNEE_CACHE, KNEE_CACHE_META)
            if self.is_case1
            else (
                CURVED_CACHE,
                CURVED_CACHE_META,
            )
        )


_REFERENCES: dict = {}
"""A run's references by (geometry, δ, c): a product grid costs a second or two."""

CASE1 = Geometry()


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


def curved_reference_rows(
    geometry: Geometry,
    deltas: Sequence[float],
    growth: Sequence[float],
    outputs: Path,
    points: int = 4000,
) -> list[dict[str, float]]:
    """E4.7's reference checks per (δ, c), on ``points`` random points of the strip.

    The product grid's elements, unknowns and build time; its agreement with
    the ``CHECK_N_X``-point Fourier grid and with ``CHECK_N_CHEB`` nodes on
    ``CHECK_MAX_WIDTH``-wide elements (max difference); its distance from
    the δ = 0 grid, the floor the δ = 0 construction sits on; and at δ = 0,
    ``c = 0`` on case 2, the distance of E2.6's cached 160,000-node
    jump-aware run from it at that run's own nodes (``CASE2_REFERENCE``,
    skipped if it is not under ``outputs``), which measures that run's error.
    """
    rng = np.random.default_rng(0)
    px, py = rng.uniform(0.0, 1.0, points), rng.uniform(0.0, 1.0, points)
    material = geometry.domain().material
    rows = []
    for c in growth:
        base = geometry.reference(0.0, c)(px, py)
        for delta in deltas:
            ref = geometry.reference(delta, c)
            medium = SmoothBand(material, delta)
            u = ref(px, py)
            fine_x = ProductGridReference(medium, c, n_x=CHECK_N_X)
            fine_eta = ProductGridReference(
                medium, c, n_cheb=CHECK_N_CHEB, max_width=CHECK_MAX_WIDTH
            )
            row = {
                "delta": delta,
                "growth": c,
                "elements": ref.elements,
                "unknowns": ref.unknowns,
                "seconds": ref.seconds,
                "n_x": float(np.abs(fine_x(px, py) - u).max()),
                "elements_check": float(np.abs(fine_eta(px, py) - u).max()),
                "distance": float(np.abs(u - base).max()),
            }
            path = outputs / CASE2_REFERENCE
            if (
                delta == 0.0
                and c == 0.0
                and geometry.name == "case 2"
                and path.exists()
            ):
                run = Reference.load(path)
                gap = run.u - ref(run.nodes.x, run.nodes.y)
                row["e26_rms"] = float(np.sqrt(np.mean(gap**2)))
                row["e26_max"] = float(np.abs(gap).max())
            rows.append(row)
    return rows


def print_curved_references(rows: list[dict[str, float]], geometry: Geometry) -> None:
    print(
        f"{geometry.name}: the product-grid reference (sheared Fourier × Chebyshev"
        f" elements), {PRODUCT_N_X} points in x, {REFERENCE_N_CHEB} nodes per element,"
        f" elements <= {REFERENCE_MAX_WIDTH}"
    )
    print(
        f"  checks: max difference against {CHECK_N_X} points in x and against"
        f" {CHECK_N_CHEB} nodes on {CHECK_MAX_WIDTH}-wide elements; distance ="
        " max |u − u at δ = 0|"
    )
    print("        δ    c  elements  unknowns      s   vs n_x   vs elements   distance")
    for r in rows:
        print(
            f"  {r['delta']:7.4f}  {r['growth']:3.1f}  {r['elements']:8d}"
            f"  {r['unknowns']:8d}  {r['seconds']:5.2f}  {r['n_x']:7.1e}"
            f"  {r['elements_check']:12.1e}  {r['distance']:9.3e}"
        )
    for r in rows:
        if "e26_rms" in r:
            print(
                "  E2.6's 160,000-node jump-aware run against the δ = 0 grid at its"
                f" own nodes: RMS {r['e26_rms']:.3e}, max {r['e26_max']:.3e}"
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
    tag: str = "",
) -> str:
    """The cache key of one line's numbers at one (problem, δ, n).

    ``tag`` is a non-case-1 geometry's (``Geometry.tag``); case 1's keys are
    E4.3's, unchanged.
    """
    d = "-" if delta is None else f"{delta:g}"
    t = f" t{t_end:g}" if PROBLEMS[problem] else ""
    key = f"{problem}{t} d{d} n{n} s{seed} i{iterations} {label}"
    return f"{tag} {key}" if tag else key


def load_knee_cache(outputs: Path, geometry: Geometry = CASE1) -> dict[str, dict]:
    """The cached numbers; empty when the file is missing or from another study."""
    name, meta = geometry.cache()
    path = outputs / name
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if data.get("meta") != meta:
        return {}
    return dict(data["entries"])


def save_knee_cache(
    outputs: Path, cache: dict[str, dict], geometry: Geometry = CASE1
) -> None:
    name, meta = geometry.cache()
    data = {"meta": meta, "entries": dict(sorted(cache.items()))}
    (outputs / name).write_text(json.dumps(data, indent=1) + "\n")


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


def seed_label(
    warp: bool = True, reach: float = TANH_REACH, tangential: bool = False
) -> str:
    """The cache label of one seed line: the chain, the warp, the reach if not 20 δ.

    Everything the seed rows depend on beyond (problem, δ, n, seed,
    iterations) rides in the label, so that ``KNEE_CACHE_META`` — and with
    it E4.3's naive and construction entries to 160,000 nodes — survives
    (§3.8's cache trap, and §4.4's "a flag that moves the numbers and not
    the key gives the earlier run's answer back in silence").
    """
    name = "tangential" if tangential else "seeds"
    name = name if warp else f"{name}-plain"
    return name if reach == TANH_REACH else f"{name}-r{reach:g}"


def seed_operators(
    nodes: NodeSet,
    medium: SmoothBand,
    stencils: Stencils,
    warps: Sequence[bool],
    reach: float = TANH_REACH,
    shape: float = GA_SHAPE,
    tangential: bool = False,
) -> tuple[dict[bool, sp.csr_array], int]:
    """``({warp: L}, the seeded rows)``: ``seed_operator``'s loop, one march per row.

    ``operators.seed_operator`` marches a row and solves for its weights; H7
    wants the same rows with plain Gaussians, and the march — 2.4–8.3 ms a
    row, everything the operator costs — does not depend on the warp. So the
    ablation is built here from one ``seed_basis`` per row and one
    ``weights_of`` per warp, which halves the sweep's marches. With a single
    warp it is ``seed_operator`` exactly (a test pins that). ``tangential``
    marches §3.10's chain (E4.11).
    """
    # Unlike ``seed_operator`` this takes no ``region_index`` shortcut for a
    # material without interfaces: the sweep only ever passes a ``SmoothBand``.
    ops = {w: direct_operator(nodes, medium, stencils, shape).tolil() for w in warps}
    seeded = 0
    for g in stencils.groups:
        if g.kind != INTERFACE_KIND:
            continue
        seen = seeded_rows(nodes, medium, g.index, reach)
        for row, idx in zip(g.rows[seen], g.index[seen], strict=True):
            sb = seed_basis(nodes.xy[idx], medium, g.spec.degree, tangential=tangential)
            seeded += 1
            for warp, op in ops.items():
                op[row, :] = 0.0
                op[row, idx] = weights_of(sb, shape, warp)
    return {w: op.tocsr() for w, op in ops.items()}, seeded


def sweep_operators(
    labels: Sequence[str],
    nodes: NodeSet,
    medium: SmoothBand,
    groups: dict[str, Stencils],
    reach: float = TANH_REACH,
    domain: Domain | None = None,
) -> dict[str, tuple[sp.csr_array, str | None, dict[str, float]]]:
    """``{label: (L, SuperLU's ordering, the readings of the build)}``, built once.

    One operator per (n, δ) for both problems: the elliptic solve and the
    parabolic march see the same matrix, and at δ = 0.04 a 40,000-node seed
    operator is three minutes of marches (§4.4), so building it per problem
    would double the sweep. ``groups`` carries the plain and crossing
    stencils of this node set; the reach group depends on δ and is built
    here. The readings are the rows the method recomputes and the seconds
    the build took, which the tables quote. ``domain`` is the geometry's
    (case 1 by default); only its material is replaced.
    """
    families = {
        chain: [w for w in (True, False) if seed_label(w, reach, chain) in labels]
        for chain in (False, True)
    }
    marched = {seed_label(w, reach, c) for c in (False, True) for w in (True, False)}
    built: dict[str, tuple[sp.csr_array, str | None, dict[str, float]]] = {}
    seed_group: list[Stencils] = []

    def stencils_of_the_rule() -> Stencils:
        """The reach group of this δ, built once: what the seed operator wants."""
        if not seed_group:
            smooth = replace(domain or case1(), material=medium)
            seed_group.append(
                build_stencils(nodes, smooth, interface=BOUNDARY, reach=reach)
            )
        return seed_group[0]

    def seeded_count(stencils: Stencils) -> int:
        return int(
            sum(
                seeded_rows(nodes, medium, g.index, reach).sum()
                for g in stencils.groups
                if g.kind == INTERFACE_KIND
            )
        )

    for label in labels:
        if label in marched:
            continue
        t0 = time.perf_counter()
        rows, permc = 0, None
        if label == "naive":
            op = naive_operator(nodes, medium, groups["plain"])
            permc = PRODUCT_ORDERING
        elif label == "direct":
            op = direct_operator(nodes, medium, groups["plain"])
        elif label == "direct-reach":
            stencils = stencils_of_the_rule()
            op = direct_operator(nodes, medium, stencils)
            rows = seeded_count(stencils)
        elif label in ("construction", "construction-flat"):
            curvature = label == "construction"
            op = interface_aware_operator(
                nodes, medium, groups["crossing"], curvature=curvature
            )
            rows = int(
                sum(
                    interface_crossings(nodes, medium, g.index).sum()
                    for g in groups["crossing"].groups
                    if g.kind == INTERFACE_KIND
                )
            )
        else:
            known = (*SWEEP_LABELS, *EXTRA_LABELS)
            raise ValueError(f"unknown line {label!r}; one of {known}")
        built[label] = (op, permc, {"rows": rows, "seconds": time.perf_counter() - t0})
    for chain, warps in families.items():
        if not warps:
            continue
        t0 = time.perf_counter()
        ops, seeded = seed_operators(
            nodes, medium, stencils_of_the_rule(), warps, reach, tangential=chain
        )
        seconds = (time.perf_counter() - t0) / len(warps)
        for warp in warps:
            built[seed_label(warp, reach, chain)] = (
                ops[warp],
                None,
                {"rows": seeded, "seconds": seconds},
            )
    return built


def probe_masks(
    nodes: NodeSet, domain: Domain, crossing: Stencils, reach: float = TANH_REACH
) -> dict[str, np.ndarray]:
    """The row sets of the truncation probe at one (n, δ) (stiff note §4.6).

    ``seeded``: the rows the seed rule marches, the same set for every line
    (for the others, the rows the seeds would rebuild, the companion's
    convention); ``crossing``: E2.3's rows, whose 30 nodes straddle a curve;
    ``bulk``: the reach stencils' interior group, 42 / 5 rows that neither see
    an edge nor sit in the boundary zone, so that the set holds one kind of
    row at every (n, δ) (empty where 20 δ covers the strip).
    """
    medium = domain.material
    stencils = build_stencils(nodes, domain, interface=BOUNDARY, reach=reach)
    seeded = np.zeros(nodes.n, dtype=bool)
    bulk = np.zeros(nodes.n, dtype=bool)
    for g in stencils.groups:
        if g.kind == INTERFACE_KIND:
            seeded[g.rows[seeded_rows(nodes, medium, g.index, reach)]] = True
        elif g.kind == INTERIOR_KIND:
            bulk[g.rows] = True
    across = np.zeros(nodes.n, dtype=bool)
    for g in crossing.groups:
        if g.kind == INTERFACE_KIND:
            across[g.rows[interface_crossings(nodes, medium, g.index)]] = True
    return {"seeded": seeded, "crossing": across, "bulk": bulk}


def truncation_probe(
    op: sp.sparray, u: np.ndarray, masks: dict[str, np.ndarray]
) -> dict[str, float]:
    """RMS of ``L u`` per row set, ``u`` the equilibrium reference at the nodes.

    ``L u = 0`` exactly, so each row's value is its local truncation error on
    the true solution: H9's probe. Absolute, since there is no rate to divide
    by; the tables set the seeded rows against the bulk rows of the same
    operator.
    """
    r = op @ u
    return {
        f"probe_{name}": float(np.sqrt(np.mean(r[mask] ** 2))) if mask.any() else 0.0
        for name, mask in masks.items()
    }


def rms_and_max(ref, u: np.ndarray, nodes: NodeSet, t: float) -> dict[str, float]:
    """The two readings of ``edge_diagnostics`` that do not assume a separable mode."""
    exact = ref(nodes.x, nodes.y, t)
    return {"rms": rms_error(u, exact), "max": float(np.abs(u - exact).max())}


def knee_sweep(
    counts: Sequence[int],
    deltas: Sequence[float],
    cache: dict[str, dict[str, float]],
    seed: int = 0,
    iterations: int = 100,
    t_end: float = T_END,
    problems: Sequence[str] = tuple(PROBLEMS),
    labels: Sequence[str] = OPERATORS,
    reach: float = TANH_REACH,
    save: Callable[[], None] | None = None,
    geometry: Geometry = CASE1,
) -> dict[str, dict[float, list[dict]]]:
    """``results[problem][δ]``: one row per count, the ``labels``, floor and uniform.

    A row holds ``n``, ``h``, ``h_over_delta``, ``floor`` (the RMS of the δ and
    δ = 0 references' difference at the nodes), ``uniform`` (the naive α ≡ 1
    run's RMS error against ``control_exact``) and, per line in ``labels``,
    the ``edge_diagnostics`` under ``"<label>/<quantity>"`` beside the
    build's ``rows`` and ``seconds``, and, when δ = 0 is swept and the naive
    line is on, ``naive/vs_jump``: the naive RMS error over the jump's on the
    same node set.
    What ``cache`` lacks is computed and added; a count whose every entry is
    cached builds no node set, and an operator is built once per (n, δ) for
    both problems. ``save`` is called after each count that computed
    something, so that a sweep interrupted at 40,000 nodes keeps the hours
    below it.

    ``geometry`` other than case 1 (E4.7) reads the product grid instead of
    the separable reference, keeps only the RMS and max errors (the
    straddling-row readings assume the separable mode), and keys its entries
    by its ``tag``. Every operator built here also carries the truncation
    probe on the elliptic entry (``probe_seeded``, ``probe_crossing``,
    ``probe_bulk``); case-1 entries cached before E4.7 have none.
    """
    domain = geometry.domain()
    results: dict[str, dict[float, list[dict]]] = {
        p: {d: [] for d in deltas} for p in problems
    }
    for n in counts:

        def key(problem, delta, label, n=n):
            return knee_key(
                problem, delta, n, label, seed, iterations, t_end, geometry.tag
            )

        needed = [key(p, None, "uniform") for p in problems] + [
            key(p, d, label)
            for p in problems
            for d in deltas
            for label in ("floor", *labels)
        ]
        if any(k not in cache for k in needed):
            t0 = time.perf_counter()
            nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
            groups = {
                "plain": build_stencils(nodes, domain),
                "crossing": build_stencils(nodes, domain, interface=BOUNDARY),
            }
            for problem in problems:
                k = key(problem, None, "uniform")
                if k not in cache:
                    op = naive_operator(nodes, Constant2D(1.0), groups["plain"])
                    ref = control_exact(PROBLEMS[problem])
                    u, t = _solve(problem, op, nodes, ref, t_end, PRODUCT_ORDERING)
                    cache[k] = {"rms": rms_error(u, ref(nodes.x, nodes.y, t))}
            for delta in deltas:
                medium = SmoothBand(domain.material, delta)
                refs = {p: geometry.reference(delta, PROBLEMS[p]) for p in problems}
                for problem in problems:
                    k = key(problem, delta, "floor")
                    if k not in cache:
                        c = PROBLEMS[problem]
                        t = t_end if c else 0.0
                        base = geometry.reference(0.0, c)
                        cache[k] = {
                            "h": nodes.h,
                            "floor": rms_error(
                                base(nodes.x, nodes.y, t),
                                refs[problem](nodes.x, nodes.y, t),
                            ),
                        }
                pending = {
                    label: [p for p in problems if key(p, delta, label) not in cache]
                    for label in labels
                }
                pending = {label: p for label, p in pending.items() if p}
                if not pending:
                    continue
                built = sweep_operators(
                    list(pending), nodes, medium, groups, reach, domain
                )
                if "elliptic" in problems:
                    smooth = replace(domain, material=medium)
                    masks = probe_masks(nodes, smooth, groups["crossing"], reach)
                    steady = refs["elliptic"](nodes.x, nodes.y)
                for label, left in pending.items():
                    op, permc, readings = built[label]
                    probe = (
                        truncation_probe(op, steady, masks)
                        if "elliptic" in left
                        else {}
                    )
                    for problem in left:
                        u, t = _solve(problem, op, nodes, refs[problem], t_end, permc)
                        if geometry.is_case1:
                            diagnostics = edge_diagnostics(
                                nodes, medium, refs[problem], u, t
                            )
                        else:
                            diagnostics = rms_and_max(refs[problem], u, nodes, t)
                        extra = probe if problem == "elliptic" else {}
                        cache[key(problem, delta, label)] = {
                            **diagnostics,
                            **readings,
                            **extra,
                        }
            print(f"  (n = {n}: {time.perf_counter() - t0:.1f} s)", flush=True)
            if save is not None:
                save()
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
                for label in labels:
                    for q, value in cache[key(problem, delta, label)].items():
                        row[f"{label}/{q}"] = value
                if 0.0 in deltas and "naive" in labels:
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
        args.counts,
        args.deltas,
        cache,
        args.seed,
        args.iterations,
        args.t_end,
        save=lambda: save_knee_cache(args.outputs, cache),
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


# --- E4.6: the flat δ sweep ---------------------------------------------------------


def _fit(rows: list[dict], name: str) -> float:
    """The least-squares order of ``name`` against ``h`` over a line's counts.

    The rate columns are per halving of h between neighbours; the fit is the
    line's own slope, which is what §2 quotes ("fit 4.77") and what a
    pre-asymptotic first count cannot hide.
    """
    usable = [r for r in rows if r.get(name, 0.0) > 0.0]
    if len(usable) < 2:
        return float("nan")
    h = np.log([r["h"] for r in usable])
    e = np.log([r[name] for r in usable])
    return float(np.polyfit(h, e, 1)[0])


def print_sweep(
    results: dict[float, list[dict]], labels: Sequence[str], title: str
) -> None:
    """Per δ: every line's RMS error (order per halving of h), and the line's fit."""
    print(f"\n{title}")
    header = "     n       h |" + "".join(f" {label:<16} |" for label in labels)
    for delta, rows in results.items():
        name = "δ = 0 (jump)" if delta == 0.0 else f"δ = {delta:g}"
        span = (
            ""
            if delta == 0.0
            else f"   h/δ {rows[0]['h_over_delta']:.2f}"
            f" … {rows[-1]['h_over_delta']:.2f}"
        )
        print(f"  {name}{span}")
        print(f"  {header}")
        columns = {label: _with_rates(rows, f"{label}/rms") for label in labels}
        for i, row in enumerate(rows):
            line = f"  {row['n']:6d}  {row['h']:.4f} |"
            for label in labels:
                line += f" {columns[label][i]:<16} |"
            print(line)
        line = "     fit          |"
        for label in labels:
            line += f" {_fit(rows, f'{label}/rms'):16.2f} |"
        print(line)


def sweep_ratios(
    results: dict[float, list[dict]], labels: Sequence[str], seeds: str
) -> list[dict]:
    """Per (δ, n): the seeds' RMS over every other line's, and what the rows cost."""
    rows = []
    for delta, lines in results.items():
        for r in lines:
            row = {
                "delta": delta,
                "n": r["n"],
                "h_over_delta": r["h_over_delta"],
                "rows": r.get(f"{seeds}/rows", float("nan")),
                "seconds": r.get(f"{seeds}/seconds", float("nan")),
            }
            row["fraction"] = row["rows"] / r["n"]
            row["ms"] = 1e3 * row["seconds"] / max(row["rows"], 1)
            for label in labels:
                if label != seeds and f"{label}/rms" in r:
                    row[f"over/{label}"] = r[f"{seeds}/rms"] / r[f"{label}/rms"]
            rows.append(row)
    return rows


def print_ratios(rows: list[dict], labels: Sequence[str], title: str) -> None:
    """The seeds against the comparators at every (δ, n), with the rows they cost."""
    print(f"\n{title}")
    header = "      δ       n    h/δ |"
    for label in labels:
        header += f" ÷ {label:<14} |"
    print(header + "  seeded rows    of N   ms a row")
    for r in rows:
        ratio = "   jump" if r["delta"] == 0.0 else f"{r['h_over_delta']:7.2f}"
        line = f"  {r['delta']:6.4f}  {r['n']:6d} {ratio} |"
        for label in labels:
            value = r.get(f"over/{label}")
            line += "                 |" if value is None else f" {value:16.3g} |"
        print(line + f"  {r['rows']:11.0f}  {r['fraction']:6.3f}  {r['ms']:9.1f}")


def print_seed_diagnostics(
    results: dict[float, list[dict]], seeds: str, title: str
) -> None:
    """The seeded solution on the straddling rows, E4.3's readings, naive beside.

    E4.3 found the naive flux on the innermost pair a function of ``h/δ``
    alone while the edge is unresolved (0.31–0.86 of it, never converging);
    the seeds' column is where that plateau is supposed to be absent.
    """
    print(f"\n{title}")
    print(
        "      δ       n    h/δ |  seeds: RMS       max   profile      flux"
        "      jump |  naive: flux      jump"
    )
    for delta, rows in results.items():
        for r in rows:
            ratio = "   jump" if delta == 0.0 else f"{r['h_over_delta']:7.2f}"
            line = f"  {delta:6.4f}  {r['n']:6d} {ratio} |"
            line += "".join(f"  {r[f'{seeds}/{q}']:8.2e}" for q in QUANTITIES)
            if "naive/flux" in r:
                line += f" |  {r['naive/flux']:10.2e}  {r['naive/jump']:8.2e}"
            print(line)


def print_resolved(rows: list[dict], seeds: str, title: str) -> None:
    """H8: where the grid resolves the edge (h ≤ δ), the seeds against the two ends.

    ``direct-reach`` is the control that matters: the same stencil groups as
    the seed operator — 30 / 4 wherever the rule seeds, 42 / 5 elsewhere —
    with the marched rows replaced by plain direct ones. The gap to
    ``direct`` is the smaller stencil's (the bulk runs 42 / 5 at degree 5,
    the seeded rows 30 / 4 at degree 4); the gap to ``direct-reach`` is the
    seeds' own.
    """
    print(f"\n{title}")
    control = "direct-reach/rms" in rows[0]
    print(
        "      δ       n    δ/h |  seeds     direct     same stencils      naive |"
        "  ÷ direct  ÷ same stencils   ÷ naive |  seeded rows of N"
    )
    for r in rows:
        same = r.get("direct-reach/rms", float("nan"))
        ratio = r[f"{seeds}/rms"] / same if control else float("nan")
        print(
            f"  {r['delta']:6.4f}  {r['n']:6d} {1.0 / r['h_over_delta']:6.2f} |"
            f" {r[f'{seeds}/rms']:9.2e} {r['direct/rms']:9.2e} {same:17.2e}"
            f" {r['naive/rms']:10.2e} |"
            f" {r['over/direct']:9.3f} {ratio:16.3f} {r['over/naive']:9.3f} |"
            f" {r['fraction']:17.3f}"
        )


def resolved_rows(
    results: dict[float, list[dict]], ratios: list[dict], seeds: str
) -> list[dict]:
    """The (δ, n) of ``results`` whose spacing resolves the edge: ``h ≤ δ``, δ > 0."""
    by = {(r["delta"], r["n"]): r for r in ratios}
    rows = []
    for delta, lines in results.items():
        for r in lines:
            if delta == 0.0 or r["h_over_delta"] > 1.0:
                continue
            if "direct/rms" not in r or "naive/rms" not in r:
                continue
            rows.append({**r, "delta": delta, **by[(delta, r["n"])]})
    return rows


def print_warp(
    results: dict[float, list[dict]], seeds: str, plain: str, title: str
) -> None:
    """H7 as two lines of the sweep: the plain rows' RMS over the warped rows'."""
    print(f"\n{title}")
    counts = [r["n"] for r in next(iter(results.values()))]
    print("      δ |" + "".join(f" {n:>10d} |" for n in counts))
    for delta, rows in results.items():
        line = f"  {delta:6.4f} |"
        for r in rows:
            line += f" {r[f'{plain}/rms'] / r[f'{seeds}/rms']:10.2f} |"
        print(line)


def print_regression(results: dict[float, list[dict]], seeds: str, title: str) -> None:
    """H4's δ = 0 half: the seed line against the construction's, which is E2.4's."""
    rows = results[0.0]
    print(f"\n{title}")
    print(f"     n       h |  {seeds:<15}construction   |  relative distance")
    for r in rows:
        built = r["construction/rms"]
        distance = abs(r[f"{seeds}/rms"] - built) / built
        print(
            f"  {r['n']:6d}  {r['h']:.4f} |  {r[f'{seeds}/rms']:.4e}  "
            f"{r['construction/rms']:.4e}  |  {distance:17.2e}"
        )
    print(
        f"     fit          |  {_fit(rows, f'{seeds}/rms'):12.2f}  "
        f"{_fit(rows, 'construction/rms'):12.2f}  |"
    )


def _marker(delta: float, positive: Sequence[float]) -> str:
    """One marker per δ > 0, in the sweep's order; the jump line carries none."""
    return "" if delta == 0.0 else MARKERS[list(positive).index(delta) % len(MARKERS)]


def _counts_axis(ax, n: np.ndarray) -> None:
    """The counts themselves as the x ticks (``plot_knee``'s: the decades collide)."""
    ax.set_xticks(n, [f"{int(k)}" for k in n], fontsize=7, rotation=45)
    ax.set_xticks([], minor=True)


def _style(label: str) -> tuple[str, str]:
    """``STYLE``'s colour and marker for a line, its ``--seed-reach`` suffix aside."""
    for chain in ("seeds", "tangential"):
        if label.startswith(f"{chain}-plain"):
            return STYLE[f"{chain}-plain"]
        if label.startswith(chain):
            return STYLE[chain]
    return STYLE[label]


def plot_seeds(
    results: dict[str, dict[float, list[dict]]],
    labels: Sequence[str],
    seeds: str,
    plain: str,
    path: Path,
    title: str | None = None,
    lines: Sequence[str] | None = None,
) -> None:
    """Top: the lines at three widths; bottom: the seeds, the rule, and the warp.

    Blue is the seeds (light blue the plain-Gaussian ablation), orange naive,
    purple the δ = 0 construction, grey the direct operator. The top row is
    the elliptic error against N at the narrowest, a middle and the widest δ,
    the bottom row the seeds' own lines at every δ (elliptic solid, parabolic
    dashed, with an ``h⁴`` guide), the seeds over the naive and direct lines
    against ``h/δ`` (the rule: no threshold, and the resolved-edge penalty),
    and the warp's factor against N. ``seeds`` and ``plain`` are the main
    line and its ablation, drawn in their own colours (the tangential chain's
    cyan in E4.11's figure); ``lines`` (default ``labels``) are the top row's
    and the legend's, which E4.11 thins to five so that no two blues crowd.
    """
    lines = list(labels if lines is None else lines)
    elliptic = results["elliptic"]
    positive = [d for d in elliptic if d > 0]
    widths = sorted(positive)
    shown = (
        list(dict.fromkeys([widths[0], widths[len(widths) // 2], widths[-1]]))
        if widths
        else []
    )
    fig = plt.figure(figsize=(11.0, 8.0))
    axes = fig.subplots(2, 3)
    for ax in axes[0][len(shown) :]:
        ax.set_visible(False)
    for ax, delta in zip(axes[0], shown, strict=False):
        rows = elliptic[delta]
        n = np.array([r["n"] for r in rows], dtype=float)
        for label in lines:
            colour, marker = _style(label)
            ax.loglog(
                n,
                [r[f"{label}/rms"] for r in rows],
                color=colour,
                marker=marker,
                ms=5,
                lw=0.9,
                label=label,
            )
        ax.loglog(n, [r["floor"] for r in rows], ":", color=CONSTRUCTION, lw=0.8)
        _counts_axis(ax, n)
        ax.set_title(f"equilibrium, δ = {delta:g} (marker per line)", fontsize=10)
        ax.set_xlabel("nodes N")
        ax.grid(True, which="both", alpha=0.25)
    axes[0][0].set_ylabel("RMS error in u")
    ax = axes[1][0]
    for delta in elliptic:
        marker = _marker(delta, positive)
        for problem, style in (("elliptic", "-"), ("parabolic", "--")):
            rows = results[problem][delta]
            ax.loglog(
                [r["n"] for r in rows],
                [r[f"{seeds}/rms"] for r in rows],
                style,
                color=_style(seeds)[0],
                marker=marker,
                ms=4,
                lw=0.9 if marker else 0.7,
            )
    rows = elliptic[0.0 if 0.0 in elliptic else positive[0]]
    n = np.array([r["n"] for r in rows], dtype=float)
    guide = rows[0][f"{seeds}/rms"] * (n / n[0]) ** -2.0
    ax.loglog(n, guide, color=REFERENCE, lw=0.7, ls="-.")
    _counts_axis(ax, n)
    ax.set_title("the seeds at every δ (marker per δ; dashed: parabolic)", fontsize=9)
    ax.set_xlabel("nodes N")
    ax.set_ylabel("RMS error in u")
    ax.grid(True, which="both", alpha=0.25)
    ax = axes[1][1]
    for delta in positive:
        rows = elliptic[delta]
        marker = _marker(delta, positive)
        for label, colour in (("naive", NAIVE), ("direct", REFERENCE)):
            if f"{label}/rms" not in rows[0]:
                continue
            ax.loglog(
                [r["h_over_delta"] for r in rows],
                [r[f"{seeds}/rms"] / r[f"{label}/rms"] for r in rows],
                color=colour,
                marker=marker,
                ms=4,
                lw=0.8,
            )
    ax.axhline(1.0, color="k", lw=0.6)
    ax.axvline(1.0, color=REFERENCE, lw=0.6, ls="--")
    ax.set_title("seeds ÷ naive (orange) and ÷ direct (grey)", fontsize=9)
    ax.set_xlabel("h / δ")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_visible(bool(positive))
    ax = axes[1][2]
    drawn = False
    for delta in elliptic:
        rows = elliptic[delta]
        if f"{plain}/rms" not in rows[0]:
            continue
        ax.semilogx(
            [r["n"] for r in rows],
            [r[f"{plain}/rms"] / r[f"{seeds}/rms"] for r in rows],
            color=_style(plain)[0],
            marker=_marker(delta, positive),
            ms=4,
            lw=0.9,
        )
        drawn = True
    # The counts come off whichever δ was swept: the δ = 0 regression run
    # (``--deltas 0 --operators … seeds``) has no positive width and no
    # ablation, and its figure keeps the panels that have something in them.
    _counts_axis(ax, np.array([r["n"] for r in next(iter(elliptic.values()))]))
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_title("the warp: plain ÷ warped seed rows", fontsize=9)
    ax.set_xlabel("nodes N")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_visible(drawn)
    handles = [
        Line2D(
            [],
            [],
            color=_style(label)[0],
            marker=_style(label)[1],
            ms=5,
            label=label,
        )
        for label in lines
    ]
    handles += [
        Line2D([], [], color=CONSTRUCTION, ls=":", label="floor: ref. δ − ref. 0"),
        Line2D([], [], color=REFERENCE, ls="-.", lw=0.7, label="h⁴ guide"),
    ]
    for delta in elliptic:
        name = "δ = 0 (jump, thin)" if delta == 0.0 else f"δ = {delta:g}"
        handles.append(
            Line2D(
                [],
                [],
                color="k",
                marker=_marker(delta, positive),
                ls="",
                ms=5,
                label=name,
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)
    top = 1.0
    if title is not None:
        fig.suptitle(title, fontsize=11)
        top = 0.96
    fig.tight_layout(rect=(0, 0.075, 1, top))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _probe_cells(rows: list[dict], name: str) -> list[str]:
    """``_with_rates`` for the probe, where an empty row set reads 0: "-" there."""
    cells = []
    for i, r in enumerate(rows):
        value = r.get(name, 0.0)
        if value == 0.0:
            cells.append("-")
            continue
        before = rows[i - 1].get(name, 0.0) if i else 0.0
        rate = f" ({_rate(rows[i - 1], r, name):5.2f})" if before > 0.0 else ""
        cells.append(f"{value:.2e}{rate}")
    return cells


def print_probe(
    results: dict[float, list[dict]], labels: Sequence[str], seeds: str, title: str
) -> None:
    """H9's probe: each line's rows applied to the equilibrium reference at the nodes.

    RMS of ``L u`` over the rows the seeds rebuild, over E2.3's crossing rows
    and over the bulk, per δ, with the order per halving of h; the seeds'
    seeded rows over their bulk rows last. A consistent row converges; one
    that is not stalls at a size set by what it gets wrong. An empty row set
    (no bulk at all where 20 δ covers the strip) prints "-".
    """
    print(f"\n{title}")
    shown = [
        label
        for label in dict.fromkeys((seeds, "seeds", "construction", "naive"))
        if label in labels
    ]
    for delta, rows in results.items():
        if f"{seeds}/probe_seeded" not in rows[0]:
            continue
        name = "δ = 0 (jump)" if delta == 0.0 else f"δ = {delta:g}"
        print(f"  {name}")
        header = "       n |"
        for label in shown:
            header += f" {label}: seeded       crossing          |"
        print(header + f" {seeds}: bulk            seeded ÷ bulk")
        cells = {
            (label, q): _probe_cells(rows, f"{label}/probe_{q}")
            for label in shown
            for q in ("seeded", "crossing")
        }
        bulk = _probe_cells(rows, f"{seeds}/probe_bulk")
        for i, r in enumerate(rows):
            line = f"  {r['n']:6d} |"
            for label in shown:
                seeded, across = cells[(label, "seeded")], cells[(label, "crossing")]
                line += f" {seeded[i]:<16} {across[i]:<16} |"
            below = r[f"{seeds}/probe_bulk"]
            ratio = r[f"{seeds}/probe_seeded"] / below if below else None
            cell = "       -" if ratio is None else f"{ratio:8.1f}"
            print(line + f" {bulk[i]:<16} {cell}")


def flat_comparison(
    results: dict[str, dict[float, list[dict]]],
    labels: Sequence[str],
    outputs: Path,
    seed: int,
    iterations: int,
    t_end: float,
) -> list[dict]:
    """The curved lines over case 1's at equal (δ, n), from E4.3–E4.6's cache.

    The ticket's "curved numbers compared with the flat ones at equal δ":
    every line of ``labels`` that case 1's cache holds at the same count,
    seed and repulsion steps; a missing entry is skipped. A tangential line
    (E4.11) is read over case 1's seed line of the same warp and reach
    (``flat_twin``): on flat lines the two are the same rows to the march's
    tolerance (§3.10, a test), and H16 asks for exactly that ratio.
    """
    flat = load_knee_cache(outputs)
    rows = []
    for problem, lines in results.items():
        for delta, entries in lines.items():
            for r in entries:
                row = {"problem": problem, "delta": delta, "n": r["n"]}
                for label in labels:
                    twin = flat_twin(label)
                    k = knee_key(problem, delta, r["n"], twin, seed, iterations, t_end)
                    if k in flat and f"{label}/rms" in r:
                        row[f"flat/{label}"] = flat[k]["rms"]
                        row[f"curved/{label}"] = r[f"{label}/rms"]
                rows.append(row)
    return rows


def flat_twin(label: str) -> str:
    """Case 1's line a curved line is read over: itself, or a tangential one's seeds."""
    return label.replace("tangential", "seeds", 1)


def print_flat_comparison(
    rows: list[dict], labels: Sequence[str], problem: str, title: str
) -> None:
    print(f"\n{title}")
    shown = [label for label in labels if any(f"flat/{label}" in r for r in rows)]
    if not shown:
        print("  (case 1's cache holds none of these lines at these counts)")
        return
    print(
        "      δ       n |"
        + "".join(f" {label + ': flat':>20}  curved  ÷ |" for label in shown)
    )
    for r in rows:
        if r["problem"] != problem:
            continue
        line = f"  {r['delta']:6.4f}  {r['n']:6d} |"
        for label in shown:
            if f"flat/{label}" in r:
                f, c = r[f"flat/{label}"], r[f"curved/{label}"]
                line += f" {f:20.2e} {c:8.2e} {c / f:5.0f} |"
            else:
                line += f" {'-':>20} {'-':>8} {'-':>5} |"
        print(line)


def figure_name(geometry: Geometry, tangential: bool = False) -> str:
    """``heat2d_stiff_seeds.png`` for case 1, the geometry's tag in it otherwise.

    With the tangential chain as the sweep's main line (E4.11) ``seeds``
    becomes ``tangential``, so E4.6's and E4.7's figures are never overwritten.
    """
    stem = "heat2d_stiff_tangential" if tangential else "heat2d_stiff_seeds"
    if geometry.is_case1:
        return f"{stem}.png"
    return f"{stem}_{geometry.tag.replace(' ', '_')}.png"


def run_seeds(args) -> dict:
    """E4.6's tables and figure: the seeds against E4.3's lines at every (n, δ).

    On another geometry than case 1 (E4.7) the same sweep against the product
    grid, without E4.3's straddling-row readings, with H9's truncation probe
    and the curved lines over case 1's at equal (δ, n). With the
    ``tangential`` line (E4.11, §3.10) the tangential chain is the sweep's
    main line: the ratios, H7's warp, H8's penalty, the δ = 0 comparison with
    E2.3, the probe and the figure are its, route (a) one of the lines.
    """
    t0 = time.perf_counter()
    geometry = args.geometry
    reach = args.seed_reach
    marched = {
        seed_label(w, TANH_REACH, c): seed_label(w, reach, c)
        for c in (False, True)
        for w in (True, False)
    }
    labels = [marched.get(label, label) for label in args.operators]
    if seed_label(True, reach) not in labels:
        raise ValueError(f"the sweep needs the {seed_label(True, reach)!r} line")
    tangential = seed_label(True, reach, True) in labels
    seeds = seed_label(True, reach, tangential)
    plain = seed_label(False, reach, tangential)
    cache = load_knee_cache(args.outputs, geometry)
    results = knee_sweep(
        args.counts,
        args.deltas,
        cache,
        args.seed,
        args.iterations,
        args.t_end,
        labels=labels,
        reach=reach,
        save=lambda: save_knee_cache(args.outputs, cache, geometry),
        geometry=geometry,
    )
    save_knee_cache(args.outputs, cache, geometry)
    tables: dict = {"sweep": results}
    chain = "the tangential chain" if tangential else "route (a)"
    where = (
        "case 1's tanh edges"
        if geometry.is_case1
        else f"the tanh edges of {geometry.name} ({chain})"
    )
    against = "the separable reference" if geometry.is_case1 else "the product grid"
    if not geometry.is_case1:
        comparison = flat_comparison(
            results, labels, args.outputs, args.seed, args.iterations, args.t_end
        )
        tables["flat"] = comparison
    for problem, lines in results.items():
        what = (
            "equilibrium"
            if problem == "elliptic"
            else f"parabolic, BD4 dt = h from the analytic history, t = {args.t_end:g}"
        )
        print_sweep(
            lines,
            labels,
            f"the seeds against E4.3's lines through {where}, {what}:"
            f" RMS error (order per halving of h) against {against},"
            f" seed rows where the 30 nodes see the edge within {reach:g} δ",
        )
        ratios = sweep_ratios(lines, labels, seeds)
        tables[f"ratios/{problem}"] = ratios
        print_ratios(
            ratios,
            [label for label in labels if label != seeds],
            f"the seeds over each line at every (δ, n), {what}; the seeded rows,"
            " their share of N, and the march per row",
        )
        if geometry.is_case1:
            print_seed_diagnostics(
                lines,
                seeds,
                f"what the seeded solution says on the straddling rows, {what}:"
                " profile = |error of the row's sin 2πx coefficient| at ±h/2"
                " (absolute); flux, jump = one-sided α ∂_y at ±h/2 and its jump"
                " across the pair, error / |α v′ at the curve|",
            )
        resolved = resolved_rows(lines, ratios, seeds)
        tables[f"resolved/{problem}"] = resolved
        if resolved:
            print_resolved(
                resolved,
                seeds,
                f"H8, the resolved-edge penalty, {what}: where h ≤ δ the seed rows"
                " tend to the standard ones, and the direct operator is the one"
                " to beat",
            )
        if plain in labels:
            print_warp(
                lines,
                seeds,
                plain,
                f"H7, the warp as two lines of the sweep, {what}: the plain-Gaussian"
                " rows' RMS error over the warped rows'",
            )
        if 0.0 in lines and "construction/rms" in lines[0.0][0]:
            print_regression(
                lines,
                seeds,
                f"H4 at δ = 0, {what}: the seed operator is the construction there"
                " (E2.3's rows), so the line is port notes §2.4–2.5's"
                if geometry.is_case1
                else f"{chain} at δ = 0 against the curved construction, {what}:"
                " E2.3 carries the curvature and alpha's Taylor table, "
                + (
                    "the tangential chain both in its ξ-series (H15)"
                    if tangential
                    else "the frozen profile neither"
                ),
            )
        if problem == "elliptic":
            print_probe(
                lines,
                labels,
                seeds,
                "H9's probe: RMS of L u over each row set, u the equilibrium"
                f" reference at the nodes ({against}); seeded = the rows the seed"
                " rule marches, crossing = E2.3's rows, bulk = the interior 42 / 5"
                " rows off the edges and the boundary zone (order per halving of h)",
            )
        if not geometry.is_case1:
            print_flat_comparison(
                comparison,
                labels,
                problem,
                f"{geometry.name} over case 1 at equal (δ, n), {what}: RMS error"
                " flat (E4.3–E4.6's cache), curved, and curved ÷ flat",
            )
    args.outputs.mkdir(parents=True, exist_ok=True)
    # E4.11's figure keeps one line per role in its top row: the ablations
    # (the two plain lines, direct-reach) are the tables' and the warp panel's.
    top = (
        [r for r in labels if "-plain" not in r and r != "direct-reach"]
        if tangential
        else None
    )
    plot_seeds(
        results,
        labels,
        seeds,
        plain,
        args.outputs / figure_name(geometry, tangential),
        None if geometry.is_case1 else f"{geometry.name}: {chain} seeds",
        top,
    )
    cache_name, _ = geometry.cache()
    print(
        f"\nseed sweep {time.perf_counter() - t0:.1f} s; figure and {cache_name}"
        f" in {args.outputs}/"
    )
    return tables


# --- E4.11: the tangential chain's own tables ---------------------------------------


TANGENTIAL_CIRCLES = (0.25, 0.35)
"""H14's two circles about the strip's centre (#81's scratch), 0.2 between them and
1 outside, so that ``RingMode`` is an exact equilibrium through both."""

CIRCLES_FROM = 2500
"""The circles' smallest count: at 1250 nodes the inner circle's curvature 4 takes
a stencil's normal line to 0.81 of its focal distance, past ``FOOT_CURVATURE``,
and the seeds refuse it (§3.10's scope)."""


def circles_problem() -> tuple[Domain, RingMode]:
    """The concentric circles with constant pieces, and the mode exact through them."""
    band = Band(
        *(Circle(r) for r in TANGENTIAL_CIRCLES), Constant2D(0.2), Constant2D(1.0)
    )
    return Domain(band, band.interfaces, STRIP), RingMode(
        TANGENTIAL_CIRCLES, (1.0, 0.2, 1.0), mode=2
    )


def crossing_rows(nodes: NodeSet, domain: Domain) -> np.ndarray:
    """``(m, 30)`` node indices of E2.3's crossing stencils on ``domain``."""
    stencils = build_stencils(nodes, domain, interface=BOUNDARY)
    (group,) = [g for g in stencils.groups if g.kind == INTERFACE_KIND]
    return group.index[interface_crossings(nodes, domain.material, group.index)]


def circle_probe_rows(counts: Sequence[int], seed: int, iterations: int) -> list[dict]:
    """H14: E2.3's, route (a)'s and the tangential rows on ``RingMode`` (§3.10).

    The mode solves ``L u = 0`` through both circles, so each crossing row
    applied to it at the nodes is that row's truncation error; the RMS over
    the rows per count, and what a tangential row costs.
    """
    domain, mode = circles_problem()
    band = domain.material
    rows = []
    for n in (n for n in counts if n >= CIRCLES_FROM):
        nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
        values: dict[str, list[float]] = {
            "construction": [],
            "seeds": [],
            "tangential": [],
        }
        seconds = 0.0
        for idx in crossing_rows(nodes, domain):
            xy = nodes.xy[idx]
            u = mode(xy[:, 0], xy[:, 1])
            values["construction"].append(stencil_weights(xy, band, 4) @ u)
            values["seeds"].append(seed_weights(xy, band) @ u)
            t0 = time.perf_counter()
            values["tangential"].append(seed_weights(xy, band, tangential=True) @ u)
            seconds += time.perf_counter() - t0
        row = {"n": n, "h": nodes.h, "rows": len(values["seeds"])}
        for label, v in values.items():
            row[label] = float(np.sqrt(np.mean(np.square(v))))
        row["ms"] = 1e3 * seconds / max(row["rows"], 1)
        rows.append(row)
    return rows


def span_rows(counts: Sequence[int], seed: int, iterations: int) -> list[dict]:
    """H15's span half: the tangential block against E2.3's translated basis, case 2.

    At δ = 0 on every crossing stencil, the sine of the largest principal
    angle between the two 15-column spans at the 30 nodes; on case 1 they are
    one span to rounding (H2, §4.3), on a curved or tangentially varying edge
    two approximations of one space, whose distance should fall with h.
    """
    domain = case2()
    band = domain.material
    rows = []
    for n in counts:
        nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
        angles = []
        for idx in crossing_rows(nodes, domain):
            xy = nodes.xy[idx]
            tb = seed_basis(xy, band, tangential=True)
            basis = interface_stencil(xy, band, 4).polynomial_block()
            angles.append(np.sin(subspace_angles(basis, tb.block).max()))
        rows.append(
            {
                "n": n,
                "h": nodes.h,
                "stencils": len(angles),
                "median": float(np.median(angles)),
                "max": float(np.max(angles)),
            }
        )
    return rows


TANGENTIAL_SPECTRUM_N = 1600
"""The node set of the tangential spectra (H6's twin on case 2): dense eigenvalues
of the interior operator in about a second each."""


def spectrum_rows(seed: int, iterations: int) -> list[dict]:
    """The interior spectra of the seed operators on case 2, per δ (H6's twin).

    Route (a) and the tangential chain, warped and plain (one march per row for
    both warps, ``seed_operators``), and E2.3's curved rows at δ = 0: the
    rightmost eigenvalue, the leftmost and the largest imaginary part in units of
    ``h⁻²``, and how many sit right of the axis. The parabolic sweep marches BD4
    on these operators; a positive eigenvalue would be its growing mode.
    """
    domain = case2()
    nodes = build_node_set(
        domain, TANGENTIAL_SPECTRUM_N, seed=seed, iterations=iterations
    )
    h2 = nodes.h**2
    rows = []
    for delta in CURVED_DELTAS:
        medium = SmoothBand(domain.material, delta)
        stencils = build_stencils(
            nodes,
            replace(domain, material=medium),
            interface=BOUNDARY,
            reach=TANH_REACH,
        )
        ops = {"seeds": seed_operators(nodes, medium, stencils, (True,))[0][True]}
        both, _ = seed_operators(
            nodes, medium, stencils, (True, False), tangential=True
        )
        ops["tangential"], ops["tangential-plain"] = both[True], both[False]
        if delta == 0.0:
            crossing = build_stencils(nodes, domain, interface=BOUNDARY)
            ops["construction"] = interface_aware_operator(nodes, medium, crossing)
        for label, op in ops.items():
            lam = interior_eigenvalues(op, nodes)
            rows.append(
                {
                    "delta": delta,
                    "operator": label,
                    "max_re": float(lam.real.max()),
                    "min_re_h2": float(h2 * lam.real.min()),
                    "max_im_h2": float(h2 * np.abs(lam.imag).max()),
                    "positive": int((lam.real > 0.0).sum()),
                }
            )
    return rows


TANGENTIAL_TIMING = (10000, 400)
"""H17's timing: the node set and how many seeded rows per width, E4.7's (§4.6)."""


def timing_table(seed: int, iterations: int) -> list[dict]:
    """H17: the median ``seed_basis`` per width on case 2, route (a) and tangential.

    Evenly spaced rows among those the rule seeds at each δ, each basis built
    once per chain; times on a quiet machine are what the notes quote (§4.6's
    lesson: concurrent runs inflate them).
    """
    n, count = TANGENTIAL_TIMING
    domain = case2()
    nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
    rows = []
    for delta in CURVED_DELTAS:
        medium = SmoothBand(domain.material, delta)
        stencils = build_stencils(
            nodes,
            replace(domain, material=medium),
            interface=BOUNDARY,
            reach=TANH_REACH,
        )
        (group,) = [g for g in stencils.groups if g.kind == INTERFACE_KIND]
        index = group.index[seeded_rows(nodes, medium, group.index, TANH_REACH)]
        index = index[:: max(1, len(index) // count)][:count]
        times: dict[bool, list[float]] = {False: [], True: []}
        for idx in index:
            for chain in (False, True):
                t0 = time.perf_counter()
                seed_basis(nodes.xy[idx], medium, tangential=chain)
                times[chain].append(1e3 * (time.perf_counter() - t0))
        rows.append(
            {
                "delta": delta,
                "rows": len(index),
                "flat_ms": float(np.median(times[False])),
                "tangential_ms": float(np.median(times[True])),
            }
        )
    return rows


def print_tangential(tables: dict) -> None:
    rows = tables["circles"]
    print(
        "\nH14, the concentric circles 0.25 and 0.35 (0.2 between, 1 outside):"
        " RMS of L u over E2.3's crossing rows on RingMode, order per halving of h"
        f" (from {CIRCLES_FROM} nodes: FOOT_CURVATURE)"
    )
    columns = {k: _with_rates(rows, k) for k in ("construction", "seeds", "tangential")}
    print(
        "       n  rows |  E2.3 curved         route (a)            tangential"
        "           | ms a tangential row"
    )
    for i, r in enumerate(rows):
        print(
            f"  {r['n']:6d} {r['rows']:5d} |  {columns['construction'][i]:<20}"
            f" {columns['seeds'][i]:<20} {columns['tangential'][i]:<20} |"
            f" {r['ms']:8.1f}"
        )
    print(
        "  fit          |  "
        + "".join(f"{_fit(rows, k):<21.2f}" for k in ("construction", "seeds"))
        + f"{_fit(rows, 'tangential'):<20.2f} |"
    )
    rows = tables["span"]
    print(
        "\nH15, case 2 at δ = 0: sin of the largest principal angle between the"
        " tangential span and E2.3's translated basis over every crossing stencil"
    )
    print("       n  stencils |  median               max")
    median = _with_rates(rows, "median")
    for i, r in enumerate(rows):
        print(f"  {r['n']:6d} {r['stencils']:9d} |  {median[i]:<20} {r['max']:.2e}")
    print(
        f"\nH6 on case 2, {TANGENTIAL_SPECTRUM_N} nodes: the interior spectrum of each"
        " seed operator per δ (h² scales the extremes)"
    )
    print("       δ  operator          |   max Re   h² min Re   h² max |Im|   Re > 0")
    for r in tables["spectra"]:
        print(
            f"  {r['delta']:6.4f}  {r['operator']:<17} | {r['max_re']:8.3f}"
            f" {r['min_re_h2']:11.2f} {r['max_im_h2']:13.3f} {r['positive']:8d}"
        )
    n, _ = TANGENTIAL_TIMING
    print(
        f"\nH17 on case 2, {n} nodes: the median seed_basis per seeded row (ms),"
        " route (a) and tangential (quote only from a quiet machine)"
    )
    print("       δ   rows |  route (a)  tangential   ratio")
    for r in tables["timing"]:
        print(
            f"  {r['delta']:6.4f} {r['rows']:6d} | {r['flat_ms']:9.1f}"
            f" {r['tangential_ms']:11.1f} {r['tangential_ms'] / r['flat_ms']:7.1f}"
        )


def run_tangential(args) -> dict:
    """E4.11's own tables (§3.10's H14, H15's span half); the sweep is ``seeds``'s."""
    t0 = time.perf_counter()
    tables = {
        "circles": circle_probe_rows(args.counts, args.seed, args.iterations),
        "span": span_rows(args.counts, args.seed, args.iterations),
        "spectra": spectrum_rows(args.seed, args.iterations),
        "timing": timing_table(args.seed, args.iterations),
    }
    print_tangential(tables)
    print(f"\ntangential tables {time.perf_counter() - t0:.1f} s")
    return {"tangential": tables}


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mode",
        choices=("all", "references", "naive", "stencils", "seeds", "tangential"),
        default="all",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=None,
        help="edge widths (default: E4.3's on case 1, CURVED_DELTAS elsewhere)",
    )
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
    parser.add_argument(
        "--operators",
        nargs="+",
        choices=(*SWEEP_LABELS, *EXTRA_LABELS),
        default=list(SWEEP_LABELS),
        help="the lines of the seed sweep; naive and construction are E4.3's cache",
    )
    parser.add_argument(
        "--seed-reach",
        type=float,
        default=TANH_REACH,
        help="how many δ a stencil sees the edge over (the seeded-row rule, §3.3)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.0,
        help="the sine pair's amplitude (0: flat lines; case 2 is 0.02, the only"
        " one validated: any other needs its own accuracy check first)",
    )
    parser.add_argument(
        "--inside",
        choices=sorted(INSIDE),
        default=None,
        help="the band's piece (default: constant on flat lines, sine on curves)",
    )
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    inside = args.inside or ("constant" if args.amplitude == 0.0 else "sine")
    try:
        args.geometry = Geometry(args.amplitude, inside)
    except ValueError as err:
        parser.error(str(err))
    if args.deltas is None:
        args.deltas = list(STUDY_DELTAS if args.geometry.is_case1 else CURVED_DELTAS)
    if not args.geometry.is_case1 and args.mode not in ("references", "seeds"):
        parser.error("another geometry than case 1 runs --mode references or seeds")
    increasing = list(args.counts) == sorted(set(args.counts))
    if args.mode == "tangential" and (min(args.counts) < 300 or not increasing):
        parser.error("give increasing counts of 300 nodes or more")
    if args.mode in ("all", "naive", "seeds"):
        if any(n < 300 for n in args.counts) or list(args.counts) != sorted(
            set(args.counts)
        ):
            parser.error("give increasing counts of 300 nodes or more")
        if 0.0 not in args.deltas:
            parser.error("the knee study needs δ = 0 (the jump line and the floors)")
    if args.mode == "seeds" and "seeds" not in args.operators:
        parser.error("the seed sweep needs the seeds line")
    args.outputs.mkdir(parents=True, exist_ok=True)
    tables: dict = {}
    if args.mode in ("all", "references"):
        if args.geometry.is_case1:
            tables["references"] = reference_rows(args.deltas, args.growth)
            print_references(tables["references"])
        else:
            tables["references"] = curved_reference_rows(
                args.geometry, args.deltas, args.growth, args.outputs
            )
            print_curved_references(tables["references"], args.geometry)
    if args.mode in ("all", "naive"):
        tables.update(run_naive(args))
    if args.mode in ("all", "stencils"):
        tables.update(run_stencils(args))
    if args.mode == "seeds":
        tables.update(run_seeds(args))
    if args.mode == "tangential":
        tables.update(run_tangential(args))
    return tables


if __name__ == "__main__":
    main()
