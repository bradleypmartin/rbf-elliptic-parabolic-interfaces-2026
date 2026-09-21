"""E2.9 (#23): the extremizing parameter of EABE eq. 40 (§3.3.3): errors against N
per s (Fig. 19), the condition numbers of the continuity matrices against s at
fixed N (Fig. 20), and what the translated basis does numerically as the ring
thins.

Eq. 40 is case 3 with the ring ``0.35 − 1/s ≤ r ≤ 0.35`` at
``α = 1/(1.5 s) + (1/(3 s)) sin 2πx sin 2πy`` (``heat2d.domain.case3(s)``;
``s = 10³`` is case 3 itself, bit for bit). The ring's resistance
``(1/s) / (1/(1.5 s))`` is 1.5 at every s, so the solutions tend to a
thin-layer limit as s grows and differ from it by O(1/s). 2016 found the
errors "almost as well, if not identically" to case 3 until s ≈ 10⁸–10⁹,
"more marked failure" at 10¹¹, and the continuity matrices' condition number
growing like s².

Two figures under ``outputs/`` and six tables on stdout.

``heat2d_extremes_convergence.png``: RMS error against the node count for
each s of ``--s`` (Fig. 19's 10³, 10⁸, 10⁹, 10¹⁰, 10¹¹), the papers' setting
(curvature, warped Gaussians, straddled rows on the midline ``0.35 − 0.5/s``)
against a fine reference at the same s (``--reference-n`` nodes, cached as
``outputs/heat2d_extremes_reference_s<s>_n<N>_seed<seed>.npz``; the case-3
file at s = 10³), next to Fig. 19's markers; the plain-Gaussian operator is
tabulated beside it. The references are read against each other at one node
set, since for s ≥ 10⁸ the solutions differ by O(1/s) only.

``heat2d_extremes_conditioning.png``: at ``--conditioning-n`` nodes and
s = 10³ … 10¹¹ by decades, over every stencil crossing the ring: the mean
2-norm condition number of the continuity matrices as built (the ring's side
and the outside's at each circle; Fig. 20 plotted "the average"), and of the
ring side's with each row scaled by its largest entry (what a
partial-pivoting solve resolves); the polynomial block's and the RBF-FD
system's condition numbers with the translated basis anchored on the
centre's region (this port), on the ring (the MATLAB's anchoring, port notes
§2.3) and with the flat frames; the largest coefficient of the far side's
basis; and the worst relative residual of ``stencil_weights`` on the matched
radial quadratic through the ring (§2.7's exactness check at every s). The
diagonal dominance ratio of the assembled operator's crossing rows (E2.8) is
one line per s.

Also per s of ``--s``: the interior spectrum at ``--spectrum-n`` nodes (the
E2.5 note on #23) and the resampling check, ``heat2d.exact.ring_exact(s=s)``
on a fine set read through its stencils at the coarse nodes near the ring
(the E2.7 note).

    uv run python scripts/heat2d_extremes.py                            # ~4.5 min
    uv run python scripts/heat2d_extremes.py --reference-n 160000 \\
        --counts 1250 2500 5000 10000 20000 40000 80000 \\
        --conditioning-n 10000                                          # ~30 min

``--conditioning-n 0``, ``--spectrum-n 0`` and ``--check-n 0`` skip those parts.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from math import log10
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from heat_interfaces.heat2d import (  # noqa: E402
    BOUNDARY,
    CASE3_S,
    INTERFACE_KIND,
    Band,
    Circle,
    Constant2D,
    Domain,
    InterfaceStencil,
    NodeSet,
    Reference,
    Stencils,
    build_node_set,
    build_stencils,
    case3,
    continuity_matrix,
    diagonal_dominance_ratio,
    interface_aware_operator,
    interface_crossings,
    interface_stencil,
    interior_eigenvalues,
    knn,
    reduced_system,
    reference_solution,
    resample,
    ring_exact,
    ring_radii,
    rms_error,
    solve_equilibrium,
    stencil_weights,
    translated_basis,
)
from heat_interfaces.plotting import NAIVE, REFERENCE  # noqa: E402

S_VALUES = (1e3, 1e8, 1e9, 1e10, 1e11)
"""Fig. 19's five values of the extremizing parameter."""

CONDITIONING_S = tuple(10.0**k for k in range(3, 12))
"""Fig. 20's nine, by decades from 10³ to 10¹¹."""

FIG19 = {
    1e3: {
        1250: 1.3e-3,
        2500: 6.0e-4,
        5000: 1.65e-4,
        10000: 5.5e-5,
        20000: 1.6e-5,
        40000: 2.75e-6,
        80000: 8.0e-7,
    },
    1e8: {
        1250: 1.45e-3,
        2500: 6.5e-4,
        5000: 2.2e-4,
        10000: 7.4e-5,
        20000: 2.4e-5,
        40000: 4.9e-6,
        80000: 1.2e-6,
    },
    1e9: {
        1250: 1.6e-3,
        2500: 5.4e-4,
        5000: 1.8e-4,
        10000: 9.0e-5,
        20000: 2.5e-5,
        40000: 9.6e-6,
        80000: 6.9e-6,
    },
    1e10: {
        1250: 1.15e-3,
        2500: 5.7e-4,
        5000: 1.65e-4,
        10000: 6.7e-5,
        20000: 4.5e-5,
        40000: 1.25e-5,
        80000: 1.1e-5,
    },
    1e11: {
        1250: 2.0e-3,
        2500: 1.2e-3,
        5000: 8.5e-4,
        10000: 7.8e-4,
        20000: 7.4e-4,
        40000: 7.4e-4,
        80000: 7.4e-4,
    },
}
"""EABE Fig. 19 read off the rendered page (2026-09-21), seven markers per s.

Reading accuracy about ±30 % (the markers overlap below 10,000 nodes). The
s = 10³ line is Fig. 14's curved line replotted; read here it is 0.7–1.0×
the §2.7 reading of Fig. 14.
"""

FIG20 = {
    1e3: 4.0e6,
    1e4: 4.5e8,
    1e5: 4.5e10,
    1e6: 4.0e12,
    1e7: 4.5e14,
    1e8: 4.5e16,
    1e9: 4.5e18,
    1e10: 4.5e20,
    1e11: 4.5e22,
}
"""EABE Fig. 20 read off the rendered page (2026-09-21): about ``4.5 s²``, ±40 %."""

MARKERS = {1e3: "o", 1e8: "^", 1e9: "x", 1e10: "+", 1e11: "D"}
"""Fig. 19's marker per s, reused for ours."""

NEAR_RING = 0.1
"""Half-width of the annulus about the ring the resampling check reads in (§2.7)."""


def edge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Eq. 39's data on both rows, ``sin 6πx`` at ``y = 0`` and ``y = 1``."""
    return np.sin(6 * np.pi * x)


VALUES = [edge, edge, 0.0]
"""Bottom row, top row, cooling circle (the order of ``case3().dirichlet``)."""


def tag(s: float) -> str:
    """``1e8`` for ``s = 10⁸``: file names and table headers."""
    return f"1e{round(log10(s))}"


def power(s: float) -> str:
    return rf"$s = 10^{{{round(log10(s))}}}$"


def constant_ring(s: float) -> Domain:
    """Eq. 40's geometry with the ring at its constant part (``ring_exact(s=s)``)."""
    inner, outer = ring_radii(s)
    band = Band(
        Circle(inner), Circle(outer), Constant2D(1.0 / (1.5 * s)), Constant2D(1.0)
    )
    return replace(case3(s), material=band)


# --- ownership and references --------------------------------------------------


def ownership(nodes: NodeSet, domain: Domain) -> dict[str, int]:
    """How many nodes each region holds; the ring's count must be zero (§2.7).

    At every s the rows sit ``±0.5 h_row`` off the midline and the ring is
    ``1/s`` wide, far thinner than ``h_row``, so no node is inside it and the
    innermost pair straddles both circles; both are checked, not assumed.
    """
    band = domain.material
    region = band.region_index(nodes.x, nodes.y)
    counts = {f"region-{k}": int((region == k).sum()) for k in range(3)}
    if counts["region-1"]:
        raise RuntimeError(f"{counts['region-1']} nodes lie inside the ring")
    if nodes.straddle_rows:
        lo, hi = nodes.rows_of(0, 0.5)
        r = np.hypot(nodes.x - 0.5, nodes.y - 0.5)
        inner, outer = band.lower.radius, band.upper.radius
        if not (np.all(r[lo.index] < inner) and np.all(r[hi.index] > outer)):
            raise RuntimeError("the innermost pair does not straddle both circles")
    return counts


def reference_path(outputs: Path, s: float, n: int, seed: int) -> Path:
    if s == CASE3_S:
        return outputs / f"heat2d_case3_reference_n{n}_seed{seed}.npz"
    return outputs / f"heat2d_extremes_reference_s{tag(s)}_n{n}_seed{seed}.npz"


def reference(
    s: float, n: int, seed: int, iterations: int, outputs: Path
) -> tuple[Reference, bool]:
    """The cached fine solution at ``s``, solved and saved if absent; ``True`` if
    reused."""
    path = reference_path(outputs, s, n, seed)
    if path.exists():
        ref = Reference.load(path)
        if ref.meta.get("iterations") == iterations and ref.meta.get("n") == n:
            return ref, True
    ref = reference_solution(case3(s), n, VALUES, seed=seed, iterations=iterations)
    ref.meta["s"] = s
    ref.save(path)
    return ref, False


# --- the convergence sweep (Fig. 19) ---------------------------------------------


def sweep(
    s: float,
    counts: tuple[int, ...],
    ref: Reference,
    ref_stencils: Stencils,
    seed: int,
    iterations: int,
) -> list[dict[str, float]]:
    """One row per count at ``s``: the curved (warped) and plain-Gaussian errors.

    ``nodes`` is the node set with its stencils, ``read`` the reading of the
    reference at the nodes (shared by both operators), ``<name>-build`` and
    ``<name>-solve`` the operator and the solve; ``region-k`` the ownership.
    """
    domain = case3(s)
    rows = []
    for n in counts:
        t0 = time.perf_counter()
        nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
        stencils = build_stencils(nodes, domain, interface=BOUNDARY)
        t1 = time.perf_counter()
        u_ref = resample(
            ref.u, ref.nodes, ref_stencils, domain.material, nodes.x, nodes.y
        )
        row: dict[str, float] = {
            "n": nodes.n,
            "h": nodes.h,
            "group": int(stencils.near_interface.sum()),
            "nodes": t1 - t0,
            "read": time.perf_counter() - t1,
            **ownership(nodes, domain),
        }
        for name, warp in (("curved", True), ("plain", False)):
            t0 = time.perf_counter()
            op = interface_aware_operator(nodes, domain.material, stencils, warp=warp)
            t1 = time.perf_counter()
            u = solve_equilibrium(op, nodes, VALUES)
            row[f"{name}-build"] = t1 - t0
            row[f"{name}-solve"] = time.perf_counter() - t1
            row[name] = rms_error(u, u_ref)
            row[f"{name}-max"] = float(np.abs(u).max())
        rows.append(row)
    return rows


def reference_spread(
    refs: dict[float, tuple[Reference, Stencils]], nodes: NodeSet
) -> tuple[dict[float, float], float]:
    """RMS difference at ``nodes`` between each reference and the first at s ≥ 10⁸.

    The solutions at s ≥ 10⁸ differ from the thin-layer limit by O(1/s), so
    their references should agree to well below their own errors; the
    s = 10³ ring is 0.001 wide and physically different.
    """
    reads = {}
    for s, (ref, stencils) in refs.items():
        material = case3(s).material
        reads[s] = resample(ref.u, ref.nodes, stencils, material, nodes.x, nodes.y)
    base = next((s for s in refs if s >= 1e8), next(iter(refs)))
    return {s: rms_error(reads[s], reads[base]) for s in refs}, base


# --- conditioning (Fig. 20) ------------------------------------------------------


def _cond(a: np.ndarray) -> float:
    return float(np.linalg.cond(a))


def _augmented(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = p.shape[1]
    return np.block([[a, p], [p.T, np.zeros((q, q))]])


def stencil_conditioning(
    st: InterfaceStencil, flat: InterfaceStencil
) -> dict[str, float]:
    """Condition numbers and magnitudes of one crossing stencil's pieces.

    ``ring-raw`` / ``out-raw``: the mean 2-norm condition number of the
    continuity matrices on the ring's side and on the outside at the two
    circles, as built (``continuity_matrix``); ``ring-eq`` / ``out-eq``: the
    same with each row scaled by its largest entry; ``all-raw``: the mean
    over all four (Fig. 20's "average"). ``far`` / ``far-flat``: the largest
    coefficient of the basis on the region across the ring from the centre;
    ``p-centre`` / ``p-band`` / ``p-flat`` and ``aug-centre`` / ``aug-band``:
    the polynomial block's and the RBF-FD system's condition numbers with
    the basis anchored on the centre's region, on the ring (the MATLAB's
    anchoring), and on the centre's region with flat frames.
    """
    out: dict[str, list[float]] = {k: [] for k in ("ring", "out")}
    eq: dict[str, list[float]] = {k: [] for k in ("ring", "out")}
    for j, it in st.interfaces.items():
        c_minus = continuity_matrix(it.minus, it.expansion)
        c_plus = continuity_matrix(it.plus, it.expansion)
        # Interface 0 is the inner circle: its ``plus`` side is the ring.
        ring, outside = (c_plus, c_minus) if j == 0 else (c_minus, c_plus)
        for name, c in (("ring", ring), ("out", outside)):
            out[name].append(_cond(c))
            eq[name].append(_cond(c / np.abs(c).max(axis=1)[:, None]))
    lowest, highest = min(st.regions), max(st.regions)
    far = [k for k in st.regions if k not in (1, st.anchor)]
    p_centre = st.polynomial_block()
    p_band = st.polynomial_block(translated_basis(st.interfaces, 1, lowest, highest))
    p_flat = flat.polynomial_block()
    a = st.gaussian_block()
    return {
        "ring-raw": float(np.mean(out["ring"])),
        "out-raw": float(np.mean(out["out"])),
        "all-raw": float(np.mean(out["ring"] + out["out"])),
        "ring-eq": float(np.mean(eq["ring"])),
        "out-eq": float(np.mean(eq["out"])),
        "far": max(np.abs(st.regions[k].coefficients).max() for k in far),
        "far-flat": max(np.abs(flat.regions[k].coefficients).max() for k in far),
        "p-centre": _cond(p_centre),
        "p-band": _cond(p_band),
        "p-flat": _cond(p_flat),
        "aug-centre": _cond(_augmented(a, p_centre)),
        "aug-band": _cond(_augmented(a, p_band)),
    }


def matched_residual(s: float, n: int, seed: int, iterations: int) -> dict[str, float]:
    """The worst relative residual of the weights on the matched quadratic at ``s``.

    ``u = r²`` inside, ``r²/α + b₁`` on the ring, ``r² + b₂`` outside with
    ``α = 1/(1.5 s)``: ``∇·(α∇u) = 4`` everywhere and ``u`` climbs by
    ``(2 × 0.35 − 1/s)(1.5 − 1/s) ≈ 1.05`` across the ring at every s
    (``tests/heat2d/test_interface.py``). Over the stencils of the interface
    spec crossing the ring on a constant-ring layout, curvature on, warp on
    and off; the climb is formed from the stored radii so that the profile
    agrees with the geometry the stencils see.
    """
    domain = constant_ring(s)
    band = domain.material
    inner, outer = band.lower.radius, band.upper.radius
    nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
    r = np.hypot(nodes.x - 0.5, nodes.y - 0.5)
    u = np.where(
        r < inner, r**2, r**2 + (outer - inner) * (inner + outer) * (1.5 * s - 1)
    )
    index, _ = knn(nodes.xy, BOUNDARY.size)
    cross = interface_crossings(nodes, band, index)
    result = {}
    for name, warp in (("curved", True), ("plain", False)):
        residuals = []
        for idx in index[cross]:
            w = stencil_weights(nodes.xy[idx], band, BOUNDARY.degree, warp=warp)
            residuals.append(abs(w @ u[idx] - 4.0) / (np.abs(w) @ np.abs(u[idx])))
        result[name] = float(np.max(residuals))
        result[f"{name}-median"] = float(np.median(residuals))
    result["stencils"] = int(cross.sum())
    return result


def conditioning(
    s_values: tuple[float, ...], n: int, seed: int, iterations: int
) -> list[dict[str, float]]:
    """One row per s at ``n`` nodes: means and maxima over the crossing stencils.

    Each entry of ``stencil_conditioning`` comes back as its mean and as
    ``<key>-max``; ``ddr-*`` is the diagonal dominance ratio of the assembled
    operator's reduced system over the crossing rows and over all interior
    rows; ``residual-*`` is ``matched_residual``.
    """
    rows = []
    for s in s_values:
        domain = case3(s)
        t0 = time.perf_counter()
        nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
        stencils = build_stencils(nodes, domain, interface=BOUNDARY)
        op = interface_aware_operator(nodes, domain.material, stencils)
        build = time.perf_counter() - t0
        system = reduced_system(op, nodes, VALUES)
        ddr = diagonal_dominance_ratio(system.a)
        group = stencils.near_interface[system.interior]
        t0 = time.perf_counter()
        samples: list[dict[str, float]] = []
        for g in stencils.groups:
            if g.kind != INTERFACE_KIND:
                continue
            cross = interface_crossings(nodes, domain.material, g.index)
            for idx in g.index[cross]:
                xy = nodes.xy[idx]
                st = interface_stencil(xy, domain.material, g.spec.degree)
                flat = interface_stencil(
                    xy, domain.material, g.spec.degree, curvature=False
                )
                samples.append(stencil_conditioning(st, flat))
        row: dict[str, float] = {
            "s": s,
            "n": nodes.n,
            "h": nodes.h,
            "stencils": len(samples),
            "build": build,
            "analysis": time.perf_counter() - t0,
            "ddr-group-min": float(ddr[group].min()),
            "ddr-group-median": float(np.median(ddr[group])),
            "ddr-all-min": float(ddr.min()),
            "ddr-all-median": float(np.median(ddr)),
            **ownership(nodes, domain),
        }
        for key in samples[0]:
            values = np.array([c[key] for c in samples])
            row[key] = float(values.mean())
            row[f"{key}-max"] = float(values.max())
        residual = matched_residual(s, n, seed, iterations)
        row["residual-curved"] = residual["curved"]
        row["residual-plain"] = residual["plain"]
        row["residual-curved-median"] = residual["curved-median"]
        row["residual-plain-median"] = residual["plain-median"]
        row["residual-stencils"] = residual["stencils"]
        rows.append(row)
    return rows


# --- the spectrum and the resampling check -------------------------------------


def spectrum(
    s_values: tuple[float, ...], n: int, seed: int, iterations: int
) -> list[dict[str, float]]:
    """The interior eigenvalues at ``n`` nodes per s, warped and plain Gaussians.

    The E2.5 note on #23: whether the plain Gaussians' loop of complex
    eigenvalues returns as the warp's slope ratio grows like s. Counts as in
    the case-1 driver: complex means ``|Im λ| > 1e-8 max |λ|``; ``positive``
    counts eigenvalues with a positive real part (growing modes).
    """
    rows = []
    for s in s_values:
        domain = case3(s)
        nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
        stencils = build_stencils(nodes, domain, interface=BOUNDARY)
        for name, warp in (("curved", True), ("plain", False)):
            t0 = time.perf_counter()
            op = interface_aware_operator(nodes, domain.material, stencils, warp=warp)
            lam = interior_eigenvalues(op, nodes)
            h = nodes.h
            rows.append(
                {
                    "s": s,
                    "name": name,
                    "count": len(lam),
                    "complex": int(np.sum(np.abs(lam.imag) > 1e-8 * np.abs(lam).max())),
                    "positive": int((lam.real > 0).sum()),
                    "max_re": float(lam.real.max()),
                    "min_re_h2": float(lam.real.min() * h**2),
                    "max_im_h2": float(np.abs(lam.imag).max() * h**2),
                    "seconds": time.perf_counter() - t0,
                }
            )
    return rows


def resampling_check(
    s_values: tuple[float, ...],
    fine_n: int,
    counts: tuple[int, ...],
    seed: int,
    iterations: int,
) -> list[dict[str, float]]:
    """``ring_exact(s=s)`` on a fine constant-ring set, read back at the coarse nodes.

    §2.7's check at every s: the harmonic mode ``R(r) cos 2θ`` through the
    ring at ``α = 1/(1.5 s)``, whose climb tends to 1.5 times the flux, read
    through the fine stencils (translated basis where they cross) at the
    coarse case-3 nodes within ``NEAR_RING`` of the ring; RMS and largest
    error, aware and blind.
    """
    rows = []
    for s in s_values:
        domain, exact = constant_ring(s), ring_exact(s=s)
        t0 = time.perf_counter()
        fine = build_node_set(domain, fine_n, seed=seed, iterations=iterations)
        u = exact(fine.x, fine.y)
        stencils = {
            "aware": build_stencils(fine, domain, interface=BOUNDARY),
            "blind": build_stencils(fine, domain),
        }
        setup = time.perf_counter() - t0
        for n in counts:
            coarse = build_node_set(case3(s), n, seed=seed, iterations=iterations)
            keep = np.abs(np.hypot(coarse.x - 0.5, coarse.y - 0.5) - 0.35) < NEAR_RING
            x, y = coarse.x[keep], coarse.y[keep]
            ref = exact(x, y)
            row: dict[str, float] = {
                "s": s,
                "n": coarse.n,
                "fine-n": fine.n,
                "setup": setup,
                "points": int(keep.sum()),
            }
            for kind, st in stencils.items():
                got = resample(u, fine, st, domain.material, x, y)
                row[kind] = rms_error(got, ref)
                row[f"{kind}-max"] = float(np.abs(got - ref).max())
            rows.append(row)
    return rows


# --- tables ----------------------------------------------------------------------


def rates(rows: list[dict[str, float]], name: str) -> list[float]:
    out = [float("nan")]
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        out.append(np.log(a[name] / b[name]) / np.log(a["h"] / b["h"]))
    return out


def _order(value: float) -> str:
    return "    -" if np.isnan(value) else f"{value:5.2f}"


def _marker(table: dict[int, float], n: int) -> str:
    value = table.get(n)
    return f"{value:8.1e}" if value else "       -"


def print_references(
    refs: dict[float, tuple[Reference, Stencils]],
    reused: dict[float, bool],
    spread: dict[float, float],
    base: float,
    nodes: NodeSet,
) -> None:
    print(
        "references per s (the papers' setting; 'spread' is the RMS difference from"
        f" the s = {tag(base)} reference read at the {nodes.n}-node set)"
    )
    print(
        "     s       n        h  group    nodes  operator   solve       "
        " max |u|     spread"
    )
    for s, (ref, _) in refs.items():
        sec = ref.meta["seconds"]
        how = "cached" if reused[s] else "solved"
        print(
            f"  {tag(s):>4s}  {ref.nodes.n:6d}  {ref.nodes.h:.5f}"
            f"  {ref.meta['interface_group']:5d}"
            f"   {sec['nodes']:5.1f}s    {sec['operator']:5.1f}s  {sec['solve']:5.1f}s"
            f"  {np.abs(ref.u).max():.9f}  {spread[s]:9.2e}  ({how})"
        )


def print_convergence(results: dict[float, list[dict[str, float]]]) -> None:
    print(
        "\neq. 40 per s: RMS error at the coarse nodes against the reference at the"
        " same s, order per halving of h ('curved' the papers' setting, 'plain' the"
        " same with plain Gaussians; times are the curved operator's)"
    )
    for s, rows in results.items():
        print(
            f"\ns = {tag(s)}\n     n       h  group  in ring |     curved  order  nodes"
            "  build  solve  read |      plain  order |  Fig. 19"
        )
        curved, plain = rates(rows, "curved"), rates(rows, "plain")
        for i, r in enumerate(rows):
            n = int(r["n"])
            times = (
                f"{r['nodes']:4.1f}s  {r['curved-build']:4.1f}s"
                f"  {r['curved-solve']:4.1f}s  {r['read']:4.1f}s"
            )
            print(
                f"{n:6d}  {r['h']:.4f}  {int(r['group']):5d}  {int(r['region-1']):7d} |"
                f"  {r['curved']:9.2e}  {_order(curved[i])}  {times} |"
                f"  {r['plain']:9.2e}  {_order(plain[i])} |  {_marker(FIG19[s], n)}"
                if s in FIG19
                else f"{n:6d}  {r['h']:.4f}  {int(r['group']):5d}"
                f"  {int(r['region-1']):7d} |  {r['curved']:9.2e}  {_order(curved[i])}"
                f"  {times} |  {r['plain']:9.2e}  {_order(plain[i])} |        -"
            )
        found = [
            f"{r['curved'] / FIG19[s][int(r['n'])]:.2f}"
            for r in rows
            if s in FIG19 and int(r["n"]) in FIG19[s]
        ]
        ratio = ", ".join(found) if found else "-"
        maxes = ", ".join(f"{r['curved-max']:.4f}" for r in rows)
        print(f"ratio to Fig. 19: {ratio}; largest |u| (curved): {maxes}")
    first = next(iter(results))
    print(
        "\nratio of each s line to the s = "
        f"{tag(first)} line at each count (curved): "
        + "; ".join(
            f"{tag(s)}: "
            + ", ".join(
                f"{a['curved'] / b['curved']:.2f}"
                for a, b in zip(rows, results[first], strict=True)
            )
            for s, rows in results.items()
            if s != first
        )
    )


def print_conditioning(rows: list[dict[str, float]]) -> None:
    r0 = rows[0]
    print(
        f"\ncontinuity matrices and stencil systems at {int(r0['n'])} nodes"
        f" ({int(r0['stencils'])} stencils crossing the ring, {BOUNDARY.size} nodes /"
        f" degree {BOUNDARY.degree}); 2-norm condition numbers, mean over the"
        " stencils (largest in the next table): the ring side's and the outside's"
        " continuity matrices as built, the ring side's with each row scaled by its"
        " largest entry, the polynomial block P and the RBF-FD system with the basis"
        " anchored on the centre's region (this port), on the ring (MATLAB) and with"
        " flat frames; 'far' the largest basis coefficient across the ring from the"
        " centre"
    )
    print(
        "     s |  ring raw   out raw  ring eq  out eq |  P centre    P band    P flat"
        " |  aug centre   aug band |       far  far flat |  Fig. 20"
    )
    for r in rows:
        print(
            f"  {tag(r['s']):>4s} |  {r['ring-raw']:8.2e}  {r['out-raw']:8.2e}"
            f"  {r['ring-eq']:7.1f}  {r['out-eq']:6.1f} |  {r['p-centre']:8.2e}"
            f"  {r['p-band']:8.2e}  {r['p-flat']:8.2e} |  {r['aug-centre']:10.2e}"
            f"  {r['aug-band']:9.2e} |  {r['far']:8.2e}  {r['far-flat']:8.2e} |"
            f"  {FIG20.get(r['s'], float('nan')):8.1e}"
        )
    print(
        "\nthe same, largest over the stencils; the matched-quadratic residual of the"
        " weights (worst relative, curvature on, warp on / off); the DDR of the"
        " assembled operator's crossing rows and of all interior rows (min / median);"
        " times for the node set with operator and for the analysis"
    )
    print(
        "     s |  ring raw   out raw  ring eq |  P centre    P band    P flat |"
        "  aug centre   aug band |       far |  residual warp  plain |"
        "  DDR group      all  | build  analysis"
    )
    for r in rows:
        print(
            f"  {tag(r['s']):>4s} |  {r['ring-raw-max']:8.2e}  {r['out-raw-max']:8.2e}"
            f"  {r['ring-eq-max']:7.1f} |  {r['p-centre-max']:8.2e}"
            f"  {r['p-band-max']:8.2e}  {r['p-flat-max']:8.2e} |"
            f"  {r['aug-centre-max']:10.2e}  {r['aug-band-max']:9.2e} |"
            f"  {r['far-max']:8.2e} |  {r['residual-curved']:8.1e}"
            f"  {r['residual-plain']:8.1e} |"
            f"  {r['ddr-group-min']:.3f} {r['ddr-group-median']:.3f}"
            f"  {r['ddr-all-min']:.3f} {r['ddr-all-median']:.3f} |"
            f"  {r['build']:4.1f}s  {r['analysis']:5.1f}s"
        )
    s = np.array([r["s"] for r in rows])
    for key in ("ring-raw", "all-raw", "p-centre", "p-band", "aug-centre", "aug-band"):
        v = np.array([r[key] for r in rows])
        slope = np.polyfit(np.log(s), np.log(v), 1)[0]
        print(f"growth of {key} with s: exponent {slope:.2f}", end="; ")
    v = np.array([r["residual-curved"] for r in rows])
    print(f"residual: {np.polyfit(np.log(s), np.log(v), 1)[0]:.2f}")
    print(
        "median residual over the stencils (warp on / off): "
        + "; ".join(
            f"{tag(r['s'])}: {r['residual-curved-median']:.1e} /"
            f" {r['residual-plain-median']:.1e}"
            for r in rows
        )
    )


def print_spectrum(rows: list[dict[str, float]]) -> None:
    print(
        f"\ninterior spectrum at {int(rows[0]['count'])} interior nodes per s (dense"
        " eigenvalues; complex means |Im| > 1e-8 max |λ|)"
    )
    print(
        "     s  operator  complex  positive     max Re   h² min Re  h² max |Im|   time"
    )
    for r in rows:
        print(
            f"  {tag(r['s']):>4s}  {r['name']:8s}  {r['complex']:7d}"
            f"  {r['positive']:8d}  {r['max_re']:9.3f}  {r['min_re_h2']:10.3f}"
            f"  {r['max_im_h2']:11.3f}"
            f"  {r['seconds']:4.1f}s"
        )


def print_check(rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    print(
        f"\nresampling check per s: ring_exact(s) on {int(rows[0]['fine-n'])} nodes"
        f" read at the coarse nodes within {NEAR_RING} of the ring; RMS / max error,"
        " aware then blind"
    )
    print("     s      n  points |       aware        max |       blind        max")
    for r in rows:
        print(
            f"  {tag(r['s']):>4s}  {int(r['n']):5d}  {int(r['points']):6d} |"
            f"  {r['aware']:10.2e}  {r['aware-max']:9.2e} |"
            f"  {r['blind']:10.2e}  {r['blind-max']:9.2e}"
        )


# --- figures ---------------------------------------------------------------------


def _colours(s_values: tuple[float, ...]) -> dict[float, tuple]:
    cmap = matplotlib.colormaps["viridis"]
    return {
        s: cmap(t)
        for s, t in zip(s_values, np.linspace(0.05, 0.85, len(s_values)), strict=True)
    }


def figure_convergence(results: dict[float, list[dict[str, float]]]):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colours = _colours(tuple(results))
    for s, rows in results.items():
        n = [r["n"] for r in rows]
        marker = MARKERS.get(s, "s")
        ax.loglog(
            n,
            [r["curved"] for r in rows],
            marker + "-",
            color=colours[s],
            markersize=6,
            label=power(s),
        )
        if s in FIG19:
            items = sorted(FIG19[s].items())
            ax.loglog(
                [k for k, _ in items],
                [v for _, v in items],
                marker + ":",
                color="k",
                markerfacecolor="none",
                markersize=6,
                linewidth=0.6,
                label="EABE Fig. 19, read off (hollow, same markers)"
                if s == next(iter(results))
                else None,
            )
    first = results[next(iter(results))]
    n0, e0 = first[0]["n"], first[0]["curved"]
    n = np.array([r["n"] for r in first], dtype=float)
    ax.loglog(n, e0 * (n / n0) ** -2.0, "k--", linewidth=0.7, label="4th order")
    ticks = sorted({int(r["n"]) for rows in results.values() for r in rows})
    ax.set_xticks(ticks, [str(k) for k in ticks], fontsize=7, rotation=45)
    ax.set_xticks([], minor=True)
    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("RMS error in u against the reference at the same s")
    ax.set_title("eq. 40: case 3 at extreme contrasts (EABE Fig. 19 twin)")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    return fig


def figure_conditioning(rows: list[dict[str, float]]):
    s = np.array([r["s"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    ax = axes[0]
    ax.loglog(
        s,
        [r["all-raw"] for r in rows],
        "o-",
        color="k",
        markersize=5,
        label="mean of all four, as built",
    )
    ax.loglog(
        s,
        [r["ring-raw"] for r in rows],
        "^-",
        color=NAIVE,
        markersize=5,
        label="ring side, as built",
    )
    ax.loglog(
        s,
        [r["out-raw"] for r in rows],
        "v-",
        color=REFERENCE,
        markersize=5,
        label="outside, as built",
    )
    ax.loglog(
        s,
        [r["ring-eq"] for r in rows],
        "s-",
        color="#2ca02c",
        markersize=5,
        label="ring side, rows equilibrated",
    )
    items = sorted(FIG20.items())
    ax.loglog(
        [k for k, _ in items],
        [v for _, v in items],
        "o:",
        color="k",
        markerfacecolor="none",
        markersize=6,
        linewidth=0.7,
        label="EABE Fig. 20 (read off)",
    )
    ax.loglog(s, 4.5 * s**2, "k--", linewidth=0.7, label=r"$4.5\,s^2$")
    ax.set_xlabel("extremizing parameter s")
    ax.set_ylabel("condition number, mean over the crossing stencils")
    ax.set_title(f"continuity matrices at N = {int(rows[0]['n'])} (Fig. 20 twin)")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    for key, style, label in (
        ("p-centre", "o-", "P, anchored on the centre's region"),
        ("p-band", "o--", "P, anchored on the ring (MATLAB)"),
        ("p-flat", "o:", "P, flat frames"),
        ("aug-centre", "s-", "RBF-FD system, centre"),
        ("aug-band", "s--", "RBF-FD system, ring"),
    ):
        ax.loglog(s, [r[key] for r in rows], style, markersize=4, label=label)
    ax.loglog(
        s,
        [r["far"] for r in rows],
        "d-",
        color="k",
        markersize=4,
        label="largest far-side coefficient",
    )
    ax.set_xlabel("extremizing parameter s")
    ax.set_ylabel("condition number / coefficient, mean")
    ax.set_title("the stencil systems")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[2]
    ax.loglog(
        s,
        [r["residual-curved"] for r in rows],
        "o-",
        color="k",
        markersize=5,
        label="warped Gaussians",
    )
    ax.loglog(
        s,
        [r["residual-plain"] for r in rows],
        "o--",
        color=REFERENCE,
        markersize=5,
        label="plain Gaussians",
    )
    ax.loglog(s, 1e-17 * s, "k:", linewidth=0.7, label=r"$10^{-17}\,s$")
    ax.set_xlabel("extremizing parameter s")
    ax.set_ylabel("worst relative residual on the matched quadratic")
    ax.set_title("the weights' exactness through the ring")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


# --- main ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--s", type=float, nargs="+", default=list(S_VALUES))
    parser.add_argument("--counts", type=int, nargs="+", default=[1250, 2500, 5000])
    parser.add_argument("--reference-n", type=int, default=20000)
    parser.add_argument(
        "--conditioning-s", type=float, nargs="+", default=list(CONDITIONING_S)
    )
    parser.add_argument("--conditioning-n", type=int, default=2500)  # 0 skips
    parser.add_argument("--spectrum-n", type=int, default=2000)
    parser.add_argument("--check-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    check_n = args.reference_n if args.check_n is None else args.check_n
    if len(args.counts) < 2 or any(n < 900 for n in args.counts):
        parser.error("give at least two counts of 900 nodes or more")
    if args.reference_n <= max(args.counts):
        parser.error("the reference must have more nodes than every count")
    if any(s <= 0 for s in args.s + args.conditioning_s):
        parser.error("s must be positive")
    args.outputs.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    refs: dict[float, tuple[Reference, Stencils]] = {}
    reused: dict[float, bool] = {}
    results: dict[float, list[dict[str, float]]] = {}
    for s in args.s:
        ref, reused[s] = reference(
            s, args.reference_n, args.seed, args.iterations, args.outputs
        )
        ownership(ref.nodes, case3(s))
        refs[s] = (ref, ref.stencils(case3(s)))
        results[s] = sweep(s, tuple(args.counts), *refs[s], args.seed, args.iterations)
    common = build_node_set(
        case3(args.s[0]), max(args.counts), seed=args.seed, iterations=args.iterations
    )
    spread, base = reference_spread(refs, common)
    print_references(refs, reused, spread, base, common)
    print_convergence(results)
    print(f"({time.perf_counter() - t0:.1f} s for the references and the sweeps)")
    fig = figure_convergence(results)
    fig.savefig(args.outputs / "heat2d_extremes_convergence.png", dpi=150)
    plt.close(fig)

    if args.conditioning_n:
        t0 = time.perf_counter()
        rows = conditioning(
            tuple(args.conditioning_s), args.conditioning_n, args.seed, args.iterations
        )
        print_conditioning(rows)
        print(f"({time.perf_counter() - t0:.1f} s for the conditioning)")
        fig = figure_conditioning(rows)
        fig.savefig(args.outputs / "heat2d_extremes_conditioning.png", dpi=150)
        plt.close(fig)

    if args.spectrum_n:
        t0 = time.perf_counter()
        print_spectrum(
            spectrum(tuple(args.s), args.spectrum_n, args.seed, args.iterations)
        )
        print(f"({time.perf_counter() - t0:.1f} s for the spectra)")
    if check_n:
        t0 = time.perf_counter()
        print_check(
            resampling_check(
                tuple(args.s), check_n, tuple(args.counts), args.seed, args.iterations
            )
        )
        print(f"({time.perf_counter() - t0:.1f} s for the resampling check)")


if __name__ == "__main__":
    main()
