"""E4.8 (#39): EABE eq. 40's ring with seeds, at δ = 0 and with smooth edges.

Stiff note §3.6 is the design and §4.8 the results. The ring ``0.35 − 1/s ≤ r ≤
0.35`` of ``heat2d.domain.case3(s)`` carries its exact width ``1/s`` as the band's
``gap``, which the seeds march across from the outer circle (``SmoothBand``'s
widths rule); every stored radius is E2.9's. The seeds are E4.11's tangential
chain (``tangential=True``), warped. The smooth ring is ``SmoothBand(…, δ,
"resistance")``: the resistivity ``1/α`` blended by the difference of the two
edges, which keeps the ring's contact resistance 1.5 at every (s, δ) (Brad's
decision on #39, 2026-09-23). "E2.3" is the jump-aware operator of the papers
(curvature, warped Gaussians), E2.9's "curved" line; on a smooth ring it reads
the edges as jumps, the δ = 0 construction of E4.3.

Four parts, each skipped by an empty or zero flag:

1. ``--counts``: Fig. 19's twin with a seeds line. Per s of ``--s``, E2.3 and the
   seeds on eq. 40 at δ = 0 against E2.9's per-s reference (``--reference-n``,
   ``heat2d_extremes.reference``, cached under ``outputs/``), read through its
   stencils at every node (``full``) and through its standard interpolant only
   where its own stencil sees no interface (``far``), the read part 4 needs.
2. ``--conditioning-n``: Fig. 20's twin with a seeds line, per s of
   ``--conditioning-s`` at δ = 0 and of ``--conditioning-smooth-s`` (none by
   default) at the δ of ``--deltas``, over every seeded row of the
   ring at its constant part: the worst and median relative residual of the
   weights on the matched radial profile (``heat2d.exact.matched_radial``,
   ``∇·(α∇u) = 4``, reference-free at any (s, δ)), for the seeds and, at δ = 0,
   for the seeds without the gap (the stored radii, the ablation) and E2.3
   (E2.9's number); the seed block's condition number raw and column-scaled, and
   the seed system's, beside E2.3's polynomial block and system.
3. ``--spectrum-n``: the seed operator's interior spectrum, warped and plain,
   and the diagonal dominance of its rows (E2.8's DDR), per s of ``--s`` at
   δ = 0 and per (s, δ) of ``--smooth-s`` and ``--deltas``.
4. ``--probe-counts``: the smooth ring. Per (s, δ) of ``--smooth-s`` and 0 and
   ``--deltas``, four operators (naive ``Dx A Dx + Dy A Dy``, the direct
   stencil, E2.3, the seeds): the truncation probe, each row applied to the
   exact mode ``R(r) cos 2θ`` through the ring at its constant part
   (``RingMode`` with the gap at δ = 0, ``SmoothRingMode`` otherwise), RMS over
   the rows the seeds rebuild; and, with ``--fine-n``, the elliptic error on
   eq. 40 itself against a fine seed run at the same (s, δ) (cached under
   ``outputs/``, s from ``--fine-s``), read ``far`` from the ring (a
   self-convergence line: at δ > 0 there is no independent reference, §3.6).

Figures ``heat2d_ring_convergence.png``, ``heat2d_ring_conditioning.png`` and
``heat2d_ring_smooth.png``; the numbers are cached in ``heat2d_ring.json`` keyed
by part, s, δ, count and line, so an extended sweep reruns only what is new, and
the tables are written to ``heat2d_ring_results.json`` (and ``--data-dir``).

    uv run python scripts/heat2d_ring.py                 # ~2 min cold; seconds cached
    uv run python scripts/heat2d_ring.py --s 1e3 1e8 1e9 1e10 1e11 \
        --counts 1250 2500 5000 10000 20000 40000 80000 --reference-n 160000 \
        --conditioning-s 1e3 1e4 1e5 1e6 1e7 1e8 1e9 1e10 1e11 \
        --conditioning-smooth-s 1e3 1e7 1e11 --conditioning-n 10000 \
        --deltas 0.0025 0.001 0.00025 --smooth-s 1e3 1e11 \
        --probe-counts 2500 5000 10000 20000 40000 --fine-n 160000 \
        --spectrum-n 5000                                    # stiff note §4.8

The last is some 20 CPU-hours cold: one part and one (s, δ) at a time, as
concurrent processes sharing ``outputs/`` (the cache merges under a lock), it
took about 75 min on 14 cores (2026-09-23), the six 160,000-node seed runs 21–30
min each; stiff note §4.8 has the times per part.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from heat_interfaces.heat1d.domain import TANH_REACH  # noqa: E402
from heat_interfaces.heat2d import (  # noqa: E402
    BOUNDARY,
    INTERFACE_KIND,
    PRODUCT_ORDERING,
    Constant2D,
    Domain,
    NodeSet,
    Reference,
    RingMode,
    SmoothBand,
    SmoothRingMode,
    Stencils,
    block_condition,
    build_node_set,
    build_stencils,
    case3,
    diagonal_dominance_ratio,
    direct_operator,
    interface_aware_operator,
    interface_crossings,
    interface_stencil,
    interior_eigenvalues,
    knn,
    matched_radial,
    naive_operator,
    reduced_system,
    resample,
    rms_error,
    saddle_system,
    seed_basis,
    seed_operator,
    seeded_rows,
    solve_equilibrium,
    stencil_weights,
    weights_of,
    with_smooth_edges,
)
from heat_interfaces.plotting import AWARE, CONSTRUCTION, NAIVE, REFERENCE  # noqa: E402
from heat_interfaces.results_cache import ResultsCache  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from heat2d_extremes import (  # noqa: E402
    MARKERS,
    VALUES,
    ownership,
    power,
    reference,
    tag,
)

COMPOSITION = "resistance"
"""The smooth ring's composition (``SmoothBand``; #39's decision)."""

RING_DELTAS = (0.0025, 0.001, 0.00025)
"""The smooth ring's edge widths: 2.5, 1 and 1/4 of case 3's ring (w = 0.001),
all below the spacing of every count to 160,000 (h = 0.0026 there)."""

LABELS = ("naive", "direct", "construction", "seeds")
"""Part 4's operators; at δ = 0 it adds ``seeds15``, the seeds without the flux
seeds (§3.11), as part 1 does beside ``construction`` (E2.3) and ``seeds``."""

FIG19_LABELS = ("construction", "seeds", "seeds15")


def probe_labels(delta: float) -> tuple[str, ...]:
    """Part 4's probe lines at δ: the four, and the seeds without the flux seeds
    at δ = 0."""
    return (*LABELS, "seeds15") if delta == 0.0 else LABELS


NAMES = {
    "naive": "naive",
    "direct": "direct",
    "construction": "E2.3",
    "seeds": "seeds",
    "seeds15": "seeds (15)",
}

COLOURS = {
    "naive": NAIVE,
    "direct": REFERENCE,
    "construction": CONSTRUCTION,
    "seeds": AWARE,
    "seeds15": "#7fb3d5",
}

CACHE = "heat2d_ring.json"
CACHE_META = {
    "study": "E4.8 EABE eq. 40 with seeds (tangential, warped) and smooth edges",
    "version": 3,
    "composition": COMPOSITION,
    "seeds": "degree 4 and the degree-5 flux seeds (20); seeds15 without them",
    "warp": "level 0 of phi01 on the ring; epsilon from the warped spacing",
}
"""Bump ``version`` after any change to the chains, their series or sampling, the
ring's gap or the composition, as the other stiff caches say. Version 2 is the
flux seeds (§3.11) and exact keys. Version 3 is the ring's warp, φ₀₁'s level 0,
and ε from the warped spacing: every seed line was rebuilt; the E2.3, naive and
direct lines, which neither touches, were carried over."""

RESULTS = "heat2d_ring_results.json"

MIN_COUNT = 1250
"""The coarsest count the seeds take on the ring: at 900 nodes a stencil's normal
line reaches 0.81 of the 0.35 circle's focal distance, past ``FOOT_CURVATURE``
(E4.11's scope, h_s < 0.14 on this circle)."""

E29_RESIDUAL = 1.5e-18
"""Port notes §2.9's fitted line for E2.3's worst matched residual, ``1.5e-18 s``
(10,000 nodes, from s = 10⁵ on)."""


# --- media -------------------------------------------------------------------------


def ring_domain(s: float, delta: float, constant: bool = False) -> Domain:
    """Eq. 40 at ``s`` (its ring at the constant part if ``constant``), edges δ wide."""
    domain = case3(s)
    if constant:
        band = replace(domain.material, inside=Constant2D(1.0 / (1.5 * s)))
        domain = replace(domain, material=band)
    if delta > 0.0:
        domain = with_smooth_edges(domain, delta, COMPOSITION)
    return domain


def exact_mode(domain: Domain) -> RingMode | SmoothRingMode:
    """``R(r) cos 2θ`` exact through the constant ring of ``domain``."""
    material = domain.material
    if isinstance(material, SmoothBand):
        return SmoothRingMode(material)
    radii = (material.lower.radius, material.upper.radius)
    alphas = (1.0, material.inside.value, 1.0)
    return RingMode(radii, alphas, widths=(material.gap,))


def without_gap(material):
    """The same ring on its stored radii alone (E2.9's geometry, the ablation)."""
    if isinstance(material, SmoothBand):
        return replace(material, band=replace(material.band, gap=None))
    return replace(material, gap=None)


def plain_stencils(nodes: NodeSet, domain: Domain) -> Stencils:
    return build_stencils(nodes, domain)


def aware_stencils(nodes: NodeSet, domain: Domain) -> Stencils:
    return build_stencils(nodes, domain, interface=BOUNDARY)


def reach_stencils(nodes: NodeSet, domain: Domain) -> Stencils:
    return build_stencils(nodes, domain, interface=BOUNDARY, reach=TANH_REACH)


def build(label: str, nodes: NodeSet, domain: Domain) -> tuple[sp.csr_array, dict]:
    """One operator on ``domain`` and what it cost: seconds and seeded rows."""
    material = domain.material
    t0 = time.perf_counter()
    info: dict[str, float] = {}
    if label == "naive":
        op = naive_operator(nodes, material, plain_stencils(nodes, domain))
    elif label == "direct":
        op = direct_operator(nodes, material, plain_stencils(nodes, domain))
    elif label == "construction":
        op = interface_aware_operator(nodes, material, aware_stencils(nodes, domain))
    elif label in ("seeds", "seeds-plain", "seeds15"):
        stencils = reach_stencils(nodes, domain)
        info["rows"] = seeded_count(nodes, material, stencils)
        with refused(nodes.n, material):
            op = seed_operator(
                nodes,
                material,
                stencils,
                warp=label != "seeds-plain",
                tangential=True,
                flux=label != "seeds15",
            )
    else:
        raise ValueError(f"unknown operator {label!r}")
    info["seconds"] = time.perf_counter() - t0
    return op, info


@contextmanager
def refused(n: int, material):
    """``FOOT_CURVATURE``'s refusal, named with the ring and the count it came from.

    At δ > 0 the reach rule seeds rows 20δ and a stencil radius from the ring,
    whose normal lines reach nearer the circle's centre; the coarsest count the
    seeds take grows with δ (1250 at δ = 0; 2500 at δ = 0.001, where ten rows
    of the 1250-node set are refused).
    """
    try:
        yield
    except ValueError as err:
        if "FOOT_CURVATURE" not in str(err):
            raise
        s = 1.0 / material.gap if material.gap else float("nan")
        delta = getattr(material, "delta", 0.0)
        raise ValueError(
            f"the seeds refuse the ring at s = {s:g}, δ = {delta:g} on {n} nodes "
            f"(FOOT_CURVATURE): take a finer count. {err}"
        ) from err


def seeded_count(nodes: NodeSet, material, stencils: Stencils) -> int:
    (group,) = [g for g in stencils.groups if g.kind == INTERFACE_KIND]
    return int(seeded_rows(nodes, material, group.index).sum())


def solve(label: str, op: sp.csr_array, nodes: NodeSet) -> np.ndarray:
    ordering = PRODUCT_ORDERING if label == "naive" else None
    return solve_equilibrium(op, nodes, VALUES, permc_spec=ordering)


# --- the cache ---------------------------------------------------------------------


def load_cache(outputs: Path) -> dict[str, dict]:
    path = outputs / CACHE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return dict(data["entries"]) if data.get("meta") == CACHE_META else {}


def save_cache(outputs: Path, cache: dict[str, dict]) -> None:
    """Merge ``cache`` into the file under a lock: several runs may share it.

    The sweeps are independent per (s, δ), so the documented runs go faster as
    concurrent processes on disjoint ``--s`` / ``--smooth-s`` / ``--deltas``;
    each keeps what the others wrote and adds its own entries.
    """
    with open(outputs / (CACHE + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merged = load_cache(outputs)
        merged.update(cache)
        cache.update(merged)
        data = {"meta": CACHE_META, "entries": dict(sorted(merged.items()))}
        path = outputs / CACHE
        path.with_suffix(".tmp").write_text(json.dumps(data, indent=1) + "\n")
        path.with_suffix(".tmp").replace(path)


def key(*parts: object) -> str:
    """The cache key: floats by ``repr``, so no two values share one."""
    return "|".join(repr(p) if isinstance(p, float) else str(p) for p in parts)


# --- reading a fine solution ---------------------------------------------------------


def far_read(
    u: np.ndarray, fine: NodeSet, stencils: Stencils, material, nodes: NodeSet
) -> tuple[np.ndarray, np.ndarray]:
    """``u`` at the coarse nodes whose nearest fine node's stencil sees no interface.

    There ``resample`` reads the fine set's own standard interpolant, exact to
    its order on the smooth solution; nearer the ring its stencils are the
    interface group's (the seeds' reach at δ > 0, the crossing rows at δ = 0),
    whose read would need the seeds' basis, and the nodes are left out.
    """
    nearest = knn(fine.xy, 1, query=nodes.xy)[0][:, 0]
    keep = ~stencils.near_interface[nearest]
    got = resample(u, fine, stencils, material, nodes.x[keep], nodes.y[keep])
    return got, keep


def reference_file(outputs: Path, s: float, delta: float, n: int, seed: int) -> Path:
    return outputs / f"heat2d_ring_reference_s{tag(s)}_d{delta:g}_n{n}_seed{seed}.npz"


def seed_reference(
    s: float, delta: float, n: int, seed: int, iterations: int, outputs: Path
) -> tuple[Reference, bool]:
    """The fine seed run on eq. 40 at (s, δ), solved and cached if absent."""
    path = reference_file(outputs, s, delta, n, seed)
    if path.exists():
        ref = Reference.load(path)
        meta = ref.meta
        if (meta.get("iterations"), meta.get("cache")) == (iterations, CACHE_META):
            return ref, True
    domain = ring_domain(s, delta)
    t0 = time.perf_counter()
    nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
    t1 = time.perf_counter()
    op, info = build("seeds", nodes, domain)
    t2 = time.perf_counter()
    u = solve("seeds", op, nodes)
    t3 = time.perf_counter()
    meta = {
        "s": s,
        "delta": delta,
        "seed": seed,
        "iterations": iterations,
        "cache": CACHE_META,
        "rows": info["rows"],
        "seconds": {"nodes": t1 - t0, "operator": t2 - t1, "solve": t3 - t2},
    }
    ref = Reference(nodes, u, meta)
    ref.save(path)
    return ref, False


# --- part 1: Fig. 19's twin ----------------------------------------------------------


def convergence(
    s: float,
    counts: Sequence[int],
    reference_n: int,
    seed: int,
    iterations: int,
    outputs: Path,
    cache: dict[str, dict],
) -> tuple[list[dict], bool]:
    """E2.3 and the seeds at δ = 0 against E2.9's reference at ``s``, per count."""
    ref, reused = reference(s, reference_n, seed, iterations, outputs)
    domain = case3(s)
    ownership(ref.nodes, domain)
    stencils = None
    rows = []
    for n in counts:
        entries = {}
        todo = []
        for label in FIG19_LABELS:
            k = key("fig19", s, 0.0, n, label, reference_n, seed, iterations)
            if k in cache:
                entries[label] = cache[k]
            else:
                todo.append((label, k))
        if todo:
            if stencils is None:
                stencils = ref.stencils(domain)
            nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
            ownership(nodes, domain)
            full = resample(
                ref.u, ref.nodes, stencils, domain.material, nodes.x, nodes.y
            )
            far, keep = far_read(ref.u, ref.nodes, stencils, domain.material, nodes)
            for label, k in todo:
                op, info = build(label, nodes, domain)
                t0 = time.perf_counter()
                u = solve(label, op, nodes)
                cache[k] = entries[label] = {
                    "n": nodes.n,
                    "h": nodes.h,
                    "full": rms_error(u, full),
                    "far": rms_error(u[keep], far),
                    "far-share": float(keep.mean()),
                    "max": float(np.abs(u).max()),
                    "solve": time.perf_counter() - t0,
                    **info,
                }
            save_cache(outputs, cache)
        rows.append({"s": s, "n": n, **{lab: e for lab, e in entries.items()}})
    return rows, reused


# --- part 2: Fig. 20's twin ----------------------------------------------------------


def _residual(w: np.ndarray, u: np.ndarray) -> float:
    return float(abs(w @ u - 4.0) / (np.abs(w) @ np.abs(u)))


def conditioning(
    s: float, delta: float, n: int, seed: int, iterations: int
) -> dict[str, float]:
    """Over every seeded row of the constant ring at (s, δ): residuals and conditions.

    The matched profile is formed from the gap for the seeds and, at δ = 0,
    from the stored radii for the seeds without the gap and for E2.3, so each
    set of weights is measured against the geometry it sees. ``seeds15`` is
    the seeds without the flux seeds (§3.11), the ablation.
    """
    domain = ring_domain(s, delta, constant=True)
    material = domain.material
    nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
    r = np.hypot(nodes.x - 0.5, nodes.y - 0.5)
    u = matched_radial(material, r)
    stored = without_gap(material)
    u_stored = matched_radial(stored, r)
    index, _ = knn(nodes.xy, BOUNDARY.size)
    rows = index[seeded_rows(nodes, material, index)]
    out: dict[str, list[float]] = {}

    def add(name: str, value: float) -> None:
        out.setdefault(name, []).append(value)

    t0 = time.perf_counter()
    for idx in rows:
        xy = nodes.xy[idx]
        with refused(nodes.n, material):
            sb = seed_basis(xy, material, tangential=True, flux=True)
            sb15 = seed_basis(xy, material, tangential=True, flux=False)
        add("seeds", _residual(weights_of(sb), u[idx]))
        raw, scaled = block_condition(sb.block)
        add("block-raw", raw)
        add("block-scaled", scaled)
        add("system", float(np.linalg.cond(saddle_system(sb))))
        add("seeds15", _residual(weights_of(sb15), u[idx]))
        add("block15-raw", block_condition(sb15.block)[0])
        add("system15", float(np.linalg.cond(saddle_system(sb15))))
        if delta == 0.0:
            plain = seed_basis(xy, stored, tangential=True, flux=True)
            add("stored", _residual(weights_of(plain), u_stored[idx]))
            add("e23", _residual(stencil_weights(xy, stored, 4), u_stored[idx]))
            st = interface_stencil(xy, stored, 4)
            p, a = st.polynomial_block(), st.gaussian_block()
            q = p.shape[1]
            aug = np.block([[a, p], [p.T, np.zeros((q, q))]])
            add("e23-block", float(np.linalg.cond(p)))
            add("e23-system", float(np.linalg.cond(aug)))
    seconds = time.perf_counter() - t0
    row: dict[str, float] = {
        "s": s,
        "delta": delta,
        "n": nodes.n,
        "h": nodes.h,
        "rows": len(rows),
        "seconds": seconds,
    }
    for name, values in out.items():
        v = np.array(values)
        if name in ("seeds", "seeds15", "stored", "e23"):
            row[f"{name}-worst"], row[f"{name}-median"] = v.max(), float(np.median(v))
        else:
            row[f"{name}-mean"], row[f"{name}-max"] = v.mean(), v.max()
    return row


# --- part 3: the spectrum ------------------------------------------------------------


def spectrum(s: float, delta: float, n: int, seed: int, iterations: int) -> list[dict]:
    """The seed operator's interior eigenvalues and DDR on eq. 40 at (s, δ)."""
    domain = ring_domain(s, delta)
    nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
    rows = []
    for label in ("seeds", "seeds-plain"):
        op, info = build(label, nodes, domain)
        t0 = time.perf_counter()
        lam = interior_eigenvalues(op, nodes)
        system = reduced_system(op, nodes, VALUES)
        ddr = diagonal_dominance_ratio(system.a)
        stencils = reach_stencils(nodes, domain)
        (group,) = [g for g in stencils.groups if g.kind == INTERFACE_KIND]
        seeded = np.zeros(nodes.n, dtype=bool)
        seeded[group.rows[seeded_rows(nodes, domain.material, group.index)]] = True
        seeded = seeded[system.interior]
        rows.append(
            {
                "s": s,
                "delta": delta,
                "label": label,
                "n": nodes.n,
                "complex": int(np.sum(np.abs(lam.imag) > 1e-8 * np.abs(lam).max())),
                "positive": int((lam.real > 0).sum()),
                "max_re": float(lam.real.max()),
                "min_re_h2": float(lam.real.min() * nodes.h**2),
                "max_im_h2": float(np.abs(lam.imag).max() * nodes.h**2),
                "ddr-seeded-min": float(ddr[seeded].min()),
                "ddr-seeded-median": float(np.median(ddr[seeded])),
                "ddr-all-min": float(ddr.min()),
                "ddr-all-median": float(np.median(ddr)),
                "seconds": info["seconds"] + time.perf_counter() - t0,
            }
        )
    return rows


# --- part 4: the smooth ring ---------------------------------------------------------


def probe_masks(nodes: NodeSet, domain: Domain) -> dict[str, np.ndarray]:
    """The rows the seeds rebuild (``seeded``) and E2.3's crossing rows."""
    material = domain.material
    seeded = np.zeros(nodes.n, dtype=bool)
    stencils = reach_stencils(nodes, domain)
    (group,) = [g for g in stencils.groups if g.kind == INTERFACE_KIND]
    seeded[group.rows[seeded_rows(nodes, material, group.index)]] = True
    crossing = np.zeros(nodes.n, dtype=bool)
    stencils = aware_stencils(nodes, domain)
    (group,) = [g for g in stencils.groups if g.kind == INTERFACE_KIND]
    crossing[group.rows[interface_crossings(nodes, material, group.index)]] = True
    return {"seeded": seeded, "crossing": crossing}


def smooth_ring(
    s: float,
    delta: float,
    counts: Sequence[int],
    fine: tuple[Reference, Stencils] | None,
    fine_n: int,
    seed: int,
    iterations: int,
    outputs: Path,
    cache: dict[str, dict],
) -> list[dict]:
    """Per count: each operator's probe on the constant ring and, with a fine
    run, its far-field error on eq. 40, at (s, δ)."""
    rows = []
    for n in counts:
        row: dict = {"s": s, "delta": delta, "n": n}
        todo = [
            lab
            for lab in probe_labels(delta)
            if key("probe", s, delta, n, lab, seed, iterations) not in cache
        ]
        if todo:
            domain = ring_domain(s, delta, constant=True)
            nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
            u = exact_mode(domain)(nodes.x, nodes.y)
            masks = probe_masks(nodes, domain)
            for label in todo:
                op, info = build(label, nodes, domain)
                residual = op @ u
                cache[key("probe", s, delta, n, label, seed, iterations)] = {
                    "n": nodes.n,
                    "h": nodes.h,
                    **{
                        name: float(np.sqrt(np.mean(residual[mask] ** 2)))
                        for name, mask in masks.items()
                    },
                    **{f"{name}-rows": int(mask.sum()) for name, mask in masks.items()},
                    **info,
                }
            save_cache(outputs, cache)
        for label in probe_labels(delta):
            row[f"probe-{label}"] = cache[
                key("probe", s, delta, n, label, seed, iterations)
            ]
        if fine is not None:
            ref, stencils = fine
            todo = [
                lab
                for lab in LABELS
                if key("error", s, delta, n, lab, fine_n, seed, iterations) not in cache
            ]
            if todo:
                domain = ring_domain(s, delta)
                nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
                far, keep = far_read(ref.u, ref.nodes, stencils, domain.material, nodes)
                for label in todo:
                    op, info = build(label, nodes, domain)
                    u = solve(label, op, nodes)
                    cache[
                        key("error", s, delta, n, label, fine_n, seed, iterations)
                    ] = {
                        "n": nodes.n,
                        "h": nodes.h,
                        "far": rms_error(u[keep], far),
                        "far-share": float(keep.mean()),
                        **info,
                    }
                save_cache(outputs, cache)
            for label in LABELS:
                k = key("error", s, delta, n, label, fine_n, seed, iterations)
                row[f"error-{label}"] = cache[k]
        rows.append(row)
    return rows


# --- tables --------------------------------------------------------------------------


def _rate(a: dict, b: dict, name: str) -> float:
    return float(np.log(a[name] / b[name]) / np.log(a["h"] / b["h"]))


def _fit(entries: list[dict], name: str) -> float:
    if len(entries) < 2:
        return float("nan")
    h = np.array([e["h"] for e in entries])
    v = np.array([e[name] for e in entries])
    return float(np.polyfit(np.log(h), np.log(v), 1)[0])


def _order(value: float) -> str:
    return "    -" if not np.isfinite(value) else f"{value:5.2f}"


def print_convergence(results: dict[float, list[dict]]) -> None:
    print(
        "\nFig. 19's twin: eq. 40 at δ = 0, RMS error against E2.9's reference at the"
        " same s, read through its stencils at every node (full) and through its"
        " standard interpolant away from the ring (far; the share of the nodes read);"
        " order per halving of h; the seeds' marched rows and their cost"
    )
    for s, rows in results.items():
        print(
            f"\ns = {tag(s)}\n     n       h |    E2.3 full  order     far |"
            "   seeds full  order     far | seeds/E2.3 | seeds (15)  order  /E2.3"
            " | far share  rows   ms/row"
        )
        for i, r in enumerate(rows):
            c, d, f = r["construction"], r["seeds"], r["seeds15"]
            rc = _rate(rows[i - 1]["construction"], c, "full") if i else float("nan")
            rs = _rate(rows[i - 1]["seeds"], d, "full") if i else float("nan")
            rf = _rate(rows[i - 1]["seeds15"], f, "full") if i else float("nan")
            print(
                f"{int(c['n']):6d}  {c['h']:.4f} |  {c['full']:10.3e}  {_order(rc)}"
                f"  {c['far']:.2e} |  {d['full']:10.3e}  {_order(rs)}  {d['far']:.2e} |"
                f"   {d['full'] / c['full']:7.2f}  |  {f['full']:9.3e}  {_order(rf)}"
                f"  {f['full'] / c['full']:5.2f} |    {d['far-share']:.2f}"
                f"  {int(d['rows']):5d}  {1e3 * d['seconds'] / max(d['rows'], 1):6.1f}"
            )
        fits = ", ".join(
            f"{NAMES[lab]} {_fit([r[lab] for r in rows], 'full'):.2f}"
            for lab in FIG19_LABELS
        )
        print(f"fit over the counts: {fits}")
    first = next(iter(results))
    print(
        f"\nratio of each s line to the s = {tag(first)} line at each count (seeds): "
        + "; ".join(
            f"{tag(s)}: "
            + ", ".join(
                f"{a['seeds']['full'] / b['seeds']['full']:.2f}"
                for a, b in zip(rows, results[first], strict=True)
            )
            for s, rows in results.items()
            if s != first
        )
    )


def print_conditioning(rows: list[dict]) -> None:
    r0 = rows[0]
    print(
        f"\nFig. 20's twin at {int(r0['n'])} nodes, the ring at its constant part: the"
        " worst and median relative residual of the weights on the matched radial"
        " profile over the seeded rows (the seeds with the gap; without the flux"
        " seeds, (15); at δ = 0 also on the stored radii and E2.3); mean 2-norm"
        " condition numbers of the seed block raw and column-scaled and of the seed"
        " system (with the 15 seeds' beside them), and of E2.3's polynomial block"
        " and system"
    )
    print(
        "     s        δ  rows |   seeds worst   median |    (15) worst |"
        "  stored worst   median |    E2.3 worst   median |"
        " block raw  scaled   system |  (15) raw  system |  E2.3 P   system | ms/row"
    )
    for r in rows:
        stored = (
            f"  {r['stored-worst']:12.2e} {r['stored-median']:8.1e}"
            if "stored-worst" in r
            else f"  {'-':>12s} {'-':>8s}"
        )
        e23 = (
            f"  {r['e23-worst']:12.2e} {r['e23-median']:8.1e}"
            if "e23-worst" in r
            else f"  {'-':>12s} {'-':>8s}"
        )
        blocks = (
            f"  {r['e23-block-mean']:7.1e}  {r['e23-system-mean']:7.1e}"
            if "e23-block-mean" in r
            else f"  {'-':>7s}  {'-':>7s}"
        )
        print(
            f"  {tag(r['s']):>4s}  {r['delta']:7g}  {int(r['rows']):4d} |"
            f"  {r['seeds-worst']:12.2e} {r['seeds-median']:8.1e} |"
            f"  {r['seeds15-worst']:12.2e} |{stored} |{e23} |"
            f"  {r['block-raw-mean']:7.1e} {r['block-scaled-mean']:7.1e}"
            f"  {r['system-mean']:7.1e} |  {r['block15-raw-mean']:7.1e}"
            f"  {r['system15-mean']:7.1e} |{blocks} |"
            f"  {1e3 * r['seconds'] / max(r['rows'], 1):5.1f}"
        )


def print_spectrum(rows: list[dict]) -> None:
    print(
        f"\nthe seed operator's interior spectrum at {int(rows[0]['n'])} nodes, warped"
        " and plain (complex: |Im| > 1e-8 max |λ|), and the DDR of its reduced rows"
        " (the seeded rows and all, least / median)"
    )
    print(
        "     s        δ  operator     complex  positive     max Re  h² min Re"
        "  h² max |Im| |  DDR seeded        all   | time"
    )
    for r in rows:
        print(
            f"  {tag(r['s']):>4s}  {r['delta']:7g}  {r['label']:11s}  {r['complex']:6d}"
            f"  {r['positive']:8d}  {r['max_re']:9.3f}  {r['min_re_h2']:9.3f}"
            f"  {r['max_im_h2']:11.3f} |  {r['ddr-seeded-min']:.3f}"
            f" {r['ddr-seeded-median']:.3f}"
            f"  {r['ddr-all-min']:.3f} {r['ddr-all-median']:.3f} | {r['seconds']:5.1f}s"
        )


def print_smooth(results: dict[tuple[float, float], list[dict]]) -> None:
    print(
        "\nthe smooth ring: per (s, δ), each operator's truncation probe (RMS of its"
        " rows on the exact mode through the constant ring, over the rows the seeds"
        " rebuild) and, with a fine run, its RMS error on eq. 40 against the fine"
        " seed run, read away from the ring; order per halving of h"
    )
    for (s, delta), rows in results.items():
        print(f"\ns = {tag(s)}, δ = {delta:g}")
        labels = probe_labels(delta)
        head = "     n       h  rows |" + "".join(
            f"  {NAMES[lab]:>10s}  order" for lab in labels
        )
        print("probe: " + head)
        for i, r in enumerate(rows):
            cells = []
            for lab in labels:
                e = r[f"probe-{lab}"]
                rate = (
                    _rate(rows[i - 1][f"probe-{lab}"], e, "seeded")
                    if i
                    else float("nan")
                )
                cells.append(f"  {e['seeded']:10.2e}  {_order(rate)}")
            e = r["probe-seeds"]
            print(
                f"       {int(e['n']):6d}  {e['h']:.4f} {int(e['seeded-rows']):5d} |"
                + "".join(cells)
            )
        entries = {lab: [r[f"probe-{lab}"] for r in rows] for lab in labels}
        print(
            "       fit: "
            + ", ".join(
                f"{NAMES[lab]} {_fit(entries[lab], 'seeded'):.2f}" for lab in labels
            )
        )
        if "error-seeds" not in rows[0]:
            continue
        head = "     n       h   far |" + "".join(
            f"  {NAMES[lab]:>10s}  order" for lab in LABELS
        )
        print("error: " + head)
        for i, r in enumerate(rows):
            cells = []
            for lab in LABELS:
                e = r[f"error-{lab}"]
                rate = (
                    _rate(rows[i - 1][f"error-{lab}"], e, "far") if i else float("nan")
                )
                cells.append(f"  {e['far']:10.2e}  {_order(rate)}")
            e = r["error-seeds"]
            print(
                f"       {int(e['n']):6d}  {e['h']:.4f}  {e['far-share']:.2f} |"
                + "".join(cells)
            )
        entries = {lab: [r[f"error-{lab}"] for r in rows] for lab in LABELS}
        print(
            "       fit: "
            + ", ".join(
                f"{NAMES[lab]} {_fit(entries[lab], 'far'):.2f}" for lab in LABELS
            )
        )


# --- figures -------------------------------------------------------------------------


def _counts_axis(ax, counts) -> None:
    """The node counts as the only ticks, as E2.9's figures label them."""
    ticks = sorted({int(n) for n in counts})
    ax.set_xticks(ticks, [str(k) for k in ticks], fontsize=7, rotation=45)
    ax.set_xticks([], minor=True)


def _colours(values: Sequence[float]) -> dict[float, tuple]:
    cmap = matplotlib.colormaps["viridis"]
    return {
        v: cmap(t)
        for v, t in zip(values, np.linspace(0.05, 0.85, len(values)), strict=True)
    }


def figure_convergence(results: dict[float, list[dict]]):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colours = _colours(tuple(results))
    for s, rows in results.items():
        n = [r["seeds"]["n"] for r in rows]
        marker = MARKERS.get(s, "s")
        ax.loglog(
            n,
            [r["construction"]["full"] for r in rows],
            marker + "--",
            color=colours[s],
            markerfacecolor="none",
            markersize=6,
            linewidth=0.8,
        )
        ax.loglog(
            n,
            [r["seeds"]["full"] for r in rows],
            marker + "-",
            color=colours[s],
            markersize=6,
            label=power(s),
        )
        ax.loglog(
            n,
            [r["seeds15"]["full"] for r in rows],
            ":",
            color=colours[s],
            linewidth=0.8,
        )
    first = results[next(iter(results))]
    n = np.array([r["seeds"]["n"] for r in first], dtype=float)
    e0 = first[0]["seeds"]["full"]
    ax.loglog(n, e0 * (n / n[0]) ** -2.0, "k:", linewidth=0.7, label="4th order")
    ax.plot([], [], "k-", label="seeds (tangential, warped)")
    ax.plot([], [], "k--", markerfacecolor="none", label="E2.3 (E2.9's curved line)")
    ax.plot([], [], "k:", linewidth=0.8, label="seeds without the flux seeds (15)")
    _counts_axis(ax, [r["seeds"]["n"] for rows in results.values() for r in rows])
    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("RMS error in u against the reference at the same s")
    ax.set_title("eq. 40 at δ = 0 with seeds (EABE Fig. 19 twin)")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    return fig


def figure_conditioning(rows: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    zero = [r for r in rows if r["delta"] == 0.0]
    s = np.array([r["s"] for r in zero])
    ax = axes[0]
    ax.loglog(s, [r["e23-worst"] for r in zero], "o-", color=CONSTRUCTION, label="E2.3")
    ax.loglog(
        s,
        [r["stored-worst"] for r in zero],
        "^-",
        color=NAIVE,
        label="seeds on the stored radii",
    )
    ax.loglog(
        s, [r["seeds-worst"] for r in zero], "s-", color=AWARE, label="seeds, δ = 0"
    )
    deltas = sorted({r["delta"] for r in rows if r["delta"] > 0.0}, reverse=True)
    for d, alpha in zip(
        deltas, np.linspace(0.8, 0.35, max(len(deltas), 1)), strict=False
    ):
        line = [r for r in rows if r["delta"] == d]
        ax.loglog(
            [r["s"] for r in line],
            [r["seeds-worst"] for r in line],
            "s:",
            color=AWARE,
            alpha=alpha,
            label=f"seeds, δ = {d:g}",
        )
    ax.loglog(
        s, E29_RESIDUAL * s, "k:", linewidth=0.7, label=r"$1.5\times10^{-18}\,s$ (E2.9)"
    )
    ax.set_xlabel("extremizing parameter s")
    ax.set_ylabel("worst relative residual on the matched radial profile")
    ax.set_title("the weights' exactness through the ring")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    ax.loglog(
        s, [r["e23-block-mean"] for r in zero], "o-", color=CONSTRUCTION, label="E2.3 P"
    )
    ax.loglog(
        s,
        [r["e23-system-mean"] for r in zero],
        "o--",
        color=CONSTRUCTION,
        label="E2.3 system",
    )
    ax.loglog(
        s,
        [r["block-raw-mean"] for r in zero],
        "s-",
        color=AWARE,
        label="seed block, raw",
    )
    ax.loglog(
        s,
        [r["block-scaled-mean"] for r in zero],
        "s:",
        color=AWARE,
        label="seed block, columns scaled",
    )
    ax.loglog(
        s, [r["system-mean"] for r in zero], "s--", color=AWARE, label="seed system"
    )
    ax.set_xlabel("extremizing parameter s")
    ax.set_ylabel("2-norm condition number, mean over the rows")
    ax.set_title(
        f"the stencil systems at δ = 0, N = {int(zero[0]['n'])} (Fig. 20 twin)"
    )
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def figure_smooth(results: dict[tuple[float, float], list[dict]]):
    deltas = sorted({d for _, d in results})
    has_error = any("error-seeds" in rows[0] for rows in results.values())
    nrows = 2 if has_error else 1
    fig, axes = plt.subplots(
        nrows, len(deltas), figsize=(3.6 * len(deltas), 3.4 * nrows), squeeze=False
    )
    s_values = sorted({s for s, _ in results})
    styles = dict(zip(s_values, ("-", "--", ":", "-."), strict=False))
    for j, d in enumerate(deltas):
        for s in s_values:
            rows = results.get((s, d))
            if not rows:
                continue
            for lab in LABELS:
                n = [r[f"probe-{lab}"]["n"] for r in rows]
                axes[0, j].loglog(
                    n,
                    [r[f"probe-{lab}"]["seeded"] for r in rows],
                    "o" + styles[s],
                    color=COLOURS[lab],
                    markersize=3,
                    label=f"{NAMES[lab]}, {power(s)}" if j == 0 else None,
                )
                if has_error and f"error-{lab}" in rows[0]:
                    axes[1, j].loglog(
                        n,
                        [r[f"error-{lab}"]["far"] for r in rows],
                        "o" + styles[s],
                        color=COLOURS[lab],
                        markersize=3,
                    )
        title = "the jump" if d == 0.0 else f"δ = {d:g}"
        axes[0, j].set_title(f"probe, {title}", fontsize=9)
        counts = [
            r["probe-seeds"]["n"]
            for (_, dd), rows in results.items()
            if dd == d
            for r in rows
        ]
        if has_error:
            if d == 0.0:
                axes[1, j].set_visible(False)
            axes[1, j].set_title(f"error away from the ring, {title}", fontsize=9)
        for ax in axes[:, j]:
            ax.grid(True, which="both", linewidth=0.3)
            ax.set_xlabel("N")
            _counts_axis(ax, counts)
    axes[0, 0].set_ylabel("RMS of L u over the seeded rows")
    if has_error:
        first = next(j for j, d in enumerate(deltas) if d > 0.0)
        axes[1, first].set_ylabel("RMS error against the fine seed run")
    axes[0, 0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    return fig


# --- main ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--s", type=float, nargs="+", default=[1e3, 1e11])
    parser.add_argument("--counts", type=int, nargs="*", default=[1250, 2500])
    parser.add_argument("--reference-n", type=int, default=20000)
    parser.add_argument("--conditioning-s", type=float, nargs="+", default=[1e3, 1e11])
    parser.add_argument("--conditioning-n", type=int, default=2500)
    parser.add_argument("--conditioning-smooth-s", type=float, nargs="*", default=[])
    parser.add_argument("--deltas", type=float, nargs="*", default=[0.001])
    parser.add_argument("--smooth-s", type=float, nargs="*", default=[1e3])
    parser.add_argument("--probe-counts", type=int, nargs="*", default=[2500])
    parser.add_argument("--fine-n", type=int, default=0)
    parser.add_argument("--fine-s", type=float, nargs="*", default=None)
    parser.add_argument("--spectrum-n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    fine_s = args.smooth_s if args.fine_s is None else args.fine_s
    single = [n for n in (args.conditioning_n, args.spectrum_n) if n]
    if any(n < MIN_COUNT for n in args.counts + args.probe_counts + single):
        parser.error(f"counts must be {MIN_COUNT} nodes or more (FOOT_CURVATURE)")
    if args.counts and args.reference_n <= max(args.counts):
        parser.error("the reference must have more nodes than every count")
    if args.fine_n and args.probe_counts and args.fine_n <= max(args.probe_counts):
        parser.error("the fine run must have more nodes than every probe count")
    if any(s <= 1.0 / 0.3 for s in args.s + args.conditioning_s + args.smooth_s):
        parser.error("s must exceed 1/0.3, the ring reaching the cooling circle")
    if any(d <= 0.0 for d in args.deltas):
        parser.error("the smooth widths must be positive (δ = 0 always runs)")
    args.outputs.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.outputs)
    results = ResultsCache("heat2d_ring", vars(args))
    tables: dict = {}

    if args.counts:
        t0 = time.perf_counter()
        fig19: dict[float, list[dict]] = {}
        for s in args.s:
            fig19[s], _ = convergence(
                s,
                args.counts,
                args.reference_n,
                args.seed,
                args.iterations,
                args.outputs,
                cache,
            )
        print_convergence(fig19)
        results.time("convergence", time.perf_counter() - t0)
        fig = figure_convergence(fig19)
        fig.savefig(args.outputs / "heat2d_ring_convergence.png", dpi=150)
        plt.close(fig)
        tables["convergence"] = fig19
        results.add("convergence", fig19)

    if args.conditioning_n:
        t0 = time.perf_counter()
        rows = []
        smooth_s = args.conditioning_smooth_s
        for s in sorted(set(args.conditioning_s) | set(smooth_s)):
            for delta in (0.0, *args.deltas) if s in smooth_s else (0.0,):
                k = key(
                    "fig20", s, delta, args.conditioning_n, args.seed, args.iterations
                )
                if k not in cache:
                    cache[k] = conditioning(
                        s, delta, args.conditioning_n, args.seed, args.iterations
                    )
                    save_cache(args.outputs, cache)
                rows.append(cache[k])
        print_conditioning(rows)
        results.time("conditioning", time.perf_counter() - t0)
        fig = figure_conditioning(rows)
        fig.savefig(args.outputs / "heat2d_ring_conditioning.png", dpi=150)
        plt.close(fig)
        tables["conditioning"] = rows
        results.add("conditioning", rows)

    if args.spectrum_n:
        t0 = time.perf_counter()
        pairs = [(s, 0.0) for s in args.s] + [
            (s, d) for s in args.smooth_s for d in args.deltas
        ]
        rows = []
        for s, delta in pairs:
            k = key("spectrum", s, delta, args.spectrum_n, args.seed, args.iterations)
            if k not in cache:
                cache[k] = {
                    "rows": spectrum(
                        s, delta, args.spectrum_n, args.seed, args.iterations
                    )
                }
                save_cache(args.outputs, cache)
            rows.extend(cache[k]["rows"])
        print_spectrum(rows)
        results.time("spectrum", time.perf_counter() - t0)
        tables["spectrum"] = rows
        results.add("spectrum", rows)

    if args.probe_counts and args.smooth_s:
        t0 = time.perf_counter()
        smooth: dict[tuple[float, float], list[dict]] = {}
        for s in args.smooth_s:
            for delta in (0.0, *args.deltas):
                fine = None
                if args.fine_n and s in fine_s and delta > 0.0:
                    ref, _ = seed_reference(
                        s, delta, args.fine_n, args.seed, args.iterations, args.outputs
                    )
                    fine = (ref, reach_stencils(ref.nodes, ring_domain(s, delta)))
                smooth[(s, delta)] = smooth_ring(
                    s,
                    delta,
                    args.probe_counts,
                    fine,
                    args.fine_n,
                    args.seed,
                    args.iterations,
                    args.outputs,
                    cache,
                )
        print_smooth(smooth)
        results.time("smooth", time.perf_counter() - t0)
        fig = figure_smooth(smooth)
        fig.savefig(args.outputs / "heat2d_ring_smooth.png", dpi=150)
        plt.close(fig)
        table = {f"{tag(s)}|{d:g}": rows for (s, d), rows in smooth.items()}
        tables["smooth"] = table
        results.add("smooth", table)

    paths = [args.outputs / RESULTS]
    if args.data_dir is not None:
        paths.append(args.data_dir / RESULTS)
    results.write(*paths)
    return tables


if __name__ == "__main__":
    main()
