"""E2.7 (#21): case 3, the 0.001-wide insulating ring around a cooling unit (EABE
eq. 37–39): FD4 / flat / curved convergence against a fine reference (EABE
Fig. 14, dissertation Fig. 5-14) and the mesh plot of the 40,000-node solution
(Fig. 13 / 5-13).

Two figures under ``outputs/`` and four tables on stdout.

The reference is the interface-aware solution (curvature, warped Gaussians,
the straddling rows on the ring's midline) on ``--reference-n`` nodes, cached
as ``outputs/heat2d_case3_reference_n<N>_seed<s>.npz`` with its node set and
timings (``heat2d.resample.Reference``), and read at the coarse nodes through
its own stencils (``heat2d.resample.resample``), the three-region translated
basis where they cross the ring. Every node set is checked for ownership
first: no node of any count lies inside the ring (the rows straddle its
midline and the free nodes keep clear), so the operator's α and the reference's
agree on every node's piece by construction.

``heat2d_case3_convergence.png``: RMS error against the node count for the
Cartesian FD4 baseline (``heat2d.fd4``, the cooling disc a staircase Dirichlet
set), the flat-interface RBF-FD variant and the curvature-included one, next
to Fig. 5-14's markers read off the rendered page; the curvature-included
operator with plain Gaussians is added dashed, since on the ring the warp
shifts the far side by 1.5 stencil radii (port notes §2.4, the E2.4 note on
#21) and the two settings differ more than on case 2. ``--fd4-counts`` may
run the grid past the RBF-FD counts (plan D7: to what one machine solves
directly in minutes).

``heat2d_case3_solution.png``: the 40,000-node solution as a surface over the
node set, the disc cut out (Fig. 13); ``--mesh-n`` picks the count, and the
reference is reused when it has that many nodes.

The resampling check reads the harmonic mode ``R(r) cos 2θ`` through the
ring at its constant part (``heat2d.exact.ring_exact``: ``α = 1/1500`` on the
ring, an exact solution regular at the centre), sampled on a node set of
``--check-n`` nodes, back at the coarse nodes and the grid points within 0.1
of the ring's midline, aware and blind, as the case-2 driver did with case
1's solution over the whole strip: what the reading itself contributes at a
1500 : 1 contrast. The mode is not periodic in x, so points whose fine
stencil wraps the seam are left out; §2.6 covered the plain read on a smooth
periodic function everywhere.

    uv run python scripts/heat2d_case3.py                              # ~1.5 min
    uv run python scripts/heat2d_case3.py --reference-n 160000 \\
        --counts 1250 2500 5000 10000 20000 40000 80000 \\
        --fd4-counts 1250 2500 5000 10000 20000 40000 80000 160000 320000 \\
        640000 1280000                                                # ~29 min
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

from heat_interfaces.heat2d import (  # noqa: E402
    BOUNDARY,
    COOLING_RADIUS,
    RING,
    Band,
    Circle,
    Constant2D,
    Domain,
    NodeSet,
    Reference,
    Stencils,
    build_node_set,
    build_stencils,
    cartesian_grid,
    case3,
    fd4_operator,
    interface_aware_operator,
    reference_solution,
    resample,
    ring_exact,
    rms_error,
    solve_equilibrium,
)
from heat_interfaces.plotting import AWARE, NAIVE, REFERENCE  # noqa: E402

FIG14 = {
    "fd4": {
        1250: 2.3e-1,
        2500: 2.3e-1,
        5000: 2.3e-1,
        10000: 2.25e-1,
        20000: 2.2e-1,
        40000: 2.2e-1,
        80000: 2.2e-1,
        160000: 2.15e-1,
        320000: 2.1e-1,
        640000: 2.1e-1,
        1280000: 2.05e-1,
        2560000: 1.85e-1,
    },
    "flat": {
        1250: 1.4e-3,
        2500: 8.0e-4,
        5000: 3.9e-4,
        10000: 2.2e-4,
        20000: 1.9e-4,
        40000: 2.0e-4,
        80000: 2.5e-4,
    },
    "curved": {
        1250: 1.35e-3,
        2500: 5.9e-4,
        5000: 2.5e-4,
        10000: 7.7e-5,
        20000: 2.0e-5,
        40000: 3.7e-6,
        80000: 8.1e-7,
    },
}
"""EABE Fig. 14 (= dissertation Fig. 5-14) read off the rendered page (2026-09-21).

Seven markers per RBF-FD line, 1250 to 80,000 nodes; twelve FD4 markers by
doublings from 1250 to 2,560,000 (1600 in each direction); reading accuracy
about ±15 % (the FD4 line sits on a coarse axis, about ±5 %).
"""

VARIANTS = (("flat", False, True), ("curved", True, True), ("plain", True, False))
"""``(name, curvature, warped Gaussians)``, all on the straddled node sets.

Fig. 14's two RBF-FD lines are ``flat`` and ``curved``; ``plain`` is the
curvature-included operator with plain Gaussians, the second setting the E2.4
note on #21 asked for.
"""

DISC = 2
"""``boundary_index`` of the cooling circle: the third Dirichlet curve of case 3."""

NEAR_RING = 0.1
"""Half-width of the annulus about the ring's midline the resampling check reads in."""


def edge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Eq. 39's data on both rows, ``sin 6πx`` at ``y = 0`` and ``y = 1``."""
    return np.sin(6 * np.pi * x)


VALUES = [edge, edge, 0.0]
"""Bottom row, top row, cooling circle (the order of ``case3().dirichlet``)."""


def constant_ring() -> Domain:
    """Case 3's geometry with the ring at its constant part (``ring_exact``)."""
    band = Band(
        Circle(RING[0]), Circle(RING[1]), Constant2D(1.0 / 1500.0), Constant2D(1.0)
    )
    return replace(case3(), material=band)


# --- ownership -----------------------------------------------------------------


def ownership(nodes: NodeSet, domain: Domain) -> dict[str, int]:
    """How many nodes each region holds; the ring's count must be zero.

    The E1 note on #21: the ring is narrower than the spacing at every count,
    so the straddling rows decide which nodes are inside it. The rows sit at
    ``±0.5 h`` off the midline and the free nodes keep ``(0.5 + √3) h`` clear,
    so no node is inside for ``h > 0.001``; the operator's α and the
    reference's are then the same piece at every node by construction.
    """
    region = domain.material.region_index(nodes.x, nodes.y)
    counts = {f"region-{k}": int((region == k).sum()) for k in range(3)}
    if counts["region-1"]:
        raise RuntimeError(f"{counts['region-1']} nodes lie inside the ring")
    if nodes.straddle_rows:
        lo, hi = nodes.rows_of(0, 0.5)
        r = np.hypot(nodes.x - 0.5, nodes.y - 0.5)
        if not (np.all(r[lo.index] < RING[0]) and np.all(r[hi.index] > RING[1])):
            raise RuntimeError("the innermost pair does not straddle both circles")
    return counts


# --- the reference ---------------------------------------------------------------


def reference_path(outputs: Path, n: int, seed: int) -> Path:
    return outputs / f"heat2d_case3_reference_n{n}_seed{seed}.npz"


def reference(
    domain: Domain, n: int, seed: int, iterations: int, outputs: Path
) -> tuple[Reference, bool]:
    """The cached fine solution, solved and saved if absent; ``True`` when reused."""
    path = reference_path(outputs, n, seed)
    if path.exists():
        ref = Reference.load(path)
        if ref.meta.get("iterations") == iterations and ref.meta.get("n") == n:
            return ref, True
    ref = reference_solution(domain, n, VALUES, seed=seed, iterations=iterations)
    ref.save(path)
    return ref, False


def error_against(
    ref: Reference,
    stencils: Stencils,
    material,
    nodes: NodeSet,
    u: np.ndarray,
    keep: np.ndarray | None = None,
) -> tuple[float, float]:
    """``(RMS error, seconds)`` of ``u`` against the reference read at ``nodes``.

    ``keep`` restricts both to a subset (the grid points outside the disc).
    """
    if keep is None:
        keep = np.ones(nodes.n, dtype=bool)
    t0 = time.perf_counter()
    u_ref = resample(ref.u, ref.nodes, stencils, material, nodes.x[keep], nodes.y[keep])
    return rms_error(u[keep], u_ref), time.perf_counter() - t0


def reference_error_estimate(
    rows: list[dict[str, float]], ref_n: int, name: str = "curved"
) -> tuple[float, int]:
    """The reference's own error from the finest ``name`` point, if ``e ∝ N⁻²``.

    As in the case-2 driver: ``e(R) = q m / (1 − q)`` with ``q = (N / R)²``.
    """
    finest = max(rows, key=lambda r: r["n"])
    q = (finest["n"] / ref_n) ** 2
    return finest[name] * q / (1.0 - q), int(finest["n"])


# --- the sweeps ------------------------------------------------------------------


def sweep(
    counts: tuple[int, ...],
    ref: Reference,
    ref_stencils: Stencils,
    seed: int = 0,
    iterations: int = 100,
) -> list[dict[str, float]]:
    """One row per count: the three variants' errors, the group size and times.

    ``nodes`` is the node set with its stencils (shared), ``<name>-build`` the
    operator, ``<name>-solve`` the solve and ``<name>-resample`` the reading
    of the reference at the nodes; ``region-k`` the ownership counts.
    """
    domain = case3()
    rows = []
    for n in counts:
        t0 = time.perf_counter()
        nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
        stencils = build_stencils(nodes, domain, interface=BOUNDARY)
        row: dict[str, float] = {
            "n": nodes.n,
            "h": nodes.h,
            "group": int(stencils.near_interface.sum()),
            "nodes": time.perf_counter() - t0,
            **ownership(nodes, domain),
        }
        for name, curvature, warp in VARIANTS:
            t0 = time.perf_counter()
            op = interface_aware_operator(
                nodes, domain.material, stencils, curvature=curvature, warp=warp
            )
            t1 = time.perf_counter()
            u = solve_equilibrium(op, nodes, VALUES)
            t2 = time.perf_counter()
            row[name], row[f"{name}-resample"] = error_against(
                ref, ref_stencils, domain.material, nodes, u
            )
            row[f"{name}-build"] = t1 - t0
            row[f"{name}-solve"] = t2 - t1
            row[f"{name}-max"] = float(np.abs(u).max())
        rows.append(row)
    return rows


def no_ring() -> Domain:
    """Case 3's geometry with ``α ≡ 1``: what a grid that misses the ring solves."""
    band = Band(Circle(RING[0]), Circle(RING[1]), Constant2D(1.0), Constant2D(1.0))
    return replace(case3(), material=band)


def fd4_sweep(
    counts: tuple[int, ...],
    ref: Reference,
    ref_stencils: Stencils,
    blind_stencils: Stencils,
) -> list[dict[str, float]]:
    """One row per count: the Cartesian FD4 error outside the disc, counts and times.

    ``disc`` is the staircase Dirichlet set's size, ``ring`` the grid points
    α samples inside the ring (FD4 sees the ring through those alone);
    ``no-ring`` is the same grid solved with ``α ≡ 1``, the floor a grid that
    misses the ring converges to; ``blind`` is the FD4 error with the
    reference read through stencils without the interface group, the way a
    plain interpolant would have moved it to the grid.
    """
    domain, control = case3(), no_ring()
    rows = []
    for n in counts:
        t0 = time.perf_counter()
        grid = cartesian_grid(domain, n)
        op = fd4_operator(grid, domain.material)
        t1 = time.perf_counter()
        u = solve_equilibrium(op, grid, VALUES)
        t2 = time.perf_counter()
        keep = grid.boundary_index != DISC
        err, seconds = error_against(
            ref, ref_stencils, domain.material, grid, u, keep=keep
        )
        blind, _ = error_against(
            ref, blind_stencils, domain.material, grid, u, keep=keep
        )
        u0 = solve_equilibrium(fd4_operator(grid, control.material), grid, VALUES)
        floor, _ = error_against(
            ref, ref_stencils, domain.material, grid, u0, keep=keep
        )
        region = domain.material.region_index(grid.x, grid.y)
        rows.append(
            {
                "n": grid.n,
                "h": grid.h,
                "fd4": err,
                "no-ring": floor,
                "blind": blind,
                "disc": int((~keep).sum()),
                "ring": int((region == 1).sum()),
                "fd4-build": t1 - t0,
                "fd4-solve": t2 - t1,
                "fd4-resample": seconds,
            }
        )
    return rows


def resampling_check(
    fine_n: int, counts: tuple[int, ...], seed: int = 0, iterations: int = 100
) -> list[dict[str, float]]:
    """The ring's harmonic mode on a fine set, read back at coarse nodes and grids.

    ``aware`` uses the fine set's stencils with the interface group, ``blind``
    the same set without it; RMS and largest error against the exact values
    over the points within ``NEAR_RING`` of the midline (the mode is not
    periodic in x, so the seam is kept out of reach).
    """
    domain, exact = constant_ring(), ring_exact()
    t0 = time.perf_counter()
    fine = build_node_set(domain, fine_n, seed=seed, iterations=iterations)
    u = exact(fine.x, fine.y)
    stencils = {
        "aware": build_stencils(fine, domain, interface=BOUNDARY),
        "blind": build_stencils(fine, domain),
    }
    setup = time.perf_counter() - t0
    rows = []
    for n in counts:
        coarse = build_node_set(case3(), n, seed=seed, iterations=iterations)
        grid = cartesian_grid(domain, n)
        row: dict[str, float] = {
            "n": coarse.n,
            "h": coarse.h,
            "fine-n": fine.n,
            "fine-h": fine.h,
            "setup": setup,
        }
        for label, pts in (("nodes", coarse), ("grid", grid)):
            keep = (
                np.abs(np.hypot(pts.x - 0.5, pts.y - 0.5) - sum(RING) / 2) < NEAR_RING
            )
            x, y = pts.x[keep], pts.y[keep]
            ref = exact(x, y)
            for kind, st in stencils.items():
                t0 = time.perf_counter()
                got = resample(u, fine, st, domain.material, x, y)
                row[f"{label}-{kind}"] = rms_error(got, ref)
                row[f"{label}-{kind}-max"] = float(np.abs(got - ref).max())
                row[f"{label}-{kind}-seconds"] = time.perf_counter() - t0
        rows.append(row)
    return rows


# --- tables ----------------------------------------------------------------------


def rates(rows: list[dict[str, float]], name: str) -> list[float]:
    """Order per halving of ``h`` between consecutive rows (``nan`` for the first)."""
    out = [float("nan")]
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        out.append(np.log(a[name] / b[name]) / np.log(a["h"] / b["h"]))
    return out


def _order(value: float) -> str:
    return "    -" if np.isnan(value) else f"{value:5.2f}"


def _marker(table: dict[int, float], n: int) -> str:
    value = table.get(n)
    return f"{value:8.1e}" if value else "       -"


def _ratios(rows: list[dict[str, float]], name: str, table: dict[int, float]) -> str:
    found = [
        f"{r[name] / table[int(r['n'])]:.1f}" for r in rows if int(r["n"]) in table
    ]
    return ", ".join(found) if found else "-"


def nearest_marker(n: int) -> int:
    """The Fig. 14 FD4 count nearest an ``m (m + 1)`` grid (1260 for 1250)."""
    return min(FIG14["fd4"], key=lambda k: abs(np.log(k / n)))


def print_reference(ref: Reference, reused: bool, rows: list[dict[str, float]]) -> None:
    s = ref.meta["seconds"]
    how = "cached" if reused else "solved now"
    counts = ownership(ref.nodes, case3())
    print(
        f"reference: {ref.nodes.n} nodes, h = {ref.nodes.h:.5f}, interface group"
        f" {ref.meta['interface_group']}, node set {s['nodes']:.1f} s, operator"
        f" {s['operator']:.1f} s, solve {s['solve']:.1f} s ({how});"
        f" max |u| = {np.abs(ref.u).max():.9f}; nodes inside the disc-side region"
        f" {counts['region-0']}, in the ring {counts['region-1']}, outside"
        f" {counts['region-2']}"
    )
    estimate, n = reference_error_estimate(rows, ref.nodes.n)
    finest = next(r["curved"] for r in rows if r["n"] == n)
    print(
        f"its own error, from the {n}-node curved point if the error falls as N^-2:"
        f" about {estimate:.0e} to one figure, the local order not being pinned"
        f" (that point's measured error is {finest:.2e})"
    )


def print_convergence(
    rows: list[dict[str, float]], fd4_rows: list[dict[str, float]]
) -> None:
    print(
        "\ncase 3, RBF-FD against the reference (RMS error at the coarse nodes,"
        " order per halving of h; 'group' is the interface group, 'in ring' the"
        " nodes inside 0.349 <= r <= 0.35; times are the curved operator's)"
    )
    print(
        "     n       h  group  in ring |       flat  order |     curved  order"
        "  nodes  build  solve  read |      plain  order |  Fig. 14 flat   curved"
    )
    orders = {name: rates(rows, name) for name, _, _ in VARIANTS}
    for i, r in enumerate(rows):
        n = int(r["n"])
        times = f"{r['nodes']:4.1f}s  " + "  ".join(
            f"{r[f'curved-{k}']:4.1f}s" for k in ("build", "solve", "resample")
        )
        line = f"{n:6d}  {r['h']:.4f}  {int(r['group']):5d}  {int(r['region-1']):7d} |"
        line += f"  {r['flat']:9.2e}  {_order(orders['flat'][i])} |"
        line += f"  {r['curved']:9.2e}  {_order(orders['curved'][i])}  {times} |"
        line += f"  {r['plain']:9.2e}  {_order(orders['plain'][i])} |"
        line += f"  {_marker(FIG14['flat'], n)} {_marker(FIG14['curved'], n)}"
        print(line)
    print(
        f"ratio to 2016: flat {_ratios(rows, 'flat', FIG14['flat'])};"
        f" curved {_ratios(rows, 'curved', FIG14['curved'])};"
        f" plain / curved {', '.join(f'{r["plain"] / r["curved"]:.2f}' for r in rows)}"
    )
    print(
        "largest |u| (the Dirichlet data bound it by 1): curved "
        + ", ".join(f"{r['curved-max']:.4f}" for r in rows)
    )
    print(
        "\ncase 3, Cartesian FD4 (Dx A Dx + Dy A Dy on m (m + 1) grid points, the"
        " disc a staircase Dirichlet set; error over the points outside it;"
        " 'no ring' the same grid with α ≡ 1, 'blind' the FD4 error with the"
        " reference read blind at the grid points)"
    )
    print(
        "      n       h   disc  ring |        fd4  order  build  solve   read |"
        "    no ring      blind |  Fig. 14 FD4"
    )
    fd4_orders = rates(fd4_rows, "fd4")
    ratios = []
    for i, r in enumerate(fd4_rows):
        n = int(r["n"])
        marker = nearest_marker(n)
        times = "  ".join(
            f"{r[f'fd4-{k}']:5.1f}s" for k in ("build", "solve", "resample")
        )
        print(
            f"{n:7d}  {r['h']:.4f}  {int(r['disc']):5d}  {int(r['ring']):4d} |"
            f"  {r['fd4']:9.2e}  {_order(fd4_orders[i])}  {times} |"
            f"  {r['no-ring']:9.2e}  {r['blind']:9.2e} |"
            f"  {_marker(FIG14['fd4'], marker)} (at {marker})"
        )
        ratios.append(f"{r['fd4'] / FIG14['fd4'][marker]:.2f}")
    print(f"ratio to 2016 (FD4 at the nearest marker count): {', '.join(ratios)}")


def print_check(rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    r0 = rows[0]
    print(
        f"\nresampling check on the ring: the harmonic mode R(r) cos 2θ (α = 1/1500"
        f" on 0.349 <= r <= 0.35) on {int(r0['fine-n'])} nodes (h = {r0['fine-h']:.5f};"
        f" {r0['setup']:.1f} s to build with both stencil sets) read at the coarse"
        f" case-3 nodes and FD4 grid points within {NEAR_RING} of the midline;"
        " RMS / max error"
    )
    print(
        "     n       h |  nodes aware        max   time |  nodes blind        max |"
        "   grid aware        max |   grid blind        max"
    )
    for r in rows:
        cells = [f"{int(r['n']):6d}  {r['h']:.4f} |"]
        for label in ("nodes-aware", "nodes-blind", "grid-aware", "grid-blind"):
            cells.append(f"  {r[label]:11.2e}  {r[label + '-max']:9.2e}")
            if label == "nodes-aware":
                cells.append(f"  {r[label + '-seconds']:4.1f}s |")
            elif label != "grid-blind":
                cells.append(" |")
        print("".join(cells))


# --- figures ---------------------------------------------------------------------


def _markers(ax, table: dict[int, float], marker: str, label: str) -> None:
    items = sorted(table.items())
    ax.loglog(
        [k for k, _ in items],
        [v for _, v in items],
        marker + ":",
        color="k",
        markerfacecolor="none",
        markersize=6,
        linewidth=0.7,
        label=label,
    )


def figure_convergence(rows: list[dict[str, float]], fd4_rows: list[dict[str, float]]):
    n = np.array([r["n"] for r in rows], dtype=float)
    n_fd4 = np.array([r["n"] for r in fd4_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.loglog(
        n_fd4,
        [r["fd4"] for r in fd4_rows],
        "s-",
        color=NAIVE,
        markersize=5,
        label="FD4, Dx A Dx + Dy A Dy, staircase disc",
    )
    ax.loglog(
        n,
        [r["flat"] for r in rows],
        "o-",
        color=REFERENCE,
        markersize=5,
        label="RBF-FD, flat interface",
    )
    ax.loglog(
        n,
        [r["curved"] for r in rows],
        "^-",
        color=AWARE,
        markersize=5,
        label="RBF-FD, curvature included",
    )
    ax.loglog(
        n,
        [r["plain"] for r in rows],
        "v--",
        color=AWARE,
        markersize=5,
        markerfacecolor="none",
        linewidth=0.9,
        label="curvature included, plain Gaussians",
    )
    _markers(ax, FIG14["fd4"], "s", "EABE Fig. 14 FD4 (read off)")
    _markers(ax, FIG14["flat"], "o", "Fig. 14 flat int.")
    _markers(ax, FIG14["curved"], "^", "Fig. 14 curv. incl.")
    e4 = rows[0]["curved"]
    ax.loglog(n, e4 * (n / n[0]) ** -2.0, "k--", linewidth=0.7, label="4th order")
    ticks = sorted({*n.astype(int), *FIG14["curved"], 640000, 2560000})
    ax.set_xticks(ticks, [str(k) for k in ticks], fontsize=7, rotation=45)
    ax.set_xticks([], minor=True)
    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("RMS error in u against the fine reference")
    ax.set_title("case 3: FD4, flat and curved RBF-FD (EABE Fig. 14 twin)")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    return fig


def figure_solution(nodes: NodeSet, u: np.ndarray, domain: Domain):
    """The solution as a surface over the node set, the cooling disc cut out."""
    tri = mtri.Triangulation(nodes.x, nodes.y)
    cx = nodes.x[tri.triangles].mean(axis=1)
    cy = nodes.y[tri.triangles].mean(axis=1)
    tri.set_mask(np.hypot(cx - 0.5, cy - 0.5) < COOLING_RADIUS)
    fig = plt.figure(figsize=(7.0, 5.2))
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(
        tri, u, cmap="viridis", linewidth=0.0, antialiased=False, shade=False
    )
    ax.view_init(elev=32, azim=-128)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title(f"case 3: the {nodes.n}-node solution (Fig. 13 twin)")
    fig.tight_layout()
    return fig


def mesh_solution(
    ref: Reference, mesh_n: int, seed: int, iterations: int
) -> tuple[NodeSet, np.ndarray, float]:
    """The reference if it has ``mesh_n`` nodes, else a fresh curved solve, timed."""
    if ref.nodes.n == mesh_n:
        return ref.nodes, ref.u, 0.0
    domain = case3()
    t0 = time.perf_counter()
    nodes = build_node_set(domain, mesh_n, seed=seed, iterations=iterations)
    stencils = build_stencils(nodes, domain, interface=BOUNDARY)
    op = interface_aware_operator(nodes, domain.material, stencils)
    u = solve_equilibrium(op, nodes, VALUES)
    return nodes, u, time.perf_counter() - t0


# --- main ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--counts", type=int, nargs="+", default=[1250, 2500, 5000, 10000]
    )
    parser.add_argument("--fd4-counts", type=int, nargs="+", default=None)
    parser.add_argument("--reference-n", type=int, default=40000)
    parser.add_argument("--check-n", type=int, default=None)
    parser.add_argument("--mesh-n", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    fd4_counts = args.counts if args.fd4_counts is None else args.fd4_counts
    check_n = args.reference_n if args.check_n is None else args.check_n
    if len(args.counts) < 2 or any(n < 900 for n in args.counts):
        parser.error("give at least two counts of 900 nodes or more")
    if any(n < 300 for n in fd4_counts):
        parser.error("the FD4 counts need 300 nodes or more")
    if args.reference_n <= max(args.counts):
        parser.error("the reference must have more nodes than every RBF-FD count")
    args.outputs.mkdir(parents=True, exist_ok=True)

    domain = case3()
    t0 = time.perf_counter()
    ref, reused = reference(
        domain, args.reference_n, args.seed, args.iterations, args.outputs
    )
    ref_stencils = ref.stencils(domain)
    blind_stencils = build_stencils(ref.nodes, domain)
    print(f"({time.perf_counter() - t0:.1f} s for the reference)")

    t0 = time.perf_counter()
    rows = sweep(tuple(args.counts), ref, ref_stencils, args.seed, args.iterations)
    fd4_rows = fd4_sweep(tuple(fd4_counts), ref, ref_stencils, blind_stencils)
    print_reference(ref, reused, rows)
    print_convergence(rows, fd4_rows)
    print(f"({time.perf_counter() - t0:.1f} s for the sweeps)")
    fig = figure_convergence(rows, fd4_rows)
    fig.savefig(args.outputs / "heat2d_case3_convergence.png", dpi=150)
    plt.close(fig)

    nodes, u, seconds = mesh_solution(ref, args.mesh_n, args.seed, args.iterations)
    fig = figure_solution(nodes, u, domain)
    fig.savefig(args.outputs / "heat2d_case3_solution.png", dpi=150)
    plt.close(fig)
    print(
        f"\nmesh plot: {nodes.n} nodes, max |u| = {np.abs(u).max():.6f}, RMS |u| ="
        f" {np.sqrt(np.mean(u**2)):.4f}, largest |u| inside the ring's disc"
        f" {np.abs(u[np.hypot(nodes.x - 0.5, nodes.y - 0.5) < RING[0]]).max():.2e}"
        + (f" ({seconds:.1f} s to solve)" if seconds else " (the reference)")
    )

    if check_n:
        t0 = time.perf_counter()
        print_check(
            resampling_check(check_n, tuple(args.counts), args.seed, args.iterations)
        )
        print(f"({time.perf_counter() - t0:.1f} s for the resampling check)")


if __name__ == "__main__":
    main()
