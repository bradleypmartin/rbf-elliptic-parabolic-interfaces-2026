"""E2.5 (#19): case 1's elliptic and parabolic convergence against the analytic
solution, and the operator's spectrum against BD4's stability region.

Two figures under ``outputs/`` and two tables on stdout: the twins of
dissertation Fig. 5-5 and Fig. 5-6 (EABE Fig. 7 is Fig. 5-5's elliptic line).

``heat2d_case1_convergence.png``: RMS error against the node count for the
interface-aware operator with the papers' settings (warped Gaussians,
straddling rows; port notes §2.3–2.4) on case 1, ``α = 0.2`` on
``0.6 ≤ y ≤ 0.8``: the elliptic problem (EABE eq. 32–34) and the parabolic one
(dissertation eq. 84–86, ``u = e^{t} sin 2πx v(y)``, ``c_t = 1``), marched by
BD4 from the analytic solution to ``t = 0.1`` with ``dt = h``, the row
spacing, next to the markers read off Fig. 5-5. A second parabolic march at
``dt = h/2`` measures the time error's share; the table also lists the BD4
step counts, which run from 3 at 1250 nodes to 13 at 20,000.

``heat2d_case1_spectrum.png``: the eigenvalues of the interior operator
(Dirichlet rows removed) at 4900 nodes, warp on and off, and of the naive
``Dx A Dx + Dy A Dy`` on the same node set: the λ plane above, and below
the zoom of Fig. 5-6 with BD4's boundary at ``dt = 0.02`` (its setting) and
at ``dt = h``. The table gives per operator the complex count, the extreme
real parts and the largest imaginary part in units of ``h⁻²``, and BD4's
largest root modulus at both steps.

    uv run python scripts/heat2d_case1.py                              # ~60 s
    uv run python scripts/heat2d_case1.py --counts 1250 2500 5000 10000 20000 40000
    uv run python scripts/heat2d_case1.py --spectrum-n 2000            # seconds
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from heat_interfaces.heat1d.march import (  # noqa: E402
    bd4_amplification,
    bd4_stability_boundary,
    march_steps,
)
from heat_interfaces.heat2d import (  # noqa: E402
    BOUNDARY,
    build_node_set,
    build_stencils,
    case1,
    case1_exact,
    interface_aware_operator,
    interior_eigenvalues,
    march,
    naive_operator,
    rms_error,
    solve_equilibrium,
)
from heat_interfaces.plotting import AWARE, NAIVE, REFERENCE  # noqa: E402

GROWTH = 1.0
"""``c_t`` of dissertation eq. 85."""

T_END = 0.1
"""Where the dissertation reports the parabolic error (Fig. 5-4, 5-5)."""

FIG56_DT = 0.02
"""The time step of Fig. 5-6's BD4 boundary at 4900 nodes."""

FIG7 = {
    1250: 1.0e-5,
    2500: 2.6e-6,
    5000: 5.5e-7,
    10000: 8e-8,
    20000: 1.8e-8,
    40000: 6e-9,
}
"""EABE Fig. 7 read off the rendered page (2026-09-21, as in heat2d_interface.py)."""

FIG55_PARABOLIC = {
    1250: 1.8e-5,
    2500: 3.8e-6,
    5000: 6.3e-7,
    10000: 1.7e-7,
    20000: 2.7e-8,
    40000: 6.6e-9,
}
"""Dissertation Fig. 5-5's parabolic line read off the rendered page (2026-09-21).

Its elliptic line is EABE Fig. 7's to reading accuracy (about ±15 %).
"""

SPECTRA = (
    ("aware-warp", "interface-aware, warped RBFs", AWARE),
    ("aware-plain", "interface-aware, plain RBFs", REFERENCE),
    ("naive", "naive Dx A Dx + Dy A Dy", NAIVE),
)
"""``(name, label, colour)`` of the three operators whose spectra are taken."""


def top(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    """Eq. 85's Dirichlet row at ``y = 1``: ``e^{c_t t} sin 2πx``."""
    return np.exp(GROWTH * t) * np.sin(2 * np.pi * x)


def problem(n: int, seed: int = 0, iterations: int = 100):
    """``(nodes, stencils)`` of the case-1 set with the interface group."""
    domain = case1()
    nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
    return nodes, build_stencils(nodes, domain, interface=BOUNDARY)


def sweep(
    counts: tuple[int, ...],
    seed: int = 0,
    iterations: int = 100,
    t_end: float = T_END,
) -> list[dict[str, float]]:
    """One row per count: elliptic and parabolic RMS errors, steps and times."""
    domain = case1()
    elliptic, parabolic = case1_exact(), case1_exact(GROWTH)
    rows = []
    for n in counts:
        t0 = time.perf_counter()
        nodes, stencils = problem(n, seed, iterations)
        op = interface_aware_operator(nodes, domain.material, stencils)
        row: dict[str, float] = {
            "n": nodes.n,
            "h": nodes.h,
            "group": len(stencils.groups[-1].rows),
            "build": time.perf_counter() - t0,
        }
        t0 = time.perf_counter()
        u = solve_equilibrium(op, nodes, [0.0, top])
        row["elliptic"] = rms_error(u, elliptic(nodes.x, nodes.y))
        row["elliptic-seconds"] = time.perf_counter() - t0
        u0 = parabolic(nodes.x, nodes.y, 0.0)
        reference = parabolic(nodes.x, nodes.y, t_end)
        for name, dt in (("parabolic", nodes.h), ("parabolic-half", nodes.h / 2)):
            t0 = time.perf_counter()
            u = march(op, nodes, u0, t_end, dt, [0.0, top], solution=parabolic)
            row[name] = rms_error(u, reference)
            row[name + "-seconds"] = time.perf_counter() - t0
            row[name + "-steps"] = march_steps(t_end, dt)[0]
        rows.append(row)
    return rows


def rates(rows: list[dict[str, float]], name: str) -> list[float]:
    """Order per halving of ``h`` between consecutive rows (``nan`` for the first)."""
    out = [float("nan")]
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        out.append(np.log(a[name] / b[name]) / np.log(a["h"] / b["h"]))
    return out


def _order(value: float) -> str:
    return "    -" if np.isnan(value) else f"{value:5.2f}"


def print_convergence(rows: list[dict[str, float]], t_end: float = T_END) -> None:
    print(
        f"case 1, interface-aware operator (warp, rows): RMS error vs the analytic"
        f" solution, elliptic and parabolic (BD4 from the analytic history to"
        f" t = {t_end}, dt = h and h/2)"
    )
    print(
        "     n       h  group |   elliptic  order  time |  parabolic  order"
        "  steps  time |  dt = h/2   ratio  steps |  Fig. 5-5 par.  Fig. 7 ell."
    )
    ell, par = rates(rows, "elliptic"), rates(rows, "parabolic")
    for i, row in enumerate(rows):
        n = int(row["n"])
        ratio = row["parabolic-half"] / row["parabolic"]
        line = f"{n:6d}  {row['h']:.4f}  {int(row['group']):5d} |"
        line += (
            f"  {row['elliptic']:9.2e}  {_order(ell[i])}"
            f"  {row['elliptic-seconds']:3.1f}s |"
        )
        line += (
            f"  {row['parabolic']:9.2e}  {_order(par[i])}"
            f"  {int(row['parabolic-steps']):5d}  {row['parabolic-seconds']:3.1f}s |"
        )
        line += (
            f"  {row['parabolic-half']:9.2e}  {ratio:5.3f}"
            f"  {int(row['parabolic-half-steps']):5d} |"
        )
        fig55, fig7 = FIG55_PARABOLIC.get(n), FIG7.get(n)
        line += f"  {fig55:12.1e}" if fig55 else "             -"
        line += f"  {fig7:11.1e}" if fig7 else "            -"
        print(line)
    for name, table in (("elliptic", FIG7), ("parabolic", FIG55_PARABOLIC)):
        ratios = [
            f"{row[name] / table[int(row['n'])]:.1f}"
            for row in rows
            if int(row["n"]) in table
        ]
        print(f"ratio to 2016, {name}: {', '.join(ratios)}")


def figure_convergence(rows: list[dict[str, float]], t_end: float = T_END):
    n = np.array([r["n"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.loglog(
        n,
        [r["parabolic"] for r in rows],
        "s:",
        color=AWARE,
        markersize=5,
        label=f"parabolic, BD4 dt = h, t = {t_end:g}",
    )
    ax.loglog(
        n,
        [r["parabolic-half"] for r in rows],
        "x--",
        color=AWARE,
        markersize=5,
        linewidth=0.6,
        label="parabolic, dt = h/2",
    )
    ax.loglog(
        n,
        [r["elliptic"] for r in rows],
        "o-",
        color=AWARE,
        markersize=5,
        label="elliptic",
    )
    for table, marker, label in (
        (FIG55_PARABOLIC, "s", "dissertation Fig. 5-5, parabolic (read off)"),
        (FIG7, "o", "EABE Fig. 7 = Fig. 5-5, elliptic (read off)"),
    ):
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
    e4 = rows[0]["elliptic"]
    ax.loglog(n, e4 * (n / n[0]) ** -2.0, "k-.", linewidth=0.7, label="4th order")
    ticks = sorted({*n.astype(int), *FIG7})
    ax.set_xticks(ticks, [str(k) for k in ticks], fontsize=8)
    ax.set_xticks([], minor=True)
    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("RMS error in u")
    ax.set_title("case 1: elliptic and parabolic convergence (Fig. 5-5 twin)")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


# --- the spectrum of Fig. 5-6 -------------------------------------------------------


def spectrum(n: int, seed: int = 0, iterations: int = 100) -> dict[str, np.ndarray]:
    """Eigenvalues of the three interior operators on one case-1 node set."""
    domain = case1()
    nodes, stencils = problem(n, seed, iterations)
    plain = build_stencils(nodes, domain)
    out: dict[str, np.ndarray] = {"h": nodes.h, "n": nodes.n}
    for name, op in (
        ("aware-warp", interface_aware_operator(nodes, domain.material, stencils)),
        (
            "aware-plain",
            interface_aware_operator(nodes, domain.material, stencils, warp=False),
        ),
        ("naive", naive_operator(nodes, domain.material, plain)),
    ):
        out[name] = interior_eigenvalues(op, nodes)
    return out


def spectrum_summary(
    spec: dict[str, np.ndarray], dt: float = FIG56_DT
) -> list[dict[str, float]]:
    h = spec["h"]
    rows = []
    for name, _, _ in SPECTRA:
        lam = spec[name]
        rows.append(
            {
                "name": name,
                "count": lam.size,
                "complex": int(np.sum(np.abs(lam.imag) > 1e-8 * np.abs(lam).max())),
                "max_re": lam.real.max(),
                "min_re_h2": lam.real.min() * h**2,
                "max_im_h2": np.abs(lam.imag).max() * h**2,
                "bd4_h": bd4_amplification(h * lam).max(),
                "bd4_dt": bd4_amplification(dt * lam).max(),
            }
        )
    return rows


def print_spectrum(spec: dict[str, np.ndarray], dt: float = FIG56_DT) -> None:
    print(
        f"\nspectra at {int(spec['n'])} nodes, h = {spec['h']:.4f}"
        f" (BD4 root moduli at dt = h and at dt = {dt:g}, Fig. 5-6's step)"
    )
    print(
        "operator        eigs  complex     max Re   h² min Re  h² max |Im|"
        f"  BD4 max |ζ| @h  @{dt:g}"
    )
    for r in spectrum_summary(spec, dt):
        print(
            f"{r['name']:14s} {r['count']:5d}  {r['complex']:7d}  {r['max_re']:9.3f}  "
            f"{r['min_re_h2']:10.2f}  {r['max_im_h2']:11.3f}  {r['bd4_h']:14.3f}"
            f"  {r['bd4_dt']:5.3f}"
        )


def figure_spectrum(spec: dict[str, np.ndarray], dt: float = FIG56_DT):
    h = spec["h"]
    fig = plt.figure(figsize=(9.0, 7.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.4])
    top_axes = [fig.add_subplot(grid[0, k]) for k in range(3)]
    zoom = fig.add_subplot(grid[1, :])
    theta = np.linspace(0.0, 2 * np.pi, 721)
    curve = bd4_stability_boundary(theta)
    for ax, (name, label, color) in zip(top_axes, SPECTRA, strict=True):
        lam = spec[name]
        ax.plot(lam.real / 1e4, lam.imag / 1e3, ".", color=color, markersize=2)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Re λ / 10⁴")
    top_axes[0].set_ylabel("Im λ / 10³")
    # The naive cloud goes down first so the interface-aware ones stay visible.
    for name, label, color in reversed(SPECTRA):
        lam = spec[name]
        zoom.plot(lam.real, lam.imag, ".", color=color, markersize=3, label=label)
    for step, style, label in (
        (dt, "k-", f"BD4 boundary, dt = {dt:g}"),
        (h, "k--", f"dt = h = {h:.4f}"),
    ):
        zoom.plot(
            curve.real / step, curve.imag / step, style, linewidth=0.9, label=label
        )
    zoom.axhline(0, color="k", linewidth=0.3)
    zoom.axvline(0, color="k", linewidth=0.3)
    zoom.set_xlim(-700, 700)
    zoom.set_ylim(-700, 700)
    zoom.set_aspect("equal")
    zoom.set_xlabel("Re λ")
    zoom.set_ylabel("Im λ")
    zoom.set_title("zoom: BD4 is stable outside the closed curve", fontsize=9)
    zoom.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        f"case 1, {int(spec['n'])} nodes: eigenvalues of the interior operators"
        " (Fig. 5-6 twin)"
    )
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--counts", type=int, nargs="+", default=[1250, 2500, 5000, 10000]
    )
    parser.add_argument("--spectrum-n", type=int, default=4900)
    parser.add_argument("--t-end", type=float, default=T_END)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    if len(args.counts) < 2 or any(n < 300 for n in args.counts):
        parser.error("give at least two counts of 300 nodes or more")
    if args.spectrum_n < 300:
        parser.error("the spectrum needs 300 nodes or more")
    args.outputs.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rows = sweep(tuple(args.counts), args.seed, args.iterations, args.t_end)
    print_convergence(rows, args.t_end)
    print(f"({time.perf_counter() - t0:.1f} s for the sweep)")
    fig = figure_convergence(rows, args.t_end)
    fig.savefig(args.outputs / "heat2d_case1_convergence.png", dpi=150)
    plt.close(fig)

    t0 = time.perf_counter()
    spec = spectrum(args.spectrum_n, args.seed, args.iterations)
    print_spectrum(spec)
    print(f"({time.perf_counter() - t0:.1f} s for the three dense eigenvalue problems)")
    fig = figure_spectrum(spec)
    fig.savefig(args.outputs / "heat2d_case1_spectrum.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
