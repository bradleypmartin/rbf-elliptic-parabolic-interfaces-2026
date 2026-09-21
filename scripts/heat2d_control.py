"""E2.2 (#16): the no-interface control, case 1 with the naive operator, and the
control's spectrum.

Two figures under ``outputs/`` and two tables on stdout for the RBF-FD
operators of ``heat_interfaces.heat2d.operators`` (42-node / degree-5
stencils, 30 / degree 4 within ``3/√N`` of a Dirichlet row, ``ε = 0.4/d``),
solved by SuperLU on the case-1 node sets of E2.1.

``heat2d_control_convergence.png``: RMS error against the node count for

- the control problem (``α ≡ 1``, ``u = sin 2πx`` on ``y = 1``, 0 on
  ``y = 0``; ``u = sin 2πx sinh 2πy / sinh 2π``) with the direct Laplacian
  stencil and with the naive product ``Dx Dx + Dy Dy``;
- case 1 (``α = 0.2`` on ``0.6 ≤ y ≤ 0.8``, EABE eq. 32–34) with the naive
  ``Dx A Dx + Dy A Dy`` (plan D3), the top line of EABE Fig. 10's analogue,
  and with the direct ``α ∇² + ∇α·∇``, which sees nothing of the jump.

``heat2d_control_spectrum.png``: the eigenvalues of both control operators
at one node count, scaled by ``dt = h``, against the BD4 stability boundary
(the α ≡ 1 baseline that epic #4's item 6 asks for before Fig. 5-6 is read).

The convergence table prints, per count, the row spacing ``h``, the size of
the boundary zone, and for every operator its RMS error, the order per
halving of ``h`` since the previous count, and the solve time; the spectrum
table prints the eigenvalue count, how many are complex, the extreme real
parts and the largest imaginary part in units of ``h⁻²``, and BD4's largest
root modulus at ``dt = h``. ``docs/port-notes.md`` §2.2 carries both.

    uv run python scripts/heat2d_control.py                      # ~15 s
    uv run python scripts/heat2d_control.py --counts 2500 5000 10000 20000
    uv run python scripts/heat2d_control.py --spectrum-n 1250    # the growing mode
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
    interior_operator,
)
from heat_interfaces.heat2d import (  # noqa: E402
    Constant2D,
    build_node_set,
    build_stencils,
    case1,
    case1_exact,
    control_exact,
    direct_operator,
    laplacian_operator,
    naive_operator,
    rms_error,
    solve_equilibrium,
)
from heat_interfaces.plotting import NAIVE, REFERENCE  # noqa: E402

ONE = Constant2D(1.0)

LINES = {
    # name: (label, material, operator builder, exact solution, style)
    "control-lap": (
        "control, Laplacian stencil",
        ONE,
        lambda nodes, mat, st: laplacian_operator(nodes, st),
        control_exact(),
        dict(color=REFERENCE, marker="s", linestyle="-"),
    ),
    "control-naive": (
        "control, Dx Dx + Dy Dy",
        ONE,
        naive_operator,
        control_exact(),
        dict(color=NAIVE, marker="s", linestyle="--"),
    ),
    "case1-naive": (
        "case 1, naive Dx A Dx + Dy A Dy",
        None,  # the case-1 band
        naive_operator,
        case1_exact(),
        dict(color=NAIVE, marker="o", linestyle="-"),
    ),
    "case1-direct": (
        "case 1, direct α∇² + ∇α·∇ (blind)",
        None,
        direct_operator,
        case1_exact(),
        dict(color=REFERENCE, marker="o", linestyle=":"),
    ),
}


def top(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)


def sweep(
    counts: tuple[int, ...], seed: int = 0, iterations: int = 100
) -> list[dict[str, float]]:
    """One table row per count: RMS errors and solve times of every line."""
    domain = case1()
    rows = []
    for n in counts:
        t0 = time.perf_counter()
        nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
        stencils = build_stencils(nodes, domain)
        row: dict[str, float] = {
            "n": nodes.n,
            "h": nodes.h,
            "zone": int(stencils.near_boundary.sum()),
            "setup": time.perf_counter() - t0,
        }
        for name, (_, material, build, exact, _) in LINES.items():
            material = domain.material if material is None else material
            t1 = time.perf_counter()
            op = build(nodes, material, stencils)
            u = solve_equilibrium(op, nodes, [0.0, top])
            row[name] = rms_error(u, exact(nodes.x, nodes.y))
            row[name + "-seconds"] = time.perf_counter() - t1
        rows.append(row)
    return rows


def rates(rows: list[dict[str, float]], name: str) -> list[float]:
    """Order per halving of ``h`` between consecutive rows (``nan`` for the first)."""
    out = [float("nan")]
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        out.append(np.log(a[name] / b[name]) / np.log(a["h"] / b["h"]))
    return out


def print_convergence(rows: list[dict[str, float]]) -> None:
    names = list(LINES)
    head = "     n       h  zone"
    for name in names:
        head += f" | {name:>13s}  order  time"
    print(head)
    r = {name: rates(rows, name) for name in names}
    for i, row in enumerate(rows):
        line = f"{row['n']:6d}  {row['h']:.4f}  {row['zone']:4d}"
        for name in names:
            order = "    -" if np.isnan(r[name][i]) else f"{r[name][i]:5.2f}"
            line += f" | {row[name]:13.2e}  {order}  {row[name + '-seconds']:4.1f}s"
        print(line)


def figure_convergence(rows: list[dict[str, float]]):
    n = np.array([r["n"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for name, (label, _, _, _, style) in LINES.items():
        ax.loglog(n, [r[name] for r in rows], label=label, markersize=4, **style)
    e1 = rows[0]["case1-naive"]
    ax.loglog(n, e1 * (n / n[0]) ** -0.5, "k--", linewidth=0.7, label="1st order")
    e4 = rows[0]["control-lap"]
    ax.loglog(n, e4 * (n / n[0]) ** -2.0, "k:", linewidth=0.7, label="4th order")
    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("RMS error in u")
    ax.set_title("control and case 1 with interface-blind operators")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def spectrum(n: int, seed: int = 0, iterations: int = 100) -> dict[str, np.ndarray]:
    """Eigenvalues of the interior control operators (Dirichlet rows removed)."""
    domain = case1()
    nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
    stencils = build_stencils(nodes, domain)
    out = {"h": nodes.h}
    for name, op in (
        ("control-lap", laplacian_operator(nodes, stencils)),
        ("control-naive", naive_operator(nodes, ONE, stencils)),
    ):
        out[name] = np.linalg.eigvals(interior_operator(op, nodes.dirichlet).toarray())
    return out


def spectrum_summary(spec: dict[str, np.ndarray]) -> list[dict[str, float]]:
    h = spec["h"]
    rows = []
    for name in ("control-lap", "control-naive"):
        lam = spec[name]
        rows.append(
            {
                "name": name,
                "count": lam.size,
                "complex": int(np.sum(np.abs(lam.imag) > 1e-8 * np.abs(lam).max())),
                "max_re": lam.real.max(),
                "min_re_h2": lam.real.min() * h**2,
                "max_im_h2": np.abs(lam.imag).max() * h**2,
                "bd4": bd4_amplification(h * lam).max(),
            }
        )
    return rows


def print_spectrum(spec: dict[str, np.ndarray]) -> None:
    print(f"\nspectra at h = {spec['h']:.4f} (dt = h for BD4)")
    print(
        "operator        eigs  complex     max Re   h² min Re  h² max |Im|  BD4 max |ζ|"
    )
    for r in spectrum_summary(spec):
        print(
            f"{r['name']:14s} {r['count']:5d}  {r['complex']:7d}  {r['max_re']:9.3f}  "
            f"{r['min_re_h2']:10.2f}  {r['max_im_h2']:11.3f}  {r['bd4']:11.3f}"
        )


def figure_spectrum(spec: dict[str, np.ndarray]):
    h = spec["h"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    theta = np.linspace(0.0, 2 * np.pi, 721)
    curve = bd4_stability_boundary(theta)
    for ax, name, color in zip(
        axes, ("control-lap", "control-naive"), (REFERENCE, NAIVE), strict=True
    ):
        z = h * spec[name]
        ax.plot(z.real, z.imag, ".", color=color, markersize=2.5)
        ax.plot(curve.real, curve.imag, "k-", linewidth=0.8, label="BD4 boundary")
        ax.axhline(0, color="k", linewidth=0.3)
        ax.axvline(0, color="k", linewidth=0.3)
        ax.set_xlabel("Re(h λ)")
        ax.set_ylabel("Im(h λ)")
        ax.set_title(LINES[name][0])
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"control operators, h = {h:.4f}; BD4 is stable outside the curve")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--counts", type=int, nargs="+", default=[1250, 2500, 5000, 10000]
    )
    parser.add_argument("--spectrum-n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    if len(args.counts) < 2 or any(n < 300 for n in args.counts):
        parser.error("give at least two counts of 300 nodes or more")
    if args.spectrum_n < 300:
        parser.error("the spectrum needs 300 nodes or more")
    args.outputs.mkdir(parents=True, exist_ok=True)

    rows = sweep(tuple(args.counts), args.seed, args.iterations)
    print_convergence(rows)
    fig = figure_convergence(rows)
    fig.savefig(args.outputs / "heat2d_control_convergence.png", dpi=150)
    plt.close(fig)

    t0 = time.perf_counter()
    spec = spectrum(args.spectrum_n, args.seed, args.iterations)
    print_spectrum(spec)
    print(f"({time.perf_counter() - t0:.1f} s for both dense eigenvalue problems)")
    fig = figure_spectrum(spec)
    fig.savefig(args.outputs / "heat2d_control_spectrum.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
