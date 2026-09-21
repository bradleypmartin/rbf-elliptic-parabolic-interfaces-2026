"""E1.3 (#13): parabolic convergence in 1-D and the twin of dissertation Fig. 5-6.

Two figures under ``outputs/``:

``heat1d_parabolic_convergence.png``: normalized ℓ2 error against node count,
naive ``Dx A Dx`` (orange) and jump-aware (blue), BD4 with ``dt = h``. Left:
the MATLAB problem (alpha = 1/9 | 1 at 0, zero start, ``u(-1, t)`` ramped to 1
over [0, 1], errors at t = 2) against the piecewise Chebyshev reference.
Right: the dissertation's eq. 75 material with ``u(-1, t) = e^t``, the 1-D
twin of 2-D case 1, against the separable solution ``e^t v(x)``.

``heat1d_spectrum.png``: the eigenvalues of both interior operators on the
dissertation problem (top) and ``dt lambda`` at ``dt = h`` against the BD4
root locus (bottom), as dissertation Fig. 5-6 does for the 4900-node 2-D
operator. BD4 is stable outside the closed curve.

    uv run python scripts/heat1d_parabolic.py                      # seconds
    uv run python scripts/heat1d_parabolic.py --counts 100 200 400 800 1600
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from heat_interfaces.heat1d import (  # noqa: E402
    DISSERTATION_BC,
    bd4_march,
    bd4_stability_boundary,
    chebyshev_equilibrium,
    chebyshev_parabolic,
    dissertation_alpha,
    equispaced_grid,
    interior_operator,
    jump_aware_operator,
    matlab_alpha,
    naive_operator,
    normalized_l2,
    ramp_boundary,
)
from heat_interfaces.plotting import AWARE, NAIVE, REFERENCE  # noqa: E402

OPERATORS = {
    "naive": (naive_operator, NAIVE),
    "jump-aware": (jump_aware_operator, AWARE),
}


def matlab_errors(counts, ramp, t_end):
    m = matlab_alpha()
    boundary = ramp_boundary(ramp)
    errs = {name: [] for name in OPERATORS}
    for n in counts:
        g = equispaced_grid(n)
        ref = chebyshev_parabolic(m, np.zeros_like, boundary, t_end, g.x, n_cheb=32)
        for name, (op, _) in OPERATORS.items():
            u = bd4_march(op(g, m), np.zeros(n), t_end, g.h, boundary)
            errs[name].append(normalized_l2(u, ref))
    return {k: np.array(v) for k, v in errs.items()}


def dissertation_errors(counts, c, t_end):
    m = dissertation_alpha()
    errs = {name: [] for name in OPERATORS}
    for n in counts:
        g = equispaced_grid(n)
        v = chebyshev_equilibrium(m, *DISSERTATION_BC, g.x, shift=c)
        for name, (op, _) in OPERATORS.items():
            u = bd4_march(op(g, m), v, t_end, g.h, lambda t: (np.exp(c * t), 0.0))
            errs[name].append(normalized_l2(u, np.exp(c * t_end) * v))
    return {k: np.array(v) for k, v in errs.items()}


def spectra(n, medium):
    g = equispaced_grid(n)
    return g.h, {
        name: np.linalg.eigvals(interior_operator(op(g, medium)).toarray())
        for name, (op, _) in OPERATORS.items()
    }


def report(title, counts, errs):
    print(f"\n{title}")
    print("  n      " + "".join(f"{name:>14s}" for name in errs))
    for i, n in enumerate(counts):
        print(f"  {n:<6d} " + "".join(f"{errs[name][i]:14.3e}" for name in errs))
    for name, e in errs.items():
        rates = np.log2(e[:-1] / e[1:])
        print(f"  {name}: rates {np.array2string(rates, precision=2)}")


def plot_convergence(panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    for ax, (title, counts, errs) in zip(np.atleast_1d(axes), panels, strict=True):
        counts = np.asarray(counts, dtype=float)
        for name, (_, colour) in OPERATORS.items():
            ax.loglog(counts, errs[name], "o-", color=colour, label=name)
        for order, anchor in ((1, errs["naive"][0]), (4, errs["jump-aware"][0])):
            ax.loglog(
                counts,
                3 * anchor * (counts[0] / counts) ** order,
                "--",
                color=REFERENCE,
                lw=0.8,
                label=f"order {order}",
            )
        ax.set_xlabel("nodes")
        ax.set_ylabel("normalized $\\ell_2$ error")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spectrum(h, eigs, n, path):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 7.2))
    for name, (_, colour) in OPERATORS.items():
        lam = eigs[name]
        top.plot(lam.real, lam.imag, ".", color=colour, ms=4, label=name)
        bottom.plot(h * lam.real, h * lam.imag, ".", color=colour, ms=4, label=name)
    top.set_title(f"eigenvalues of the {n}-node interior operators", fontsize=10)
    top.set_xlabel("Re $\\lambda$")
    top.set_ylabel("Im $\\lambda$")
    z = bd4_stability_boundary(np.linspace(0, 2 * np.pi, 720))
    bottom.plot(z.real, z.imag, "-", color="k", lw=1, label="BD4 boundary")
    bottom.set_xlim(z.real.min() - 0.5, z.real.max() + 0.5)
    bottom.set_ylim(z.imag.min() - 0.5, z.imag.max() + 0.5)
    bottom.set_title(f"$dt\\,\\lambda$ at $dt = h = {h:.3g}$; stable outside the curve")
    bottom.set_xlabel("Re $dt\\,\\lambda$")
    bottom.set_ylabel("Im $dt\\,\\lambda$")
    for ax in (top, bottom):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--counts", type=int, nargs="+", default=[100, 200, 400, 800])
    parser.add_argument(
        "--dissertation-counts", type=int, nargs="+", default=[101, 201, 401, 801]
    )
    parser.add_argument("--ramp", type=float, default=1.0)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument(
        "--growth", type=float, default=1.0, help="c in u(-1, t) = e^{ct}"
    )
    parser.add_argument("--growth-t-end", type=float, default=1.0)
    parser.add_argument("--spectrum-n", type=int, default=101)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.outputs.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    matlab = matlab_errors(args.counts, args.ramp, args.t_end)
    report(f"MATLAB problem, t = {args.t_end}, dt = h", args.counts, matlab)
    growth = dissertation_errors(
        args.dissertation_counts, args.growth, args.growth_t_end
    )
    report(
        f"dissertation problem, u(-1, t) = e^{{{args.growth} t}}, "
        f"t = {args.growth_t_end}, dt = h",
        args.dissertation_counts,
        growth,
    )
    plot_convergence(
        [
            (f"MATLAB problem, $t = {args.t_end}$", args.counts, matlab),
            (
                f"eq. 75 material, $u(-1, t) = e^{{{args.growth:g} t}}$, "
                f"$t = {args.growth_t_end}$",
                args.dissertation_counts,
                growth,
            ),
        ],
        args.outputs / "heat1d_parabolic_convergence.png",
    )

    h, eigs = spectra(args.spectrum_n, dissertation_alpha())
    print(f"\nspectra at n = {args.spectrum_n}, dissertation problem")
    for name, lam in eigs.items():
        pairs = int(np.sum(np.abs(lam.imag) > 1e-8))
        print(
            f"  {name}: max Re {lam.real.max():.4g},"
            f" min Re h^2 {lam.real.min() * h**2:.4g},"
            f" max |Im| {np.abs(lam.imag).max():.4g}, complex {pairs}"
        )
    plot_spectrum(h, eigs, args.spectrum_n, args.outputs / "heat1d_spectrum.png")
    print(f"\n{time.perf_counter() - t0:.1f} s; figures in {args.outputs}/")


if __name__ == "__main__":
    main()
