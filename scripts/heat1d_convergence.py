"""E1.4 (#14): the 1-D equilibrium reproduction, twins of dissertation Fig. 4-5 to 4-7.

Three figures under ``outputs/`` for the eq. 75 problem (``alpha = 0.1 + 0.4
sin 2πx`` on the closed layer [0, 0.5], 1 elsewhere; ``(alpha u_x)_x = 0``,
``u(-1) = 1``, ``u(1) = 0``), jump-aware in blue and naive ``Dx A Dx`` in
orange, every error against the quadrature reference (exact to rounding,
in place of the dissertation's 6400-node run):

``heat1d_solutions.png`` (Fig. 4-5): the 101-node solutions, jump-aware on top
and naive below, over the reference.

``heat1d_errors.png`` (Fig. 4-6): the errors of those two solutions.

``heat1d_convergence.png`` (Fig. 4-7): the RMS error ``||e||_2 / sqrt(n)``
against the node count, 101 to 3201 with both interfaces on nodes, with
first- and fourth-order guides. The figure's "normalized ℓ2 error" is the
RMS, not ``||e||_2 / ||u||_2``; ``docs/port-notes.md`` §1.3 has the evidence.
The report also prints the relative error and the direct ``(alpha u_x)_x``
stencil's error, which does not converge. Node counts that move either
interface off a node are refused (``(n - 1) % 4 == 0`` is required), since
the figure's placement is part of what is being reproduced.

    uv run python scripts/heat1d_convergence.py                      # < 1 s
    uv run python scripts/heat1d_convergence.py --counts 101 201 401 801 1601 3201 6401
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
    direct_operator,
    dissertation_alpha,
    equilibrium_exact,
    equispaced_grid,
    jump_aware_operator,
    naive_operator,
    node_counts,
    normalized_l2,
    rms_error,
    solve_equilibrium,
)
from heat_interfaces.plotting import AWARE, NAIVE, REFERENCE  # noqa: E402

LINES = {
    "jump-aware": (jump_aware_operator, AWARE),
    "naive": (naive_operator, NAIVE),
}
"""The two lines of Fig. 4-7, in the panel order of Fig. 4-5 and 4-6."""

REPORTED = {**{k: op for k, (op, _) in LINES.items()}, "direct": direct_operator}
"""Every operator the report tabulates; the direct stencil is for the record."""

DEFAULT_COUNTS = (101, 201, 401, 801, 1601, 3201)
"""Fig. 4-7's six abscissae (labelled 100 to 3200) with both interfaces on nodes."""


def require_interfaces_on_nodes(parser, counts, medium):
    """Refuse counts whose grid puts an interface anywhere but on a node."""
    bad = [
        n
        for n in counts
        if any(equispaced_grid(n).placement(xi) != "node" for xi in medium.interfaces)
    ]
    if bad:
        lo, hi = min(bad) // 2, 2 * max(bad)
        parser.error(
            f"node counts {bad} put an interface of {medium.interfaces} off a node; "
            f"admissible counts in [{lo}, {hi}]: "
            f"{node_counts(medium.interfaces, 'node', lo, hi)}"
        )


def solutions(n, medium):
    """``(grid, reference, {name: u})`` for the ``n``-node equilibrium solves."""
    g = equispaced_grid(n)
    ref = equilibrium_exact(medium, *DISSERTATION_BC, g.x)
    us = {
        name: solve_equilibrium(op(g, medium), *DISSERTATION_BC)
        for name, op in REPORTED.items()
    }
    return g, ref, us


def sweep(counts, medium):
    """``{name: {"rms": array, "relative": array}}`` over ``counts``, in count order."""
    errs = {name: {"rms": [], "relative": []} for name in REPORTED}
    for n in counts:
        _, ref, us = solutions(n, medium)
        for name, u in us.items():
            errs[name]["rms"].append(rms_error(u, ref))
            errs[name]["relative"].append(normalized_l2(u, ref))
    return {
        name: {norm: np.array(values) for norm, values in d.items()}
        for name, d in errs.items()
    }


def report(counts, errs):
    print("\nequilibrium problem, both interfaces on nodes, quadrature reference")
    header = "  n      " + "".join(
        f"{name + ' ' + norm:>20s}" for name in errs for norm in ("rms", "relative")
    )
    print(header)
    for i, n in enumerate(counts):
        row = "".join(
            f"{errs[name][norm][i]:20.3e}"
            for name in errs
            for norm in ("rms", "relative")
        )
        print(f"  {n:<6d} " + row)
    for name, d in errs.items():
        rates = np.log2(d["rms"][:-1] / d["rms"][1:])
        print(f"  {name}: RMS rates {np.array2string(rates, precision=2)}")


def report_solutions(grid, ref, us):
    print(f"\n{grid.n}-node solutions")
    for name in LINES:
        e = us[name] - ref
        i = int(np.argmax(np.abs(e)))
        flips = int(np.sum(np.diff(np.sign(e)) != 0))
        monotone = bool(np.all(np.diff(us[name]) < 0))
        print(
            f"  {name}: max |e| {abs(e[i]):.3e} at x = {grid.x[i]:.2f},"
            f" {flips} sign changes, monotone {monotone},"
            f" RMS {rms_error(us[name], ref):.3e}"
        )


def _mark_interfaces(ax, medium):
    for xi in medium.interfaces:
        ax.axvline(xi, color=REFERENCE, lw=0.6, alpha=0.5)


def plot_solutions(grid, ref, us, medium, path):
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 6.4), sharex=True)
    for ax, (name, (_, colour)) in zip(axes, LINES.items(), strict=True):
        ax.plot(grid.x, ref, "--", color=REFERENCE, lw=0.8, label="reference")
        ax.plot(grid.x, us[name], "-", color=colour, lw=1.2, label=name)
        _mark_interfaces(ax, medium)
        ax.set_ylabel("$u(x)$")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{grid.n} nodes, {name}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("$x$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(grid, ref, us, medium, path):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    lim = 1.1 * max(np.abs(us[name] - ref).max() for name in LINES)
    for ax, (name, (_, colour)) in zip(axes, LINES.items(), strict=True):
        ax.plot(grid.x, us[name] - ref, "-", color=colour, lw=1.0)
        _mark_interfaces(ax, medium)
        ax.set_xlabel("$x$")
        ax.set_ylim(-lim, lim)
        ax.set_title(f"error in $u$, {name}", fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("error in $u$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(counts, errs, path):
    counts = np.asarray(counts, dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for name, (_, colour) in LINES.items():
        ax.loglog(counts, errs[name]["rms"], "o-", color=colour, label=name)
    # As in Fig. 4-7: the first-order guide above the naive line, the
    # fourth-order guide below the jump-aware one.
    guides = ((1, 3 * errs["naive"]["rms"][0]), (4, errs["jump-aware"]["rms"][0] / 3))
    for order, anchor in guides:
        ax.loglog(
            counts,
            anchor * (counts[0] / counts) ** order,
            "--",
            color=REFERENCE,
            lw=0.8,
            label=f"order {order}",
        )
    ax.set_xticks(counts)
    ax.set_xticklabels([str(int(n)) for n in counts])
    ax.minorticks_off()
    ax.set_xlabel("nodes")
    ax.set_ylabel("RMS error in $u$")
    ax.set_title("equilibrium problem, dissertation eq. 75", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--n", type=int, default=101, help="Fig. 4-5 / 4-6's node count"
    )
    parser.add_argument("--counts", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    medium = dissertation_alpha()
    require_interfaces_on_nodes(parser, [args.n, *args.counts], medium)
    args.outputs.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    grid, ref, us = solutions(args.n, medium)
    report_solutions(grid, ref, us)
    plot_solutions(grid, ref, us, medium, args.outputs / "heat1d_solutions.png")
    plot_errors(grid, ref, us, medium, args.outputs / "heat1d_errors.png")

    errs = sweep(args.counts, medium)
    report(args.counts, errs)
    plot_convergence(args.counts, errs, args.outputs / "heat1d_convergence.png")
    print(f"\n{time.perf_counter() - t0:.2f} s; figures in {args.outputs}/")


if __name__ == "__main__":
    main()
