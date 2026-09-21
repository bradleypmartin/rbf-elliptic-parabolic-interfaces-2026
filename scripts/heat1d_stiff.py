"""E3 (#5): the 1-D stiff-edge study. E3.2 (#27): the references at any δ.

The media are ``SmoothEdges(matlab_alpha(), δ)``, the ``1/9 | 1`` jump at 0
with a tanh edge of width δ, and the same for eq. 75's layer; the parabolic
problem is the MATLAB one of port notes §1.4 (zero start, ``u(-1, t)`` ramped
to 1 over [0, 1], the solution at t = 2). The references (plan D4) are the
quadrature for the equilibrium problem and the Chebyshev-element solution
with Radau in time for the parabolic one, cut at ``EDGE_CUTS`` about every
edge, wide elements split, and cached under ``outputs/`` as
``heat1d_stiff_reference_<medium>_d<δ>_n<n_cheb>_w<max_width>.npz`` + ``.json``.

This driver builds the parabolic references for the study's δ values and,
per medium and δ, reports the run (elements, unknowns, Radau steps, seconds)
and the two checks the stiff note's P1 asks for: the agreement with a finer
resolution (more nodes, narrower elements; cached too) and the Chebyshev
equilibrium against the quadrature. The knee sweep (E3.3), the seeds (E3.4),
the treatments (E3.5) and the figures (E3.6) are added by their tickets.

    uv run python scripts/heat1d_stiff.py            # 20 s; caches the references
    uv run python scripts/heat1d_stiff.py --deltas 0 0.0025 --media matlab
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from heat_interfaces.heat1d import (
    RADAU_ATOL,
    RADAU_RTOL,
    ParabolicReference,
    SmoothEdges,
    chebyshev_equilibrium,
    dissertation_alpha,
    equilibrium_exact,
    matlab_alpha,
    parabolic_reference,
    ramp_boundary,
)

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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deltas", type=float, nargs="+", default=list(STUDY_DELTAS))
    parser.add_argument("--media", nargs="+", default=list(MEDIA), choices=list(MEDIA))
    parser.add_argument("--n-cheb", type=int, default=N_CHEB)
    parser.add_argument("--max-width", type=float, default=MAX_WIDTH)
    parser.add_argument("--check-n-cheb", type=int, default=CHECK_N_CHEB)
    parser.add_argument("--check-max-width", type=float, default=CHECK_MAX_WIDTH)
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


if __name__ == "__main__":
    main()
