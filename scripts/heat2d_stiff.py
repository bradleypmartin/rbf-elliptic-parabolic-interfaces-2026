"""E4 (#6): the 2-D stiff-edge study. E4.2: the smooth flat band and its references.

The medium is case 1's band with tanh edges of width δ in the signed normal
distance (``SmoothBand(case1().material, δ)``, stiff note §3.1), which on
case 1 is E3.2's 1-D medium in ``y`` bit for bit. The reference is the
separable ``u = e^{ct} sin 2πx v(y)`` with ``(α v′)′ − (4π² α + c) v = 0``,
``v(0) = 0``, ``v(1) = 1``, by E3.2's Chebyshev elements in ``y``
(``case1_reference``): the elliptic problem at ``c = 0`` and the parabolic
one at E2.5's ``c_t = 1``, exact in time.

The driver reports, per δ and ``c``: the reference's elements, interior
unknowns and build time; its agreement with a finer resolution (24 nodes on
0.05-wide elements, ``CHECK_N_CHEB``, ``CHECK_MAX_WIDTH``); its distance
from the δ = 0 reference, ``sup |v_δ − v_0|`` and that over δ (the O(δ)
floor the δ = 0 construction sits on, H10), and at δ = 0 its distance from
the analytic ``case1_exact``; and how far the band's midline is from the
plateau, ``α(0.7) − 0.2``. Stiff note §4.1 records the table.

    uv run python scripts/heat2d_stiff.py                           # < 1 s
    uv run python scripts/heat2d_stiff.py --deltas 0 0.001 0.0005
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import numpy as np

from heat_interfaces.heat2d import (
    REFERENCE_MAX_WIDTH,
    REFERENCE_N_CHEB,
    SmoothBand,
    case1,
    case1_exact,
    case1_reference,
)

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


def main(argv: Sequence[str] | None = None) -> list[dict[str, float]]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deltas", type=float, nargs="+", default=list(STUDY_DELTAS))
    parser.add_argument("--growth", type=float, nargs="+", default=list(GROWTH))
    args = parser.parse_args(argv)
    rows = reference_rows(args.deltas, args.growth)
    print_references(rows)
    return rows


if __name__ == "__main__":
    main()
