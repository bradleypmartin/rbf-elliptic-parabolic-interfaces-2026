"""Reading a fine solution at other points: interface-aware interpolation (E2.6).

Cases 2 and 3 have no analytic solution, so their errors are measured against a
fine RBF-FD run, 160,000 nodes as in EABE §3.2 (plan D4), and the coarse
solutions live on other node sets. ``u`` has a kink at every interface, so a
polynomial or RBF interpolant whose stencil crosses one is first order there
(port notes §2.6), which would swamp errors of 1e-8. The fine set's own
stencils are reused instead: the RBF-FD system with the identity in place of
the operator (``rbf.rbf_interpolation_weights``) on the fine node nearest each
point, and the translated basis with the warped Gaussians of E2.3–E2.4
(``interface.interpolation_weights``) where that stencil crosses an interface,
so the interpolant upholds the interface conditions to the stencil's order.
``Reference`` is such a run with its node set and timings, saved under
``outputs/`` so the drivers solve it once.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .domain import Domain, NodeSet, build_node_set
from .interface import interpolation_weights
from .neighbors import knn, offsets, periodic_dx, wrap_x
from .operators import (
    INTERFACE_KIND,
    BoundaryValue,
    Stencils,
    build_stencils,
    interface_aware_operator,
    interface_crossings,
)
from .rbf import BOUNDARY, GA_SHAPE, rbf_interpolation_weights
from .solve import solve_equilibrium


def resample(
    u: np.ndarray,
    nodes: NodeSet,
    stencils: Stencils,
    material,
    x: np.ndarray,
    y: np.ndarray,
    shape: float = GA_SHAPE,
    curvature: bool = True,
    warp: bool = True,
) -> np.ndarray:
    """``u`` of the node set read at ``(x, y)`` through the nearest node's stencil.

    ``stencils`` are the node set's (``build_stencils`` with the interface spec
    when ``material`` has interfaces): a point whose stencil crosses an
    interface is read through the translated basis of its region, every other
    one through the standard RBF-FD interpolant of its stencil's degree, so a
    fine solution can be compared with a coarse one at the coarse nodes
    without the kink polluting the difference. Without an interface group
    every point is read blind, the first-order interpolation the notes measure.
    """
    x, y = wrap_x(x), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be matching 1-D arrays")
    u = np.asarray(u, dtype=float)
    if u.shape != (nodes.n,):
        raise ValueError("u must have one value per node")
    centre = knn(nodes.xy, 1, query=np.column_stack([x, y]))[0][:, 0]
    group_of = np.full(nodes.n, -1)
    position = np.zeros(nodes.n, dtype=int)
    for gi, g in enumerate(stencils.groups):
        group_of[g.rows] = gi
        position[g.rows] = np.arange(len(g.rows))
    out = np.empty(x.size)
    for gi, g in enumerate(stencils.groups):
        sel = np.flatnonzero(group_of[centre] == gi)
        if not sel.size:
            continue
        idx = g.index[position[centre[sel]]]
        cross = np.zeros(sel.size, dtype=bool)
        if g.kind == INTERFACE_KIND:
            cross = interface_crossings(nodes, material, idx)
        plain, plain_idx = sel[~cross], idx[~cross]
        if plain.size:
            c = centre[plain]
            dx, dy = offsets(nodes.xy, plain_idx, centre=nodes.xy[c])
            w = rbf_interpolation_weights(
                dx,
                dy,
                periodic_dx(x[plain] - nodes.x[c]),
                y[plain] - nodes.y[c],
                g.spec.degree,
                shape,
            )
            out[plain] = np.einsum("ij,ij->i", w, u[plain_idx])
        crossing = sel[cross]
        if not crossing.size:
            continue
        # Points sharing a fine node share its stencil: one translated-basis
        # system per centre, with every point of the centre on its right-hand
        # side, as the plain branch batches its systems.
        centres, first, inverse = np.unique(
            centre[crossing], return_index=True, return_inverse=True
        )
        order = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[order], np.arange(centres.size + 1))
        for k, row in enumerate(idx[cross][first]):
            pts = crossing[order[bounds[k] : bounds[k + 1]]]
            w = interpolation_weights(
                nodes.xy[row],
                material,
                g.spec.degree,
                (x[pts], y[pts]),
                shape,
                curvature,
                warp,
            )
            out[pts] = w @ u[row]
    return out


@dataclass(frozen=True)
class Reference:
    """A fine solution with its node set and how it was made, for ``resample``.

    ``nodes`` carries the coordinates, kinds and Dirichlet indices; after a
    ``load`` its row tuples are empty, which ``build_stencils`` and
    ``resample`` do not need. ``meta`` records the count, seed, settings and
    timings, written as JSON next to the arrays.
    """

    nodes: NodeSet
    u: np.ndarray
    meta: dict

    def stencils(self, domain: Domain) -> Stencils:
        """The stencils the solution was built with, the interface group included."""
        return build_stencils(self.nodes, domain, interface=BOUNDARY)

    def save(self, path: Path) -> None:
        path = Path(path)
        np.savez_compressed(
            path,
            x=self.nodes.x,
            y=self.nodes.y,
            kind=self.nodes.kind,
            boundary_index=self.nodes.boundary_index,
            u=self.u,
        )
        meta = {**self.meta, "h": self.nodes.h, "n": self.nodes.n}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> Reference:
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path) as arrays:
            nodes = NodeSet(
                x=arrays["x"],
                y=arrays["y"],
                kind=arrays["kind"],
                h=float(meta["h"]),
                boundary_index=arrays["boundary_index"],
            )
            u = arrays["u"]
        return cls(nodes, u, meta)


def reference_solution(
    domain: Domain,
    n: int,
    values: Sequence[BoundaryValue],
    seed: int = 0,
    iterations: int = 100,
    curvature: bool = True,
    warp: bool = True,
    shape: float = GA_SHAPE,
) -> Reference:
    """The interface-aware equilibrium solution on an ``n``-node set, timed.

    The papers' settings by default: curvature, warped Gaussians, straddling
    rows. ``meta["seconds"]`` holds the node set, operator and solve times.
    """
    t0 = time.perf_counter()
    nodes = build_node_set(domain, n, seed=seed, iterations=iterations)
    t1 = time.perf_counter()
    stencils = build_stencils(nodes, domain, interface=BOUNDARY)
    op = interface_aware_operator(
        nodes, domain.material, stencils, curvature=curvature, warp=warp, shape=shape
    )
    t2 = time.perf_counter()
    u = solve_equilibrium(op, nodes, values)
    t3 = time.perf_counter()
    meta = {
        "seed": seed,
        "iterations": iterations,
        "curvature": curvature,
        "warp": warp,
        "shape": shape,
        "interface_group": int(stencils.near_interface.sum()),
        "seconds": {"nodes": t1 - t0, "operator": t2 - t1, "solve": t3 - t2},
    }
    return Reference(nodes, u, meta)
