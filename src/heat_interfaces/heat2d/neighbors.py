"""Nearest neighbours on the x-periodic strip.

The 2-D test domains (dissertation §5.4, EABE §3) are periodic in x with
period 1 and closed in y. The MATLAB found periodic neighbours by tiling
copies of the nodes across x = 0 and x = 1 (``findperiodicneighbors6``);
``cKDTree`` does the same with a box size in x and none in y, and returns
distances measured across the seam. Every routine here takes nodes as an
``(n, 2)`` array with ``0 <= x < period``; ``wrap_x`` puts them there.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

PERIOD = 1.0
"""The x-period of every test domain."""


def wrap_x(x: np.ndarray, period: float = PERIOD) -> np.ndarray:
    """``x`` reduced into ``[0, period)``."""
    return np.mod(np.asarray(x, dtype=float), period)


def periodic_dx(dx: np.ndarray, period: float = PERIOD) -> np.ndarray:
    """An x-difference reduced to the nearest image, in ``[-period/2, period/2)``."""
    return np.mod(np.asarray(dx, dtype=float) + period / 2, period) - period / 2


def periodic_tree(xy: np.ndarray, period: float = PERIOD) -> cKDTree:
    """A kd-tree on the strip: periodic in x, not in y."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("nodes must be an (n, 2) array")
    if np.any(xy[:, 0] < 0) or np.any(xy[:, 0] >= period):
        raise ValueError("node x-coordinates must lie in [0, period); use wrap_x")
    return cKDTree(xy, boxsize=[period, 0.0])


def knn(
    xy: np.ndarray,
    k: int,
    period: float = PERIOD,
    query: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` nearest nodes to each query point, wrapping in x only.

    Returns ``(index, distance)``, each ``(m, k)``, nearest first. With no
    ``query`` the nodes are their own queries and column 0 is the node itself
    at distance 0 (unless two nodes coincide). ``k`` counts that self entry,
    as the paper's "42 nodes" does.
    """
    xy = np.asarray(xy, dtype=float)
    if not 1 <= k <= len(xy):
        raise ValueError(f"k = {k} must be between 1 and the node count {len(xy)}")
    tree = periodic_tree(xy, period)
    q = xy if query is None else np.asarray(query, dtype=float)
    dist, idx = tree.query(q, k=k, workers=-1)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    return idx, dist


def offsets(
    xy: np.ndarray,
    idx: np.ndarray,
    period: float = PERIOD,
    centre: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour minus centre, ``(dx, dy)`` of ``idx``'s shape, x to the nearest image.

    The local coordinates a stencil is built in (E2.2) and the repulsion
    directions of the node generator. ``centre`` defaults to the node whose
    row of ``idx`` it is, i.e. ``xy[i]`` for row ``i``.
    """
    xy = np.asarray(xy, dtype=float)
    c = xy if centre is None else np.asarray(centre, dtype=float)
    dx = periodic_dx(xy[idx, 0] - c[:, 0][:, None], period)
    dy = xy[idx, 1] - c[:, 1][:, None]
    return dx, dy


def nearest_spacing(xy: np.ndarray, period: float = PERIOD) -> np.ndarray:
    """Distance from each node to its nearest other node (the ``d`` of EABE eq. 31)."""
    _, dist = knn(xy, 2, period)
    return dist[:, 1]
