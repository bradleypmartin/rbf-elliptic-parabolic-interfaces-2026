"""Time stepping on a node set: the 1-D BD4 behind the node set's Dirichlet mask.

Dissertation §5.4 integrated its one time-dependent problem (case 1, eq. 84–86,
``c_t = 1``) by BD4 with the time step scaled with the node spacing and one
LU per operator. The marcher is ``heat1d.march``'s (epic #4, item 7): the
node set's Dirichlet rows are its ``dirichlet`` mask and ``dirichlet_boundary``
turns per-curve values that may depend on time into the ``boundary(t)`` it
expects, in index order. ``analytic_history`` hands BD4 its three starting
values from a known solution, so every step of a verification run is BD4;
without them the 1-D module's RK4 start-up runs, which on these operators
means 1000–2000 sub-steps per step (``rk4_dt_limit`` is the ∞-norm bound,
and the row sums of the 42-node stencils are about ``20 / h²``, so the
start-up takes about ``30 / h`` sub-steps per step). ``interior_eigenvalues``
is the dense eigenvalue problem of Fig. 5-6.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse as sp

from ..heat1d.march import (
    Boundary,
    Forcing,
    bd4_march,
    interior_operator,
    march_steps,
)
from .domain import NodeSet

TimeBoundaryValue = float | Callable[[np.ndarray, np.ndarray, float], np.ndarray]
"""A constant, or ``(x, y, t) -> u`` on the nodes of one Dirichlet curve."""

Solution = Callable[[np.ndarray, np.ndarray, float], np.ndarray]
"""``(x, y, t) -> u`` on the nodes, such as ``exact.LayeredExact``."""


def dirichlet_boundary(nodes: NodeSet, values: Sequence[TimeBoundaryValue]) -> Boundary:
    """``t -> g`` on the Dirichlet nodes in index order, for ``heat1d.march``.

    ``values[k]`` belongs to the ``k``-th Dirichlet curve, as in
    ``operators.dirichlet_values``: a constant or ``(x, y, t) -> u``.
    """
    if len(values) != len(nodes.dirichlet_rows):
        raise ValueError(
            f"{len(values)} boundary values for "
            f"{len(nodes.dirichlet_rows)} Dirichlet curves"
        )
    fixed = np.flatnonzero(nodes.dirichlet)
    x, y, curve = nodes.x[fixed], nodes.y[fixed], nodes.boundary_index[fixed]

    def boundary(t: float) -> np.ndarray:
        g = np.empty(fixed.size)
        for k, value in enumerate(values):
            on = curve == k
            g[on] = value(x[on], y[on], t) if callable(value) else value
        return g

    return boundary


def analytic_history(
    solution: Solution, nodes: NodeSet, t_end: float, dt: float
) -> list[np.ndarray]:
    """``u`` at ``t = -3 dt, -2 dt, -dt`` on the nodes, ``dt`` rounded as marched."""
    _, dt = march_steps(t_end, dt)
    return [solution(nodes.x, nodes.y, -k * dt) for k in (3, 2, 1)]


def march_parabolic(
    operator: sp.sparray,
    nodes: NodeSet,
    u0: np.ndarray,
    t_end: float,
    dt: float,
    values: Sequence[TimeBoundaryValue],
    forcing: Forcing | None = None,
    solution: Solution | None = None,
    permc_spec: str | None = None,
) -> np.ndarray:
    """BD4 from ``u0`` at ``t = 0`` to ``t_end`` on the node set, steps of about ``dt``.

    The Dirichlet rows carry ``values`` at every step. With ``solution``, the
    analytic solution of a verification run, the three starting values are
    its (``analytic_history``) and every step is BD4; without it the RK4
    start-up of ``heat1d.march.bd4_march`` supplies them. ``permc_spec`` is
    SuperLU's ordering for the step's LU (``solve.PRODUCT_ORDERING`` for the
    naive operator; ``None`` keeps SuperLU's default).
    """
    history = None
    if solution is not None:
        history = analytic_history(solution, nodes, t_end, dt)
    return bd4_march(
        operator,
        u0,
        t_end,
        dt,
        dirichlet_boundary(nodes, values),
        forcing,
        dirichlet=nodes.dirichlet,
        history=history,
        permc_spec=permc_spec,
    )


def interior_eigenvalues(operator: sp.sparray, nodes: NodeSet) -> np.ndarray:
    """The eigenvalues of ``L`` without its Dirichlet rows and columns, dense."""
    return np.linalg.eigvals(interior_operator(operator, nodes.dirichlet).toarray())
