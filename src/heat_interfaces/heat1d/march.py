"""Time stepping for ``u_t = (alpha u_x)_x + f`` with Dirichlet ends (plan D5).

BD4 (fourth-order backward differentiation) with the time step scaled with
the node spacing and one sparse LU per operator, as in dissertation §5.4;
the first three steps come from RK4 sub-cycled below its stability limit so
the start-up is fourth order too, unless the caller supplies the starting
values (``history``) from an analytic solution. Explicit RK4 on its own is
the integrator of the MATLAB solver (``FD4heat1DAC.m``) and is kept for
small node counts.
The Dirichlet values may depend on time and are imposed on the Dirichlet
rows at every stage, as that code does. Those rows are the two ends by
default; a ``dirichlet`` mask or index list names them in 2-D (the node set's
Dirichlet rows), with ``boundary(t)`` returning the values in the same
order, so ``heat2d`` marches with this module rather than a second BD4
(epic #4, item 7). ``bd4_stability_boundary`` and ``bd4_amplification`` serve
the Fig. 5-6 twin: BD4 is stable where the amplification is below one,
outside the closed curve.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

Boundary = Callable[[float], tuple[float, float] | np.ndarray]
"""``t -> (u(-1, t), u(1, t))``, or the values on the Dirichlet nodes in index order."""

Forcing = Callable[[float], np.ndarray]
"""``t -> f(x, t)`` on the nodes; the Dirichlet entries are ignored."""

Dirichlet = np.ndarray | None
"""A boolean mask or an index array of the Dirichlet nodes; ``None`` means both ends."""

BD4_HISTORY = np.array([48.0, -36.0, 16.0, -3.0]) / 25.0
"""Weights of ``u^n, u^{n-1}, u^{n-2}, u^{n-3}`` in the BD4 update."""

BD4_STEP = 12.0 / 25.0
"""The coefficient of ``dt f^{n+1}`` in the BD4 update."""

STARTUP_FRACTION = 0.5
"""BD4's RK4 start-up runs at this fraction of ``rk4_dt_limit``."""

RK4_SAFE_RADIUS = 1.4
"""RK4 is stable on the left half-disk ``|dt lambda| <= 1.4``.

The disk is well inside the region ``|1 + z + z^2/2 + z^3/6 + z^4/24| <= 1``,
which reaches -2.785 on the real axis and ±2.83i on the imaginary one; the
∞-norm of the interior operator bounds every eigenvalue's modulus, so the
sub-step ``1.4 / ||L||_∞`` is safe for complex spectra (the naive operator's)
as well as real ones.
"""


def constant_boundary(u_left: float, u_right: float) -> Boundary:
    """Time-independent Dirichlet values."""
    return lambda t: (u_left, u_right)


def smooth_step(t: np.ndarray | float, duration: float) -> np.ndarray | float:
    """A C^∞ ramp from 0 at ``t <= 0`` to 1 at ``t >= duration``.

    ``S(τ) = e^{-1/τ} / (e^{-1/τ} + e^{-1/(1-τ)})``: every derivative vanishes
    at both ends, so a boundary value ramped this way from a zero initial
    condition is compatible with the heat equation at the corner ``(x, t) =
    (-1, 0)`` to all orders. The MATLAB solver ramps with a logistic in
    time, whose value at ``t = 0`` is small but not zero and whose derivatives
    there are not either; that corner mismatch would cap a fourth-order
    convergence study, so the port uses this ramp instead.
    """
    tau = np.clip(np.asarray(t, dtype=float) / duration, 0.0, 1.0)
    out = np.zeros_like(tau)
    inside = (tau > 0.0) & (tau < 1.0)
    ti = tau[inside]
    a, b = np.exp(-1.0 / ti), np.exp(-1.0 / (1.0 - ti))
    out[inside] = a / (a + b)
    out[tau >= 1.0] = 1.0
    return out if out.ndim else float(out)


def ramp_boundary(
    duration: float, u_left: float = 1.0, u_right: float = 0.0
) -> Boundary:
    """The MATLAB problem's ends.

    ``u(-1, t)`` rises from 0 to ``u_left`` by ``smooth_step`` over ``duration``;
    ``u(1, t)`` stays ``u_right``.
    """
    return lambda t: (u_left * float(smooth_step(t, duration)), u_right)


def dirichlet_index(n: int, dirichlet: Dirichlet = None) -> np.ndarray:
    """The Dirichlet nodes as an increasing index array; ``None`` gives ``[0, n-1]``."""
    if dirichlet is None:
        return np.array([0, n - 1])
    d = np.asarray(dirichlet)
    if d.dtype == bool:
        if d.shape != (n,):
            raise ValueError("a Dirichlet mask must have one entry per node")
        return np.flatnonzero(d)
    d = np.unique(d.astype(int))
    if d.size and (d[0] < 0 or d[-1] >= n):
        raise ValueError("Dirichlet indices must lie in range(n)")
    return d


def interior_operator(
    operator: sp.sparray, dirichlet: Dirichlet = None
) -> sp.csr_array:
    """The rows and columns that evolve: ``L`` without the Dirichlet nodes.

    With the boundary values known, ``u_I' = L_II u_I + L_IB g(t)``; the
    spectrum of ``L_II`` is what the time integrator sees.
    """
    n = operator.shape[0]
    keep = np.setdiff1d(np.arange(n), dirichlet_index(n, dirichlet))
    return sp.csr_array(operator)[keep][:, keep]


def rk4_dt_limit(operator: sp.sparray, dirichlet: Dirichlet = None) -> float:
    """The largest RK4 step the ∞-norm bound guarantees stable for this operator."""
    l_ii = interior_operator(operator, dirichlet)
    row_sums = np.abs(l_ii).sum(axis=1)
    return RK4_SAFE_RADIUS / float(np.max(row_sums))


def _stepper(
    operator: sp.sparray,
    boundary: Boundary,
    forcing: Forcing | None,
    dirichlet: Dirichlet = None,
):
    n = operator.shape[0]
    fixed = dirichlet_index(n, dirichlet)
    mask = np.ones(n)
    mask[fixed] = 0.0

    def impose(t: float, u: np.ndarray) -> np.ndarray:
        u = u.copy()
        u[fixed] = boundary(t)
        return u

    def rate(t: float, u: np.ndarray) -> np.ndarray:
        du = operator @ u
        if forcing is not None:
            du = du + forcing(t)
        return mask * du

    def rk4_step(t: float, u: np.ndarray, dt: float) -> np.ndarray:
        # u carries the boundary values of time t already.
        k1 = rate(t, u)
        k2 = rate(t + dt / 2, impose(t + dt / 2, u + dt / 2 * k1))
        k3 = rate(t + dt / 2, impose(t + dt / 2, u + dt / 2 * k2))
        k4 = rate(t + dt, impose(t + dt, u + dt * k3))
        return impose(t + dt, u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    return fixed, mask, impose, rk4_step


def march_steps(t_end: float, dt: float) -> tuple[int, float]:
    """``(steps, dt)``: the step count rounded so the march lands on ``t_end``.

    Both marchers round this way, so a caller supplying BD4's starting
    ``history`` evaluates it at multiples of the ``dt`` returned here.
    """
    steps = max(1, round(t_end / dt))
    return steps, t_end / steps


def rk4_march(
    operator: sp.sparray,
    u0: np.ndarray,
    t_end: float,
    dt: float,
    boundary: Boundary,
    forcing: Forcing | None = None,
    dirichlet: Dirichlet = None,
) -> np.ndarray:
    """Explicit RK4 from ``u0`` at ``t = 0`` to ``t_end`` in steps of about ``dt``.

    The caller keeps ``dt`` below ``rk4_dt_limit(operator)``; nothing here
    checks it. The boundary values are re-imposed at every stage.
    """
    _, _, impose, rk4_step = _stepper(operator, boundary, forcing, dirichlet)
    steps, dt = march_steps(t_end, dt)
    u = impose(0.0, np.array(u0, dtype=float))
    for k in range(steps):
        u = rk4_step(k * dt, u, dt)
    return u


def bd4_march(
    operator: sp.sparray,
    u0: np.ndarray,
    t_end: float,
    dt: float,
    boundary: Boundary,
    forcing: Forcing | None = None,
    dirichlet: Dirichlet = None,
    history: Sequence[np.ndarray] | None = None,
    permc_spec: str | None = None,
) -> np.ndarray:
    """BD4 from ``u0`` at ``t = 0`` to ``t_end`` in steps of about ``dt``, one LU.

    ``u^{n+1} - (48 u^n - 36 u^{n-1} + 16 u^{n-2} - 3 u^{n-3}) / 25
    = (12/25) dt (L u^{n+1} + f^{n+1})`` inside; the Dirichlet rows carry
    the boundary values at ``t^{n+1}``. Without ``history`` the three
    starting values come from RK4 sub-cycled at half ``rk4_dt_limit`` or
    less, so the whole march is fourth order in ``dt``, and a run of three
    steps or fewer is all RK4. A caller that knows ``u`` before ``t = 0``
    (a run started from an analytic solution) passes ``history``, the values
    at ``t = -3 dt, -2 dt, -dt`` in that order with ``dt`` as ``march_steps``
    rounds it; then every step is BD4. ``permc_spec`` is SuperLU's column
    ordering for the one LU (``None``: its default, ``COLAMD``).
    """
    fixed, mask, impose, rk4_step = _stepper(operator, boundary, forcing, dirichlet)
    steps, dt = march_steps(t_end, dt)
    n = operator.shape[0]

    if history is not None:
        if len(history) != 3:
            raise ValueError("BD4 needs the three values before u0, oldest first")
        state = [
            impose(-(3 - k) * dt, np.array(u, dtype=float))
            for k, u in enumerate(history)
        ]
        state.append(impose(0.0, np.array(u0, dtype=float)))
        started = 0
    else:
        state = [impose(0.0, np.array(u0, dtype=float))]
        # Half the stability limit: at the limit the stage error of a
        # time-dependent end value is about a hundred times the interior's
        # at the node next to it, and it falls sixteen-fold per halving of
        # the sub-step.
        limit = rk4_dt_limit(operator, dirichlet)
        substeps = max(1, int(np.ceil(dt / (STARTUP_FRACTION * limit))))
        for k in range(min(3, steps)):
            u = state[-1]
            for j in range(substeps):
                u = rk4_step(k * dt + j * dt / substeps, u, dt / substeps)
            state.append(u)
        if steps <= 3:
            return state[-1]
        started = 3

    matrix = sp.diags_array(mask) @ (
        sp.eye_array(n, format="csr") - BD4_STEP * dt * sp.csr_array(operator)
    ) + sp.diags_array(1.0 - mask)
    lu = splu(sp.csc_array(matrix), permc_spec=permc_spec)
    for k in range(started, steps):
        t_next = (k + 1) * dt
        rhs = mask * (
            BD4_HISTORY[0] * state[-1]
            + BD4_HISTORY[1] * state[-2]
            + BD4_HISTORY[2] * state[-3]
            + BD4_HISTORY[3] * state[-4]
        )
        if forcing is not None:
            rhs += mask * (BD4_STEP * dt * forcing(t_next))
        rhs[fixed] = boundary(t_next)
        state.append(lu.solve(rhs))
        del state[0]
    return state[-1]


def bd4_stability_boundary(theta: np.ndarray) -> np.ndarray:
    """``z = dt lambda`` on the root locus ``sum_{j=1}^{4} (1 - e^{-iθ})^j / j``.

    The closed curve of dissertation Fig. 5-6 (bottom); BD4 is stable
    outside it.
    """
    w = 1.0 - np.exp(-1j * np.asarray(theta, dtype=float))
    return sum(w**j / j for j in range(1, 5))


def bd4_amplification(z: np.ndarray) -> np.ndarray:
    """The largest root modulus of BD4's characteristic polynomial at each ``z``.

    ``(1 - (12/25) z) ζ^4 - (48/25) ζ^3 + (36/25) ζ^2 - (16/25) ζ + 3/25``;
    a value below one means the mode ``dt lambda = z`` is damped.
    """
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    lead = 1.0 - BD4_STEP * z
    companion = np.zeros((z.size, 4, 4), dtype=complex)
    companion[:, 0, :] = -np.array([-48.0, 36.0, -16.0, 3.0]) / 25.0 / lead[:, None]
    companion[:, 1, 0] = companion[:, 2, 1] = companion[:, 3, 2] = 1.0
    return np.max(np.abs(np.linalg.eigvals(companion)), axis=1)
