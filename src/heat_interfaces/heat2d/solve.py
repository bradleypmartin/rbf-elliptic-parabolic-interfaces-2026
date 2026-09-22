"""The 2-D equilibrium solve: Dirichlet rows on the node set, SuperLU, and the
iterative solvers of dissertation §5.4.4 / EABE §3.3.2.

Dissertation §5.4 used MATLAB's backslash; SuperLU through ``spsolve`` is the
direct baseline of plan D6. The error norms are the 1-D module's: the 2016
"normalized ℓ2 error" is ``rms_error`` (epic #4, item 1).

The iterative study (E2.8) works on the *reduced* system, the interior block
of the operator with the Dirichlet columns moved to the right-hand side.
Solving the identity-row form of ``dirichlet_system`` iteratively is a trap:
its right-hand side lives on the Dirichlet rows alone, so BiCGSTAB's shadow
residual ``r̂ = b`` is exactly orthogonal to every residual after the first
step (``r̂ · (b − A b) = 0`` when the Dirichlet rows are the identity) and the
method breaks down at iteration 1 whether or not an interface is present.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, spilu, spsolve

from ..heat1d.solve import normalized_l2, rms_error
from .domain import NodeSet
from .operators import BoundaryValue, dirichlet_system, dirichlet_values

__all__ = [
    "ILU_ORDERING",
    "IterativeResult",
    "PRODUCT_ORDERING",
    "ReducedSystem",
    "ilu_preconditioner",
    "normalized_l2",
    "reduced_system",
    "rms_error",
    "solve_equilibrium",
    "solve_iterative",
]

METHODS = ("gmres", "bicgstab")
"""The two MATLAB solvers of dissertation §5.4.4, by their SciPy names."""


def solve_equilibrium(
    operator: sp.sparray,
    nodes: NodeSet,
    values: Sequence[BoundaryValue],
    forcing: np.ndarray | None = None,
    permc_spec: str | None = None,
) -> np.ndarray:
    """Solve ``L u = forcing`` off the Dirichlet rows with ``u = values`` on them.

    ``values`` follows the domain's ``dirichlet`` order (``dirichlet_values``).
    ``permc_spec`` is SuperLU's column ordering (``None``: its default,
    ``COLAMD``; ``PRODUCT_ORDERING`` for the naive operator).
    """
    g = dirichlet_values(nodes, values)
    a, b = dirichlet_system(operator, nodes.dirichlet, g, forcing)
    return np.asarray(spsolve(a, b, permc_spec=permc_spec))


PRODUCT_ORDERING = "MMD_ATA"
"""SuperLU's ordering for the naive ``Dx A Dx + Dy A Dy``: minimum degree on ``AᵀA``.

The product reaches the neighbours of the neighbours, about 147 nonzeros per
row against 41 for a stencil (port notes §2.2), and ``COLAMD``'s factor of it
is slow: on case 1 at δ = 0.01 one factor and solve takes 11.6 s at 20,000
nodes and 52 s at 40,000 with ``COLAMD``, 3.7 s and 10.4 s with this
ordering, and the two solutions agree to 5e-12 (E4.3, stiff note §4.2).
``MMD_AT_PLUS_A`` is no faster than ``COLAMD`` here (12 s and 72 s).
"""


@dataclass(frozen=True)
class ReducedSystem:
    """``a u_I = b`` on the interior nodes, the Dirichlet values already applied.

    ``a`` is the operator's interior block, ``b`` the forcing there minus the
    Dirichlet columns times their values ``g`` (a full-length vector, zero
    off the Dirichlet nodes); ``interior`` are the unknowns' node indices.
    ``expand`` puts an interior solution back on the whole node set. A
    left-preconditioned system (``left_preconditioned``) keeps the system it
    came from as ``original``, so residuals can be measured in the
    unweighted norm; it is ``None`` on a system built from the operator.
    """

    a: sp.csr_array
    b: np.ndarray
    interior: np.ndarray
    g: np.ndarray
    original: ReducedSystem | None = None

    @property
    def n(self) -> int:
        return len(self.interior)

    def expand(self, u_interior: np.ndarray) -> np.ndarray:
        u = np.array(self.g, dtype=float)
        u[self.interior] = u_interior
        return u

    def solve_direct(self) -> np.ndarray:
        """SuperLU on the reduced system, expanded: ``solve_equilibrium``'s answer."""
        return self.expand(np.asarray(spsolve(sp.csc_array(self.a), self.b)))

    def left_preconditioned(self, p: sp.sparray) -> ReducedSystem:
        """``P a u_I = P b`` (dissertation eq. 99) for a sparse interior ``P``."""
        p = sp.csr_array(p)
        if p.shape != (self.n, self.n):
            raise ValueError("the preconditioner must be square on the interior")
        return ReducedSystem(
            sp.csr_array(p @ self.a),
            p @ self.b,
            self.interior,
            self.g,
            self.original or self,
        )

    def residual(self, u_interior: np.ndarray) -> float:
        """``|b − a u| / |b|`` of the *original* system (itself if unpreconditioned)."""
        base = self.original or self
        return float(
            np.linalg.norm(base.b - base.a @ u_interior) / np.linalg.norm(base.b)
        )


def reduced_system(
    operator: sp.sparray,
    nodes: NodeSet,
    values: Sequence[BoundaryValue],
    forcing: np.ndarray | None = None,
) -> ReducedSystem:
    """The interior system of ``L u = forcing`` with ``u = values`` on the boundary.

    The same solution as ``solve_equilibrium`` (which keeps the Dirichlet rows
    as identity rows), in the form the iterative solvers need.
    """
    g = dirichlet_values(nodes, values)
    op = sp.csr_array(operator)
    interior = np.flatnonzero(~nodes.dirichlet)
    boundary = np.flatnonzero(nodes.dirichlet)
    rows = op[interior]
    f = np.zeros(nodes.n) if forcing is None else np.asarray(forcing, dtype=float)
    b = f[interior] - rows[:, boundary] @ g[boundary]
    return ReducedSystem(sp.csr_array(rows[:, interior]), b, interior, g)


@dataclass(frozen=True)
class IterativeResult:
    """One iterative solve: the expanded solution and how it went.

    ``iterations`` counts inner iterations (GMRES's Arnoldi steps across
    restarts, BiCGSTAB's steps); ``residual`` is the final ``|b − a u| / |b|``
    of the *unpreconditioned* system (``ReducedSystem.residual``), which on
    a left-preconditioned system differs from the ``P``-weighted residual
    the solver's own test used; ``info`` is SciPy's flag on the system the
    solver saw (0 converged, positive: the iteration cap, negative:
    breakdown); ``seconds`` is the solver's wall-clock alone.
    """

    u: np.ndarray
    method: str
    iterations: int
    residual: float
    info: int
    seconds: float

    @property
    def converged(self) -> bool:
        return self.info == 0


def solve_iterative(
    system: ReducedSystem,
    method: str = "gmres",
    rtol: float = 1e-8,
    maxiter: int = 3000,
    restart: int | None = None,
    m: LinearOperator | None = None,
    x0: np.ndarray | None = None,
) -> IterativeResult:
    """``gmres`` or ``bicgstab`` on the reduced system to ``|r| <= rtol |b|``.

    ``restart=None`` is full GMRES (MATLAB's default: one Arnoldi cycle of up
    to ``maxiter`` steps); an integer restarts every that many steps, with
    ``maxiter`` still the cap on inner iterations. ``m`` is SciPy's approximate
    inverse (``ilu_preconditioner``), whose convergence test SciPy makes on
    the true residual; a left preconditioner in matrix form goes through
    ``ReducedSystem.left_preconditioned`` instead (dissertation eq. 99), and
    the solver then tests ``|P b − P a u| <= rtol |P b|`` while the result's
    ``residual`` reports the unweighted one.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    count = [0]

    def tick(_: object) -> None:
        count[0] += 1

    a, b = system.a, system.b
    t0 = time.perf_counter()
    if method == "gmres":
        # 'legacy' makes maxiter the cap on inner iterations and lets a cycle
        # that ends on the preconditioned residual be followed by another
        # until the true residual passes; the cycle length is the restart.
        u, info = gmres(
            a,
            b,
            x0=x0,
            rtol=rtol,
            atol=0.0,
            restart=maxiter if restart is None else min(restart, maxiter),
            maxiter=maxiter,
            M=m,
            callback=tick,
            callback_type="legacy",
        )
    else:
        u, info = bicgstab(
            a, b, x0=x0, rtol=rtol, atol=0.0, maxiter=maxiter, M=m, callback=tick
        )
    seconds = time.perf_counter() - t0
    u = np.asarray(u, dtype=float)
    return IterativeResult(
        system.expand(u), method, count[0], system.residual(u), int(info), seconds
    )


ILU_ORDERING = "MMD_AT_PLUS_A"
"""``spilu``'s column ordering here: minimum degree on ``AᵀA``'s pattern.

SuperLU's default ``COLAMD`` gives an incomplete factor that is nonsense on
case 3 from 20,000 nodes on (``M b`` off the solution by 1e67 at the default
drop tolerance and fill; E2.8, port notes §2.8), and degrades on the control
past 10,000; the symmetric-pattern ordering at the same drop tolerance and
fill factor stays at two or three BiCGSTAB iterations to 40,000 nodes.
"""


def ilu_preconditioner(
    system: ReducedSystem,
    drop_tol: float = 1e-4,
    fill_factor: float = 10.0,
    permc_spec: str = ILU_ORDERING,
) -> tuple[LinearOperator, float]:
    """SuperLU's incomplete LU as SciPy's ``M``, with its build time in seconds.

    The off-the-shelf comparison of plan D6: ``spilu`` at its default drop
    tolerance and fill factor, the ordering ``ILU_ORDERING`` unless told
    otherwise (``"COLAMD"`` is SuperLU's own default).
    """
    t0 = time.perf_counter()
    ilu = spilu(
        sp.csc_array(system.a),
        drop_tol=drop_tol,
        fill_factor=fill_factor,
        permc_spec=permc_spec,
    )
    op = LinearOperator(system.a.shape, matvec=ilu.solve, dtype=float)
    return op, time.perf_counter() - t0
