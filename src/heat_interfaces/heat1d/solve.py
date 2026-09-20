"""The 1-D equilibrium solve: Dirichlet rows at both ends and a sparse direct solve.

Dissertation §4.2 (``u_t = 0``); SuperLU through ``spsolve`` is the direct
baseline of plan D6.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def dirichlet_system(
    operator: sp.sparray,
    u_left: float,
    u_right: float,
    forcing: np.ndarray | None = None,
) -> tuple[sp.csc_array, np.ndarray]:
    """``(A, b)``: the operator with its end rows replaced by the Dirichlet values.

    Interior rows solve ``L u = forcing`` (zero by default, the equilibrium
    problem; a manufactured right-hand side otherwise).
    """
    n = operator.shape[0]
    interior = np.ones(n)
    interior[[0, -1]] = 0.0
    a = sp.diags_array(interior) @ operator + sp.diags_array(1.0 - interior)
    b = np.zeros(n) if forcing is None else np.array(forcing, dtype=float)
    b[0], b[-1] = u_left, u_right
    return sp.csc_array(a), b


def solve_equilibrium(
    operator: sp.sparray,
    u_left: float,
    u_right: float,
    forcing: np.ndarray | None = None,
) -> np.ndarray:
    """Solve ``L u = forcing`` inside with ``u(-1) = u_left`` and ``u(1) = u_right``."""
    a, b = dirichlet_system(operator, u_left, u_right, forcing)
    return np.asarray(spsolve(a, b))


def normalized_l2(u: np.ndarray, reference: np.ndarray) -> float:
    """``||u - ref||_2 / ||ref||_2``, the ordinate of dissertation Fig. 4-7."""
    return float(np.linalg.norm(u - reference) / np.linalg.norm(reference))
