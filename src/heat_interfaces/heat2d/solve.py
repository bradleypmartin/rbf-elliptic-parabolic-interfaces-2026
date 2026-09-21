"""The 2-D equilibrium solve: Dirichlet rows on the node set and a sparse direct solve.

Dissertation §5.4 used MATLAB's backslash; SuperLU through ``spsolve`` is the
direct baseline of plan D6. The error norms are the 1-D module's: the 2016
"normalized ℓ2 error" is ``rms_error`` (epic #4, item 1).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from ..heat1d.solve import normalized_l2, rms_error
from .domain import NodeSet
from .operators import BoundaryValue, dirichlet_system, dirichlet_values

__all__ = ["normalized_l2", "rms_error", "solve_equilibrium"]


def solve_equilibrium(
    operator: sp.sparray,
    nodes: NodeSet,
    values: Sequence[BoundaryValue],
    forcing: np.ndarray | None = None,
) -> np.ndarray:
    """Solve ``L u = forcing`` off the Dirichlet rows with ``u = values`` on them.

    ``values`` follows the domain's ``dirichlet`` order (``dirichlet_values``).
    """
    g = dirichlet_values(nodes, values)
    a, b = dirichlet_system(operator, nodes.dirichlet, g, forcing)
    return np.asarray(spsolve(a, b))
