"""The 1-D heat operator ``d/dx (alpha d/dx)`` on [-1, 1] with Dirichlet ends.

``domain``: grids and piecewise materials (dissertation eq. 64, 75);
``operators``: FD4 ``Dx``, ``A``, the naive ``Dx A Dx`` (eq. 76) and the direct
stencil; ``solve``: Dirichlet rows and the sparse solve; ``exact``: the
quadrature reference ``u = A + B ∫ dξ/alpha``. The jump-aware stencils
(E1.2), time stepping (E1.3), seeds (E3) and treatments (E3.5) follow.
"""

from .domain import (
    DISSERTATION_BC,
    PLACEMENT_TOL,
    Constant,
    Grid1D,
    Medium1D,
    PiecewiseAlpha,
    Sinusoid,
    Smooth,
    dissertation_alpha,
    eabe_alpha,
    equispaced_grid,
    grid_for,
    jump_alpha,
    node_counts,
)
from .exact import equilibrium_exact, equilibrium_flux, inverse_alpha_integral
from .operators import (
    alpha_matrix,
    derivative_matrix,
    direct_operator,
    dx_matrix,
    dxx_matrix,
    naive_operator,
)
from .solve import dirichlet_system, normalized_l2, solve_equilibrium

__all__ = [
    "DISSERTATION_BC",
    "PLACEMENT_TOL",
    "Constant",
    "Grid1D",
    "Medium1D",
    "PiecewiseAlpha",
    "Sinusoid",
    "Smooth",
    "alpha_matrix",
    "derivative_matrix",
    "direct_operator",
    "dirichlet_system",
    "dissertation_alpha",
    "dx_matrix",
    "dxx_matrix",
    "eabe_alpha",
    "equilibrium_exact",
    "equilibrium_flux",
    "equispaced_grid",
    "grid_for",
    "inverse_alpha_integral",
    "jump_alpha",
    "naive_operator",
    "node_counts",
    "normalized_l2",
    "solve_equilibrium",
]
