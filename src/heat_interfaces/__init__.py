"""RBF-FD for elliptic and parabolic PDEs in domains with interfaces.

Library layout (filled in by the epics in ``docs/plan.md``):

``heat1d/``
    The 1-D heat operator ``d/dx (alpha d/dx)`` on ``[-1, 1]`` with Dirichlet
    ends: naive FD4, the dissertation's translated piecewise-polynomial
    stencils for a jump (ch. 4), seed stencils for a sub-grid smooth edge,
    coefficient treatments, equilibrium and time-dependent solves.
``heat2d/``
    The 2-D operator ``div(alpha grad)`` on the x-periodic unit strip with
    Dirichlet rows at ``y = 0, 1``: scattered nodes, Gaussian RBF-FD with
    polynomial augmentation, interface-aware stencils with curvature
    (dissertation ch. 5, Martin & Fornberg 2017), scalar seeds, solvers.
"""

__version__ = "0.1.0"
