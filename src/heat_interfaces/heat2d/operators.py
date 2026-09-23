"""Discrete 2-D operators on a node set: ``Dx``, ``Dy``, the Laplacian, ``A``, and
the naive ``Dx A Dx + Dy A Dy``.

Stencils follow EABE §3: a node's 41 nearest neighbours with polynomials
through degree 5, or 29 and degree 4 within ``3/√n`` of a Dirichlet curve
(the MATLAB's ``boundVec``, ``3 * hApprox``), where the stencil is one-sided.
The MATLAB also shrank stencils within ``3/√n`` of ``x = 0`` and ``x = 1``,
which the x-periodic neighbour search makes unnecessary; the paper's
"close to the domain boundary" is followed instead. Every stencil, including
those of the Dirichlet nodes, gets a row in ``Dx`` and ``Dy``, because the
naive operator applies ``Dx`` twice (dissertation eq. 76, plan D3). The
Dirichlet rows of the assembled operator are then replaced by the identity
(``dirichlet_system``).

``direct_operator`` is ``α ∇² + ∇α · ∇`` on the owning piece's smooth values,
the 2-D twin of the 1-D direct stencil: fourth order wherever α is smooth
and blind to a jump. With ``α ≡ 1`` it is the Laplacian of the control
problem. ``interface_aware_operator`` (E2.3, dissertation §5.3) is the same
operator with the rows of the stencils that cross an interface recomputed
on the translated basis of ``interface.py``: ``build_stencils`` given an
``interface`` spec puts every node whose stencil at the largest size would
cross an interface into a third group of 30-node / degree-4 stencils (EABE
§3, "across interfaces"), so no standard stencil ever sees a jump, and the
members of that group whose own 30 nodes cross are translated.
``seed_operator`` (E4.5, #36) is the same dispatch for a *smooth* edge of
width δ below the spacing: the rows whose stencils see the edge
(``seeded_rows``, the reach rule of ``docs/stiff-diffusion.md`` §3.3) are
``seeds.seed_weights``'s, the rest the direct operator's, and at δ = 0 the
two agree to rounding. ``build_operator`` is the dispatch by name.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np
import scipy.sparse as sp

from ..heat1d.domain import TANH_REACH
from ..heat1d.stiff import SEED_ATOL, SEED_RTOL, edge_width
from .domain import Domain, NodeSet
from .interface import stencil_weights
from .neighbors import knn, offsets
from .rbf import BOUNDARY, GA_SHAPE, INTERIOR, StencilSpec, rbf_fd_weights
from .seeds import seed_weights

BOUNDARY_ZONE = 3.0
"""Half-width of the boundary zone in units of ``1/√n`` (MATLAB ``3 * hApprox``)."""


INTERIOR_KIND, BOUNDARY_KIND, INTERFACE_KIND = "interior", "boundary", "interface"
"""``StencilGroup.kind``: off the zones, in the boundary zone, across an interface."""


@dataclass(frozen=True)
class StencilGroup:
    """The stencils sharing one ``spec``: their centre nodes and neighbour lists.

    ``index[i]`` are the ``spec.size`` nodes of the stencil centred on
    ``rows[i]``, nearest first, so ``index[i, 0] == rows[i]``. ``kind`` says
    which zone the group serves; only the ``interface`` group's rows are
    recomputed on the translated basis.
    """

    spec: StencilSpec
    rows: np.ndarray
    index: np.ndarray
    kind: str = INTERIOR_KIND


@dataclass(frozen=True)
class Stencils:
    """Every node's stencil, grouped by spec and kind.

    ``near_boundary`` marks the boundary zone; ``near_interface`` the nodes
    whose stencil at the largest size crosses an interface (all False unless
    ``build_stencils`` was given an ``interface`` spec).
    """

    n: int
    groups: tuple[StencilGroup, ...]
    near_boundary: np.ndarray
    near_interface: np.ndarray

    def spec_of(self, node: int) -> StencilSpec:
        return self.group_of(node).spec

    def group_of(self, node: int) -> StencilGroup:
        for g in self.groups:
            if node in g.rows:
                return g
        raise KeyError(node)


def boundary_zone(
    nodes: NodeSet, domain: Domain, width: float = BOUNDARY_ZONE
) -> np.ndarray:
    """Nodes within ``width / √n`` of a Dirichlet curve, its own nodes included."""
    zone = np.zeros(nodes.n, dtype=bool)
    for curve in domain.dirichlet:
        zone |= np.abs(curve.signed_distance(nodes.x, nodes.y)) < width / sqrt(nodes.n)
    return zone


def interface_crossings(nodes: NodeSet, material, index: np.ndarray) -> np.ndarray:
    """True per row of ``index`` whose nodes lie in more than one region of the band.

    A material without interfaces (a single ``Piece2D``) crosses nothing.
    Regions are the band's ``region_index``, by exact level signs.
    """
    if not hasattr(material, "region_index"):
        return np.zeros(len(index), dtype=bool)
    region = material.region_index(nodes.x, nodes.y)
    return np.ptp(region[index], axis=1) > 0


def seeded_rows(
    nodes: NodeSet,
    material,
    index: np.ndarray,
    reach: float = TANH_REACH,
) -> np.ndarray:
    """True per row of ``index`` whose nodes see an edge of ``material`` (§3.3).

    A stencil sees an edge when the span of its nodes' signed distances to
    that curve meets the edge's support ``[−reach δ, reach δ]``, δ the
    medium's width (``heat1d.stiff.edge_width``; past ``TANH_REACH`` δ alpha
    is the piece bit for bit). At δ = 0 the support is the curve itself and
    the rule is ``interface_crossings`` exactly, the stencils that straddle
    it; that test is also OR-ed in at δ > 0, so a stencil straddling an edge
    far thinner than the spacing is seeded whatever its nodes' distances.
    The rule needs no δ of its own (H8): at δ = 0.04 on case 1, ``20 δ = 0.8``
    covers the strip and every row is seeded.
    """
    seen = interface_crossings(nodes, material, index)
    half = reach * edge_width(material)
    if half <= 0.0:
        return seen
    for curve in getattr(material, "interfaces", ()):
        d = curve.signed_distance(nodes.x, nodes.y)[index]
        seen |= (d.min(axis=1) <= half) & (d.max(axis=1) >= -half)
    return seen


def build_stencils(
    nodes: NodeSet,
    domain: Domain,
    interior: StencilSpec = INTERIOR,
    boundary: StencilSpec = BOUNDARY,
    zone: float = BOUNDARY_ZONE,
    interface: StencilSpec | None = None,
    reach: float = 0.0,
) -> Stencils:
    """Nearest-neighbour stencils: ``interior`` off the zones, ``boundary`` in the zone.

    With an ``interface`` spec, every node whose stencil at the largest of
    the three sizes crosses an interface of ``domain.material`` gets that
    spec instead (the interface group takes precedence over the boundary
    zone), so no stencil of any group but the interface group can see a
    jump; without one the interfaces are ignored, as the naive operator
    wants. With ``reach`` the membership test is ``seeded_rows``'s instead,
    so that no 42 / 5 stencil sees an unresolved edge either (§3.3); at
    δ = 0 the two tests agree.
    """
    near = boundary_zone(nodes, domain, zone)
    specs = [interior, boundary] + ([interface] if interface is not None else [])
    k = max(spec.size for spec in specs)
    if k > nodes.n:
        raise ValueError(f"{nodes.n} nodes cannot hold a {k}-node stencil")
    index, _ = knn(nodes.xy, k)
    cross = np.zeros(nodes.n, dtype=bool)
    if interface is not None:
        cross = (
            seeded_rows(nodes, domain.material, index, reach)
            if reach
            else interface_crossings(nodes, domain.material, index)
        )
    groups = []
    for spec, rows, kind in (
        (interior, np.flatnonzero(~near & ~cross), INTERIOR_KIND),
        (boundary, np.flatnonzero(near & ~cross), BOUNDARY_KIND),
        (interface, np.flatnonzero(cross), INTERFACE_KIND),
    ):
        if spec is not None and len(rows):
            groups.append(StencilGroup(spec, rows, index[rows, : spec.size], kind))
    return Stencils(nodes.n, tuple(groups), near, cross)


def derivative_matrices(
    nodes: NodeSet,
    stencils: Stencils,
    ops: tuple[str, ...],
    shape: float = GA_SHAPE,
) -> dict[str, sp.csr_array]:
    """Sparse ``(n, n)`` matrices of the RBF-FD ``ops``, one row per node."""
    rows = {op: [] for op in ops}
    cols = {op: [] for op in ops}
    vals = {op: [] for op in ops}
    for g in stencils.groups:
        dx, dy = offsets(nodes.xy, g.index, centre=nodes.xy[g.rows])
        w = rbf_fd_weights(dx, dy, ops, g.spec.degree, shape)
        r = np.repeat(g.rows, g.spec.size)
        c = g.index.ravel()
        for i, op in enumerate(ops):
            rows[op].append(r)
            cols[op].append(c)
            vals[op].append(w[i].ravel())
    n = nodes.n
    return {
        op: sp.csr_array(
            (
                np.concatenate(vals[op]),
                (np.concatenate(rows[op]), np.concatenate(cols[op])),
            ),
            shape=(n, n),
        )
        for op in ops
    }


def derivative_matrix(
    nodes: NodeSet, stencils: Stencils, op: str, shape: float = GA_SHAPE
) -> sp.csr_array:
    return derivative_matrices(nodes, stencils, (op,), shape)[op]


def alpha_matrix(nodes: NodeSet, material) -> sp.dia_array:
    """``A = diag(α(x_j, y_j))``: ``material`` is a ``Band`` or a single ``Piece2D``."""
    return sp.diags_array(material.alpha(nodes.x, nodes.y))


def naive_operator(
    nodes: NodeSet, material, stencils: Stencils, shape: float = GA_SHAPE
) -> sp.csr_array:
    """``L = Dx A Dx + Dy A Dy``, dissertation eq. 76 in 2-D (plan D3).

    Its rows reach the neighbours of the neighbours, about four times the
    stencil size; a jump in α inside that reach makes the row first order.
    """
    d = derivative_matrices(nodes, stencils, ("dx", "dy"), shape)
    a = alpha_matrix(nodes, material)
    return (d["dx"] @ a @ d["dx"] + d["dy"] @ a @ d["dy"]).tocsr()


def direct_operator(
    nodes: NodeSet, material, stencils: Stencils, shape: float = GA_SHAPE
) -> sp.csr_array:
    """``α ∇² + α_x ∂_x + α_y ∂_y`` on the owning piece: ``∇·(α ∇)`` where α is smooth.

    ``material.gradient`` is the owning piece's, so the rows are exact to the
    stencil's order where α is smooth and see nothing of a jump.
    """
    d = derivative_matrices(nodes, stencils, ("dx", "dy", "lap"), shape)
    a = alpha_matrix(nodes, material)
    ax, ay = material.gradient(nodes.x, nodes.y)
    op = a @ d["lap"] + sp.diags_array(ax) @ d["dx"] + sp.diags_array(ay) @ d["dy"]
    return op.tocsr()


def laplacian_operator(
    nodes: NodeSet, stencils: Stencils, shape: float = GA_SHAPE
) -> sp.csr_array:
    """``∇²`` by RBF-FD: the control problem's operator (``α ≡ 1``)."""
    return derivative_matrix(nodes, stencils, "lap", shape)


def interface_aware_operator(
    nodes: NodeSet,
    material,
    stencils: Stencils,
    curvature: bool = True,
    warp: bool = True,
    shape: float = GA_SHAPE,
) -> sp.csr_array:
    """The §5.3 operator: direct rows, translated-basis rows across interfaces.

    ``stencils`` should come from ``build_stencils`` with an ``interface``
    spec; the rows of its interface group whose nodes reach more than one
    region are recomputed by ``interface.stencil_weights`` (across both
    interfaces when they reach three regions), the rest are the direct
    operator's. ``curvature=False`` is the flat-interface variant EABE
    Fig. 10 and 14 compare against; on flat interfaces the two coincide.
    ``warp=False`` replaces the warped Gaussians of EABE §2.2.4, the papers'
    default, by plain ones (the "no warp" line of Fig. 11). A material
    without interfaces gives the direct operator back.
    """
    op = direct_operator(nodes, material, stencils, shape)
    if not hasattr(material, "region_index"):
        return op
    op = op.tolil()
    for g in stencils.groups:
        if g.kind != INTERFACE_KIND:
            continue
        cross = interface_crossings(nodes, material, g.index)
        for row, idx in zip(g.rows[cross], g.index[cross], strict=True):
            w = stencil_weights(
                nodes.xy[idx],
                material,
                g.spec.degree,
                shape,
                curvature=curvature,
                warp=warp,
            )
            op[row, :] = 0.0
            op[row, idx] = w
    return op.tocsr()


def seed_operator(
    nodes: NodeSet,
    material,
    stencils: Stencils,
    warp: bool = True,
    reach: float = TANH_REACH,
    shape: float = GA_SHAPE,
    rtol: float = SEED_RTOL,
    atol: float = SEED_ATOL,
    tangential: bool = False,
    flux: bool | None = None,
) -> sp.csr_array:
    """The seed operator (E4.5, §3.4): direct rows, seed rows where an edge is seen.

    ``interface_aware_operator``'s shape with ``seeds.seed_weights`` in place
    of ``interface.stencil_weights`` and ``seeded_rows`` in place of
    ``interface_crossings``: the rows of the interface group whose own nodes
    see an edge of width δ within ``reach`` δ are marched, the rest are the
    direct operator's. ``stencils`` should come from ``build_stencils`` with
    an ``interface`` spec and the same ``reach``, so that the group holds
    every row the rule will seed. ``warp=False`` is the plain-Gaussian
    ablation (H7). At δ = 0 on a jump ``Band`` the rows are E2.3's to
    rounding (H2), so the operator is ``interface_aware_operator``'s.
    ``tangential=True`` marches §3.10's coupled chain in the foot curve's
    coordinates instead (E4.11, #81), the seeds of a curved or tangentially
    varying edge; ``flux`` adds the degree-5 flux seeds (``seeds.flux_exponents``,
    E4.8), by default on a band with a ``gap``.
    """
    op = direct_operator(nodes, material, stencils, shape)
    if not hasattr(material, "region_index"):
        return op
    op = op.tolil()
    for g in stencils.groups:
        if g.kind != INTERFACE_KIND:
            continue
        seen = seeded_rows(nodes, material, g.index, reach)
        for row, idx in zip(g.rows[seen], g.index[seen], strict=True):
            w = seed_weights(
                nodes.xy[idx],
                material,
                g.spec.degree,
                shape,
                warp,
                rtol,
                atol,
                tangential,
                flux,
            )
            op[row, :] = 0.0
            op[row, idx] = w
    return op.tocsr()


OPERATOR_MODES: dict[str, Callable[..., sp.csr_array]] = {
    "naive": naive_operator,
    "direct": direct_operator,
    "jump-aware": interface_aware_operator,
    "seeds": seed_operator,
}
"""The operators by name, ``heat1d.stiff.OPERATOR_MODES``'s twin: eq. 76's
product, the blind stencil, §5.3's translated rows, the seeds."""


def build_operator(
    nodes: NodeSet,
    material,
    stencils: Stencils,
    mode: str = "seeds",
    **options: object,
) -> sp.csr_array:
    """``OPERATOR_MODES[mode](nodes, material, stencils, **options)``.

    The dispatch the plan's E4.5 names. Each mode wants the stencils it was
    built for: ``naive`` and ``direct`` the plain groups, ``jump-aware`` and
    ``seeds`` an interface group (``build_stencils(interface=BOUNDARY)``,
    with ``reach`` for the seeds).
    """
    try:
        build = OPERATOR_MODES[mode]
    except KeyError:
        raise ValueError(
            f"unknown operator {mode!r}; one of {sorted(OPERATOR_MODES)}"
        ) from None
    return build(nodes, material, stencils, **options)


BoundaryValue = float | Callable[[np.ndarray, np.ndarray], np.ndarray]
"""A constant, or ``(x, y) -> u`` on the nodes of one Dirichlet curve."""


def dirichlet_values(nodes: NodeSet, values: Sequence[BoundaryValue]) -> np.ndarray:
    """``g`` on the Dirichlet nodes, zero elsewhere.

    ``values[k]`` belongs to the ``k``-th curve of the domain's ``dirichlet``
    tuple (bottom row, top row, then case 3's circle), as a constant or a
    function of ``(x, y)``.
    """
    if len(values) != len(nodes.dirichlet_rows):
        raise ValueError(
            f"{len(values)} boundary values for "
            f"{len(nodes.dirichlet_rows)} Dirichlet curves"
        )
    g = np.zeros(nodes.n)
    for k, value in enumerate(values):
        on = nodes.boundary_index == k
        g[on] = value(nodes.x[on], nodes.y[on]) if callable(value) else value
    return g


def dirichlet_system(
    operator: sp.sparray,
    dirichlet: np.ndarray,
    values: np.ndarray,
    forcing: np.ndarray | None = None,
) -> tuple[sp.csc_array, np.ndarray]:
    """``(A, b)``: the operator's rows on ``dirichlet`` replaced by ``u = values``.

    ``dirichlet`` is a boolean mask; ``values`` a full-length vector read on
    the masked nodes; the other rows solve ``L u = forcing`` (zero by default).
    """
    dirichlet = np.asarray(dirichlet, dtype=bool)
    n = operator.shape[0]
    if dirichlet.shape != (n,):
        raise ValueError("the Dirichlet mask must have one entry per node")
    inside = (~dirichlet).astype(float)
    a = sp.diags_array(inside) @ operator + sp.diags_array(1.0 - inside)
    b = np.zeros(n) if forcing is None else np.array(forcing, dtype=float)
    b[dirichlet] = np.asarray(values, dtype=float)[dirichlet]
    return sp.csc_array(a), b
