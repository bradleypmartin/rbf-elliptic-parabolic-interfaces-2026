"""The scalar seeds of E4.4 (#35) and the seed rows of E4.5 (#36):
`docs/stiff-diffusion.md` §3.7's H1–H3, and H2's and H7's row-level halves.

Every check runs on real 30-node stencils of the 2500-node case-1 set (seed 0,
``h = 1/48``) at δ ∈ {h/8, h, 8h}, the ticket's widths, and at δ = 0.
"""

import time
from dataclasses import replace
from math import comb

import numpy as np
import pytest
from scipy.linalg import subspace_angles

from heat_interfaces.fd_weights import fornberg_weights
from heat_interfaces.heat1d.stiff import seed_profiles as seed_profiles_1d
from heat_interfaces.heat2d.domain import (
    Band,
    Constant2D,
    FlatLine,
    SineGraph,
    SineProduct,
    SmoothBand,
    build_node_set,
    case1,
    case2,
    case3,
)
from heat_interfaces.heat2d.exact import profile_medium
from heat_interfaces.heat2d.interface import interface_stencil, stencil_weights
from heat_interfaces.heat2d.neighbors import knn
from heat_interfaces.heat2d.rbf import (
    GA_SHAPE,
    augmented_solve,
    gaussian,
    gaussian_derivative,
    polynomial_block,
    polynomial_exponents,
)
from heat_interfaces.heat2d.seeds import (
    WARP_TOL,
    block_condition,
    chain,
    seed_basis,
    seed_coordinates,
    seed_profiles,
    seed_weights,
    weights_of,
)

N = 2500
H = 1.0 / 48.0
RATIOS = (1.0 / 8.0, 1.0, 8.0)
EXPONENTS = [tuple(int(v) for v in e) for e in polynomial_exponents(4)]
# Anchors on the innermost straddling rows, below and above each of case 1's
# lines: 0.5896 and 0.8104 outside the band, 0.6104 and 0.7896 inside it.
ANCHORS = (0.5896, 0.6104, 0.7896, 0.8104)
THIN = Band(FlatLine(0.6), FlatLine(0.62), Constant2D(0.2), Constant2D(1.0))


@pytest.fixture(scope="module")
def nodes():
    return build_node_set(case1(), N)


def stencil(nodes, y, x=0.5):
    """The 30 nearest nodes of the node nearest ``(x, y)``, that node first."""
    target = np.argmin((nodes.x - x) ** 2 + (nodes.y - y) ** 2)
    idx, _ = knn(nodes.xy, 30, query=nodes.xy[target : target + 1])
    return nodes.xy[idx[0]]


def column(a, b):
    return EXPONENTS.index((a, b))


def seed_solve_weights(sb, st):
    """The seed rows' weights with ``st``'s Gaussian block (E2.3's, plain)."""
    b_rbf = sb.alpha_e * gaussian_derivative(st.xi, st.eta, st.eps, "lap")
    w = augmented_solve(
        st.gaussian_block()[None],
        sb.block[None],
        b_rbf[None, :, None],
        sb.rhs[None, :, None],
    )
    return w[0, :, 0] / sb.scale**2


def centred_derivative(f, step, axis, half=6):
    """Twelfth-order centred d/dx along ``axis``; NaN where the stencil does not fit."""
    w = fornberg_weights(0.0, step * np.arange(-half, half + 1), 1)[1]
    f = np.moveaxis(f, axis, -1)
    m = f.shape[-1]
    out = np.full_like(f, np.nan)
    out[..., half : m - half] = sum(
        wk * f[..., k : m - 2 * half + k] for k, wk in enumerate(w)
    )
    return np.moveaxis(out, -1, axis)


def physical(frame, xi, eta):
    c, s = np.cos(frame.angle), np.sin(frame.angle)
    return (
        frame.x0 + frame.scale * (c * xi - s * eta),
        frame.y0 + frame.scale * (s * xi + c * eta),
    )


# --- the chain ----------------------------------------------------------------


def test_the_chain_is_22_levels_and_nilpotent():
    # §3.2: 5 + 4 + 6 + 4 + 3 levels for a = 0 … 4, each seed's from j = a
    # down by twos; the coupling only reaches lower degree or a higher level
    # of the same seed, so the system is nilpotent (triangular) with frozen α.
    ch = chain(4)
    assert ch.size == 22
    counts = [sum(1 for a, _, _ in ch.levels if a == k) for k in range(5)]
    assert counts == [5, 4, 6, 4, 3]
    assert all(j <= a and (a - j) % 2 == 0 for a, _, j in ch.levels)
    block = np.block(
        [
            [np.zeros((22, 22)), np.eye(22)],
            [ch.source - 0.7 * ch.lower, np.zeros((22, 22))],
        ]
    )
    assert not np.linalg.matrix_power(block, 2 * 22).any()
    assert ch.index(2, 0, 0) == ch.levels.index((2, 0, 0))


@pytest.mark.parametrize("ratio", (0.0, *RATIOS))
def test_constant_alpha_gives_the_monomials(nodes, ratio):
    # H1, first clause: with α the same on both sides the chain is the
    # monomials themselves at any δ (§3.2, by induction on the degree).
    flat = Band(FlatLine(0.6), FlatLine(0.8), Constant2D(0.37), Constant2D(0.37))
    sb = seed_basis(stencil(nodes, 0.5896), SmoothBand(flat, ratio * H))
    np.testing.assert_allclose(
        sb.block, polynomial_block(sb.xi, sb.eta, 4), rtol=0, atol=5e-15
    )
    top = np.array([j == a for a, _, j in sb.profiles.chain.levels])
    assert np.abs(sb.profiles.g[~top]).max() < 1e-15
    np.testing.assert_array_equal(sb.rhs, [0, 0, 0, 0.74, 0, 0.74, *[0] * 9])


# --- H1: the march is the chain ---------------------------------------------------


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("y", (0.5896, 0.6104))
def test_the_shift_identity_holds_through_the_edge(nodes, ratio, y):
    # §3.2: the frozen problem is invariant under shifts along ξ, so every
    # level is a binomial multiple of a level-zero function; the 44-state
    # march must reproduce that to rounding (a wrong level coupling would not).
    sb = seed_basis(stencil(nodes, y), SmoothBand(case1().material, ratio * H))
    eta = np.union1d(sb.eta, np.linspace(-1.0, 1.0, 41))
    p = seed_profiles(sb.profile, eta, alpha_e=sb.alpha_e)
    ch = p.chain
    scale = np.abs(p.g).max()
    for a, b, j in ch.levels:
        lhs = p.g[ch.index(a, b, j)]
        m = a - j
        rhs = comb(a, j) * p.g[ch.index(m, b, 0)]
        assert np.abs(lhs - rhs).max() < 5e-15 * scale, (a, b, j)
    # The constant-flux seed's flux is α_e everywhere: the warp's α η̃′ ≡ α_e.
    np.testing.assert_allclose(p.psi[ch.index(0, 1, 0)], sb.alpha_e, rtol=1e-15)


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("band", (case1().material, THIN), ids=("case1", "thin"))
def test_the_residual_on_a_tensor_grid_is_the_differences_floor(nodes, ratio, band):
    # H1: L φ_e − α_e Σ C φ_{e′} on 25 × 801 points of (ξ, η) ∈ [−1, 1]², with
    # twelfth-order differences in both directions and α from the medium at
    # the physical points, sharing no coefficient bookkeeping with the march;
    # the thin band puts both of its edges inside the grid (one march).
    medium = SmoothBand(band, ratio * H)
    sb = seed_basis(stencil(nodes, 0.5896), medium)
    xi, eta = np.linspace(-1.0, 1.0, 25), np.linspace(-1.0, 1.0, 801)
    phi = seed_profiles(sb.profile, eta, alpha_e=sb.alpha_e).values(xi[:, None])
    alpha = medium.alpha(*physical(sb.frame, xi[:, None], eta[None, :]))
    worst = 0.0
    for k, (a, b) in enumerate(EXPONENTS):
        f = phi[..., k]
        lf = sum(
            centred_derivative(
                alpha * centred_derivative(f, d[1] - d[0], ax), d[1] - d[0], ax
            )
            for ax, d in ((0, xi), (1, eta))
        )
        rhs = np.zeros_like(f)
        if a >= 2:
            rhs += a * (a - 1) * phi[..., column(a - 2, b)]
        if b >= 2:
            rhs += b * (b - 1) * phi[..., column(a, b - 2)]
        rhs *= sb.alpha_e
        worst = max(worst, np.nanmax(np.abs(lf - rhs)) / max(np.abs(rhs).max(), 1.0))
    assert worst < 2e-10


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("y", (0.5896, 0.6104))
def test_the_right_hand_side_is_the_true_operator_at_the_anchor(nodes, ratio, y):
    # §3.4, decision 4: with α varying along the tangent (the sine product
    # inside a flat band) the seeds solve the frozen profile's equation, and
    # the true L on each seed at the anchor is 2 α_e on the two quadratics and
    # h_s α_ξ on ξ; differences on a 25 × 25 patch of spacing 0.004 read it.
    band = Band(FlatLine(0.6), FlatLine(0.8), SineProduct(0.2, 0.1), Constant2D(1.0))
    medium = SmoothBand(band, ratio * H)
    sb = seed_basis(stencil(nodes, y, x=0.4), medium)
    t = 0.004 * np.arange(-12, 13)
    phi = seed_profiles(sb.profile, t, alpha_e=sb.alpha_e).values(
        np.broadcast_to(t[:, None], (25, 25))
    )
    alpha = medium.alpha(*physical(sb.frame, t[:, None], t[None, :]))
    w = fornberg_weights(0.0, 0.004 * np.arange(-6, 7), 1)[1]
    at_anchor = []
    for k in range(15):
        fx = alpha * centred_derivative(phi[..., k], 0.004, 0)
        fy = alpha * centred_derivative(phi[..., k], 0.004, 1)
        at_anchor.append(w @ fx[6:19, 12] + w @ fy[12, 6:19])
    assert np.abs(np.array(at_anchor) - sb.rhs).max() < 1e-11
    if y > 0.6:  # inside the band the tangential term is O(1)
        assert abs(sb.rhs[column(1, 0)]) > 1e-3
    others = np.delete(sb.rhs, [column(1, 0), column(2, 0), column(0, 2)])
    assert not others.any()


@pytest.mark.parametrize("ratio", (0.0, *RATIOS))
def test_the_2d_march_is_the_1d_march_on_the_y_profile(nodes, ratio):
    # On case 1 the seeds of ηᵇ are E3.4's 1-D seeds of the profile in y
    # (``profile_medium``: E3.2's medium, the stops the 1-D march's).
    medium = SmoothBand(case1().material, ratio * H)
    sb = seed_basis(stencil(nodes, 0.5896), medium)
    eta = np.unique(sb.eta)
    one = seed_profiles_1d([sb.frame.y0], [sb.scale], eta, profile_medium(medium), 5)
    two = seed_profiles(sb.profile, eta, alpha_e=sb.alpha_e).values(np.zeros(eta.size))
    cols = [column(0, b) for b in range(5)]
    assert np.abs(one[0].T - two[:, cols]).max() < 5e-12 * np.abs(one).max()


def test_one_stencils_march_takes_milliseconds(nodes):
    # #35's acceptance line: 1.7 ms at δ = 0, 2.7–6.8 ms through the edge,
    # on the 2026-09-22 machine; a tenfold margin here.
    medium = case1().material
    for ratio in (0.0, *RATIOS):
        smooth = SmoothBand(medium, ratio * H)
        times = []
        for y in ANCHORS:
            xy = stencil(nodes, y)
            start = time.perf_counter()
            seed_basis(xy, smooth)
            times.append(time.perf_counter() - start)
        assert np.median(times) < 0.03, (ratio, times)


# --- H2: the jump limit is E2.3 ---------------------------------------------------


@pytest.mark.parametrize(
    ("band", "y"),
    [(case1().material, y) for y in ANCHORS]
    + [(THIN, y) for y in (0.5896, 0.6104, 0.63)],
    ids=[f"case1-{y}" for y in ANCHORS] + [f"thin-{y}" for y in (0.5896, 0.6104, 0.63)],
)
def test_at_delta_zero_the_seeds_are_e23s_stencil(nodes, band, y):
    # H2: on constant pieces across flat lines the seed span is the
    # translated basis's (the thin band's stencils reach all three regions,
    # translated twice by E2.3, one march here), the weights with the same
    # plain Gaussian block are stencil_weights(warp=False)'s, and φ₀₁ at the
    # nodes is E2.4's warped normal coordinate (§3.4).
    xy = stencil(nodes, y)
    regions = set(band.region_index(xy[:, 0], xy[:, 1]).tolist())
    assert regions == ({0, 1, 2} if band is THIN else {0, 1} if y < 0.7 else {1, 2})
    sb = seed_basis(xy, band)
    assert np.array_equal(sb.block, seed_basis(xy, SmoothBand(band, 0.0)).block)
    st = interface_stencil(xy, band, 4, warp=False)
    assert np.sin(subspace_angles(st.polynomial_block(), sb.block).max()) < 1e-13
    w_ref = stencil_weights(xy, band, 4, warp=False)
    w = seed_solve_weights(sb, st)
    assert np.abs(w - w_ref).max() < 2e-12 * np.abs(w_ref).max()
    warped = interface_stencil(xy, band, 4, warp=True)
    np.testing.assert_allclose(sb.xi, warped.xi, rtol=0, atol=1e-15)
    np.testing.assert_allclose(sb.warp, warped.eta, rtol=0, atol=5e-15)


def test_the_jump_limit_is_first_order_in_delta_over_h(nodes):
    # H2, the δ > 0 half (P4's ladder in 2-D): the sine of the largest
    # principal angle between the seed span and E2.3's, and the weights'
    # distance, fall with δ/h and are linear in it below h/64, with no floor
    # down to 1e-5 h (2026-09-22: angles 0.78, 0.30, 0.059 at 8h, h, h/8,
    # then 4.76e-4, 4.76e-5, 4.76e-6).
    band = case1().material
    xy = stencil(nodes, 0.5896)
    st = interface_stencil(xy, band, 4, warp=False)
    w_ref = stencil_weights(xy, band, 4, warp=False)
    ratios = (8.0, 1.0, 1 / 8, 1 / 64, 1e-3, 1e-4, 1e-5)
    angles, weights = [], []
    for r in ratios:
        sb = seed_basis(xy, SmoothBand(band, r * H))
        angles.append(np.sin(subspace_angles(st.polynomial_block(), sb.block).max()))
        w = seed_solve_weights(sb, st)
        weights.append(np.abs(w - w_ref).max() / np.abs(w_ref).max())
    angles, weights = np.array(angles), np.array(weights)
    assert np.all(np.diff(angles) < 0)
    np.testing.assert_allclose(angles[-3:], [4.76e-4, 4.76e-5, 4.76e-6], rtol=0.01)
    np.testing.assert_allclose(
        angles[-4:-1] / angles[-3:], [15.7, 10.0, 10.0], rtol=0.02
    )
    np.testing.assert_allclose(weights[-3:], [3.65e-3, 3.64e-4, 3.64e-5], rtol=0.01)


# --- H3: the seed blocks condition like polynomial blocks ---------------------------


@pytest.mark.parametrize("y", ANCHORS)
def test_seed_blocks_condition_like_polynomial_blocks(nodes, y):
    # H3 (P5's twin): column-scaled, the 30 × 15 seed block stays within a
    # factor three of the monomial block's condition number on the same
    # nodes from δ = 1e-5 h to 8 h (2026-09-22: 29–149 against 58–63), and
    # tends to the δ = 0 block's, the jump's span in another basis.
    band = case1().material
    xy = stencil(nodes, y)
    ratios = (8.0, 1.0, 0.5, 1 / 8, 1 / 64, 1e-3, 1e-5)
    conds = []
    for r in ratios:
        sb = seed_basis(xy, SmoothBand(band, r * H))
        conds.append(block_condition(sb.block))
    raw, scaled = np.array(conds).T
    monomial = np.linalg.cond(polynomial_block(sb.xi, sb.eta, 4))
    assert np.all((scaled > monomial / 3) & (scaled < 3 * monomial)), scaled
    jump = block_condition(seed_basis(xy, band).block)
    np.testing.assert_allclose([raw[-1], scaled[-1]], jump, rtol=1e-4)


# --- E4.5 (#36): the seed rows ------------------------------------------------------


SINE_BAND = Band(FlatLine(0.6), FlatLine(0.8), SineProduct(0.2, 0.1), Constant2D(1.0))


def epsilon(sb, shape=GA_SHAPE):
    """E2.3's shape parameter of a stencil: ``shape h_s / d`` (``rbf``'s, in units
    of the radius), read off the physical offsets as ``interface_stencil`` reads it."""
    r = np.hypot(sb.xy[:, 0] - sb.xy[0, 0], sb.xy[:, 1] - sb.xy[0, 1])
    return shape * sb.scale / np.where(r > 0.0, r, np.inf).min()


def cancelled_weights(sb, warp=True):
    """The seed row with §3.4's *cancelled* right-hand side, assembled here.

    ``α_e (G_ξξ + G_η̃η̃) + α_ξ G_ξ`` in the warped coordinates, which is what
    the chain-rule form in ``seeds.weights_of`` must come to once
    ``α η̃′ ≡ α_e`` kills the ``G_η̃`` term.
    """
    xi, eta = (sb.xi, sb.warp) if warp else (sb.xi, sb.eta)
    eps = epsilon(sb)
    b_rbf = sb.alpha_e * gaussian_derivative(xi, eta, eps, "lap")
    b_rbf = b_rbf + sb.gradient[0] * gaussian_derivative(xi, eta, eps, "dx")
    if not warp:
        b_rbf = b_rbf + sb.gradient[1] * gaussian_derivative(xi, eta, eps, "dy")
    block = gaussian(xi[:, None] - xi[None, :], eta[:, None] - eta[None, :], eps)
    w = augmented_solve(
        block[None], sb.block[None], b_rbf[None, :, None], sb.rhs[None, :, None]
    )
    return w[0, :, 0] / sb.scale**2


@pytest.mark.parametrize("warp", (True, False))
@pytest.mark.parametrize(
    ("band", "y"),
    [(case1().material, y) for y in ANCHORS] + [(THIN, y) for y in (0.6104, 0.63)],
    ids=[f"case1-{y}" for y in ANCHORS] + [f"thin-{y}" for y in (0.6104, 0.63)],
)
def test_at_delta_zero_a_seed_row_is_e23s_row(nodes, band, y, warp):
    # H2 at the row level: with the φ₀₁ warp the Gaussian block is E2.4's
    # warped one and with it off E2.3's plain one, so the whole row is
    # ``stencil_weights``'s, three-region stencils included.
    xy = stencil(nodes, y)
    w = seed_weights(xy, band, 4, warp=warp)
    ref = stencil_weights(xy, band, 4, warp=warp)
    assert np.abs(w - ref).max() < 1e-11 * np.abs(ref).max()


def test_at_delta_zero_every_crossing_row_is_e23s_row(nodes):
    # The sweep behind the single anchors: every eighth crossing stencil of
    # the 2500-node set, the warp on (E4.5's operator setting).
    band = case1().material
    region = band.region_index(nodes.x, nodes.y)
    index, _ = knn(nodes.xy, 30)
    crossing = np.flatnonzero(np.ptp(region[index], axis=1) > 0)
    worst = 0.0
    for row in crossing[::8]:
        xy = nodes.xy[index[row]]
        ref = stencil_weights(xy, band, 4)
        w = seed_weights(xy, band, 4)
        worst = max(worst, np.abs(w - ref).max() / np.abs(ref).max())
    assert len(crossing[::8]) > 60
    assert worst < 1e-11, worst


@pytest.mark.parametrize("ratio", (0.0, *RATIOS))
@pytest.mark.parametrize("warp", (True, False))
def test_the_chain_rule_right_hand_side_is_the_cancelled_form(nodes, ratio, warp):
    # §3.8 decision 5: ``weights_of`` applies L to G(ξ, η̃(η)) by the chain
    # rule, every coefficient read off the march, and the ``G_η̃`` term must
    # cancel against the warp's curvature. The row is therefore the one
    # assembled here from α_e Δ_{ξη̃} G + α_ξ G_ξ alone.
    medium = SmoothBand(SINE_BAND, ratio * H)
    sb = seed_basis(stencil(nodes, 0.6104, x=0.4), medium)
    w = weights_of(sb, warp=warp)
    ref = cancelled_weights(sb, warp=warp)
    # The two right-hand sides are the same number summed in a different
    # order, so the rows agree to the saddle-point solve's rounding.
    assert np.abs(w - ref).max() < 1e-10 * np.abs(ref).max()
    if warp:  # the term that cancels: (α η̃′)′ at the anchor, from the chain
        ch = sb.profiles.chain
        segment = int(sb.profile.segment(np.zeros(1))[0])
        rate = ch.rate(sb.alpha_e, sb.profile.alpha_function(segment))
        state = rate(0.0, ch.initial(sb.alpha_e))
        assert state[ch.size + ch.index(0, 1, 0)] == 0.0
        assert abs(sb.gradient[1]) > 1e-3 if ratio else True


@pytest.mark.parametrize("warp", (True, False))
def test_a_seed_row_is_the_operator_on_a_function_outside_its_span(nodes, warp):
    # The row is exact on its 15 seeds by construction; on a smooth function
    # that is not in their span it must still be the operator, to the
    # stencil's order. α varies along the tangent here (the sine product
    # inside the band), so a wrong sign on either gradient term shows. Only
    # at a resolved edge (δ = 8h): a generic smooth function has no business
    # being differentiated across an unresolved one, and the seed rows, like
    # the jump's, are built for the functions that satisfy the edge's
    # conditions (E4.6 measures the rows on those).
    ratio = 8.0
    medium = SmoothBand(SINE_BAND, ratio * H)
    xy = stencil(nodes, 0.6104, x=0.4)
    x, y = xy[:, 0], xy[:, 1]
    u = np.sin(2 * np.pi * x) * np.cos(3.0 * y)
    ux = 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(3.0 * y)
    uy = -3.0 * np.sin(2 * np.pi * x) * np.sin(3.0 * y)
    lap = -(4 * np.pi**2 + 9.0) * u
    ax, ay = (g[0] for g in medium.gradient(x[:1], y[:1]))
    exact = medium.alpha(x[:1], y[:1])[0] * lap[0] + ax * ux[0] + ay * uy[0]
    w = seed_weights(xy, medium, 4, warp=warp)
    assert abs(w @ u - exact) < 0.02 * abs(exact), (ratio, warp, w @ u, exact)


def test_seed_coordinates_are_the_warp_and_the_flux_is_checked(nodes):
    sb = seed_basis(stencil(nodes, 0.5896), SmoothBand(case1().material, H / 8))
    xi, eta = seed_coordinates(sb, warp=True)
    assert xi is sb.xi and np.array_equal(eta, sb.warp)
    xi, eta = seed_coordinates(sb, warp=False)
    assert np.array_equal(eta, sb.eta) and not np.array_equal(eta, sb.warp)
    ch = sb.profiles.chain
    psi = np.array(sb.profiles.psi)
    psi[ch.index(0, 1, 0), 3] *= 1.0 + 10 * WARP_TOL
    drifted = replace(sb, profiles=replace(sb.profiles, psi=psi))
    with pytest.raises(RuntimeError, match="the march, not the geometry"):
        seed_coordinates(drifted, warp=True)
    seed_coordinates(drifted, warp=False)  # the plain rows never read the flux


def test_a_seed_row_leaves_e23s_row_as_the_edge_widens(nodes):
    # H2's δ > 0 half at the row level: the seed row is the jump's at δ = 0
    # and a different row once the edge is a fraction of the spacing, so the
    # operator of E4.6 is not silently the δ = 0 construction.
    band = case1().material
    xy = stencil(nodes, 0.5896)
    jump = seed_weights(xy, band, 4)
    distances = []
    for ratio in (1.0 / 64.0, *RATIOS):
        w = seed_weights(xy, SmoothBand(band, ratio * H), 4)
        distances.append(np.abs(w - jump).max() / np.abs(jump).max())
    # 0.006, 0.060, 0.31, 0.50 at δ/h = 1/64, 1/8, 1 and 8 (2026-09-22): first
    # order in δ/h below h, with the warp moving with the block and taking
    # roughly a factor four off E4.4's fixed-Gaussian ladder (§4.4).
    assert distances == sorted(distances), distances
    assert distances[0] < 0.02 and distances[-1] > 0.25, distances


# --- the contract -------------------------------------------------------------------


def test_the_basis_refuses_what_it_cannot_build(nodes):
    xy = stencil(nodes, 0.5896)
    band = case1().material
    with pytest.raises(ValueError, match="15 seeds"):
        seed_basis(xy[:14], band)
    with pytest.raises(ValueError, match="15 seeds"):
        seed_basis(xy[:, :1], band)
    with pytest.raises(ValueError, match="coincide"):
        seed_basis(np.vstack([xy, xy[3:4]]), band)
    with pytest.raises(NotImplementedError, match="E4.8"):
        seed_basis(xy, SmoothBand(case3().material, 0.01))
    sb = seed_basis(xy, band)
    with pytest.raises(ValueError, match="1-D"):
        seed_profiles(sb.profile, sb.eta[None, :])
    with pytest.raises(ValueError, match="15 seeds"):
        seed_weights(xy[:14], band)


# --- route (a): curved features (E4.7, #38) -----------------------------------------


@pytest.fixture(scope="module")
def curved_nodes():
    return build_node_set(case2(), N)


def test_curved_seeds_are_the_flat_seeds_of_the_same_local_stencil(curved_nodes):
    # Route (a) (stiff note §3.5): along the foot point's normal the blend is
    # the tanh in that curve's distance, so a curved stencil's seeds are the
    # flat seeds of its own local coordinates. The companion's test, here on
    # a real case-2 stencil tilted with the lower sine, the upper curve far
    # beyond 20 δ (alpha is the piece there to the bit).
    delta = 0.002
    band = Band(SineGraph(0.6), SineGraph(0.8), Constant2D(0.2), Constant2D(1.0))
    xy = stencil(curved_nodes, 0.6 + 0.02 * np.sin(2 * np.pi * 0.03) - 0.01, x=0.03)
    sb = seed_basis(xy, SmoothBand(band, delta))
    assert sb.interface == 0 and abs(sb.profile.nx) > 0.1
    d_e = float(band.lower.signed_distance(xy[:1, 0], xy[:1, 1])[0])
    flat = Band(FlatLine(0.6), FlatLine(0.8), Constant2D(0.2), Constant2D(1.0))
    twin = np.column_stack([0.5 + sb.scale * sb.xi, 0.6 + d_e + sb.scale * sb.eta])
    tb = seed_basis(twin, SmoothBand(flat, delta))
    np.testing.assert_allclose(tb.xi, sb.xi, atol=1e-13)
    np.testing.assert_allclose(tb.eta, sb.eta, atol=1e-13)
    scale = np.abs(sb.block).max(axis=0)
    assert np.abs(sb.block - tb.block).max(axis=0).__truediv__(scale).max() < 1e-12
    np.testing.assert_allclose(sb.rhs, tb.rhs, atol=1e-12)


@pytest.mark.parametrize("delta", (0.0, 0.0025))
def test_the_seed_frame_is_e23s_on_every_crossing_stencil_of_case2(curved_nodes, delta):
    # The E4.4 breadcrumb's decision: ``nearest_interface`` frames on the curve
    # nearest the anchor, E2.3 on the curve bordering its region. On case 2's
    # 0.2-thick band a 30-node stencil crosses one curve at most, and the two
    # rules pick the same curve for every crossing stencil.
    from heat_interfaces.heat2d.operators import (
        INTERFACE_KIND,
        build_stencils,
        interface_crossings,
    )
    from heat_interfaces.heat2d.rbf import BOUNDARY
    from heat_interfaces.heat2d.seeds import nearest_interface

    domain = case2()
    band = domain.material
    stencils = build_stencils(curved_nodes, domain, interface=BOUNDARY)
    (group,) = [g for g in stencils.groups if g.kind == INTERFACE_KIND]
    index = group.index[interface_crossings(curved_nodes, band, group.index)]
    medium = SmoothBand(band, delta)
    mismatched = 0
    for idx in index:
        xy = curved_nodes.xy[idx]
        st = interface_stencil(xy, band, 4)
        e23 = st.regions[st.anchor].frame
        mismatched += nearest_interface(medium, *xy[0]) != e23
    assert len(index) > 400 and mismatched == 0
