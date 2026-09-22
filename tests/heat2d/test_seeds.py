"""The scalar seeds of E4.4 (#35): `docs/stiff-diffusion.md` §3.7's H1–H3.

Every check runs on real 30-node stencils of the 2500-node case-1 set (seed 0,
``h = 1/48``) at δ ∈ {h/8, h, 8h}, the ticket's widths, and at δ = 0.
"""

import time
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
    SineProduct,
    SmoothBand,
    build_node_set,
    case1,
    case2,
)
from heat_interfaces.heat2d.exact import profile_medium
from heat_interfaces.heat2d.interface import interface_stencil, stencil_weights
from heat_interfaces.heat2d.neighbors import knn
from heat_interfaces.heat2d.rbf import (
    augmented_solve,
    gaussian_derivative,
    polynomial_block,
    polynomial_exponents,
)
from heat_interfaces.heat2d.seeds import (
    block_condition,
    chain,
    seed_basis,
    seed_profiles,
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
    with pytest.raises(NotImplementedError, match="E4.7"):
        seed_basis(xy, SmoothBand(case2().material, 0.01))
    sb = seed_basis(xy, band)
    with pytest.raises(ValueError, match="1-D"):
        seed_profiles(sb.profile, sb.eta[None, :])
