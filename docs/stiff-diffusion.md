# Stiff smooth edges in diffusion: seed stencils for `∂ₓ α ∂ₓ` (E3–E4)

The sub-grid-edge study of this repo: a material edge of width δ smaller
than the node spacing h, and stencils on the unchanged grid that keep
fourth order through it because their polynomial basis is replaced by
*seeds*, functions continued through the edge by ODEs. Canonical for E3–E4
(plan D11); the manuscript quotes this note and never becomes a second
source of truth. §1 is the formulation (E3.1, #26); §2 holds the 1-D
results (E3.2, #27, to E3.6, #31, which closes it in §2.5); §3 is the
2-D design (E4.1, #32) and §4–5 will hold the 2-D results (E4.2, #33,
to E4.10, #41). The port of the 2016 methods this builds on is in
`docs/port-notes.md`.

The construction is the one of the wave-equation companion, *Seed
stencils: high-order finite differences and RBF-FD through material edges
too steep for the grid* (B. P. Martin, 2026; packaged for arXiv on
2026-09-20, cited below as "the companion"), specialised to the diffusion
operator. Nothing is imported from it (plan D2): the chain, the marchers
and the tests are re-derived and re-implemented here, and the companion is
cited for the wave case and for the well-posedness argument it proved in
the two-field setting.

## 1. Formulation (E3.1, #26)

### 1.1 Setting

The 1-D heat equation of dissertation ch. 4 (eq. 51; EABE eq. 1), on
[−1, 1] with Dirichlet ends and a positive diffusivity α(x):

    u_t = ∂ₓ (α u_x) ≡ L u.

At an interface, temperature and heat flux are continuous, and so is every
time derivative of both (eq. 52–55): for every k, `Dᵏu` and `α ∂ₓ Dᵏu`
match from the two sides. The equilibrium problem (`u_t = 0`) has constant
flux `α u_x = B`, so `u = u(−1) + B ∫_{−1}^{x} dξ/α`, which is the
quadrature reference of `heat1d.exact.equilibrium_exact` (exact at any
δ); the parabolic reference is Chebyshev collocation per smooth piece
(`chebyshev_parabolic`, port notes §1.6).

**The smooth edge.** E3.2 (#27) replaces the jump at an interface `x_c`
by a tanh transition of width δ between the two smooth pieces
`α⁻`, `α⁺` of `PiecewiseAlpha`:

    α(x) = (1 − s(x)) α⁻(x) + s(x) α⁺(x),    s(x) = ½ [1 + tanh((x − x_c)/δ)],

so that δ = 0 is the jump bit for bit and the pieces keep their own
variation (constants for the MATLAB problem `1/9 | 1`; `1` and
`0.1 + 0.4 sin 2πx` for eq. 75, whose layer [0, 0.5] has both edges at
contrast `1 | 0.1` since the sinusoid vanishes at 0 and 0.5). The tails
round to the pieces in double precision beyond about 19δ from the centre
(`1 − tanh z = 2e^{−2z}` to leading order, below 2.2e-16 at z = 18.4).
Both edges of a layer are two such transitions; a layer thinner than the
stencil is the double-cross of §1.3.

**The three regimes.** For δ ≫ h a standard FD4 stencil sees a smooth
coefficient and is fourth order. For δ ≪ h it samples α on the two sides
of what is, to it, a jump, and treats the coefficient as smooth between
nodes: the first-order failure of Fig. 4-7. The dissertation's translated
basis (E1.2) is exact for δ = 0 and, as §1.7 shows, off at first order in
δ for a smooth edge. Between the two, at δ ≈ h and below, nothing standard
is fourth order, and that is the case the seeds are built for.

**Baselines, as named throughout.** *Naive* is `Dx A Dx`, dissertation
eq. 76 with `A = diag α(x_j)` sampled at the nodes (plan D3;
`operators.naive_operator`); *jump-aware* or *the δ = 0 construction* is
E1.2's operator with the translated basis of eq. 74 built from the pieces'
Taylor coefficients at `x_c` (`operators.jump_aware_operator`); *seeds*
is what E3.4 (#29) builds, below.

### 1.2 The chain: seeds as the profiles of solutions polynomial in time

Look for solutions polynomial in time, `u(x, t) = Σ_{j=0}^{J} tʲ g_j(x)`.
Matching powers of t in `u_t = L u` gives

    (j + 1) g_{j+1} = L g_j,     L g_J = 0,

so `g_J ∈ ker L` and the t = 0 profile satisfies `L^{J+1} g₀ = 0`. The
heat equation is first order in time, so one power of L is spent per power
of t (the wave equation spent two: `(j+2)(j+1) g_{j+2} = L g_j`), but the
spaces the profiles fill are the same nested kernels `ker Lᵐ` as in the
wave case, and with constant α they are exactly the polynomials: the
time-polynomial solutions of `u_t = α u_xx` are the classical heat
polynomials of Rosenbloom & Widder (1959), `v_n(x, t) = Σ_k n!/(k! (n−2k)!)
x^{n−2k} (α t)^k`, whose t = 0 profiles are the monomials. The seeds are
the t = 0 profiles of the variable-coefficient heat polynomials, anchored
at the stencil's evaluation point.

**Definition (seeds at x_e).** With `α_e = α(x_e)`,

    φ₀ = 1;
    α φ₁′ = α_e,   φ₁(x_e) = 0;
    L φ_k = k (k − 1) α_e φ_{k−2},   φ_k(x_e) = φ_k′(x_e) = 0   (k ≥ 2).

Constant α gives `φ_k = (x − x_e)ᵏ` by induction (`L (x − x_e)ᵏ =
α k (k − 1) (x − x_e)^{k−2}` and the anchor conditions fix the two
constants). This is the companion's u-field chain with ρ = 1 and K = α,
so `c_e² = α_e`, in a single field (plan §3.1).

Read one seed at a time, each is the profile of one time-polynomial
solution and says what one monomial becomes through the edge:

- **φ₀**: `u ≡ C`.
- **φ₁, the x-like seed**: `L φ₁ = 0`, the equilibrium of constant flux,
  `φ₁(x) = α_e ∫_{x_e}^{x} dξ/α(ξ)`. It is the flux `α u_x` that stays
  smooth through the edge, not the slope; over one cell this seed is the
  harmonic-mean conductance of the finite-volume literature (§1.8).
- **φ₂, the x²-like seed**: `u = φ₂ + 2 α_e t` solves the heat equation
  with `u_t ≡ 2 α_e`: the profile on which the temperature's rate is
  constant. The constant matters: `u_t ≡ 0` would only return `ker L`.
- **φ₃**: `u = φ₃ + 6 α_e t φ₁`, with `u_t = 6 α_e φ₁ ∈ ker L`: the flux
  of `u_t` is constant, `α ∂ₓ u_t = 6 α_e²`.
- **φ₄**: `u = φ₄ + 12 α_e t φ₂ + 12 α_e² t²`, with `u_tt ≡ 24 α_e²`.

So the even seeds are the conditions `∂ₜᵏu = const` (k = 0, 1, 2) and the
odd seeds the conditions `α ∂ₓ ∂ₜᵏu = const` (k = 0, 1): in that order
they are the five continuity conditions `(u, 0), (flux, 0), (u, 1),
(flux, 1), (u, 2)` that `interface.continuity_conditions(4)` interleaves
for the degree-4 continuity matrices (dissertation eq. 71, EABE eq. 23).
The interleaving of the wave case is the same, but there it alternated
between two fields; here both conditions are on the one field u, so there
is no even/odd field swap to route and one chain serves every stencil. A
six-point stencil adds `φ₅ ↔ (flux, 2)`.

**The stencil.** The weights approximate `L u` at `x_e` from `u` at the
nodes `x_1 < … < x_p` (p = 5 for FD4) by reproducing the operator on the
seeds, the same moment conditions E1.2's `stencil_weights` imposes on the
translated basis:

    Σ_i w_i φ_k(x_i) = (L φ_k)(x_e) = 2 α_e δ_{k,2},    k = 0 … p − 1,

since `L φ_k = k (k − 1) α_e φ_{k−2}` vanishes at the anchor for every k
but 2. Only the span of the seeds enters, and §1.4 shows the spans are
canonical, so a different anchor or normalisation changes nothing in the
weights.

### 1.3 Numerics of the chain: the first-order form (for E3.4)

The chain is integrated as a first-order system that never differentiates
α. The state is `(φ_k, ψ_k)` with `ψ_k = α φ_k′` the flux of the seed:

    φ_k′ = ψ_k / α,     ψ_k′ = k (k − 1) α_e φ_{k−2},

with initial data `φ₀(x_e) = 1`, `ψ₁(x_e) = α_e` and zero otherwise. All
p seeds march together (2p states), from `x_e` outward to each side of the
stencil, since `φ_k` only needs `φ_{k−2}`. A jump is a segment boundary
across which `(φ, ψ)` is continuous, which is the weak reading of §1.4, so
the same march covers δ = 0 (E3.4 may instead dispatch δ = 0 to E1.2's
algebra and check the two agree to rounding on constant pieces). A stencil
that straddles both edges of a thin layer marches through both: the
diffusion twin of E1.2's translate-twice (`test_a_window_that_sees_both_
interfaces_of_a_thin_layer_translates_twice`).

**Stencil coordinate.** In `ξ = (x − x_e)/h_s` the chain is invariant:
`φ̃_k(ξ) = φ_k(x)/h_sᵏ` satisfies the same chain with `α(x_e + h_s ξ)`, so
`φ̃_k ≈ ξᵏ` and the interpolation matrix `A_{ki} = φ̃_k(ξ_i)` is a small
perturbed Vandermonde matrix; the weights are `w = w̃/h_s²` with
`A w̃ = 2 α_e e₂`. The constant-α condition numbers are 23.5 for the
half-width normalisation `ξ_i = 0, ±½, ±1` (the companion's choice) and
42.5 for E1.2's units of h, `ξ_i = 0, ±1, ±2`; marching in physical x at
h = 0.01 instead gives 8e7 (a scratch check, 2026-09-21; E3.4 confirms),
so the march is done in ξ. Prediction P5 (§1.9): `cond A` stays within a small factor of
the constant-α value at every δ (the companion saw 20 to 60 for δ from
1e-5 to 1).

**Segments and tolerances**, as in the companion: boundaries at every edge
centre and at ±10δ from it, so the adaptive step never has to find the
edge inside a long step over flat material; an explicit order-8 Runge–Kutta
march (`DOP853`) at rtol 1e-13 and atol 1e-15. Which rows: every row one
of whose window nodes lies within 19δ of an edge centre, so the seeded
region reaches 19δ + 2h from each centre; for δ ≳ h every row near the
layer is rebuilt and for δ ≳ 0.05 nearly every row in the domain, which
is harmless (§1.4's last paragraph). Cost in the companion's
implementation: 5 to 15 ms per stencil, seconds per operator; the same is
expected here.

### 1.4 Nested kernels, and the jump as the δ → 0 limit

Since L sends each seed two steps down the chain,
`span{φ₀, …, φ_k}` is the (k + 1)-th space of

    {1} ⊂ ker L ⊂ {L g = const} ⊂ ker L² ⊂ {L² g = const} ⊂ ker L³ ⊂ ⋯

and the spaces, not the basis, determine the weights.

**The jump.** For δ = 0 read L in the weak sense, `φ` and `α φ′`
continuous across `x_c`. Then `ker L` is spanned by 1 and a piecewise
linear function whose slopes are in the ratio `α⁺/α⁻`: the translated x of
dissertation eq. 74 (the 0.5 in its second row is `α_R/α_L` for the
`1 | 0.5` example). `L φ₂ = 2 α_e` is `u_t` continuous with its flux, and
so on up the chain, each `∂ₜᵏu = const` or `α ∂ₓ ∂ₜᵏu = const` being one
of the continuity conditions the translated basis was built from. For
constant pieces the chain for a jump *is* the translated basis of eq. 74:
one construction with two implementations, algebra for a jump
(`interface.translated_basis`; `test_translated_basis_keeps_u_flux_and_
their_time_derivatives_continuous` is its check) and an ODE march for a
smooth edge.

**Smoothly varying pieces are not quite the same limit.** E1.2's anchor
side carries the plain monomials `(x − x_c)ᵏ` and applies the operator
truncated at degree 4 (`interface.coefficient_operator`), while the seed
`φ₂` solves `(α φ₂′)′ = 2 α_e` exactly: `φ₂ = (x − x_e)² − ⅔ (α_e′/α_e)
(x − x_e)³ + ⋯`. Within a stencil the two five-dimensional spaces differ
at relative `O(h α′/α)`, first order in h with a fixed constant, the same
"different space, both fourth order on solutions" as the companion's
resolved-edge remark (its weights differed from Fornberg's by 17 % at
δ = 8h). So E3.4's δ → 0 test compares with E1.2's weights on the MATLAB
medium, where the limit is exact, and on eq. 75 records the floor at which
the difference saturates instead of claiming a limit.

**Rate of approach: first order in δ/h.** A scratch check of the chain
above (2026-09-21, not committed; E3.4 reproduces it in `tests/`): for the
`1/9 | 1` jump half a cell right of the evaluation node and the tanh edge
of §1.1, `max|w_seed − w_jump| / max|w_jump|` is 84 %, 47 %, 11 %,
1.1 % and 0.11 % at δ/h = 1, 0.5, 0.1, 0.01 and 0.001, and constant α
returns Fornberg's second-derivative weights times α to 2e-15. The
companion measured 22 %, 10 %, 0.98 %, 0.098 % at δ/h = 1, 0.1, 0.01,
0.001 for contrast 4; the constant grows with the contrast, the order does
not. Two consequences carried over: an edge below about `1e-5 h` is better
served by the jump weights directly, and for δ of the order of the stencil
or wider every row is seeded and the standard weights would have done as
well, so seeding a resolved edge costs accuracy nothing.

### 1.5 Well-posedness: the seeds form an extended complete Chebyshev system

**Proposition.** Let α be positive and continuous on an interval I
containing `x_e`, or positive and piecewise continuous on I with finitely
many jumps, L read in the weak sense. Then for every k the seeds
`φ₀, …, φ_k` form an extended complete Chebyshev (ECT) system on I:
interpolation in their span at any k + 1 distinct points of I is uniquely
solvable. Hence the stencil solve `A w̃ = 2 α_e e₂` is nonsingular for
every δ ≥ 0 and every node set, not only the ones tried.

*Proof sketch* (the companion's, with ρ = 1). Each space of the chain is
the kernel of an operator in Pólya form, a product of `∂ₓ` and
multiplications by positive functions:

    Lᵐ    = ∂ₓ α ∂ₓ ∂ₓ α ∂ₓ ⋯ ∂ₓ α ∂ₓ      (2m derivatives, weights 1 and 1/α alternating),
    ∂ₓ Lᵐ = ∂ₓ ∂ₓ α ∂ₓ ⋯ ∂ₓ α ∂ₓ           (2m + 1 derivatives),

with `ker Lᵐ` the even spaces and `{Lᵐ g = const} = ker ∂ₓ Lᵐ` the odd
ones. The kernel of such an operator has the nested basis `w₀, w₀ ∫ w₁,
w₀ ∫ w₁ ∫ w₂, …` in its weights, here `1, ∫ 1/α, ∫ (1/α) ∫ 1, ∫ (1/α) ∫ 1
∫ (1/α), …`, and is an ECT system on any interval where the weights are
positive (Pólya 1922; the structure theorem in Karlin & Studden 1966; the
disconjugacy side in Coppel 1971); in particular it has the Haar property.
The seeds span these kernels: their number matches the order, and they
are independent because their leading behaviour at `x_e` is triangular,
`φ_k = c_k (x − x_e)ᵏ (1 + o(1))` with `c_k ≠ 0`, by integrating
`(α φ_k′)′ = k (k − 1) α_e φ_{k−2}` twice from the anchor; only the
first-order system is used. For the weak reading the nested integrals are
unaffected by finitely many jumps of the weights, and the argument is the
same.

Two remarks specific to this repo. When the evaluation node sits *on* a
jump, which E1.2's rows at an interface node do (rows 50 and 75 of the
101-node eq. 75 grid), `α_e` is the owner's value (`region_index` puts a
point on a jump to its right, `PiecewiseAlpha.at_interface` names the
owner) and `ψ₁(x_e) = α_e` gives `φ₁` slope 1 on the owner's side and
`α_e/α_other` on the other; the leading-behaviour argument runs on the
owner's side and nothing else changes. And the Haar property of the
nested-integral system survives jumps in the weights for Lagrange
interpolation at distinct points (the Rolle argument between consecutive
zeros only needs the derivative to exist almost everywhere with the sign
of the next weight), which is why the confluent ("extended") part of ECT
is the only part the jump costs and the stencil solves are never
threatened.

The seeds are close relatives of two classical objects, cited and not
claimed: splines whose pieces lie in `ker L` are the L-splines of Schultz
& Varga (1967), and the harmonic-mean coefficient `h / ∫ dx/α` of the
conservative difference schemes of Tikhonov & Samarskii (1962) is the
two-point version of `φ₁` (§1.8). The literature ledger (E5.2, #43)
places the construction; nothing here claims novelty.

### 1.6 Truncation error: third order locally on the seeded rows, fourth globally

Interpolation in an ECT system has a Newton form whose remainder carries
the operator that annihilates the space in place of the (k + 1)-th
derivative (the generalised divided differences of Mühlbach 1973). For
the five-point stencil the annihilator of `span{φ₀, …, φ₄} = {L² g =
const}` is `M = ∂ₓ L²`, five derivatives, so on a function g the stencil's
local truncation error is

    τ(g) = Σ_i w_i g(x_i) − (L g)(x_e) = c₅ h³ (M g) + c₆ h⁴ (⋯) + ⋯,

with constants the remainder formula expresses through the weights `1` and
`1/α` inside the stencil. **This is one order lower than the companion's
`O(h⁴ ∂ₓ L² g)`**, and the ticket's wording carried that over: the wave
case differentiated once on five points, which is fourth order for any
node placement, while a second derivative on five points is generically
third order and reaches fourth only when the `h³` term cancels by
symmetry, as it does for the centred constant-α stencil (`Σ_i w_i ξ_i⁵ =
0`). A jump or a sub-grid edge breaks the symmetry. Checked on E1.2's own
rows: with the degree-4 weights and the sixth function of the degree-6
translated basis as g (in `ker L³`, outside the reproduced space), the
`h³` moment `Σ_i w_i g(ξ_i) − (L g)(0)` is 0.32, 0.084 and 172 for the
`1/9 | 1` jump at ½, 3/2 and 0 cells from the evaluation node, and
zero to rounding for equal pieces, where the `h⁴` moment on the seventh
function is −8, FD4's `−h⁴ u⁽⁶⁾/90`
(`tests/heat1d/test_interface.py::test_jump_rows_have_an_h3_moment_and_
equal_pieces_do_not`).

The global order is still four. A third-order local error confined to O(1)
rows in 1-D, or to a strip of O(1/h) rows along an interface in 2-D, costs
one order less in the solution than the same error everywhere (the
discrete Green's function is bounded and the strip has measure O(h)):
`O(h⁴)`, which is what E1 measured on the jump (rates 3.85–4.19 elliptic
and parabolic, port notes §1.3–1.4) and what the 2016 2-D results show.
So the prediction reads: seeded rows `O(h³ ∂ₓ L² u)` locally, plain rows
`O(h⁴ u⁽⁶⁾)`, solution `O(h⁴)`. Its two readings are what the plan (§3.2)
built the epics around:

- **Equilibrium: the seed rows are exact.** `L u = 0` gives `M u = 0`,
  and more than that, `u = A + B φ₁ ∈ span{φ₀, φ₁}` in every seeded
  window, so the moment conditions reproduce `L u` exactly there at any δ
  and any h. Where the pieces are constants (the MATLAB medium) every
  plain row sees a linear function and is exact too, and the discrete
  equilibrium solution is the exact one to solver precision. On eq. 75 the
  plain FD4 rows inside the layer keep their `O(h⁴ u⁽⁶⁾)` smooth-α error,
  which does not depend on δ: the elliptic seed line is the smooth
  problem's fourth-order line, one line for every δ. E3.4's acceptance
  "exact to 1e-12 at every δ" is therefore a statement about constant
  pieces (or a seed-every-row mode), and §1.8 says why no 1-D equilibrium
  experiment can rank the exact methods.
- **Parabolic: the driving derivative is bounded independently of δ.**
  `u_t = L u` gives `L² u = u_tt` and `M u = ∂ₓ u_tt`, the x-derivative of
  another solution of the heat equation (`u_tt` is continuous with a
  continuous flux, like u), which is bounded independently of δ: it
  changes by the contrast ratio across the edge whatever the width. FD4's
  `u⁽⁵⁾` is `∼ δ⁻⁴` inside the edge, and the naive `Dx A Dx` rows that see
  an unresolved edge miss an O(1) fraction of the slope jump, O(1/h) in
  `L u`. Whether the seeded rows' constant `c₅` is in fact independent of
  δ/h is not a consequence of the proposition (the Peano kernel involves
  `1/α` across the stencil); it is the prediction E3.4 tests, P7 below.

Diffusion smooths, so for t > 0 every `∂ₜᵏu` is itself a solution and the
second reading holds for the whole chain; and across an edge of width δ
the flux changes only by `∫_edge u_t dx = O(δ)`, so in diffusion a
sub-grid edge is close to a jump at all times. That is why the δ = 0
construction is a serious baseline here (§1.7) and why the seeds' margin
over it is set by δ, not by a propagating O(1) signal as in the wave case.

### 1.7 The knee, and the δ = 0 construction on a smooth edge (predictions for E3.3)

**The naive knee.** With `h` the node spacing and `δ` the edge width,
naive FD4 converges at about first order while `h ≳ δ`, as it does at the
jump (E1: rates 1.00–1.35), and at fourth order once `h ≲ δ`, with a knee
at `h ≈ δ`; the companion saw a factor 200 between `h = 2δ` and `h = δ`.
The pre-knee rates are not cleanly first order because the nodes sample
the tanh at grid-dependent positions, so the effective jump moves with n.
Elliptic and parabolic lines both show the knee: the degeneracy of §1.8 is
a property of the exact schemes, not of the naive one.

**The δ = 0 construction on a smooth edge**, E1.2's operator applied at
the edge centre with the pieces' Taylor data as if δ were 0, is off in two
measurable ways. Its weights differ from the seeds' at first order in δ/h
(§1.4's 84 % … 0.11 %). Its solution converges at fourth order to the
*jump* medium's solution, not to the true one, so against the true-δ
reference its error saturates at the difference of the two problems'
solutions, which is first order in δ: with `F_δ(x) = ∫_{−1}^{x} dξ/α_δ`,

    F_δ(x) − F₀(x) → −c δ  for x past the edge,
    c = ∫_ℝ [ 1/α₀(z) − 1/(α⁻ + (α⁺ − α⁻) s(z)) ] dz,

a quadrature of the closed-form integrand: c = 8.79 for the 9 : 1 contrast
of the MATLAB medium and 10.36 for the 10 : 1 contrast of eq. 75's edges,
in either direction (2026-09-21, scratch; E3.3 found the closed form
`c = (a − b) ln(a/b) / (2ab)`, `exact.edge_resistance_deficit`, §2.2). On
the MATLAB medium `F₀(1) = 10`, so the flux and the solution shift by about
0.9 δ relative, 2e-3 at δ = 0.0025: comparable to the naive error on the
coarsest grids and five orders above the seeds'. E3.3 tabulates this floor
exactly from the two quadrature references, elliptic; the parabolic floor
is O(δ) with a time-dependent constant and is measured, not predicted.
(§2.2 qualifies the saturation: the construction sits on the floor only
while `h ≳ 2δ`; once the grid resolves the edge its error grows to an
O(1) constant instead, since the rebuilt rows keep enforcing a kink.)

**Seeds.** Fourth order at every δ with a constant that does not depend on
δ; the δ = 0.0025 and 0.01 parabolic lines should agree with the δ = 0
jump-aware line to three digits at every n, as the companion's did, and
the elliptic lines are exact on the MATLAB medium.

### 1.8 The 1-D elliptic degeneracy and its finite-volume twin (plan §3.2)

Three constructions are exact on the 1-D equilibrium problem at any h and
any δ, for a piecewise-constant α:

1. **The seed rows** (§1.6, first reading), and hence the seed operator
   on constant pieces.
2. **The translated basis** of E1.2 (`test_a_jump_between_constants_is_
   solved_exactly`, `test_a_two_cell_layer_between_constants_is_solved_
   exactly`), which is the seeds at δ = 0.
3. **The conservative three-point scheme with exact face conductances**
   `a_{i+½} = h / ∫_{x_i}^{x_{i+1}} dξ/α`: its discrete flux
   `a_{i+½} (u_{i+1} − u_i)/h` equals the exact flux B at the exact nodal
   values, so the exact solution satisfies every discrete equation to
   rounding. This is Tikhonov & Samarskii's homogeneous scheme (1962) and,
   for piecewise-constant α, Patankar's interface conductivity, the
   harmonic mean of the two cell values (his 1980 book, ch. 4). It is
   exact at δ = 0 and at every δ > 0 alike, since the face integral is
   exact for any profile.

The 1-D elliptic problem therefore cannot rank the methods (plan R2): it
is the sanity check and the limit case, one figure and a remark, and the
headline 1-D figure is parabolic, where the FV scheme is second order and
the seeds fourth.

**A correction for E3.5 (#30).** The plan's T1, "the cell harmonic mean
of α over one and two cells, a material the naive operator samples", is
not item 3. `Dx A Dx` applied to a kinked solution is not exact for any
nodal α, because `Dx` of the kink is already O(1) off at the nodes beside
it: on the MATLAB jump sitting mid-cell at 100 nodes, the residual
`max |L_h u_exact|` over the interior rows is 5.6 with the one-cell
harmonic mean at the nodes and 1.2 with the two-cell mean, where the
jump-aware operator gives rounding
(`tests/heat1d/test_operators.py::test_no_nodal_alpha_makes_dx_a_dx_
exact_on_a_kinked_equilibrium` pins both at 0.5 and 1e-9). Any nonzero
residual disproves exactness; nothing below depends on the two values or
their order. So #30's acceptance line "T1 is exact on the 1-D equilibrium
problem with a piecewise-constant α" holds only for the conservative
form. The recommendation: build both, T1 as the changed medium under
`Dx A Dx` (the like-for-like comparator, expected first order at the edge
like every nodal treatment) and T1-FV, the three-point conservative
scheme with exact face conductances (item 3: second order at every δ,
exact at equilibrium by construction), which is the strongest low-order
comparator on that ground and the finite-volume twin the manuscript
should show. Brad decides; #30's text is amended either way (breadcrumb
posted with this ticket). E3.5 built both; §2.4 has the tables.

### 1.9 What the later sections check

The predictions above, numbered so §2 can tick them off:

- **P1 (E3.2)** δ = 0 reproduces the jump medium bit for bit; the tanh
  tails reach the pieces' values at about 19δ; the parabolic reference
  converges to 1e-10 between two Chebyshev resolutions at every δ.
- **P2 (E3.3)** Naive `Dx A Dx`: rate ≈ 1 for `h ≳ δ`, a knee at
  `h ≈ δ`, fourth order below, elliptic and parabolic (§1.7).
- **P3 (E3.3)** The δ = 0 construction: weights first order in δ/h from
  the seeds'; elliptic error saturating at the floor `c δ` of §1.7 with
  c = 8.79 (9 : 1) and 10.36 (10 : 1), reproduced exactly from the two
  quadrature references; parabolic error O(δ), measured. (§2.2: the
  saturation holds for `h ≳ 2δ` only.)
- **P4 (E3.4)** Constant α gives Fornberg's weights times α to rounding;
  on constant pieces the seed weights reach E1.2's at first order in δ/h
  (84 %, 47 %, 11 %, 1.1 %, 0.11 % at δ/h = 1, ½, 0.1, 0.01, 0.001 for
  `1/9 | 1` half a cell off the node); on eq. 75's pieces the difference
  saturates at an `O(h α′/α)` floor to be recorded (§1.4).
- **P5 (E3.4)** `cond A` in the stencil coordinate within a small factor
  of the constant-α Vandermonde value (23.5 half-width, 42.5 units of h)
  at every δ from 1e-5 to 1 (§1.3).
- **P6 (E3.4)** Equilibrium exact to solver precision at every δ on the
  MATLAB medium; on eq. 75 one δ-independent fourth-order line equal to
  the smooth problem's (§1.6).
- **P7 (E3.4)** Parabolic error fourth order at every δ with a
  δ-independent constant, the δ = 0.0025 and 0.01 lines on the δ = 0
  jump-aware line to three digits (§1.7); local truncation on the seeded
  rows `O(h³)` with a δ-independent constant, plain rows `O(h⁴)`.
- **P8 (E3.4)** The double-cross: a layer thinner than the stencil
  (both tanh edges inside one window) handled by one march, its δ → 0
  limit E1.2's translate-twice weights at first order in δ/h.
- **P9 (E3.4)** The seed operator's spectrum is real and negative like
  the jump-aware one (port notes §1.5: extreme `−16/3 h⁻²` in the α = 1
  material), damped by BD4 at dt = h; any complex pairs come from the
  one-sided end rows, not the edge.
- **P10 (E3.5)** T1 under `Dx A Dx` is first order at the edge and not
  exact at equilibrium; T1-FV is exact at equilibrium and second order in
  the parabolic problem at every δ; T0 (widening to `max(δ, mh)`) trades
  the knee for the §1.7 floor with `mh` in place of δ, first order in h
  with the constant `c m`; T2 (arithmetic mean) is the worst of them,
  being the "regularise α itself" case. The parabolic ranking predicted:
  seeds < T1-FV < T1 ≈ T0 < T2 ≈ naive in error at every δ below h.

### 1.10 References for §1

Verified entries are those of the companion's `paper/references.bib`
(fetched 2026-09-20) plus two Crossref records fetched 2026-09-21; the
E5.2 ledger re-verifies before any enters `paper/references.bib`.

- Martin, B. P. (2016), *Application of RBF-FD to Wave and Heat Transport
  Problems in Domains with Interfaces*, PhD thesis, CU Boulder; ch. 4.
- Martin, B. & Fornberg, B. (2017), Eng. Anal. Bound. Elem. 79, 38–48,
  doi:10.1016/j.enganabound.2017.03.005.
- Martin, B. P. (2026), *Seed stencils: high-order finite differences and
  RBF-FD through material edges too steep for the grid*, arXiv preprint
  (packaged 2026-09-20; identifier to be filled in when announced).
- Rosenbloom, P. C. & Widder, D. V. (1959), Expansions in terms of heat
  polynomials and associated functions, Trans. Amer. Math. Soc. 92,
  220–266, doi:10.1090/s0002-9947-1959-0107118-2 (Crossref, 2026-09-21).
- Pólya, G. (1922), Trans. Amer. Math. Soc. 24, 312–324,
  doi:10.1090/S0002-9947-1922-1501228-5.
- Karlin, S. & Studden, W. J. (1966), *Tchebycheff Systems*, Interscience.
- Coppel, W. A. (1971), *Disconjugacy*, LNM, Springer, doi:10.1007/BFb0058618.
- Mühlbach, G. (1973), J. Approx. Theory 9, 165–172,
  doi:10.1016/0021-9045(73)90104-4.
- Schultz, M. H. & Varga, R. S. (1967), L-splines, Numer. Math. 10,
  345–369, doi:10.1007/BF02162033.
- Tikhonov, A. N. & Samarskii, A. A. (1962), Homogeneous difference
  schemes, USSR Comput. Math. Math. Phys. 1, 5–67,
  doi:10.1016/0041-5553(62)90005-8.
- Patankar, S. V. (1980), *Numerical Heat Transfer and Fluid Flow*,
  Hemisphere; the Crossref record found is the CRC Press 2018 edition,
  doi:10.1201/9781482234213, ch. 4 "Heat Conduction" (the interface
  conductivity). Section number and page to be pinned in E5.2.

## 2. Results in 1-D (E3.2–E3.6)

### 2.1 The smooth medium and the references at any δ (E3.2, #27)

`heat1d.domain.SmoothEdges(jump, delta)` is §1.1's medium: the pieces of a
`PiecewiseAlpha` blended across each interface with `s = ½ [1 + tanh((x −
x_c)/δ)]`, the edges folded in from the left so the pieces' weights are a
partition of unity even for a layer thinner than δ. `alpha_x` is analytic
(`s′ = 2 s (1 − s)/δ`). On both study media alpha stays positive at every
δ (eq. 75's minimum is 0.12 at δ = 0.0025, 0.28 at 0.04; the sinusoid piece
is negative on part of (−0.5, 0) but its weight there is below 0.12).
Decisions, each pinned by a test in `tests/heat1d/test_domain.py`:

- **δ = 0 is the jump bit for bit** in `alpha`, `alpha_x`, `taylor`,
  `elements` and `interfaces` (delegation, not a limit).
- **The tails.** The blend is computed from the near piece on each side, so
  beyond `TANH_REACH` = 20 δ from a centre it *is* the piece to the bit for
  values within a factor ten of each other; at the 19δ of §1.1 it is
  within two ulps. `alpha_x` there is the true derivative of the blend,
  3e-15 at 20δ for δ = 0.0025, not zero.
- **`taylor` returns the pieces' data**, not the smooth alpha's expansion
  (whose k-th coefficient is O(δ⁻ᵏ) and would serve no stencil), and
  `interfaces` are the edge centres: `jump_aware_operator(grid,
  SmoothEdges(m, δ))` is therefore §1.7's δ = 0 construction on a smooth
  edge with no further code, which is what E3.3 (#28) measures.
- **Elements.** Every medium reports `elements()`, the intervals on which
  alpha is smooth up to their ends, and both references cut on them: the
  pieces for a jump; for a smooth edge the cuts `x_c ± m δ`, `m ∈
  EDGE_CUTS = (1, 3, 9, 27)`, with cuts closer than δ/2 to a kept one
  merged (the two edges of eq. 75 at δ = 0.04 share cuts) and the medium
  itself as every element's piece. Each element spans at most a factor
  three in `z = (x − x_c)/δ` away from `[−δ, δ]`, which keeps the
  transition's pole at `z = iπ/2` and, for contrasts of order ten, the
  complex zero of alpha at Bernstein-ellipse parameter ρ ≳ 3.4 from every
  element; past 27δ the tails are below 1e-23.

**The equilibrium reference** is the quadrature of E1.1 on those cuts. For
a tanh edge between constants `a | b` the integral has the closed form
`F = δ [z/a + (a − b)/(2ab) ln(a + b e^{2z})]` + const, and the 24-point
Gauss panels reproduce it to 1e-14 at δ from 0.1 to 1e-6 for the `1/9 | 1`
contrast and to 1e-13 at contrast 100 (`test_quadrature_on_a_tanh_edge_
matches_the_closed_form`). The first-order approach to the jump of §1.7
shows in it: `F₀(1) − F_δ(1)` is positive and scales exactly with δ
(ratios 10.000 between δ = 0.01, 0.001, 0.0001); E3.3 records the constant.

**The parabolic reference** is E1.3's Chebyshev collocation generalised
from pieces to elements (`ChebyshevPieces` on `elements()`, `u` and
`α u_x` matched at every element edge, which on a smooth edge's cuts is C¹
continuity), integrated by Radau, kept as `ParabolicReference` (element
edges, nodal values, JSON metadata; `evaluate` interpolates to any grid)
and cached under `outputs/` by `parabolic_reference(..., cache=)`, reused
when medium (its `repr`), problem label, `t_end`, resolution and tolerances
match. Two decisions, 2026-09-21:

- **Radau's tolerances are (rtol, atol) = (1e-9, 1e-11) (`RADAU_RTOL`,
  `RADAU_ATOL`), not E1.3's (1e-12, 1e-13).** The linear system's Newton
  iteration converges in one step and the next update is the round-off of
  the solve, about `eps · h ‖A‖` in absolute terms; Radau measures it
  against `atol + rtol |y|`, counts the iteration as failed where it
  exceeds that, and cuts the step, so it is the absolute tolerance that
  has to sit above the floor. On the `1/9 | 1` edge at δ = 0.0025
  (unsplit elements; ‖A‖ up to 6e10 on the δ-wide ones), with 24 and 32
  nodes per element:

  | rtol, atol | steps, 24 nodes | steps, 32 nodes | time |
  | --- | --- | --- | --- |
  | 1e-12, 1e-13 | fails after 12 s: "required step size is less than spacing between numbers" | 20,614 | 53 s |
  | 1e-10, 1e-12 | 3,291 | 13,127 | 6 s / 46 s |
  | 1e-10, 1e-11 | 519 | 709 | ≤ 1 s |
  | 1e-9, 1e-11 | 400 | 399 | 0.1–0.2 s |

  Every run agrees with the (1e-9, 1e-11) one to 1.4e-13 or better; at
  δ = 0 the (1e-12, 1e-13) run (1805 steps) and the (1e-9, 1e-11) run
  (436) agree to 1e-13; and E1.3's accuracy tests pass with margin at the
  new default (the decaying mode to 2e-13, the separable solution to
  2e-11, the t = 30 equilibrium to 1.4e-11). Port notes §1.4 now say so.
  (The first draft of this note blamed rtol and put the failure at 32
  nodes; the review caught it, and the table is the re-measurement.)
- **Resolution: 20 nodes per element, elements wider than 0.1 split
  (`ChebyshevPieces.build(max_width=)`), checked against 24 nodes and
  0.05.** The collocation system's round-off floor grows with the node
  count on the δ-wide elements (the Chebyshev equilibrium against the
  quadrature on the `1/9 | 1` edge at δ = 0.0025: 8e-12, 7e-12, 1e-11
  at 24, 32, 48 unsplit nodes; on eq. 75: 5e-11, 1e-10, 2e-10, and
  the 48-node Radau march there takes 2529 steps and 76 s) while the
  truncation error falls with it on the wide pieces (32 unsplit nodes leave
  eq. 75's half-unit sinusoid layer at 6e-9, 48 at 5e-13). Fewer nodes on
  narrower elements serve both: 20 nodes resolve a tanh element to 1e-11
  (ρ ≥ 3.4) and a 0.1-wide piece of the sinusoid to 1e-12, 16 do not
  (1e-9).

`scripts/heat1d_stiff.py` (20 s, both media at the four δ of E3.3, the
check resolutions cached too) reports, per medium and δ, the reference's
elements, interior unknowns, Radau steps and seconds, the check's seconds,
the max difference between the two resolutions on 2001 points
(*agreement*) and the Chebyshev equilibrium's max difference from the
quadrature at the reference's resolution (*elliptic*). The element,
unknown and step counts reproduce exactly from run to run; the two floor
columns are round-off and move by 10–30 % between runs of the same code
on the same machine (threaded BLAS reductions), so they are quoted to one
figure:

| medium | δ | elements | unknowns | steps | s (ref / check) | agreement | elliptic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `1/9 \| 1` | 0 | 20 | 380 | 436 | 0.25 / 1.24 | 8e-13 | 5e-13 |
| | 0.04 | 23 | 437 | 433 | 0.31 / 1.47 | 1e-12 | 8e-13 |
| | 0.01 | 25 | 475 | 429 | 0.35 / 1.87 | 5e-12 | 6e-12 |
| | 0.0025 | 27 | 513 | 427 | 0.46 / 2.13 | 2e-12 | 1e-12 |
| eq. 75 | 0 | 20 | 380 | 473 | 0.27 / 1.35 | 2e-11 | 9e-12 |
| | 0.04 | 27 | 513 | 473 | 0.50 / 2.23 | 3e-11 | 3e-11 |
| | 0.01 | 30 | 570 | 470 | 0.51 / 2.77 | 5e-11 | 1e-11 |
| | 0.0025 | 33 | 627 | 468 | 0.63 / 3.26 | 6e-11 | 1e-11 |

**P1 holds**: δ = 0 is the jump bit for bit, the tails are the pieces to
the bit beyond 20δ (two ulps at 19δ), and the parabolic reference agrees
between two resolutions to about 6e-11 at worst (eq. 75 at δ = 0.0025;
5.5e-11 and 6.3e-11 in two runs), 2e-12 on the `1/9 | 1` medium, against
the ticket's 1e-10. The floor is a property of collocation on δ-wide
elements in double precision, not of the tolerance; eq. 75's is 5–10×
the constant-piece medium's. For the study
that is enough by four orders on eq. 75, where every method's error stays
above 1e-6 (the plain FD4 rows inside the layer, §1.6), and by two on the
`1/9 | 1` medium, where the jump-aware line reached 6e-12 at 800 nodes in
E1.3: the 800-node points of E3.4's seed line will sit on the reference's
floor and are read as such.

Tests (`tests/heat1d/test_exact.py`): the closed form above at five δ and
two contrasts; the δ = 0 medium passing the jump's closed form bit for bit
with the jump; the Chebyshev equilibrium on a smooth edge within 3e-11 of
the quadrature at δ = 0.04 and 0.0025; the two-resolution agreement below
1e-10 at δ = 0, 0.01, 0.0025; the cache round trip (files written, reused
with the recorded run time, rebuilt when `t_end` changes, a dot in the
file stem kept). `tests/test_heat1d_stiff.py` runs the driver at two δ and
checks that the second run reads every reference from the cache.

### 2.2 The naive knee, and the δ = 0 construction on a smooth edge (E3.3, #28)

![knee](figures/heat1d_stiff_knee.png)

`scripts/heat1d_stiff.py` (the knee study appended to the reference check;
45 s cold, 4 s with everything cached; the sweep to 6400 nodes behind
`--counts`, 3 min once, its parabolic errors kept in
`outputs/heat1d_stiff_knee.json`). Two operators on the smooth medium at
δ ∈ {0, 0.04, 0.01, 0.0025}: naive `Dx A Dx` (`naive_operator`, the α
sampled at the nodes) and the δ = 0 construction (`jump_aware_operator`,
E1.2's rows across the edge centre with the pieces' data, the direct rows
on the smooth α elsewhere; §1.7). Both problems of §2.1: the equilibrium
against the quadrature, the ramp problem at t = 2 (BD4, dt = h) against the
cached Chebyshev reference, errors `‖e‖₂/‖u‖₂` at the nodes. Node counts
double from 50 to 6400 with E1's placements, the MATLAB edge mid-cell
(even counts) and eq. 75's two edges on nodes (`4k + 1`); eq. 75 starts at
101 because at 49 nodes `h α′/α ≈ 1` where the sinusoid meets the layer's
edges and the jump-aware operator's spectrum crosses into the right
half-plane (largest real part 248; 8.8 at 53 nodes, −2.1 at 101, −2.06 from
201 to 801, none positive from 101 on; measured on the interior operator
with the Dirichlet rows removed, `interior_operator`, as BD4 steps it and
as port notes §1.5 plots it), which E1 never ran either. Each δ > 0 line is read against its **floor**, the two
exact solutions' difference at the nodes, `‖u₀ − u_δ‖₂/‖u_δ‖₂` (quadrature
against quadrature; reference against reference for the ramp), which is
what the δ = 0 construction converges to while the grid does not resolve
the edge. Rates are `log₂` of consecutive errors; `h = δ` falls at n = 51,
201 and 801 for the three δ.

**The naive knee (P2 holds).** Errors and rates, both media:

| n | δ = 0 (jump) | δ = 0.04 | δ = 0.01 | δ = 0.0025 |
| --- | --- | --- | --- | --- |
| *MATLAB medium, equilibrium* | | | | |
| 50 | 8.26e-3 | 8.58e-4 | 7.64e-3 | 8.68e-3 |
| 100 | 4.08e-3 (1.02) | 1.97e-5 (5.44) | 3.15e-4 (4.60) | 4.23e-3 (1.04) |
| 200 | 2.04e-3 (1.00) | 1.23e-6 (4.01) | 1.85e-4 (0.77) | 1.69e-3 (1.32) |
| 400 | 1.02e-3 (1.00) | 7.81e-8 (3.97) | 2.51e-6 (6.20) | 8.70e-5 (4.28) |
| 800 | 5.12e-4 (1.00) | 4.90e-9 (3.99) | 1.46e-7 (4.11) | 4.47e-5 (0.96) |
| 1600 | 2.56e-4 (1.00) | 3.07e-10 (4.00) | 9.34e-9 (3.97) | 3.90e-7 (6.84) |
| 3200 | 1.28e-4 (1.00) | 1.93e-11 (3.99) | 5.88e-10 (3.99) | 1.80e-8 (4.43) |
| 6400 | 6.42e-5 (1.00) | 1.50e-11 (0.36) | 1.05e-10 (2.49) | 1.16e-9 (3.96) |
| *MATLAB medium, ramp, t = 2* | | | | |
| 50 | 3.86e-3 | 4.50e-4 | 3.09e-3 | 3.85e-3 |
| 100 | 1.97e-3 (0.98) | 1.29e-5 (5.12) | 1.79e-4 (4.11) | 2.04e-3 (0.92) |
| 200 | 9.90e-4 (0.99) | 8.19e-7 (3.98) | 9.25e-5 (0.95) | 7.90e-4 (1.37) |
| 400 | 4.97e-4 (0.99) | 5.22e-8 (3.97) | 1.54e-6 (5.91) | 4.36e-5 (4.18) |
| 800 | 2.49e-4 (1.00) | 3.28e-9 (3.99) | 9.32e-8 (4.04) | 2.19e-5 (0.99) |
| 1600 | 1.25e-4 (1.00) | 2.05e-10 (4.00) | 5.96e-9 (3.97) | 2.20e-7 (6.64) |
| 3200 | 6.24e-5 (1.00) | 1.35e-11 (3.93) | 3.76e-10 (3.99) | 1.14e-8 (4.27) |
| 6400 | 3.12e-5 (1.00) | 1.64e-11 (−0.28) | 2.57e-11 (3.87) | 7.27e-10 (3.96) |
| *eq. 75 medium, equilibrium* | | | | |
| 101 | 2.06e-2 | 6.12e-5 | 5.83e-3 | 1.02e-2 |
| 201 | 8.05e-3 (1.35) | 3.26e-6 (4.23) | 1.38e-4 (5.40) | 3.44e-3 (1.56) |
| 401 | 3.98e-3 (1.01) | 2.12e-7 (3.95) | 7.38e-6 (4.22) | 1.06e-3 (1.70) |
| 801 | 1.98e-3 (1.01) | 1.34e-8 (3.98) | 4.77e-7 (3.95) | 2.85e-5 (5.22) |
| 1601 | 9.87e-4 (1.00) | 8.39e-10 (4.00) | 3.07e-8 (3.96) | 1.04e-6 (4.77) |
| 3201 | 4.92e-4 (1.00) | 5.30e-11 (3.98) | 1.94e-9 (3.99) | 6.08e-8 (4.10) |
| 6401 | 2.46e-4 (1.00) | 1.32e-11 (2.01) | 1.52e-10 (3.67) | 3.90e-9 (3.96) |
| *eq. 75 medium, ramp, t = 2* | | | | |
| 101 | 1.93e-2 | 6.06e-5 | 5.68e-3 | 9.61e-3 |
| 201 | 7.30e-3 (1.40) | 3.25e-6 (4.22) | 1.34e-4 (5.40) | 3.23e-3 (1.57) |
| 401 | 3.62e-3 (1.01) | 2.11e-7 (3.95) | 7.36e-6 (4.19) | 1.01e-3 (1.68) |
| 801 | 1.80e-3 (1.01) | 1.33e-8 (3.98) | 4.76e-7 (3.95) | 2.74e-5 (5.20) |
| 1601 | 8.96e-4 (1.00) | 8.36e-10 (3.99) | 3.06e-8 (3.96) | 1.03e-6 (4.73) |
| 3201 | 4.47e-4 (1.00) | 5.62e-11 (3.90) | 1.93e-9 (3.99) | 6.07e-8 (4.09) |
| 6401 | 2.24e-4 (1.00) | 4.54e-11 (0.31) | 2.02e-10 (3.26) | 3.90e-9 (3.96) |

While `h ≳ 4δ` the naive line is the jump's, at about first order (rates
0.9–1.7; the pre-knee rates are noisy because the nodes sample the tanh at
grid-dependent positions, as §1.7 said, and on the MATLAB medium the
δ = 0.01 line even shows a 4.6 followed by a 0.77). Across the knee the
error drops by two to three orders between `h = 2δ` and `h = δ/2` (MATLAB
δ = 0.0025: 8.70e-5 → 3.90e-7, a factor 220; eq. 75: 1.06e-3 → 1.04e-6, a
factor 1000; the companion saw 200 between `h = 2δ` and `h = δ`), the
single doubling `h = δ → δ/2` carrying most of it (rates 5.9–6.8 there).
Below `h ≈ δ/2` every line is fourth order (3.9–4.4) down to the direct
solve's round-off at 6400 nodes (1–5e-11, where the δ = 0.04 rates
collapse). The ramp problem repeats the equilibrium's numbers and rates
to two digits at every count: at t = 2 the ramp solution is close to
equilibrium, and the knee is a property of the operator, not of the
problem. On eq. 75 the knee sits at the same `h/δ`, one to two orders
higher in error, since the sinusoid piece is not resolved to rounding by
any of these grids.

**The δ = 0 construction (P3 holds for `h ≳ 2δ`, and fails below).** The
same runs, the floor quoted at the finest count (it moves by 3 % over the
sweep as the nodal norm converges):

| n | δ = 0.04, floor 2.92e-2 | δ = 0.01, floor 7.47e-3 | δ = 0.0025, floor 1.88e-3 |
| --- | --- | --- | --- |
| *MATLAB medium, equilibrium* | | | |
| 50 | 1.68e-2 | 7.35e-3 | 1.836e-3 |
| 100 | 6.85e-2 | 7.21e-3 | 1.861e-3 |
| 200 | 1.05e-1 | 1.23e-2 | 1.869e-3 |
| 400 | 1.18e-1 | 6.86e-2 | 1.743e-3 |
| 800 | 1.23e-1 | 1.00e-1 | 1.42e-2 |
| 1600 | 1.25e-1 | 1.12e-1 | 6.87e-2 |
| 3200 | 1.26e-1 | 1.16e-1 | 9.91e-2 |
| 6400 | 1.27e-1 | 1.18e-1 | 1.10e-1 |

| n | δ = 0.04, floor 1.14e-2 | δ = 0.01, floor 2.87e-3 | δ = 0.0025, floor 7.19e-4 |
| --- | --- | --- | --- |
| *MATLAB medium, ramp, t = 2* | | | |
| 50 | 8.80e-3 | 2.823e-3 | 6.970e-4 |
| 100 | 2.51e-2 | 2.782e-3 | 7.122e-4 |
| 200 | 3.91e-2 | 4.79e-3 | 7.158e-4 |
| 400 | 4.45e-2 | 2.34e-2 | 7.009e-4 |
| 800 | 4.67e-2 | 3.54e-2 | 4.65e-3 |
| 1600 | 4.77e-2 | 3.99e-2 | 2.30e-2 |
| 3200 | 4.81e-2 | 4.17e-2 | 3.45e-2 |
| 6400 | 4.84e-2 | 4.25e-2 | 3.88e-2 |

| n | δ = 0.04, floor 3.05e-2 / 2.71e-2 | δ = 0.01, floor 1.20e-2 / 1.06e-2 | δ = 0.0025, floor 3.59e-3 / 3.20e-3 |
| --- | --- | --- | --- |
| *eq. 75 medium, equilibrium / ramp* | | | |
| 101 | 1.08e-1 / 1.12e-1 | 1.62e-2 / 1.53e-2 | 7.39e-3 / 7.51e-3 |
| 201 | 1.68e-1 / 1.73e-1 | 3.89e-2 / 4.09e-2 | 3.79e-3 / 3.41e-3 |
| 401 | 1.99e-1 / 2.04e-1 | 1.21e-1 / 1.27e-1 | 4.60e-3 / 4.29e-3 |
| 801 | 2.13e-1 / 2.18e-1 | 1.70e-1 / 1.77e-1 | 4.24e-2 / 4.60e-2 |
| 1601 | 2.20e-1 / 2.24e-1 | 1.90e-1 / 1.97e-1 | 1.25e-1 / 1.32e-1 |
| 3201 | 2.23e-1 / 2.27e-1 | 1.99e-1 / 2.06e-1 | 1.67e-1 / 1.75e-1 |
| 6401 | 2.24e-1 / 2.29e-1 | 2.03e-1 / 2.09e-1 | 1.84e-1 / 1.92e-1 |

Two regimes, and the prediction covered only the first:

- **`h ≳ 4δ`: on the floor, to three digits.** MATLAB δ = 0.0025 at 50,
  100, 200 nodes: 1.836e-3, 1.861e-3, 1.869e-3 against floors 1.836e-3,
  1.861e-3, 1.869e-3; the ramp 6.970e-4, 7.122e-4, 7.158e-4 against
  6.983e-4, 7.122e-4, 7.158e-4; eq. 75 at 101 nodes 7.39e-3 against
  3.60e-3 (the 101-node grid also carries E1's own 4.3e-3 at δ = 0, so
  it is not yet on the floor), at 201 nodes 3.79e-3 against 3.60e-3. The
  construction converges to the *jump's* solution, and its error is the
  distance between the two problems, first order in δ: the elliptic floor
  is 0.73–0.75 δ on the MATLAB medium (2.92e-2, 7.47e-3, 1.88e-3), the
  ramp's at t = 2 is 0.28 δ (1.14e-2, 2.87e-3, 7.19e-4), the
  "time-dependent constant" of §1.7 measured at one time. On eq. 75 the
  floors are 0.76 δ, 1.20 δ, 1.44 δ (elliptic) and 0.68 δ, 1.06 δ, 1.28 δ
  (ramp): not proportional to δ at these widths, for the reason the next
  paragraph gives.
- **`h ≲ δ`: off the floor and growing, to an O(1) constant.** Once the
  plain rows resolve the edge (they are fourth-order rows on the smooth α,
  nothing wrong with them), the three or four rebuilt rows still impose
  the translated basis, a kink with slope ratio `α⁺/α⁻` at the centre,
  on a solution whose slope ratio across the centre cell tends to one.
  The discrete solution takes the kink, and the error saturates at a
  constant of the contrast and the problem: 0.127 (equilibrium) and 0.048
  (ramp) on the MATLAB medium, 0.22 and 0.23 on eq. 75, reached from
  below with rate −0.6, −0.2, −0.06, −0.02 per doubling. Between the two
  regimes the error dips slightly *below* the floor at `h ≈ 2δ` (MATLAB
  δ = 0.0025: 1.743e-3 against 1.871e-3; the partly resolved plain rows
  pull the solution part of the way to the truth) and is 7–10× the floor
  at `h = δ`. On constant pieces the error is nearly a function of `h/δ`
  alone: 6.85e-2, 6.86e-2, 6.87e-2 at `h/δ ≈ ½` for the three δ, 0.105,
  0.100, 0.099 at ¼. So "treat the edge as a jump" is the right baseline
  exactly where the grid cannot see the edge, and the worst of all the
  lines where it can; to use it one has to know δ and switch it off at
  `h ≈ 2δ`, which the seeds never need (§1.4's last paragraph: for
  δ ≳ h every row is seeded and the seeds' weights tend to the standard
  ones). The naive line never does anything that bad, and the
  jump-aware line at δ = 0 is E1's: exact to rounding on the two
  constants (6e-15 to 3e-10 as the solve's conditioning grows), fourth
  order on eq. 75 (3.85–4.12).

**The floor's constant.** With `F(1) = ∫_{−1}^{1} dξ/α` the total
resistance, `(F₀(1) − F_δ(1)) / δ` against §1.7's `c`, which E3.3 found
in closed form: the antiderivative of the blended integrand in
`z = (x − x_c)/δ` is `z/a + (a − b)/(2ab) ln(a + b e^{2z})`
(`tests/heat1d/test_exact.py::tanh_edge_integral`), and subtracting the
jump's `z/a`, `z/b` on the two sides leaves, as `Z → ∞`, only the
logarithms' difference:

    c(a, b) = (a − b) ln(a/b) / (2ab)    (`exact.edge_resistance_deficit`),

symmetric in `a ↔ b`, positive since `1/α` is convex (the blend's
resistance is below the jump's), 8.788898 for `1/9 | 1` and 10.361633 for
`1 | 0.1`. On the MATLAB medium the measured ratio is 8.788898 at all
three δ (ratio 1.0000: the tails are exponentially small and the shift is
`c δ` to rounding). On eq. 75, whose two edges both have contrast ten so
that `Σ c = 20.72`, the measured ratio is 9.06, 15.31, 18.95 at δ = 0.04,
0.01, 0.0025 (0.44, 0.74, 0.91 of the limit; 19.96 at δ = 0.001, 20.64 at
0.0001): the sinusoid piece has `α′/α = 25` where it meets the layer's
edges, so at δ = 0.01 its value changes by 75 % across `±3δ` and the
"constant contrast" the closed form assumes is not what the edge sees.
The approach is first order in δ with a constant of that slope; the
floors above are the exact quantity and carry it. E3.5's T0 (widen δ to
`m h`) inherits `c m h` on constant pieces and this slower approach on
eq. 75.

**"How far off in weights" as a residual (the first-order-in-δ/h defect).**
The seed weights the ticket compares against are E3.4's (P4 keeps §1.4's
scratch numbers, 84 % … 0.11 % at δ/h = 1 … 0.001); what E3.3 measures
without seeds is the same defect as the rows' residual on the true-δ
equilibrium, which is what enters the solution error. At fixed h (200 and
201 nodes, h = 0.01) with δ = (δ/h) h, `max |L_h (u_δ − u₀)|` over the
rows whose windows straddle a centre, `h²` of it over δ, `h² |L_h u₀|`
on the same rows (rounding on the constants, E1.2's exactness; the
sinusoid's own third-order truncation on eq. 75, which is why the
residual is taken on the difference), and the naive rows' `h · max |L_h
u_δ|` for the scale of a row that misses the slope jump outright:

| δ/h | MATLAB: residual | h² res / δ | h² \|L_h u₀\| | naive h \|L_h u_δ\| | eq. 75: residual | h² res / δ | h² \|L_h u₀\| | naive h \|L_h u_δ\| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.58 | 0.066 | 4e-17 | 7.0e-3 | 20.2 | 0.20 | 3.6e-6 | 2.3e-2 |
| 0.5 | 5.34 | 0.107 | 4e-17 | 8.2e-3 | 20.5 | 0.41 | 3.6e-6 | 0.11 |
| 0.1 | 2.38 | 0.239 | 4e-17 | 7.9e-2 | 5.02 | 0.502 | 3.6e-6 | 0.32 |
| 0.01 | 0.238 | 0.2387 | 4e-17 | 0.109 | 0.515 | 0.515 | 3.6e-6 | 0.39 |
| 0.001 | 0.0238 | 0.2387 | 4e-17 | 0.112 | 0.0516 | 0.516 | 3.6e-6 | 0.39 |

`h² · residual / δ` settles to 0.2387 (MATLAB) and 0.516 (eq. 75) by
δ/h = 0.01, so the rows are off by `C δ/h²` against the naive rows'
`0.11/h` and `0.39/h`: a first-order-in-δ/h fraction of a wrong row,
ratio about `2 δ/h` on the MATLAB medium, and `u_δ − u₀ → −B c δ` past the
edge is what the row sees (a step of that size across the window). At
δ = h the scaled residual is a third of the constant: the edge is partly
inside the window and the rows are less wrong per unit δ, consistent with
the dip below the floor at `h ≈ 2δ` above.

**What this changes downstream.** P2 is ticked. P3 is ticked for the
unresolved regime and corrected for the resolved one; the manuscript's
baseline paragraph should say both, and the reason. The knee figure is
the frame E3.4's seed lines go on (one δ-independent fourth-order line
through the whole plot is the claim, P7), and E3.5's treatments are read
against the same floors. On eq. 75 the study's δ are not in the
asymptotic range of the floor constant, so any statement there quotes the
measured floor, not `c δ`.

Tests: `tests/test_heat1d_stiff.py` pins the grids' placements and the
101 floor, the knee on both media (pre-knee rates averaging below 2, a
first post-knee rate above 3.8, a factor above 100 between `h = 2δ` and
`h = δ/2`, E1's δ = 0 line), the construction on the floor to 1 % for
`h ≥ 4δ` and above 10× it at `h = δ/2` (elliptic, δ = 0.01 and 0.0025;
ramp at δ = 0.01 with the floor at 0.28 δ), the floor constants (exact on
the MATLAB medium, a first-order approach on eq. 75), the scaled residual
constant to 5 % between δ/h = 0.01 and 0.001 and below half of it at
δ/h = 1, and the driver's figure and cache;
`tests/heat1d/test_exact.py` pins the closed form against the quadrature
gap in both directions of the contrast.

### 2.3 The seed stencils (E3.4, #29)

![knee, with the seeds](figures/heat1d_stiff_knee.png)

`heat1d/stiff.py`: the chain of §1.2 marched as §1.3 prescribes, and the
seed operator on the knee sweep of §2.2 as its third line (blue; the δ = 0
construction moves to purple). The figure above replaces §2.2's. The
implementation, so the numbers below can be read:

- **The march.** One first-order system in `(φ_k, ψ_k = α φ_k′)`, DOP853
  at rtol 1e-13 / atol 1e-15 (`SEED_RTOL`, `SEED_ATOL`) in the stencil
  coordinate `ξ = (x − x_e)/h_s`, `h_s = max |x_i − x_e|` (the companion's
  half-width normalisation, `ξ_i = 0, ±½, ±1` on an interior row), with a
  fresh segment at every edge centre and at `±10δ` from it (`EDGE_STOP`).
  On each segment α is read from the piece of the medium's element that
  contains it (`Medium1D.elements`), which is what makes the evaluation
  one-sided at a jump: δ = 0 is marched like any other width, the centre a
  stop across which `(φ, ψ)` is continuous, and it reproduces E1.2's
  algebra to rounding (P4 below) rather than being dispatched to it.
- **Batched rows.** Every seeded row whose window has the same node pattern
  `ξ_i` marches in one system (`seed_profiles`, `(rows, 2, 5)` states):
  the rows' stops, spaced `h/h_s = ½` apart in ξ, union to at most six per
  side, so an operator costs some sixteen `solve_ivp` calls whatever the
  count of seeded rows: 0.11 s for the 1284 rows of δ = 0.04 at 1600 nodes
  (1442 on eq. 75), 0.04 s for the 8 rows of δ = 0.0025 at 100. The batch
  agrees with single-row marches to 6e-15 on φ where the edge is resolved
  and to 2e-13 where it is not (DOP853's error norm is an RMS over the
  batch, so the few rows crossing the steep part are controlled to a factor
  sqrt(rows) more loosely); the weights agree to 7e-12 relative at worst
  and the solutions to rounding (`test_the_batched_march_agrees_with_
  single_stencils`; the equilibrium errors below are the same whether the
  rows are marched together, in eights or singly).
- **Which rows.** Those whose window span overlaps `(x_c − 20δ, x_c + 20δ)`,
  `TANH_REACH` (E3.2's bit-exact reach, not §1.3's 19δ): at δ = 0 exactly
  `straddling_windows`; 8 rows at δ = 0.0025 and 84 at δ = 0.04 on 100
  nodes; every row, the one-sided end rows in batches of their own, once
  the reach covers the domain. `reach` is the operator's one knob.
- **Cost of the study.** The driver reruns in 3.7 s with everything
  cached and took 18 s to add the 24 seed marches of the ramp sweep to the
  knee cache; the 41 tests of `tests/heat1d/test_stiff.py` and
  `tests/test_heat1d_stiff.py` run in 17 s.

**P4, the weights (holds).** Constant α = 0.7 gives 0.7 × Fornberg's
second-derivative weights to 1e-15 on the centred window and 2e-16 with
the centre off the nodes (the chain is nilpotent, so the order-8 march
reproduces the monomials exactly). On the MATLAB window of §1.4 (the
`1/9 | 1` jump half a cell right of the node, h = 0.01), `max |w_seed −
w_jump| / max |w_jump|` against E1.2's weights, with the stencil solve's
condition number as built and with the rows of A scaled to unit max norm
(the seeds carry the normalisation `α_e^{⌈k/2⌉}`, §1.2, which the weights
never see):

| δ/h | difference | cond A | rows scaled |
| --- | --- | --- | --- |
| 0 | 7e-16 | 90.3 | 90.3 |
| 1 | 0.843 | 100.4 | 147.0 |
| 0.5 | 0.465 | 137.9 | 206.7 |
| 0.1 | 0.114 | 114.2 | 114.2 |
| 0.01 | 0.0107 | 92.2 | 92.2 |
| 0.001 | 0.00107 | 90.5 | 90.5 |

§1.4's scratch numbers to three digits (84 %, 47 %, 11 %, 1.1 %, 0.11 %),
first order in δ/h (ratios 10.6 and 10.0 over the last two decades), and
the δ = 0 march equal to the translated basis to rounding at offsets ½,
3/2 and 0 cells (the last with the jump *on* the evaluation node, `α_e`
the owner's). The double-cross of P8 is the same limit twice over.

On eq. 75 the difference does not go to zero. At the layer's edge on a
node (x = 0, 201 nodes, `α_e` = 0.1 the layer's, `α′/α = 25` on that side)
it is 2.24, 2.04, 1.69, 1.31, 1.26 at δ/h = 1, ½, 0.1, 0.01, 0.001 and
1.25 at δ = 0: the `O(h α′/α)` floor of §1.4, at which E1.2's
degree-4-truncated operator on plain monomials and the exact chain part
company. It is first order in h (at x = 0.5: 1.75, 1.25, 0.50, 0.22 for
h = 0.02, 0.01, 0.005, 0.0025), O(1) on every grid of the study, and both
operators are fourth order on solutions (P6, P7): the two spaces are
different and equally good, as §1.4 said, and the seeds' one is the
better of the two in the solution error by 6–44× (below).

**P5, conditioning (holds, with the constant).** `cond A` in the stencil
coordinate is 23.5 at constant α for the centred window (the Vandermonde
value) and, on the MATLAB window, 90 at every δ ≤ 0.01 h (the translated
basis' own value at δ = 0), rising to 138 at δ = h/2 (207 rows-scaled) and
back to 24 for δ ≥ 10 h, where the seeds are the monomials again (§1.4's
last paragraph): within a factor six of the Vandermonde value from δ =
1e-5 h to 100 h, the companion's "20 to 60" at contrast 4 becoming 24 to
138 at contrast 9. On eq. 75's edge node the as-built numbers are 84 to
648 and the rows-scaled 117 to 452: the sinusoid's `h α′/α = 0.25` bends
every seed away from `ξ^k` inside the window, and the δ = 0 value (648) is
inflated by `α_e` being the owner's 0.1 rather than the blend's 0.55, a
row scaling the weights do not see (the rows-scaled 452 against 450 at
δ = 0.001 h). No solve in the study is threatened; the equilibrium
residuals below are rounding.

**P6, equilibrium (holds on constant pieces; on eq. 75 it is the plain
rows' line).** On the MATLAB medium the seed line is exact at every δ:

| n | δ = 0 | δ = 0.04 | δ = 0.01 | δ = 0.0025 | E1.2 at δ = 0 |
| --- | --- | --- | --- | --- | --- |
| 50 | 1.4e-14 | 1.7e-14 | 1.9e-14 | 4.5e-14 | 6.2e-15 |
| 100 | 5.9e-14 | 2.2e-14 | 4.2e-14 | 5.3e-14 | 5.1e-14 |
| 200 | 2.6e-13 | 3.2e-13 | 2.2e-13 | 2.6e-13 | 2.9e-13 |
| 400 | 1.2e-12 | 4.7e-14 | 8.9e-13 | 1.1e-12 | 6.3e-13 |
| 800 | 1.2e-11 | 1.2e-12 | 8.8e-12 | 1.0e-11 | 1.0e-11 |
| 1600 | 2.3e-11 | 3.8e-11 | 1.7e-11 | 2.3e-11 | 1.9e-11 |

Below 1e-12 to 400 nodes at every δ, then growing exactly as E1.2's own
δ = 0 line does (last column): the direct solve's conditioning, not the
seeds. The seed rows' residual on the exact solution, `h² max |L_h u_δ|`
over the seeded rows at 200 nodes, is 1.8e-16, 2.5e-16, 9.7e-17 at δ/h =
1, ½, 0.1 and 2.4e-13, 5.8e-13 at 0.01, 0.001 (against the δ = 0 rows'
6.6 … 0.024 of §2.2): rounding while the march crosses the edge in a few
steps, and the march's own floor once it takes many, which the solution
sees at δ ≲ h/40 (1.0e-11 at 101 nodes and δ = 5e-4, 4e-11 at 100 nodes
and δ = 1e-4; tightening rtol to SciPy's 100 eps changes neither, and
§1.4's remark that an edge below 1e-5 h is better served by the jump
weights stands, with the bound nearer 1e-2 h for 1e-12 work). The
ticket's "exact to 1e-12 at every δ" is met where the direct solve allows
it and for δ ≳ h/40.

On eq. 75 the seed rows are exact too, and the line is the error of the
rows that are *not* seeded, which depends on δ through the reach:

| n | δ = 0 | δ = 0.04 | δ = 0.01 | δ = 0.0025 | E1.2 at δ = 0 |
| --- | --- | --- | --- | --- | --- |
| 101 | 9.69e-5 | 2.2e-13 | 2.0e-14 | 5.05e-6 | 4.25e-3 |
| 201 | 1.58e-5 (2.61) | 6.2e-13 | 3.6e-10 | 8.65e-7 (2.55) | 2.45e-4 (4.12) |
| 401 | 1.79e-6 (3.14) | 1.8e-13 | 8.0e-11 | 7.54e-8 (3.52) | 1.67e-5 (3.88) |
| 801 | 1.57e-7 (3.51) | 9.6e-12 | 6.7e-12 | 5.61e-9 (3.75) | 1.13e-6 (3.89) |
| 1601 | 1.18e-8 (3.74) | 3.4e-11 | 3.4e-11 | 3.63e-10 (3.95) | 7.39e-8 (3.93) |

At δ = 0.04 the reach (0.8) covers the layer and the solve is exact at
every count. At δ = 0.01 it covers it at 101 nodes (2e-14) and from 201
on leaves the rows with `x ∈ (0.2, 0.3)` plain: FD4 on the sinusoid at
h = 0.01, pre-asymptotic (`u⁽⁶⁾ ∼ (2π)⁶`), 3.6e-10, falling to the solve's
floor by 801; seeding every row (`reach = 40`) at 201 nodes returns
1.3e-13. At δ = 0.0025 the reach is 0.05 and the layer's interior is plain
from 101 nodes on: the line is the sinusoid's own FD4 error with the rows
nearest the edges (`h α′/α = 0.5` at 101 nodes) removed, rate 2.55 → 3.95
as those rows become asymptotic. At δ = 0 only the six straddling rows are
seeded and the line is 44× below E1.2's at 101 nodes and 6× at 1601, at
rates 2.6 → 3.7 against E1.2's 3.9–4.1: E1.2's error was dominated by its
straddling rows' `O(h³)` local term, the seeds' by the plain rows next to
them. So "one δ-independent fourth-order line equal to the smooth
problem's" (§1.9) is right about the seeded rows and wrong about the
line, which is whichever plain rows the reach leaves inside the layer;
the note's prediction assumed the seeded set did not move with δ.

**P7, parabolic (holds; three digits at δ = 0.0025, 1.5 % at 0.01).** The
ramp problem at t = 2, BD4 at dt = h, against the true-δ reference:

| n | δ = 0 | δ = 0.04 | δ = 0.01 | δ = 0.0025 | construction at 0.0025 |
| --- | --- | --- | --- | --- | --- |
| *MATLAB medium* | | | | | |
| 50 | 2.487e-6 | 2.388e-6 | 2.462e-6 | 2.480e-6 | 6.97e-4 |
| 100 | 2.951e-8 (6.40) | 3.049e-8 (6.29) | 2.989e-8 (6.36) | 2.954e-8 (6.39) | 7.12e-4 |
| 200 | 1.614e-9 (4.19) | 1.690e-9 (4.17) | 1.638e-9 (4.19) | 1.616e-9 (4.19) | 7.16e-4 |
| 400 | 9.757e-11 (4.05) | 1.027e-10 (4.04) | 9.886e-11 (4.05) | 9.741e-11 (4.05) | 7.01e-4 |
| 800 | 6.1e-12 (4.00) | 6.5e-12 (3.97) | 6.3e-12 (3.97) | 5.6e-12 (4.12) | 4.65e-3 |
| 1600 | 4.2e-12 | 2.2e-11 | 5.8e-12 | 4.2e-12 | 2.30e-2 |
| *eq. 75 medium* | | | | | |
| 101 | 1.280e-4 | 2.606e-8 | 2.922e-8 | 8.497e-6 | 7.51e-3 |
| 201 | 1.991e-5 (2.68) | 1.531e-9 (4.09) | 1.373e-8 (1.09) | 1.281e-6 (2.73) | 3.41e-3 |
| 401 | 2.201e-6 (3.18) | 1.107e-10 (3.79) | 1.301e-9 (3.40) | 1.082e-7 (3.57) | 4.29e-3 |
| 801 | 1.910e-7 (3.53) | 2.1e-11 | 1.017e-10 (3.68) | 7.953e-9 (3.77) | 4.60e-2 |
| 1601 | 1.421e-8 (3.75) | 2.2e-11 | 3.3e-11 | 5.064e-10 (3.97) | 1.32e-1 |

On the MATLAB medium the seed line is one line: the δ = 0.0025 points sit
on the δ = 0 ones (the jump-aware operator's, which the seeds *are* at
δ = 0 on constant pieces) to 0.1–0.3 % at every count to 400, the δ = 0.01
ones to 1.0–1.5 % and the δ = 0.04 ones (δ ≈ h at 50 nodes, a genuinely
different solution) to 3–5 %; the rates are 6.4, 4.2, 4.05, 4.0 (E1.3's
super-convergent first pair, then four), and from 800 nodes every δ sits
on the reference's 2e-12 floor of §2.1 (6e-12, then 4e-12 to 2e-11), as
that section said the 800-node points would. The construction's δ =
0.0025 line is on its floor (7e-4) to 400 nodes, seven orders above the
seeds there and ten once h ≤ δ; the naive line crosses the seed line
nowhere. On eq. 75 the
reading of P6 repeats: δ = 0.04 (every row seeded) is the fourth-order
line 2.6e-8 → 2e-11 (4.09, 3.79, then the reference's 2–6e-11 floor), δ =
0.01 has the plain rows' 1.4e-8 at 201 nodes (1.7e-9 with every row
seeded) and is fourth order past it, δ = 0.0025 runs 2.7 → 4.0, and δ = 0
is 38× below E1.2's 4.93e-3 … 8.53e-8 at rates 2.7 → 3.75 against 3.9–4.1.

Two library-level checks without the Chebyshev reference, on the separable
solution `e^{ct} v(x)` (E1.3's pattern, `chebyshev_equilibrium(shift=c)`
on the smooth medium's elements; MATLAB medium, δ = 0, 0.01, 0.0025, 50 to
400 nodes): the seed lines converge at 4.95, 4.84, 4.15 and agree across
δ to 0.07 %, 0.17 %, 0.85 %, 2.3 % as the errors fall from 1.6e-6 to
1.0e-10 and the profiles `v_δ` themselves differ at O(δ). And §1.6's
local truncation on v (`L v = c v`): the seeded rows' residual is third
order (rates 3.06, 3.78 at δ = 0.01 and 3.15, 3.51 at 0.0025 over 100 →
400 nodes; `residual · n³` 0.27, 0.26, 0.15 and 0.43, 0.38, 0.27, the same
within a factor two between the widths, the "δ-independent constant"
P7 asked for) and the plain rows' fourth (3.94, 3.97 and 3.94, 4.01); at
800 nodes both are on the 48-node collocation profile's floor.

**P8, the double-cross (holds).** A layer two cells thick (`1 | 0.1 | 1`
on [0, 2h] at h = 0.05) with both tanh edges inside one window: the
window's seed weights are E1.2's translate-twice weights to 7.6e-15 at
δ = 0 and differ from them by 22.2 %, 1.86 %, 0.182 % at δ/h = 0.1, 0.01,
0.001 (first order in δ/h with constant 1.8 against the single edge's 1.1:
two edges, each a 10 : 1 contrast, inside one window);
the seed operator annihilates the smooth layer's exact equilibrium to
3e-13 at δ/h = 0, 0.1 and 0.5, the windows seeing both centres numbering
1, 5 and 23. One march, two stops per edge.

**P9, the spectrum (holds, and more).** The interior operator's
eigenvalues (Dirichlet rows removed, `interior_operator`) at δ = 0 and
0.0025:

| medium | n | max Re λ | max \|Im λ\| | min Re λ · h² | BD4 max amplification |
| --- | --- | --- | --- | --- | --- |
| MATLAB | 50 | −0.831 / −0.834 | 0 | −5.302 | 0.967 |
| | 100 | −0.831 / −0.834 | 0 | −5.326 | 0.983 |
| | 400 | −0.831 / −0.834 | 0 | −5.333 | 0.996 |
| eq. 75 | 49 | −2.05 / −2.08 | 0 | −5.302 | 0.918 |
| | 53 | −2.05 / −2.08 | 0 | −5.306 | 0.924 |
| | 101 | −2.06 / −2.08 | 0 | −5.326 | 0.960 |
| | 401 | −2.06 / −2.08 | 0 | −5.333 | 0.990 |

Real to the bit (no complex pair anywhere, the one-sided end rows
included), negative, the extreme FD4's `−16/3 h⁻²` in the α = 1 material
to 0.6 % at 50 nodes and 0.01 % at 400, BD4-damped at dt = h; the
least-damped eigenvalue moves by 0.4 % (MATLAB) and 1.5 % (eq. 75) between
the jump and the sub-grid edge. And the seed operator is stable on eq. 75 at
49 and 53 nodes (largest real part −2.05), where the δ = 0 construction
has eigenvalues at +248 and +8.8 (§2.2, `MIN_COUNT`): E1.2's under-resolved
translated basis, `h α′/α ≈ 1` at the layer's edges, is what crossed the
axis, and the exact chain does not. The sweep still starts at 101 (its
grids are shared by the three operators); E3.6 may run the seeds coarser
if the figure wants it.

**What this changes downstream.** P4, P5, P7, P8 and P9 are ticked as
predicted, P6 on constant pieces too; two readings are corrected. On
eq. 75 the seed line is the plain rows' line and moves with the reach, so
any statement about it names δ and the reach (the manuscript's 1-D
elliptic remark should stay on the MATLAB medium, where the exactness is
clean, plan R2). P7's "three digits at every n" holds for δ = 0.0025 and
becomes 1.5 % at δ = 0.01 and 5 % at 0.04, where δ ≈ h and the solutions
differ; the claim to make is the one line through the whole plot in the
figure, not a digit count. Three things for the manuscript: the seeds
need no δ to be switched off where the construction does (§2.2's h ≲ 2δ
regime is the seeds' at cost nothing: 1e-12 elliptic and the same
parabolic line at every h/δ from 16 to 1/32); they are the jump-aware
operator itself at δ = 0 on constant pieces (one construction, two
implementations, checked to rounding) and a better operator than it on
smoothly varying pieces (6–44× on eq. 75) and at coarse counts (stable at
49 nodes); and the march's floor at δ ≲ h/40 (a row residual of 1e-12,
a solution error of 1e-11) is the one place the ODE implementation shows,
which a collocation integration of the same chain would remove and E3.6
may mention as not built. E3.5 (#30) reads its treatments against the
tables above with the seed line as the target; E3.6 (#31) inherits
`COLOURS` (purple for the construction, blue for the seeds) and the
figure.

Tests: `tests/heat1d/test_stiff.py` (Fornberg at four centres; the δ = 0
march against E1.2 at three offsets; the 84 % … 0.11 % limit and its
ratios; the condition numbers' range, ends and peak; eq. 75's first-order
floor; the equilibrium below 1e-12 at three counts and three δ with the
residual below 1e-14, and the march's floor at δ = 1e-4; eq. 75's exact,
plain-row and δ = 0 lines against E1.2's; the separable solution's rates
and cross-δ agreement; the local truncation orders and constants; the
double-cross; the seeded windows at δ = 0 and with the reach; the spectra
at four grids with the construction's instability pinned at 49 nodes; the
batch against single stencils at 14 and at 1284 rows; six-point stencils
against Fornberg and against the degree-5 translated basis; the dispatch;
validation) and
`tests/test_heat1d_stiff.py` (the driver's seed line on the ramp problem
at the study's reference resolution, the weights and conditioning table,
the eq. 75 floor, the spectra at 49 and 101 nodes).

### 2.4 The coefficient treatments (E3.5, #30)

![treatments](figures/heat1d_stiff_treatments.png)

`heat1d/treatments.py`, and the comparator tables appended to
`scripts/heat1d_stiff.py` (30 s cold per medium for the 144 extra BD4
marches, 2 s cached; the whole driver 2 min cold, 6 s cached). Plan §3.4's
"change the medium, keep the scheme" comparators on §2.2's grids and
problems, read against the true-δ references with the seeds as the
target:

- **The nodal treatments are a material.** `NodalAlpha(grid, values)`: a
  table at the nodes, linear between them, no interfaces, elements cut at
  the nodes. `naive_operator` samples it to the bit, so a treatment is
  `Dx A Dx` on a changed A and nothing else. T1 `harmonic_cells(grid,
  medium, cells)` is `(b − a) / ∫_a^b dξ/α` over the window `x_j ± cells
  h/2` (one cell: the node's own cell; two: to the neighbours; clipped at
  the domain ends), T2 `arithmetic_cells` the same window's `∫ α / (b −
  a)` (`exact.alpha_integral`, the resistance quadrature with the other
  integrand). Both are differences of cumulative quadratures, so they
  carry a relative rounding of about `n ε` (1e-12 at 6400 nodes), below
  anything the tables read.
- **T0** `widened_edge(grid, medium, m)` is `SmoothEdges(jump, max(δ,
  m h))`: the same class, a wider edge, so the references and the windows
  stay consistent; below `δ = m h` it does not depend on δ at all.
- **T1-FV** `face_conductance_operator` is the three-point conservative
  scheme with `a_{i+½} = h / ∫_{x_i}^{x_{i+1}} dξ/α` (§1.8 item 3), an
  operator, not a medium; zero end rows for the Dirichlet rows to replace.
- **T3**, the band-limited α, is not built: the ticket keeps it only if
  E5.2 (#43) finds it in use for diffusion.

Errors `‖e‖₂/‖u‖₂` at the nodes (the rate in parentheses), the ramp
problem at t = 2, MATLAB medium, the comparator columns in P10's order
with the untreated naive line and the seeds as the two ends:

| n | h/δ | naive | T1 1c | T1 2c | T2 1c | T0 m=1 | T0 m=2 | T1-FV | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *δ = 0 (jump, mid-cell)* | | | | | | | | | |
| 50 | ∞ | 3.86e-3 | 3.86e-3 | 6.26e-4 | 3.86e-3 | 1.12e-2 | 2.22e-2 | 1.31e-4 | 2.49e-6 |
| 100 | ∞ | 1.96e-3 (0.98) | 1.96e-3 (0.98) | 2.30e-4 (1.44) | 1.96e-3 (0.98) | 5.67e-3 (0.98) | 1.13e-2 (0.98) | 3.24e-5 (2.02) | 2.95e-8 (6.40) |
| 200 | ∞ | 9.90e-4 (0.99) | 9.90e-4 (0.99) | 8.32e-5 (1.47) | 9.90e-4 (0.99) | 2.85e-3 (0.99) | 5.69e-3 (0.99) | 8.10e-6 (2.00) | 1.61e-9 (4.19) |
| 400 | ∞ | 4.97e-4 (0.99) | 4.97e-4 (0.99) | 2.97e-5 (1.48) | 4.97e-4 (0.99) | 1.43e-3 (0.99) | 2.86e-3 (0.99) | 2.02e-6 (2.00) | 9.76e-11 (4.05) |
| 800 | ∞ | 2.49e-4 (1.00) | 2.49e-4 (1.00) | 1.06e-5 (1.49) | 2.49e-4 (1.00) | 7.18e-4 (1.00) | 1.43e-3 (1.00) | 5.06e-7 (2.00) | 6.11e-12 (4.00) |
| 1600 | ∞ | 1.25e-4 (1.00) | 1.25e-4 (1.00) | 3.75e-6 (1.50) | 1.25e-4 (1.00) | 3.59e-4 (1.00) | 7.18e-4 (1.00) | 1.26e-7 (2.00) | 4.18e-12 (0.55) |
| *δ = 0.01* | | | | | | | | | |
| 50 | 4.08 | 3.09e-3 | 1.53e-3 | 7.03e-4 | 1.79e-3 | 8.65e-3 | 2.00e-2 | 1.30e-4 | 2.46e-6 |
| 100 | 2.02 | 1.79e-4 (4.11) | 1.23e-4 (3.63) | 2.37e-4 (1.57) | 8.21e-4 (1.13) | 2.93e-3 (1.56) | 8.69e-3 (1.20) | 3.27e-5 (2.00) | 2.99e-8 (6.36) |
| 200 | 1.01 | 9.25e-5 (0.95) | 5.97e-5 (1.04) | 6.99e-5 (1.76) | 1.54e-4 (2.41) | 9.12e-5 (5.00) | 2.92e-3 (1.57) | 8.14e-6 (2.00) | 1.64e-9 (4.19) |
| 400 | 0.50 | 1.54e-6 (5.91) | 4.38e-6 (3.77) | 1.91e-5 (1.87) | 3.48e-5 (2.15) | 1.54e-6 (5.89) | 7.34e-6 (8.63) | 2.03e-6 (2.00) | 9.89e-11 (4.05) |
| 800 | 0.25 | 9.32e-8 (4.04) | 1.20e-6 (1.87) | 4.94e-6 (1.95) | 8.74e-6 (1.99) | 9.32e-8 (4.04) | 9.32e-8 (6.30) | 5.09e-7 (2.00) | 6.32e-12 (3.97) |
| 1600 | 0.13 | 5.96e-9 (3.97) | 3.09e-7 (1.96) | 1.25e-6 (1.99) | 2.19e-6 (2.00) | 5.96e-9 (3.97) | 5.96e-9 (3.97) | 1.27e-7 (2.00) | 5.76e-12 (0.13) |
| *δ = 0.0025* | | | | | | | | | |
| 50 | 16.3 | 3.85e-3 | 3.28e-3 | 6.35e-4 | 2.78e-3 | 1.05e-2 | 2.16e-2 | 1.31e-4 | 2.48e-6 |
| 100 | 8.08 | 2.04e-3 (0.92) | 1.37e-3 (1.26) | 2.38e-4 (1.42) | 1.11e-3 (1.33) | 4.99e-3 (1.08) | 1.06e-2 (1.02) | 3.20e-5 (2.03) | 2.95e-8 (6.39) |
| 200 | 4.02 | 7.90e-4 (1.37) | 3.95e-4 (1.79) | 8.94e-5 (1.41) | 4.59e-4 (1.27) | 2.16e-3 (1.21) | 5.02e-3 (1.09) | 8.00e-6 (2.00) | 1.62e-9 (4.19) |
| 400 | 2.01 | 4.36e-5 (4.18) | 2.74e-5 (3.85) | 2.96e-5 (1.60) | 1.99e-4 (1.20) | 7.23e-4 (1.58) | 2.16e-3 (1.22) | 2.01e-6 (2.00) | 9.74e-11 (4.05) |
| 800 | 1.00 | 2.19e-5 (0.99) | 1.40e-5 (0.97) | 8.69e-6 (1.77) | 3.76e-5 (2.41) | 2.18e-5 (5.05) | 7.21e-4 (1.58) | 5.01e-7 (2.00) | 5.61e-12 (4.12) |
| 1600 | 0.50 | 2.20e-7 (6.64) | 5.50e-7 (4.67) | 2.37e-6 (1.87) | 8.59e-6 (2.13) | 2.20e-7 (6.63) | 4.96e-7 (10.5) | 1.25e-7 (2.00) | 4.22e-12 (0.41) |

The δ = 0.04 block (h/δ = 1.02 at 50 nodes, resolved from 100 on) and both
elliptic tables are in the driver's output; the seed line is §2.3's, on
the reference's floor from 800 nodes.

**P10, T1-FV (holds).** Exact at equilibrium at every δ and both
placements: 1e-14 at 50 nodes, rising with n to 6.5e-12 at 1600 (MATLAB)
and 1.4e-11 at 1601 (eq. 75), the cumulative quadrature's `n ε` on top of
the direct solve's growth that the seed line shows too (§2.3); the
discrete flux equals `equilibrium_flux` at every face to 2e-12
(`test_the_face_conductance_scheme_is_exact_at_equilibrium`). Second order
in the ramp problem at every δ with one constant: on the MATLAB medium the
δ = 0, 0.01 and 0.0025 columns agree to 1 % at every count and the δ = 0.04
one (1.40e-4 → 1.35e-7) sits 7 % above them; on eq. 75 the rate is 2.00
at every δ from 401 nodes on but the constant moves with δ by up to 40 %
(6.47e-8, 3.93e-8, 4.62e-8, 4.99e-8 at 1601 nodes for δ = 0, 0.04, 0.01,
0.0025: the sinusoid varies across the edge, §2.2's slow approach again),
with one wobble (rate 0.91 at 201 nodes, h = 4δ, δ = 0.0025) where the
face across the edge changes character. **On eq. 75 it is ahead of the seeds at coarse counts:**
at δ = 0, 1.63e-5, 4.12e-6, 1.03e-6 against the seeds' 1.28e-4, 1.99e-5,
2.20e-6 at 101, 201, 401 nodes, the seeds ahead from 801 (1.91e-7 against
2.59e-7); at δ = 0.0025, 6.35e-6 against 8.50e-6 at 101, the seeds ahead
from 201. That is §2.3's pre-asymptotic sinusoid line (rates 2.7 → 3.75
on the plain rows, the `O(h α′/α)` floor), not the edge: at δ = 0.04 and
0.01 the seeds lead at every count, by 400× at 101 nodes and δ = 0.01
(2.92e-8 against 1.28e-5). On the MATLAB medium they lead everywhere, by
50× at 50 nodes and 3e4× at 1600.

**P10, T1 under `Dx A Dx` (holds, with a reading).** Not exact: the
module reproduces §1.8's residuals 5.58 and 1.24 (one and two cells, the
mid-cell jump at 100 nodes; `test_no_nodal_alpha_makes_dx_a_dx_exact_
the_section_1_8_residuals`). With the edge mid-cell the one-cell window
ends *on* the jump, so at δ = 0 T1 (one cell) and T2 (one cell) are the
naive operator to rounding in every row of the table; on eq. 75, whose
edges sit on nodes, the one-cell window straddles them and T1 (one cell)
is 9–35× below naive at δ = 0 (2.07e-3 → 2.56e-5, rates → 1.5). The
**two-cell harmonic mean converges at order 1.5** in this norm (rates 1.44
→ 1.50 parabolic, 1.50 elliptic, both media), a local defect and not a
global one: its max-norm error is O(h) (1.18e-3 → 7.28e-5 at 100 → 1600
nodes, rates 1.00) and sits on the two nodes beside the edge, with
rounding (1e-12) beyond |x| > 0.2, so `‖e‖₂/‖u‖₂ ∝ h · n^{−1/2}`. A
max-norm table would call it first order; the manuscript says which norm.
It is the best nodal treatment while the edge is unresolved (7.03e-4
against naive's 3.09e-3 at 50 nodes, δ = 0.01), 6–33× ahead of naive at
δ = 0 (the gap widening with n, the orders differing by ½) and 4–9× at
h ≥ 4δ.

**P10, T0 (holds, with the constant).** T0's own floor is the widened
medium's exact equilibrium against the true one, `‖u_{max(δ, m h)} −
u_δ‖/‖u_δ‖`, and T0's elliptic error *is* that floor: to 0.1 % for m = 2
at every h > δ/2 (ratios 1.000 on both media, 1.006 at h = δ/2 where the
widened edge is `1.01 δ` wide and the operator is at its own knee) and to
1 % for m = 1 at h ≥ 2δ (1.006, 0.998, 0.999, 1.000 at δ = 0.0025), then
above it at
h ≈ δ (ratio 4.9 at h = 1.01 δ, 19 at 1.00 δ: `m h − δ` is a sliver and the
naive operator's knee error is what is left), and the naive operator
itself once m h ≤ δ. The floor is `c (m h − δ)` in resistance (§2.2's
closed form, `test_the_widened_edge_sits_a_resistance_deficit_from_the_
true_medium` to 1e-10) and 0.70–0.74 `(m h − δ)` in this norm on the
MATLAB medium, drifting to 0.66 by a width of 0.08 as the widened tails
reach the boundary. So T0 is first order with the constant `c m`, as
predicted, and that makes it the **worst** treatment, not T1's equal: 2.9×
(m = 1) and 5.7× (m = 2) above naive at δ = 0 on the MATLAB medium, on
every grid. Widening an edge the grid already fails to resolve adds
resistance error without removing the knee; it only ever helps a scheme
whose failure is on the resolved side, which `Dx A Dx` is not.

**P10, T2 (holds where it says "≈ naive", not elsewhere).** Equal to
naive at δ = 0 with the edge mid-cell (the one-cell window again), 20–40 %
below it on eq. 75, and at δ > 0 the worst nodal treatment for h ≲ 2δ
(1.54e-4 against T1's 5.97e-5 at h = δ, δ = 0.01), second order once the
edge is resolved.

**Every nodal treatment caps the naive operator at second order once the
edge is resolved.** For h ≲ δ/4 the naive line is fourth order and the
treated lines are second order with the FV constant times 1–5 (δ = 0.04,
1600 nodes, ramp: naive 2.05e-10, T1-FV 1.35e-7, T1 1c 1.62e-7, T2
5.76e-7, T1 2c 6.50e-7; the seeds 2.2e-11 on the reference's floor): the
cell mean perturbs a smooth α by O(w²) and the fourth-order operator
faithfully solves the perturbed problem. The nodal treatments therefore
have the δ = 0 construction's property (§2.2): they help only while
h ≳ δ and must know δ to be switched off, which the seeds need not.

**The ranking as measured.** With the edge unresolved (h ≥ 4δ, δ = 0
included), on both media and both problems: *seeds < T1-FV < T1 (two
cells) < naive ≳ T1 (one cell) ≈ T2 < T0 (m = 1) < T0 (m = 2)*, the middle
three within a factor two of each other; at 200 nodes and δ = 0.0025 on
the ramp problem, 1.6e-9, 8.0e-6, 8.9e-5, 7.9e-4 / 4.0e-4 / 4.6e-4, 2.2e-3,
5.0e-3. Across the knee (h from 2δ to δ/2) the naive line and T0 (m = 1)
drop through the cell-mean lines, and in §2.2's dip at h ≈ 2δ naive can
come in below T1 (two cells) by a quarter (δ = 0.01 at 100 nodes; not at
δ = 0.0025 and 400). Resolved
(h ≤ δ/4): *seeds < naive = T0 < T1-FV ≈ T1 (one cell) < T2 ≈ T1 (two
cells)*. P10's "T1 ≈ T0" and "T2 ≈ naive at every δ" are corrected; its
order of the ends stands, with the one exception above (eq. 75 below 800
nodes at δ = 0 and 0.0025, where the seeds' sinusoid line is still
pre-asymptotic and the second-order FV is ahead).

**What this changes downstream.** P10 is ticked for T1-FV's exactness and
order and for T0's floor and constant, and corrected for the ranking.
Three things for the manuscript: the strongest low-order comparator is
the finite-volume scheme with exact face conductances, exact at
equilibrium and second order at every δ with one constant, and on the
smoothly varying medium it is ahead of the seeds until the seeds' own
sinusoid error has converged (the honest comparator paragraph says so;
E4.9's scattered-node treatments, #40, have no such twin); the cell means
and the widened edge are first-order-family treatments (1.5 in this norm
for the two-cell mean) that help only while h ≳ δ and cap the fourth-order
operator at second order once h ≲ δ/4, so they share the construction's
need to know δ; and the 1-D elliptic table cannot rank the methods (plan
§3.2), T1-FV and the seeds both exact and growing together with n, so the
elliptic remark stays one line and the ranking is the parabolic table's.
E3.6 (#31) inherits `COMPARATORS`, `COMPARATOR_STYLE` and the figure; E5.2
(#43) pins the sources (Tikhonov & Samarskii 1962 for the exact
conductances, Patankar 1980 ch. 4 for the harmonic mean) and decides T3.

Tests: `tests/heat1d/test_treatments.py` (every treatment the identity on
a constant α, T1-FV then α times the three-point second difference; the
`NodalAlpha` table, interpolant, slopes, elements and validation; the
windows' clipping; T1-FV exact at equilibrium at three δ on both
placements and both media with the flux to 1e-11; Patankar's harmonic mean
at δ = 0 mid-cell and the pieces' own values on a node; §1.8's residuals
5.58 and 1.24 from the module; the δ = 0 forms bit-equal through
`SmoothEdges(jump, 0)`, the first-order approach as δ → 0 for the means
and the conductances, T0 δ-independent below `m h`; AM ≥ HM with the 7/3
ratio beside the mid-cell jump, both means second order on a sinusoid
with the 4× between one and two cells; T1-FV second order on a
manufactured smooth problem; the widened medium's resistance deficit
`c (m h − δ)`; the one-cell mean seeing the tanh tail and nothing beyond
it; a smooth-edged eq. 64) and `tests/test_heat1d_stiff.py` (the
comparator table complete at every δ with T1-FV's rate 2 and δ-independent
constant, the ranking where the edge is unresolved, the one-cell means
equal to naive at δ = 0 mid-cell, the elliptic exactness of T1-FV and the
seeds; T0 on its floor to 1 % for m = 2 and the floor constant in
0.65–0.76; the driver's second figure and the cache keys).

### 2.5 Closing the 1-D study: the seed functions, a snapshot, the driver, and what §2 states (E3.6, #31)

![seed functions](figures/heat1d_stiff_seeds.png)

![snapshot](figures/heat1d_stiff_snapshot.png)

E3.6 adds the two figures the manuscript draws its construction and its
headline from, the run's results file, and this closing section, which
states what §2 has established with the numbers and the subsection each
traces to. `scripts/heat1d_stiff.py` now runs, in this order: the
references and their checks (§2.1), the knee sweep with the seed line and
the P4–P9 tables (§2.2–2.3), the comparators (§2.4), the seed functions and
the snapshot (this section), then writes `outputs/heat1d_stiff.json`. From
a populated `outputs/` the whole driver takes 6.0 s (references 0.1 s,
knee 3.3 s, comparators 2.1 s, the two figures 0.6 s), well under the
minute #31 asks for; cold it is about 2 min, all of it the parabolic
references (20 s) and the 396 BD4 marches of the knee and comparator
sweeps, every one cached in `heat1d_stiff_knee.json` afterwards.

- **The seed functions (first figure).** P4's window on the MATLAB medium
  (200 nodes, h = 0.01005, the `1/9 | 1` edge half a cell right of the
  evaluation node, `h_s = 2h`), in the stencil coordinate `ξ = (x −
  x_e)/h_s` in which the march works (§1.3): α across the window on top,
  and below it seeds `φ₁ … φ₄` at δ/h = ½ and 0.1 against the monomials
  `ξ^k` (constant α) and E1.2's translated basis (δ = 0, `translated_basis`
  re-centred at `x_e` with `shift_matrix`, which is what the initial
  conditions `φ_k(x_e) = φ_k′(x_e) = 0` pick out of its span). Each seed is
  multiplied by `(α₀/α_e)^⌈k/2⌉`, `α₀` the jump's value at `x_e` and `α_e`
  the blend's, so that the lines are comparable across δ: the seeds carry
  the normalisation `α_e^⌈k/2⌉` of §1.2, every moment condition is
  homogeneous in it, and the weights never see it (without the factor the
  δ/h = ½ seeds sit a factor two off, `α_e` being 0.22 there against the
  owner's 0.11). On the node's side of the edge the δ = 0 seeds are the
  monomials themselves (checked to 1e-13), across it they are the
  translated polynomials with the 9 : 1 slope, and the δ > 0 seeds bend
  through the edge instead of kinking at it. The sup distances over the
  window (`seed_functions`; the driver prints δ/h = 0, ½, 0.1 and the
  test the ladder below it):

  | δ/h | vs E1.2, k = 1 | k = 2 | k = 3 | k = 4 | vs `ξ^k`, k = 1 | k = 2 |
  | --- | --- | --- | --- | --- | --- | --- |
  | 0 | 2.9e-15 | 2.7e-15 | 2.1e-15 | 1.4e-15 | 0.667 | 0.833 |
  | 1 | 0.312 | 0.198 | 0.559 | 0.440 | 0.831 | 0.861 |
  | 0.5 | 0.167 | 0.045 | 0.204 | 0.094 | 0.822 | 0.870 |
  | 0.1 | 5.12e-2 | 2.12e-2 | 1.67e-2 | 1.02e-2 | 0.715 | 0.853 |
  | 0.01 | 5.12e-3 | 2.51e-3 | 1.81e-3 | 1.20e-3 | 0.672 | 0.836 |
  | 0.001 | 5.12e-4 | 2.55e-4 | 1.83e-4 | 1.22e-4 | 0.667 | 0.834 |

  At δ = 0 the march *is* E1.2's algebra function by function, to 3e-15
  (P4 checked the weights only, §2.3); from δ/h = 0.1 down the distance is
  first order in δ/h to three digits (P4's weights: 11 %, 1.1 %, 0.11 %);
  the monomials stay O(1) away at every δ, the kink. On eq. 75's window
  (201 nodes, the edge on the node) the δ = 0 distance is 0.19–0.40 and does not shrink with δ: that is the
  `O(h α′/α)` gap between E1.2's degree-4-truncated pieces and the exact
  chain that P4 recorded in the weights (1.25, i.e. 125 %, at h = 0.01), so
  the figure
  is the MATLAB window's and the eq. 75 seeds are compared to the
  reference solution, not to E1.2, throughout §2.
- **The snapshot (second figure).** The ramp problem at t = 2 on 100 nodes
  (h = 0.0202) with δ = 0.0025, `h = 8δ`: the reference, the four nodal
  solutions (naive, the δ = 0 construction, T1-FV, the seeds) with a zoom
  at the edge, and the interior nodes' errors. The point of the grid is
  that the four regimes of §2 are in one picture: naive on its first-order
  line, the construction on its `c δ` floor, T1-FV on its second-order
  line, the seeds on the δ = 0 jump-aware line, and the errors are the
  sweeps' own numbers for this row (`snapshot` repeats them to 1e-12):

  | operator | ‖e‖₂/‖u‖₂ | max \|e\| | at x | share of ‖e‖₂² within 2h of the edge |
  | --- | --- | --- | --- | --- |
  | naive `Dx A Dx` | 2.04e-3 | 2.78e-3 | −0.030 | 0.16 |
  | δ = 0 construction | 7.12e-4 | 8.79e-4 | −0.010 | 0.23 |
  | T1-FV | 3.20e-5 | 2.16e-5 | −0.616 | 0.03 |
  | seeds | 2.95e-8 | 1.80e-8 | −0.394 | 0.07 |

  What it shows: the naive error is largest beside the edge (1.5 h to its
  left) and carries a node-to-node oscillation, the checkerboard mode of
  `Dx A Dx` excited by the sampled edge, but 84 % of its energy lies more
  than 2h away, on the α = 1/9 side where the profile is steep; the
  construction's error is smooth and global, the profile shifted by the
  resistance deficit `c δ` (§1.7) with its maximum at the edge; T1-FV's
  is largest far from the edge (its second-order truncation on the
  curved profile, the edge itself handled exactly by the face
  conductances); the seeds' 1e-8 is the BD4 time error at `dt = h`, the
  δ = 0 line's 2.95e-8 at this count, flat across the domain with two
  sign changes (the dips). The zoom makes the naive and construction
  errors visible to the eye at u ≈ 0.05: this is what an unresolved edge
  costs at fourth order's price.
- **The results file and `--data-dir`.** `results_cache.py` (plan D1;
  E5.3, #44) is now in the package: `ResultsCache(driver, args)` collects
  every table the driver prints (`add`) and its phases' seconds (`time`),
  and `write` puts `{"schema": 1, "driver", "date", "git": {"sha",
  "dirty"}, "args", "timings", "tables"}` at every path it is given, float
  keys as `%g` (`"0.0025"`), numpy as Python. The driver writes
  `outputs/heat1d_stiff.json` on every run and, with `--data-dir
  paper/data`, the same file there for the manuscript's number check; its
  tables are `references`, `knee/equilibrium`, `knee/ramp`,
  `floor_constants/<medium>`, `row_residuals/<medium>`,
  `weights_vs_jump/<medium>`, `seed_spectra/<medium>`,
  `comparators/equilibrium`, `comparators/ramp`,
  `widened_floors/<medium>/<δ>`, `seed_functions` and `snapshot`, each a
  list of row dicts or a medium → δ → rows map. Two caveats carried from
  §2.2 and §2.4: the knee cache is a working cache keyed by label and
  reference resolution and does not notice a change to an operator's
  construction or to the marcher (delete `heat1d_stiff_knee.json` after
  one); the results file is never read back by a driver, only by E5.3's
  scripts, and records `dirty: true` when the tree has uncommitted
  changes, which `paper_numbers.py` should refuse.

**What §2 states.** The four things #31 asks the section to say, with
their numbers and where they were measured:

1. **The knee (§2.2, P2 ✓).** Naive `Dx A Dx` on a smooth edge of width δ
   is first order in h while `h ≳ δ`, turns at `h ≈ δ` and is fourth
   order below, elliptic and parabolic, on both media: on the MATLAB ramp
   problem 1.8e-4 at h = 2δ to 1.5e-6 at h = δ/2 (δ = 0.01), 4.4e-5 to
   2.2e-7 (δ = 0.0025), a 100–200× drop across the knee, and on eq. 75
   5.7e-3 to 7.4e-6 (δ = 0.01), 1.0e-3 to 1.0e-6 (δ = 0.0025), 800–1000×.
   The δ = 0 construction (E1.2's rows at the edge centre) sits on the
   resistance-deficit floor `‖u₀ − u_δ‖/‖u_δ‖ = c δ`, `c = (a − b)
   ln(a/b)/(2ab) = 8.79` on the MATLAB medium, reproduced to six digits
   from the two quadratures, for `h ≳ 2δ` only (P3, corrected in §2.2):
   once the grid resolves the edge its rebuilt rows enforce a kink the
   solution lacks and its error grows to O(1) (at 1600 nodes and δ = 0.01,
   0.11 at equilibrium and 0.04 on the ramp problem), so it needs δ to be
   switched off. Both baselines are in the
   snapshot above.
2. **The seeds' rates (§2.3, P4–P9 ✓).** The seed operator is fourth
   order at every δ with a δ-independent constant: on the MATLAB ramp
   problem one line for every δ, rates 6.4, 4.2, 4.05 and 4.0–4.1 from 50
   to 800 nodes (2.5e-6 → 6e-12, then the reference's 2e-12 floor), with δ =
   0.0025 on the δ = 0 jump-aware line to 0.1–0.3 %, δ = 0.01 to 1–1.5 %
   and δ = 0.04 to 3–5 % where δ ≈ h and the solutions genuinely differ
   (P7 as measured: "one line", not "three digits at every n"). At
   equilibrium the seeds are exact to solver precision at every δ (P6:
   below 1e-12 to 400 nodes, then the direct solve's own growth). The
   seeds are E1.2's operator at δ = 0 on constant pieces (2e-14 in the
   weights, 3e-15 in the functions above) and need no δ to be switched
   off: §2.2's `h ≲ 2δ` regime costs them nothing. On eq. 75 the seeded
   rows are exact where the reach covers the layer and the line is the
   plain rows' sinusoid line, still pre-asymptotic at these counts (rates
   2.7 → 3.75 at δ = 0, 1.3e-4 → 1.4e-8 from 101 to 1601 nodes); with the
   edge rows seeded at δ = 0.04 the line is 2.6e-8, 1.5e-9, 1.1e-10 and
   the 2e-11 reference floor (rates 4.1, 3.8). Two floors to quote: the
   references' own error (2e-12 MATLAB, 6e-11 eq. 75, §2.1) and the DOP853
   march's at δ ≲ h/40 (row residual 1e-12, solution 4e-11, §2.3).
3. **The comparator ranking (§2.4, P10 corrected).** With the edge
   unresolved (h ≥ 4δ, δ = 0 included), on both media and both problems,
   in ‖e‖₂/‖u‖₂: *seeds < T1-FV < T1 (two cells) < naive ≳ T1 (one cell)
   ≈ T2 < T0 (m = 1) < T0 (m = 2)*; at 200 nodes and δ = 0.0025 on the
   MATLAB ramp problem 1.6e-9, 8.0e-6, 8.9e-5, 7.9e-4 / 4.0e-4 / 4.6e-4,
   2.2e-3, 5.0e-3. T1-FV, the conservative scheme with exact face
   conductances, is exact at equilibrium and second order at every δ with
   one constant (rate 2.00; 1.31e-4 → 1.26e-7 from 50 to 1600 nodes), the
   strongest low-order comparator; every nodal treatment (cell means,
   widened edge) caps the naive operator at second order once `h ≲ δ/4`
   and so shares the construction's need to know δ; T0 is the worst,
   sitting on the floor `c (m h − δ)`. Two caveats the manuscript's
   comparator paragraph must carry: on eq. 75 T1-FV is *ahead of the
   seeds* at δ = 0 below 801 nodes and at δ = 0.0025 at 101, where the
   seeds' sinusoid line is pre-asymptotic (item 2); and the two-cell
   harmonic mean's order 1.5 is a local O(h) defect on the two nodes
   beside the edge, first order in the max norm, so the norm is named
   wherever the ranking is quoted.
4. **The elliptic remark (§1.8, §2.2, §2.4; plan §3.2).** The 1-D
   equilibrium problem cannot rank the methods: its solution has constant
   flux and lies in `span{φ₀, φ₁} = ker L`, so the seeds are exact at any
   δ and any h (1.9e-14 at 50 nodes, δ = 0.01, growing with n to 1.7e-11
   at 1600 through the solver), and so is any finite-volume scheme with
   exact face conductances (T1-FV, 1e-14 → 6.5e-12). What the elliptic
   tables do show is the same knee for the naive operator (7.6e-3 at
   h = 4δ to 9.3e-9 at h = δ/8, δ = 0.01, MATLAB) and the construction on
   the same `c δ` floor, so the elliptic row of the knee figure is the
   sanity check and the limit case, one figure row and one remark; the
   ranking is the parabolic tables'. In 2-D the tangential variation takes
   the equilibrium solution out of the normal operator's kernel and the
   elliptic problem becomes a real test (E4).

**The predictions of §1.9, as they stand.**

| | Section | Status |
| --- | --- | --- |
| P1 the medium and the references | §2.1 | holds: δ = 0 bit for bit, references agree to 6.3e-11 (eq. 75) and 2e-12 (MATLAB) between resolutions |
| P2 the naive knee at h ≈ δ | §2.2 | holds, elliptic and parabolic, both media |
| P3 the δ = 0 construction on the `c δ` floor | §2.2 | holds for `h ≳ 2δ` with c = 8.79 to six digits; corrected: O(1) growth once `h ≲ 2δ` |
| P4 the seed weights → E1.2's at first order in δ/h | §2.3, §2.5 | holds (84 % … 0.11 %), and function by function to 3e-15 at δ = 0; eq. 75's `O(h α′/α)` floor recorded |
| P5 `cond A` Vandermonde-like | §2.3 | holds: 90 → 138 (peak at δ = h/2) → 23.5 |
| P6 equilibrium exact at every δ | §2.3 | holds below 1e-12 to 400 nodes; the march floor at δ ≲ h/40 added |
| P7 fourth order at every δ, one line | §2.3 | holds as "one line for every δ" (0.3 %, 1.5 %, 5 %), not three digits; local truncation O(h³) seeded / O(h⁴) plain |
| P8 the double-cross | §2.3 | holds: 7.6e-15 at δ = 0, first order in δ/h |
| P9 spectra real and negative | §2.3 | holds, and the seeds are stable on eq. 75 at 49 and 53 nodes where the construction is not |
| P10 the treatments' ranking | §2.4 | T1-FV exact and second order, T0's floor and constant hold; ranking corrected: T0 is the worst, T2 ≈ T1 (one cell) ≈ naive, T1 (two cells) ahead of naive; eq. 75 exception recorded |

**Limitations to carry into the manuscript.** The DOP853 march's floor
at δ ≲ h/40 (a collocation integration of the same chain would remove it;
not built); the eq. 75 seed line being the plain rows' pre-asymptotic
sinusoid line, so the medium's fourth-order claim rests on its δ > 0
rows and the MATLAB medium; the seed-function figure's `(α₀/α_e)^⌈k/2⌉`
normalisation, stated in its caption; the snapshot's grid chosen at
`h = 8δ` because it is the row of the sweep at which the four lines are
ordered and do not overlap, the naive operator and the δ = 0 construction
2.9× apart (near the knee those two are never more than about 5× apart,
which is what a knee is), T1-FV a further 22× below and the seeds 1000×
below that (at h = 2δ the naive operator is in §2.2's dip); and the knee
cache's blindness to operator changes.

Tests: `tests/test_results_cache.py` (`jsonable` on numpy, paths and float
keys; write-and-read round trip with the provenance, a second `add`
replacing the first, `read_results` refusing another file; `git_state`
naming this checkout and nothing outside one) and `tests/test_heat1d_stiff.py`
(the seed functions on E1.2's translated basis to 1e-13 at δ = 0 and first
order in δ/h down the ladder 0.1, 0.01, 0.001 with 5.12e-4 for φ₁, the
monomials on the node's side, the window's stencil units; the snapshot
repeating the sweep's four errors to 1e-12 with the ranking, the seeds
four orders below naive, the naive and construction errors mostly away
from the edge and the naive maximum beside it; `main` with `--data-dir`
writing the two figures and the same results file to both directories
with the schema's tables, timings and args).

**What this changes downstream.** E3 (#5) is complete: §2 is the
canonical 1-D account and every number the manuscript's §3 (E5.5, #46)
will quote is in it with a subsection to trace to; the figures it lifts
are the knee with the seeds (§2.3), the seed functions and the snapshot
(§2.5), the treatments (§2.4), all regenerated by one driver in seconds.
E5.1 (#42) can scaffold `paper/` against this section. E5.3 (#44) reads
`outputs/heat1d_stiff.json` into `paper/data/` and adds the content hash
the knee cache lacks. E4.10 (#41) gives `heat2d_stiff.py` the same
`--data-dir` and `ResultsCache`. Corners stay out (E2.10's decision).

## 3. Two dimensions: the design of the scalar seeds (E4.1, #32)

The design note for E4.2–E4.10, written before any 2-D seed code, as §1
was written before the 1-D code. It fixes the notation (the frame, the
anchor, the flux variables), writes the straight-feature chain out, says
how the seeds enter the 2016 stencil of `heat2d.interface`, decides the
curved and double-edged cases, and turns plan §3.3's elliptic questions
into the numbered hypotheses H1–H12 of §3.7, each with the experiment and
the ticket that answers it. §3.8 is for whoever implements E4.2–E4.10:
the decisions that are already taken, the traps §2 and the port found,
and where each piece goes. Nothing here is measured; §4–5 will tick the
hypotheses off as §2.5 ticked P1–P10.

### 3.1 Setting

The 2-D problems are the port's (port notes §2): the x-periodic unit
strip, Dirichlet rows at `y = 0` and `y = 1` (and on the cooling circle of
case 3), `u_t = ∇·(α ∇u) ≡ L u`, with α one smooth piece inside a closed
band between two interfaces and another outside it (`heat2d.domain.Band`).
Case 1 is the flat band `0.6 ≤ y ≤ 0.8` with `α = 0.2` inside and 1
outside; case 2 bends both interfaces into `c + 0.02 sin 2πx` and puts
`0.2 + 0.1 sin 2πx sin 2πy` inside; case 3 is the ring `0.349 ≤ r ≤ 0.35`
at contrast 1500 : 1, extremised by `s` in E2.9. The 2016 method
(dissertation §5.3, EABE §2.2.3–2.2.4, port notes §2.3–2.4) rebuilds the
rows of the 30-node / degree-4 stencils that cross an interface: the 15
monomials of the RBF-FD saddle-point system are replaced by the
translated basis, piecewise polynomials continuous in `u`, in the normal
flux and in `D^k u` and its flux along the interface, and the Gaussians
are warped so that their `α ∂_n` is continuous too. Everything is written
in a local frame at the foot point of the interface nearest the stencil
centre, `x′` along the tangent, `y′` along the normal into `level > 0`, in
units of the stencil radius.

**The smooth edge in 2-D.** E4.2 (#33) replaces each interface by a tanh
transition of width δ in the *signed normal distance* to the curve
(`Curve.signed_distance`, exact for the graphs through the foot-point
Newton iteration and for the circles), composed edge by edge exactly as
`heat1d.domain.SmoothEdges._blend` composes them: from the outside piece,
across the lower curve into the inside piece with weight
`s(d₁/δ)`, then across the upper curve back out with weight `s(d₂/δ)`,
`s(z) = ½ (1 + tanh z)`, so that a band thicker than its edges has two
independent transitions and a band thinner than its edge (case 3's ring
at δ > w = 0.001, §3.6) has the product profile `α_out + (α_in − α_out)
s(d₁/δ) (1 − s(d₂/δ))` whose peak is below the full contrast. On case 1
this is E3.2's 1-D medium in `y`, bit for bit, so the separable reference
`u = sin 2πx v(y)` of E4.2 is a 1-D problem in `y` with E3.2's α. δ = 0 is
the jump medium bit for bit, in `alpha`, `gradient` and the pieces'
`taylor` tables, so every 2016 driver runs on the smooth medium unchanged
at δ = 0. The smooth medium keeps the band's *protocol* (`region_index`,
`region_piece`, `interfaces`, `taylor`) with the jump's values: that is
what makes the δ = 0 construction below free.

**The three baselines, named as in §1.1.** *Naive* is `Dx A Dx + Dy A Dy`
with `A = diag α(x_j, y_j)` sampled at the nodes (plan D3;
`operators.naive_operator`, on §2.2's stencils, no interface group).
*The δ = 0 construction* is `interface_aware_operator` run on the smooth
medium: its crossing rows read the pieces' Taylor tables and the curves'
frames as if δ were 0, its direct rows read the smooth α, and it needs no
new code. *Seeds* is what E4.4–E4.5 build: the same operator with the
crossing rows, and every other row whose stencil sees an edge, rebuilt on
the seed basis of §3.2. The comparators of E4.9 are the disc means and
the widened edge of plan §3.4 (§3.8 lists them).

**What the seeds keep and what they change.** Kept, from the port: the
node sets with their straddling rows, the stencil groups (42 / 5 interior,
30 / 4 in the boundary zone and across interfaces), the Gaussian block
with `ε = 0.4/d`, the saddle-point solve `rbf.augmented_solve`, the
Dirichlet rows, BD4 at `dt = h` from the analytic history, SuperLU and the
iterative solvers of E2.8. Changed: the 15 augmenting functions of a
crossing stencil (seeds in place of translated polynomials), the warp's
coordinate (§3.4), and which rows are rebuilt (those that see the edge,
not only those that cross a curve). Nothing about hyperviscosity: the
diffusion operator is dissipative and BD4 implicit, so the companion's
§5.3 problem (the hyperviscosity row's footprint) has no twin here; what
replaces it is the implicit solve's conditioning (H5) and the spectrum's
right edge (H6).

### 3.2 The local frame, the anchor, and the straight-feature ansatz

**Frame and anchor.** A seeded stencil keeps the 2016 frame's orientation
(`interface.frame_at` at the foot point of the nearest interface, `x′`
tangential, `y′` normal into `level > 0`) and moves its origin to the
*evaluation node*, which lies on the normal through that foot point: the
seeds are anchored where the operator is evaluated, as in §1.2, and in
the stencil coordinate

    ξ = (x′ − x′_e) / h_s,    η = (y′ − y′_e) / h_s,    h_s = the stencil radius,

in which the chain is invariant (§1.3) and every seed is O(1) with
`φ ≈ ξᵃ ηᵇ` on constant α. The 2016 basis keeps its origin at the foot
point; the two are one span with a shift of origin between them
(`interface.frame_change` between two frames of the same angle is the
2-D `shift_matrix`), which is how E4.4 compares the seeds with E2.3's
translated basis in the jump limit (H2). The material along the normal,

    α_n(η) = α(x_e + h_s η n),   n the unit normal at the foot point,

is what the march samples; `α_e = α_n(0)` is the anchor value.

**The ansatz.** Where the material depends on the normal coordinate alone
(exactly on case 1's flat edges; locally, to the order §3.5 states, on a
curved one) the seed of the monomial `ξᵃ ηᵇ` is a polynomial in ξ with
coefficient functions of η,

    φ_{ab}(ξ, η) = Σ_j g_j(η) ξʲ,    j = a, a − 2, …, a mod 2,

because `L (g(η) ξʲ) = ξʲ L_n g + j (j − 1) α g ξ^{j−2}` with
`L_n = ∂_η α ∂_η`: the operator lowers the ξ-degree by two or not at all,
so only the levels of `a`'s parity below `a` are ever driven. The chain of
§1.2 read in 2-D is

    L φ_{ab} = α_e [ a (a − 1) φ_{a−2,b} + b (b − 1) φ_{a,b−2} ],

the constant-coefficient operator's action on the monomial with the
lower monomials' *seeds* on the right, frozen at the anchor, exactly
`L φ_k = k (k − 1) α_e φ_{k−2}` with the seeds of `x^{k−2}` on the right.
Collecting the coefficient of `ξʲ` gives, per seed and per level, one
second-order ODE in η that never differentiates α once it is written with
the flux variable `ψ_j = α g_j′`:

    g_j′ = ψ_j / α_n(η),
    ψ_j′ = α_e [ a (a − 1) g_j^{(a−2,b)} + b (b − 1) g_j^{(a,b−2)} ] − (j + 2)(j + 1) α_n(η) g_{j+2}^{(a,b)},

with the anchor data `g_a(0) = [b = 0]`, `ψ_a(0) = α_e [b = 1]` and every
other `g_j(0)`, `ψ_j(0)` zero: the monomial's jet in η at its top level,
and for `b ≥ 2` zero data with the source generating `ηᵇ`, as
`φ_k(x_e) = φ_k′(x_e) = 0` did in §1.2. The right-hand side of seed
`(a, b)` at level `j` needs the *same level* of the seeds `(a − 2, b)` and
`(a, b − 2)`, both of `a`'s parity, and the level `j + 2` of itself: the
system is triangular in total degree and in level, and all 15 seeds of
degree ≤ 4 march together as one linear first-order system of

    22 levels  (5 + 4 + 6 + 4 + 3 for a = 0 … 4),  44 states,

against about 600 for the companion's elastic seeds (plan §3.1's "about
70" counted every level below `a`; the parity trim halves it). With
constant α the solution is the monomial itself (`g_a = ηᵇ`, every lower
level zero, by induction on the degree as in §1.2), which is E4.4's first
check.

**Four seeds in closed form, and the one that is not a product.** The
linear seeds are the 1-D ones stretched along the tangent: `φ₁₀ = ξ`
through any edge (`ψ₁ ≡ 0` gives `g₁ ≡ 1`), `φ₀₁ = φ₁(η) = α_e ∫₀^η
dη′/α_n`, the constant-flux profile of §1.2, and `φ₁₁ = ξ φ₁(η)`;
`φ₀₂ = φ₂(η)` is the 1-D `x²`-seed. The first seed that is not a 1-D seed
times a power of ξ is

    φ₂₀ = ξ² + g₀(η),    (α_n g₀′)′ = 2 (α_e − α_n),  g₀(0) = g₀′(0) = 0:

`L ξ² = 2 α_n(η)` is not constant across the edge, and `g₀` is the normal
profile that restores `L φ₂₀ ≡ 2 α_e`, the "`u_t = const`" condition of
§1.2 for a temperature quadratic *along* the edge. It vanishes on
constant α, is O(1) in stencil units across an unresolved edge, and is
what a 1-D construction cannot supply: the 2-D content of the seeds is in
the coupling of tangential monomials to normal profiles, and E4.6's
elliptic test (plan §3.2) is a test of exactly these seeds, since
`sin 2πx v(y)` is not in the span of the 1-D seeds at any resolution.

**A symmetry that tests the march.** The frozen-α problem is invariant
under shifts along ξ, so the seed of `(ξ + c)ᵃ ηᵇ` anchored at the same
node is the binomial combination of lower seeds, and matching powers of
`c` gives

    g_j^{(a,b)}(η) = C(a, j) · g₀^{(a−j, b)}(η)   for every level j,

with `g₀^{(m,b)} ≡ 0` for odd `m`. So only the nine level-zero functions
`g₀^{(m,b)}`, `m ∈ {0, 2, 4}`, `m + b ≤ 4`, are independent (18 states),
and the 44-state march must reproduce the identity to rounding: an
E4.4 test that costs nothing and catches a wrong factor in the level
coupling, the twin of the companion's injected-error test (its §4.2).
The 44-state form is the one to implement, since it is the one that
generalises to a tangentially varying α (§3.5) and to the elastic case.

### 3.3 Numerics of the march (for E4.4)

Everything §1.3 and §2.3 settled in 1-D carries over unchanged; the
differences are in the bookkeeping of a scattered stencil.

- **One march per stencil, both directions from the anchor**, the state
  `(g_j, ψ_j)` for the 22 levels, DOP853 at `SEED_RTOL = 1e-13`,
  `SEED_ATOL = 1e-15` in the stencil coordinate (the 1-D constants,
  re-used, not re-tuned), restarted at every node's η so that node
  values are integrated and never interpolated (the companion's choice;
  DOP853's dense output is one order below its step and was not trusted
  at 1e-13), and at every edge centre and its `±EDGE_STOP δ = ±10δ`
  flanks along the normal. On case 1 the stops are the two lines
  `y = 0.6, 0.8` read in the frame, `η_c = (y_c − y_e)/h_s` (`cos θ = 1`);
  for a curved interface they are the crossings of the normal line with
  each curve, by Newton (the companion's `crossings`), and both curves
  of a band are stops of the same march whatever the anchor (§3.6).
- **α on each segment from the smooth medium's normal profile**, and at
  δ = 0 from the piece of the region the segment lies in, so the
  evaluation is one-sided at a jump and the march reproduces E2.3's
  algebra (H2) the way the 1-D march reproduced E1.2's to 2e-14 (§2.3).
  The 2-D medium therefore needs the "smooth piece per region" contract
  of `Medium1D.elements`: `region_piece(j).alpha` on a segment, never the
  blend, when δ = 0.
- **Batching.** The 1-D trick (§2.3: every row with the same node pattern
  in one `solve_ivp` system) has one exact 2-D use: the nodes of one
  straddling row share `y_e` and, on a flat edge, the same α_n, so their
  stencils' seeds are the same functions of (ξ, η) and one march with the
  union of their η targets serves the row (the targets are exact, not
  interpolated). Free nodes have their own `y_e` and march alone;
  stacking different anchors into one system is possible (block
  diagonal, the RMS-norm dilution of §2.3) and not worth it at 44 states.
  Cost estimate: 44 states, a few hundred steps, a few milliseconds per
  stencil; the interface group is 414–2294 rows at 1250–40,000 nodes on
  case 1 (port notes §2.3), so an operator's seeds cost seconds, against
  the 1.5 ms per row the 2016 rows cost already. #35's "milliseconds per
  stencil" is the acceptance line.
- **Which stencils are seeded** (the E3.4 breadcrumb): those with a node
  within `TANH_REACH δ = 20δ` of an edge centre in signed normal distance,
  for either curve; at δ = 0 that is exactly the crossing test
  `operators.interface_crossings`. The stencil *groups* follow the same
  rule one size up, as `build_stencils` already does with the 42-node
  crossing test: a node whose 42-node stencil sees the edge joins the
  interface group and gets the 30 / 4 stencil, so that no 42 / 5 stencil
  ever sees an unresolved edge, and of the group the members whose own 30
  nodes see it are seeded, the rest keep the direct row. A boundary-zone
  node whose one-sided 30 / 4 stencil sees an edge is seeded on its own
  nodes; the chain is anchored at the node and marches only where nodes
  are, so one-sidedness costs nothing. At δ = 0.04, `20δ = 0.8` covers
  the strip and every row is seeded (§1.3's last paragraph; P4 says it
  costs accuracy nothing, only the marches' time, about 40,000 × a few
  ms at the largest count); a `--seed-reach` knob for the resolved end of
  the sweep is E4.6's call, with the resolved-edge penalty measured
  either way (H8).
- **Normalisation and conditioning.** The seeds carry `α_e^{⌈(a+b)/2⌉}`
  through the chain constants, as in 1-D (§1.2, §2.5), and across a
  9 : 1 or 1500 : 1 contrast the normal profiles are O(contrast) in
  stencil units: the seed block's condition number sees the scaling and
  the weights do not (the moment conditions are homogeneous in it).
  Report `cond` of the 30 × 15 seed block raw and column-scaled next to
  the polynomial block's (32–34 in the companion's case; the 2016 basis
  here has the continuity matrices' 25–223, port notes §2.3), and expect
  the companion's picture (H3): within a small factor of the polynomial
  block at every δ, tending to it as δ → 0.
- **The floor.** The 1-D march's floor at `δ ≲ h/40` (row residual 1e-12
  in units of h⁻², solution 4e-11, §2.3) is the same integrator on the
  same chain and will be here too; case 3's ring at s = 10³ with
  δ = 0.0025 on 40,000 nodes is at `δ/h = 0.5`, well above it, and the
  extremised ring of E4.8 is where it may show. Below about `1e-2 h`
  dispatch to the jump construction, as §1.4 says.

### 3.4 The seeds in the RBF-FD system, and the warp they bring with them

**The saddle-point system.** EABE eq. 2 with the seed block in place of
the polynomial block (`rbf.augmented_solve` takes any `P`; E2.3 already
passes the translated block through it):

    [ A   S ] [ w ]   [ (L G_j)(x_e) ]
    [ Sᵀ  0 ] [ λ ] = [ (L φ_e)(x_e) ],     S_{ie} = φ_e(ξ_i, η_i),

`A` the Gaussian block in the (warped) coordinates below, the 30 nodes'
seed values read off the march, and the right-hand side the *true*
operator applied to each function at the anchor. For the seeds that is
the chain's constant term plus one correction:

    (L φ_e)(x_e) = 2 α_e [e ∈ {(2,0), (0,2)}] + h_s α_ξ [e = (1,0)]    (stencil units),

`α_ξ` the tangential derivative of α at the anchor. The chain gives the
first term (`a (a − 1) φ_{a−2,b} + b (b − 1) φ_{a,b−2}` at the anchor is
nonzero only for the two quadratics, where the lower seed is `φ₀₀ = 1`);
the second is what the frozen normal profile leaves out, since
`∂_ξ (α ∂_ξ φ) = α_ξ φ_ξ + α φ_ξξ` and at the anchor `α = α_e` exactly
while `φ_ξ ≠ 0` only for `φ₁₀ = ξ`. The normal derivative `α_η` never
appears: the seeds satisfy `∂_η α_n ∂_η` exactly with the true profile.
On case 1 `α_ξ = 0`; on case 2's inside piece it is the sine product's
tangential gradient, rotated into the frame as `stencil_weights` already
rotates `∇α` (`Frame.rotate_in`). So the seeds carry the operator's
`∇α · ∇` term through the ODE and need it at the anchor only once, in
one entry, where the 2016 rows apply `α ∇² + ∇α · ∇` to all 15 functions.
The weights come back as `w = w̃ / h_s²`.

**The Gaussian part, and the smooth warp for free.** The E2.4 breadcrumb
on #32 conjectured that the seeds' warp is the quadrature
`η̃(η) = ∫₀^η α_e / α_n dη′`, continuous through the edge and equal to
`interface.Warp`'s piecewise-linear stretch at δ = 0. That integral is
the seed `φ₀₁` itself. So the warped coordinate of a seeded stencil is

    (ξ, η̃) = (ξ, φ₀₁(η)),

read off the same march at every node, no `region` and no second
quadrature; at δ = 0 on constant pieces it is `Warp.apply` bit for bit
(slope 1 on the anchor's side, `α_e / α_across` beyond the jump,
continuous at it), and at δ > 0 it is the smooth stretch. The Gaussian
block is `G(ξ_i − ξ_j, η̃_i − η̃_j)` as in E2.4, and its right-hand side is
`L` applied to `G(ξ, φ₀₁(η))` at the anchor by the chain rule. Written out
with `η̃′ = α_e / α_n`, at any point of the normal line

    L G(ξ, η̃(η)) = α_ξ G_ξ + α G_ξξ + ∂_η(α η̃′) G_η̃ + α η̃′² G_η̃η̃,
    α η̃′ = α_e  ⇒  ∂_η(α η̃′) = 0,

so the term in `G_η̃` that the 2016 rows carry as `α_η G_η` (the
`g_eta * dy` of `stencil_weights`) is cancelled *exactly* by the warp's
curvature `η̃″(0) = −α_η / α_e`, and at the anchor (`η̃′ = 1`)

    (L G)(x_e) = α_e (G_ξξ + G_η̃η̃) + α_ξ G_ξ,

the plain Gaussian's Laplacian at the warped offsets times `α_e`, plus
the same tangential correction as the seeds'. E4.5 should implement the
general chain-rule form and assert the cancellation, not assume it; the
piecewise warp at δ = 0 has `η̃″ = 0` on the anchor's side and keeps the
`α_η G_η` term, and the two forms agree there because `α_n` is the
anchor's piece. This is the "smooth warp" of the ablation twin of Fig. 11
(H7): *seeds + φ₀₁-warp* against *seeds + plain Gaussians*, with E2.4's
2.3–6.9× on case 1 at δ = 0 as the expectation for what the warp is worth
and E2.9's finding that on the ring the warp decides the *sign* of the
spectrum as the reason not to run the seeds plain there.

**The jump limit is E2.3's stencil.** At δ = 0 on constant pieces across
a flat interface the seed of `ξᵃ ηᵇ` is the monomial on the anchor's
side and, on the far side, a polynomial of degree ≤ 4 (the chain with
constant coefficients and polynomial sources) that is continuous with
its flux at every ξ and, by induction down the chain, has `D^k φ` and
its flux continuous too: the 15 conditions the continuity matrices
impose, all exact for constant pieces on a flat line. `C` being square
and nonsingular, the far side is E2.3's translated polynomial, the two
15-dimensional spans coincide, and with the same Gaussian block (warp on
both, or off both) the weights agree to rounding (H2): the 2-D form of
"one construction, two implementations" (§1.4), with the three-region
stencils of a thin band included since both curves are stops of one
march. On a *curved* interface E2.3 carries the curvature through the
expansion `f` and the seeds of §3.5 do not, so they differ at O(κ h_s);
on case 2's tangentially varying inside piece E2.3 truncates α's Taylor
table at degree 4 where the seeds freeze it along the normal, §1.4's
`O(h α′/α)` remark. So the rounding-level agreement is case 1's, and on
cases 2 and 3 E4.4 records the distance and its order instead of
claiming a limit, exactly as §2.3 did for eq. 75.

**Stencil size: 30 / 4, and why the companion's 19 / 3 warning does not
carry over.** The companion found degree-4 seed stencils unstable across
a sharp edge in the wave case, with hyperviscosity, and settled on
19 nodes and degree 3 (its §5.3). The 2016 diffusion method uses 30 / 4
across the jump in every experiment and the port reproduces its fourth
order with them (port notes §2.3–2.9), the seed rows tend to those rows
as δ → 0 (H2), and there is no hyperviscosity row whose footprint could
be wrong. The seed stencils are therefore 30 / 4, the interface group's
spec, and the spectrum study of H6 is where a surprise would show.

### 3.5 Curved features: route (a), and what it leaves out

Plan §3.1 and the companion's §4.1 name three routes for a curved edge:
(a) the straight-feature seeds along the true normal through the foot
point, curvature dropped; (b) a local boundary-value problem per stencil
on a fine patch (the multiscale / oversampling construction); (c) global
harmonic coordinates. The companion built (a), measured it on the same
sine pair at amplitude 0.02, and found route (b) unnecessary on these
node sets (its §5.6.2). E4.7 (#38) takes route (a) first and measures the
same things.

**Route (a) here.** The frame is E2.3's at the foot point of the nearest
curve, the anchor the evaluation node on its normal, the profile `α_n`
sampled along that normal with the stops where the line meets each curve
and its images (Newton; the flat formula kept for `θ = 0` so that case 1
is bit for bit the flat march). Along that line the blend is exactly the
medium's tanh in the normal distance of the curve the line is normal to,
so a curved-edge stencil's seeds are the *flat* seeds of its local
coordinates (the companion's `test_curved_seeds_are_the_flat_seeds_of_
the_same_local_stencil`, 1e-12): nothing in §3.2–3.4 changes.

**Two things it gets wrong, with their sizes.** A node at tangential
offset `x′` sits at true normal distance `y′ − κ x′²/2 + O(κ²)` from the
curve while its seed value assumes `y′`; over a 30-node stencil of radius
`r ≈ √(30 / πN)` on the case-2 curves, `κ ≤ 0.02 (2π)² = 0.79` (tilt to
7.2°), the geometric error `κ r²/2` at the outer nodes is

    N        1250    2500    5000   10,000  20,000  40,000
    r        0.087   0.062   0.044   0.031   0.022   0.015
    κ r²/2   3.0e-3  1.5e-3  7.5e-4  3.8e-4  1.9e-4  9.4e-5

to be read against δ: larger than a δ = 0.0025 edge at 1250 nodes,
0.3 δ at 5000 and 0.075 δ at 20,000, and 0.075 δ at 5000 for δ = 0.01.
It shrinks with N only as h², so route (a)'s geometry shows first at the
sharpest δ on the coarsest sets, and where it shows the seed value at an
outer node is off by that fraction of the edge profile. The companion
saw exactly this pattern: at the seed floor through δ = 0.005 and 0.01
at every N, and 1.4–1.5× the flat seeds' error at δ = 0.0025 on its
10,000- and 19,600-node sets. The second thing: case 2's inside piece
`0.2 + 0.1 sin 2πx sin 2πy` varies along the tangent, the frozen profile
does not, and §3.4's anchor correction `α_ξ` restores consistency of the
moment conditions at the anchor only; away from it the seeds solve the
wrong problem by `O(x′ α_ξ)`, first order in `h_s` with a fixed constant.
By §1.4's argument (the spaces differ at O(h), the annihilator's
coefficients at O(1) on lower-order terms, the local error stays
`O(h³)`) the order survives and the constant changes; the 2016 rows,
which carry α's full Taylor table, do not have this term. It is the
diffusion twin of the companion's oblique-incidence question and is
measured, not predicted, in H9.

**If route (a) is not enough.** The first step is not route (b) but the
companion's route (a′): keep the marches, evaluate each seed at the
node's true `(arclength, normal distance)` instead of its tangent
coordinates, and scale the tangential jets at the anchor by
`1/(1 − κ y′_e)`, which makes the material exact at every node and
leaves only the operator's curvature terms (`κ ∂_n` and the metric of
the tangential derivative) as the error, at no march cost. For the
tangential variation of α the matching step is the ξ-Taylor ansatz: with
`α = Σ_m a_m(η) ξᵐ` the levels couple through `a_m` (`ξʲ` is driven by
`g_{j+2−m}` and by `∂_η(a_m ∂_η g_{j−m})`), the system is no longer
triangular but is still 44 linear ODEs in η, truncated at the operator's
consistency order as `interface.coefficient_operator` truncates. Neither
is built unless H9's residual probe asks for it; route (b) stays named
and unbuilt, with the companion's reasons.

### 3.6 The double-stiff ring, and the extremizing sweep with seeds

**Two edges in one march.** Case 3's ring is 0.001 wide and thinner than
the node spacing at every count (`h ≥ 0.0026` at 160,000 nodes; port
notes §2.7), its straddling rows sit `±0.5 h` off the midline and no
node lies inside it. A stencil crossing the midline reaches regions 0
and 2, and E2.3 translates it across both circles with a frame change
between them (port notes §2.3). The seeds have no frame change and no ring
polynomial: the normal through the foot point of the *nearest* circle
crosses both, both crossings and their `±10δ` flanks are stops, and one
march carries the 44 states through the ring, the diffusion twin of
§1.3's double-cross and of P8 (7.6e-15 at δ = 0). For the concentric
circles the normal is radial and the crossings are exact; the seeds'
march parameters are carried as *widths* `(w, δ)` from the outer radius,
never as two absolute radii, because `Circle(0.35 − 1/s)` stores the
ring's width to only 8e-8 relative at `s ≥ 10¹⁰` (E2.9's breadcrumb on
#39, item 3).

**The smooth ring.** With δ of the order of `w` or above, §3.1's product
profile `α_out + (α_in − α_out) s(d₁/δ) (1 − s(d₂/δ))` never reaches the
ring's plateau: the effective contrast is reduced and the "ring" is a
resistive bump of height `(α_in − α_out) tanh(w/2δ)` to leading order.
That is a different problem from the jump ring, on purpose (it is what a
sub-grid smooth layer *is*), and the notes must say at each δ what the
bump's contact resistance `∫ dr/α` across it is, since that, not the
plateau, is what the solution feels (port notes §2.9: the ring's resistance 1.5 at
every s is what the s-sweep holds fixed).

**What the s-sweep asks of the seeds.** Port notes §2.9 found that the
continuity matrices' `O(s²)` (Fig. 20) is row scaling a pivoting solve
never sees (16.5 row-equilibrated at every s), and that what does grow
into the 2016 weights is the far side's translated basis, `O(s κ² scale)`
from the ring polynomial's normal terms coupling through the curvature
and the frame shift, with the weights' residual on the matched radial
quadratic growing like `1.4e-18 s` at the worst stencil. The seeds have
no such intermediate. What they carry instead is physical: across a
resistive ring the constant-flux seed `φ₀₁` climbs by `α_e w / α_ring`,
i.e. `O(s w / h_s)` in stencil units (about `2e9` at `s = 10¹¹`), in one
column of `S`, and the seeds built on it (`φ₁₁`, `φ₂₁`, `φ₀₃`, …) the
same. So the seed block's raw condition number should grow like
`s w / h_s`, a *column* scaling, and be O(1) column-equilibrated (H11);
the march itself is exact in the `(g, ψ)` form, `ψ` continuous and
`g′ = s ψ` a scale, not a stiffness; and the number to report is E2.9's
own, the worst and median relative residual of the seed weights on the
matched radial quadratic per s (`matched_residual` in
`scripts/heat2d_extremes.py`, formed from the stored radii), next to
port notes §2.9's `1.4e-18 s` line. With a smooth ring the matched profile is the
radial solution of `∇·(α ∇u) = 4` regular at the centre,
`u′ = 2r/α(r)`, `u = ∫ 2r/α dr`, one quadrature with the smooth `α(r)`:
the reference-free instrument of E4.8 at any `(s, δ)`.

**The reference problem, stated now.** Fig. 19's twin at δ = 0 needs no
new reference: the per-s 160,000-node jump-aware runs of E2.9 are cached,
and the seeds must sit on E2.9's line (every s the s = 10³ line to
0.73–1.15×). At δ > 0 there is *no independent reference for the smooth
ring*: it is not separable, and a Fourier × Chebyshev product grid would
have to resolve a 0.0025-wide tanh at r = 0.35 across the whole strip. A
self-convergence line against a fine seed run (160,000 nodes, `h ≈ δ`)
through the resampling of E2.6 is what can be had, and it measures the
seeds against themselves. So E4.8's evidence for δ > 0 is the matched
radial residual (exact) first and the self-convergence line second, and
if the fine run is too costly the ticket scales back to the δ = 0 sweep
with seeds plus the residual at every `(s, δ)`, which already answers
"does the breakdown move and is the march the new limit". Brad decides
at E4.8; the plan's "Fig. 19–20 twins with a seeds line" reads as
"Fig. 20's twin at every δ, Fig. 19's at δ = 0" until then.

### 3.7 Hypotheses for E4.4–E4.9, with the experiment that tests each

Plan §3.3's elliptic questions and §3.2–3.6's predictions, numbered so
§4–5 can tick them off as §2.5 ticked P1–P10. "Reference" means the
separable Chebyshev reference of E4.2 on the flat cases, the cached
160,000-node jump-aware runs at δ = 0 on the curved ones, and the matched
radial quadratic on the ring.

| | Hypothesis | Experiment | Ticket |
| --- | --- | --- | --- |
| H1 | **The march is the chain.** Constant α returns the 15 monomials to rounding; the shift identity `g_j^{(a,b)} = C(a,j) g₀^{(a−j,b)}` holds to rounding through a δ = h/8 edge; `L φ_e − α_e Σ C φ_{e′}` evaluated on a tensor grid through the edge with high-order finite differences (the companion's injected-error lesson: no coefficient bookkeeping shared with the march) is ≤ 1e-10 relative; one stencil marches in milliseconds. | `tests/heat2d/test_seeds.py` at δ ∈ {h/8, h, 8h}; the residual grid 25 × 801 | E4.4 |
| H2 | **The jump limit is E2.3.** At δ = 0 on case 1 the seed span equals the translated basis's span (each column in the other's span to 1e-12, three-region stencils included) and the stencil weights agree to rounding with the same Gaussian block; at δ > 0 the distance is first order in δ/h (P4's ladder, 84 % … 0.11 % in 1-D at δ/h = 1 … 0.001; the companion's 2-D 2.3e-2 → 2.4e-4 at δ = 1e-3 → 1e-5 with h = 0.05). On case 2 the distance saturates at O(κ h_s) and O(h α_ξ/α), recorded, not claimed as a limit. | seed block vs `InterfaceStencil.basis`, both re-centred; `stencil_weights(warp=False)` vs the seed solve | E4.4, E4.5 |
| H3 | **Seed blocks condition like polynomial blocks** (P5's twin): the 30 × 15 block's condition number column-scaled within a small factor of the constant-α polynomial block's at every δ from 1e-5 h to 8 h, tending to the translated block's as δ → 0; raw, it carries the `α_e^{⌈k/2⌉}` and contrast scaling. | table per δ/h on one real stencil of each case | E4.4 |
| H4 | **The seed operator is the aware operator at δ = 0 and the direct operator at δ ≫ h** (P4, P7 in 2-D): on case 1 at δ = 0 the seed operator's elliptic and parabolic errors equal port notes §2.5's warped line (1.60e-5 → 5.28e-9, fit 4.77) to rounding; at δ = 0.0025, 0.005, 0.01 the lines are fourth order with a δ-independent constant and lie on the δ = 0 line to a few per cent where δ ≪ h; the elliptic and parabolic lines coincide at `dt = h` as port notes §2.5's do. The elliptic line is a real test (plan §3.2): `sin 2πx v(y)` is outside the 1-D seeds' span, so the error is O(h⁴), not zero. | the flat sweep, every (n, δ) | E4.6 |
| H5 | **Global solvability does not degrade as δ → 0** (plan R3): the seed rows' diagonal dominance (least and median DDR) and the reduced system's `gmres` / `bicgstab` iteration counts, with and without `spilu` and Appendix B's `P`, are monotone in δ/h between the direct rows' values (δ ≫ h) and the jump-aware rows' (δ = 0: least 0.08, median 0.61 on the 42 / 5 + 30 / 4 operator; bicgstab 98–319 iterations on case 3, gmres 1.1–1.3× the control's), with no breakdown; SuperLU's solve time and residual do not move with δ. If this fails at some δ/h the plan is revised before E4.6 (#36's done-when). | DDR and iteration tables per δ/h ∈ {8, 1, 1/8, 1/64, 0} on case 1 at 10,000 nodes and case 3 at 20,000 | E4.5 |
| H6 | **Spectra** (P9's twin): the seed operator's eigenvalues at 4900 nodes on case 1 sit where the warped aware operator's do (max Re −7.27, `h² min Re` −13.2, `h² max |Im|` ≤ 0.4, every eigenvalue within ±700 of the origin real), at every δ; the slowest mode is the physical −7.27 and BD4's largest root modulus 0.897 at `dt = h`; no positive eigenvalue with the warp on. With plain Gaussians the crossing rows' complex loop of port notes §2.5 (`h² max |Im|` 1.49) returns and on the ring positive eigenvalues may (E2.9: 50–87 at s = 10¹¹ on 1250–4000 nodes). | `interior_eigenvalues` per operator and δ, the Fig. 5-6 twin with a seeds panel | E4.5 |
| H7 | **The warp is worth on seeds what it is worth on polynomials** (the E2.4 breadcrumb): seeds with the `φ₀₁`-warp against seeds with plain Gaussians on case 1, 2.3–6.9× at 1250–20,000 nodes at δ = 0 (E2.4's numbers, since H2 makes the two operators equal there) and a comparable factor at δ > 0; the chain-rule right-hand side equals `α_e Δ_{ξη̃} G + α_ξ G_ξ` at the anchor to rounding; at δ = 0 the warp coordinate equals `Warp.apply` bit for bit. | the Fig. 11 twin with two seed lines; a unit test on the cancellation | E4.5, E4.6 |
| H8 | **The rule needs no δ**: seeded rows are those that see the edge (reach 20δ), the seed operator needs no threshold to be switched off, and the resolved-edge penalty at δ = 0.04 (every row seeded) is ≤ 1.2× the direct operator's error (P4: the seed weights tend to the standard ones as δ/h grows; §1.4's different-space remark bounds the rest). | the δ = 0.04 column of the flat sweep, seeded vs direct | E4.6 |
| H9 | **Route (a) reproduces the flat numbers** at δ ≥ 0.005 for n ≥ 5000 and shows its geometric floor `κ r²/2` (§3.5's table) at δ = 0.0025 on the coarse sets as a factor ≤ 1.5 above the flat line; the seed rows' residual on the true curved solution (E4.7's probe) converges at the bulk rows' rate; the tangential-α term of case 2's inside piece changes the constant, not the order. Route (a′) is built only if the probe's residual stalls. | the curved sweep against the flat one at equal δ; the residual probe | E4.7 |
| H10 | **The 2-D knee** (P2's twin, measured first): naive `Dx A Dx + Dy A Dy` on scattered nodes is first order while `h ≳ δ` and fourth order once `h ≲ δ`, elliptic and parabolic; the δ = 0 construction sits on an O(δ) floor (the two references' difference, exact from the separable solves) for `h ≳ 2δ` and grows once the grid resolves the edge (§2.2); what separates resolved from unresolved most sharply is the flux jump across the edge read from the discrete solution. Watch the naive operator's coarse-set growing mode (+847 at 900 nodes, +17.7 at 1250) before quoting a parabolic naive number. | the naive and construction lines of the flat sweep | E4.3, E4.6 |
| H11 | **The ring**: the seed march through both edges reproduces E2.9's s = 10³ line at δ = 0 (the Fig. 19 twin) and the matched radial residual stays at or below port notes §2.9's `1.4e-18 s` worst-stencil line at every s, with no `O(s κ² scale)` term; the raw seed block conditions like `s w/h_s` (one column) and O(1) column-scaled; the march floor of §3.3 is the first limit to appear, at the largest s and smallest δ, and the stored width's 8e-8 the second. | `matched_residual` per (s, δ); Fig. 20's twin with a seeds line | E4.8 |
| H12 | **The comparators** (P10's twin): the disc harmonic and arithmetic means and the widened edge cap the naive operator at second order once `h ≲ δ/4` and are first order while the edge is unresolved; T0 is the worst, sitting on the widened floor; there is no conservative scheme on scattered nodes, so T1-FV has no twin and the strongest low-order comparator is the two-cell disc harmonic mean; the seeds are 3–4 orders below every treatment at δ ≤ h/4 on the parabolic problem. The ranking is quoted in the RMS norm with the max norm beside it. | the comparator tables per δ, elliptic and parabolic | E4.9 |

### 3.8 For the implementer of E4.2–E4.10

Written for whoever picks up the next tickets (Brad expects a different
model to), so that the decisions above are read as decisions and the
traps §2 and the port paid for are not paid for twice.

**Checked in scratch, 2026-09-22** (a 190-line script marching the
44-state chain of §3.2 with `solve_ivp`, not committed; E4.4 reproduces
every line in `tests/heat2d/test_seeds.py`): constant α returns the 15
monomials to 1.3e-15; the shift identity holds to 2.2e-16 through a
tanh edge half a stencil radius from the anchor at δ = h_s/24 (δ = h/8),
where the largest seed is 10 in stencil units for the 1 : 0.2 contrast;
the PDE residual `L φ_e − α_e Σ C φ_{e′}` on a 25 × 1601 tensor grid with
eighth-order differences in η is 2.2e-10 relative, the differences'
own floor; `α η̃′ = α_e` holds to 1.1e-16 with `η̃ = φ₀₁`, the chain-rule
form of `L G` on the normal line agrees with `α G_ξξ + α_e η̃′ G_η̃η̃` to
4.5e-12 and with `α_e Δ G` at the anchor to 1e-10; and on a real 30-node
stencil of the 2500-node case-1 set at δ = 0 (anchor at y = 0.5896, the
edge 0.17 radii away, regions 0 and 1) the seed block's columns lie in
the translated basis's span and conversely to 3e-16, the weights of the
seed solve equal `stencil_weights(warp=False)` to 5.1e-13 relative, and
the condition numbers are 176 (E2.3's block), 212 (seeds, raw) and 142
(seeds, columns scaled). One stencil's march, restarted at all 30 node
η's: 3.2 ms at δ = 0, 12 ms through the tanh edge. So H1, H2 (its δ = 0
half) and H7's cancellation are established on one stencil before any
package code exists, and #35's "milliseconds" is met with a margin.

**Decisions taken here, not to be re-derived.**

1. The smooth medium keeps the `Band` protocol with the jump's
   `region_index`, `region_piece`, `interfaces` and `taylor`, and adds
   the blended `alpha` and `gradient` (§3.1). Consequence: naive, the
   δ = 0 construction and every 2016 driver run on it unchanged, and
   δ = 0 is the jump bit for bit. Name it as E4.2 likes
   (`SmoothBand(band, delta)` is the obvious one); give it
   `normal_profile(j, x, y)` returning the callable `α_n(η)` and the
   stops for a stencil anchored at `(x, y)` on the normal of interface
   `j`, in stencil units, and at δ = 0 the piece-per-segment rule.
2. The seed frame is E2.3's orientation with the origin at the
   evaluation node, in units of the stencil radius; the march is in
   `(ξ, η)`; the α_e normalisation stays in the seeds; `cond` is
   reported raw and column-scaled (§3.2–3.3).
3. The 44-state chain with the flux variables, DOP853 at the 1-D
   tolerances, restarted at every node η and at the centres ± 10δ; one
   march per stencil, the straddling rows batched by shared `y_e` if it
   ever matters (§3.3).
4. The right-hand side of the seed block is `2 α_e` on the two
   quadratics plus `h_s α_ξ` on `ξ`, nothing else (§3.4). Test it against
   `stencil_weights`'s polynomial right-hand side at δ = 0, where the two
   must agree.
5. The Gaussian coordinate is `(ξ, φ₀₁(η))`, its right-hand side the
   chain-rule form with the cancellation asserted, `warp=False` the plain
   ablation (§3.4). At δ = 0 assert equality with `Warp.apply`.
6. Seed stencils are 30 / 4 (§3.4). Seeded rows: the reach-20δ rule on
   the 30 nodes; the interface group: the same rule on the 42 nodes.
7. Curved edges take route (a) with the anchor correction, and (a′)
   before (b) if H9 fails (§3.5). The ring: widths, not radii; both
   circles as stops of one march (§3.6).
8. E4.8's δ > 0 evidence is the matched radial residual first; the
   self-convergence reference is Brad's call (§3.6).

**Traps carried over from §2 and the port, in the order they will bite.**

- *The naive operator's growing mode* on the coarse sets (+847 at 900
  nodes, +17.7 at 1250; port notes §2.2, §2.5): check
  `interior_eigenvalues(op, nodes).real.max()` before quoting a
  parabolic naive number at 1250 nodes (the E2.5 breadcrumb on #34).
- *`atol`, not `rtol`, trips a stiff integrator* on a reference whose
  solution is small (E3.2's Radau lesson, §2.1): the separable
  parabolic reference in `y` inherits E3.2's 1e-9 floor and its
  20-node / 0.1-split recipe.
- *The knee cache is blind to operator changes* (§2.2, §2.5): key the
  2-D operator cache on δ, n, seed, mode, warp and reach, and delete it
  after any change to the marcher.
- *`spilu` needs `MMD_AT_PLUS_A`* and *the iterative solvers see the
  reduced system*, never the identity-row form (port notes §2.8); the
  E2.8 breadcrumb on #36 has the machinery's names.
- *On the ring the warp decides the sign of the spectrum* (port notes
  §2.7, §2.9): never run the seeds plain on case 3 without looking at
  `max Re λ`.
- *E1.2's and E2.3's rows differ from the exact chain at O(h α′/α) on a
  smoothly varying piece* (§1.4, §2.5): on case 2 compare the seeds
  with the reference, not with E2.3, and record the distance.
- *The march floor at δ ≲ h/40* (§2.3) and *the stored ring width at
  8e-8* (port notes §2.9) are the two floors of E4.8; name which one a
  number sits on.
- *A blind read of a reference floors every curve at 1e-4* (port notes
  §2.6): any reference on another node set goes through the
  interface-aware resampling, and at δ > 0 through a seed-aware one
  (`interpolation_weights` with the seed block: the same system with
  the identity on the right, as E2.6 built for the jump).
- *Every 2-D error is the RMS over all nodes, Dirichlet rows included*,
  `h = 1/round(0.95 √N)`, orders per halving of h (port notes §2.10);
  the 1-D study's `‖e‖₂/‖u‖₂` is a different norm and the manuscript
  must not mix them in one table.

**Where the pieces go** (the plan's module names, CLAUDE.md's layout).

| Piece | Module | Ticket |
| --- | --- | --- |
| the smooth band, `normal_profile`, the Chebyshev separable references in `y` (elliptic and parabolic), `dirichlet_boundary` from the reference | `heat2d/domain.py`, `heat2d/exact.py` | E4.2 (#33) |
| the naive sweep, `--delta`, the knee table, the flux-jump diagnostic | `scripts/heat2d_stiff.py --mode naive` | E4.3 (#34) |
| the chain, `seed_profiles` (one stencil), `seed_basis` (values and the anchor right-hand side), the shift-identity and residual tests | `heat2d/seeds.py`, `tests/heat2d/test_seeds.py` | E4.4 (#35) |
| `seed_weights`, the `φ₀₁` warp, `build_operators(mode="seeds")` and the seeded-row rule, DDR / iteration / spectrum tables | `heat2d/seeds.py`, `heat2d/operators.py`, `scripts/heat2d_stiff_eigenvalues.py` | E4.5 (#36) |
| the flat sweep, cached operators, the rule and the resolved-edge penalty | `scripts/heat2d_stiff.py --mode seeds` | E4.6 (#37) |
| route (a) profiles by Newton on the sine curves, the product-grid reference for δ > 0, `--amplitude` | `heat2d/seeds.py`, `heat2d/exact.py`, the driver | E4.7 (#38) |
| the smooth ring, widths from the outer radius, `matched_residual` through smooth α, the s-sweep with seeds, `--ring` | `heat2d/domain.py`, `scripts/heat2d_extremes.py` | E4.8 (#39) |
| disc harmonic / arithmetic means (radius h/2, h), the widened edge, band-limited α if E5.2 keeps it | `heat2d/treatments.py` | E4.9 (#40) |
| the figures, §4–5 of this note, `--data-dir` and `ResultsCache` as in `heat1d_stiff.py` | `scripts/heat2d_stiff.py`, `docs/figures/` | E4.10 (#41) |

Every driver default runs in seconds (plan D7); the sweeps and the
references sit behind flags and cache under `outputs/`, and the notes
record the run times as §2 did.

### 3.9 References for §3

Those of §1.10, plus the companion's 2-D sections (its `docs/stiff-
features.md` §4.1–4.2, §5.3, §5.6; cited as the arXiv preprint in the
manuscript, never as files, plan D2), dissertation §5.3 and EABE
§2.2.3–2.2.4 for the frame, the continuity matrices and the warp, and,
for routes (b) and (c) of §3.5, the multiscale FEM of Hou & Wu and the
harmonic coordinates of Owhadi & Zhang named in plan R1; those two
enter `paper/references.bib` only through E5.2's verification, and
nothing in §3 claims novelty over them.
