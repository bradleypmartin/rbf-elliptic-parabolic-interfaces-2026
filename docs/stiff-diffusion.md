# Stiff smooth edges in diffusion: seed stencils for `∂ₓ α ∂ₓ` (E3–E4)

The sub-grid-edge study of this repo: a material edge of width δ smaller
than the node spacing h, and stencils on the unchanged grid that keep
fourth order through it because their polynomial basis is replaced by
*seeds*, functions continued through the edge by ODEs. Canonical for E3–E4
(plan D11); the manuscript quotes this note and never becomes a second
source of truth. §1 is the formulation (E3.1, #26); §2 holds the 1-D
results as the tickets land (E3.2, #27 onward, closed by E3.6, #31) and
later sections the 2-D design and results (E4.1, #32; E4.10, #41). The
port of the 2016 methods this builds on is in `docs/port-notes.md`.

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
posted with this ticket).

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
