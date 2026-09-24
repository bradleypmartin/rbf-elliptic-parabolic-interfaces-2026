# Plan: RBF-FD for elliptic and parabolic interface problems, a decade later

Scoped 2026-09-20. This is the working plan: what the 2016 work did, what is
new here, the decisions taken up front, the risks, and the epics and tickets
in dependency order. Section 6 is machine-readable: `scripts/publish_issues.py`
turns each `## E<k>:` heading into a GitHub epic and each `### E<k>.<n>` into
a ticket linked to it (dry run by default). Published 2026-09-20: the issue
number sits at the end of each heading (epics #2–#7, tickets #8–#52).

## 1. Purpose and deliverables

The elliptic and parabolic analogue of the 2026-09 wave-equation work: port
the RBF-FD heat-transfer methods of the 2016 dissertation (ch. 4–5) and of
Martin & Fornberg, EABE 2017 to Python, verify them against the published
figures, then extend them to the case neither paper touched, a material edge
that is smooth but too steep for the node spacing (a *sub-grid smooth edge*,
width δ), and write the whole thing up as an arXiv manuscript.

Deliverables, in order:

1. `src/heat_interfaces/` with tests: the 1-D and 2-D solvers, three stencil
   families (naive, jump-aware, seeds), the coefficient treatments.
2. Reproductions of the 2016–2017 figures (dissertation Fig. 4-5–4-7,
   5-5–5-6, 5-9–5-11, 5-14–5-18, 5-22; EABE Fig. 7, 10, 11, 14, 16–20) with
   the numbers recorded in `docs/port-notes.md`.
3. The stiff-edge study in 1-D and 2-D, canonical in `docs/stiff-diffusion.md`.
4. `paper/`: the manuscript, with a novelty ledger `LITERATURE.md`, a results
   cache and scripted figures, arXiv packaging.

Not a deliverable: a talk or slides. Nothing here feeds a deck.

## 2. What exists

Sources and where to read them are in `docs/paper-index.md`; checksums in
`papers/README.md`. In one paragraph each:

**1-D (dissertation ch. 4).** `u_t = (α u_x)_x` on [−1, 1]. Stencils that
cross an interface are built on a piecewise-polynomial basis: Taylor
coefficients on one side are translated to the other by continuity matrices
that encode continuity of every time derivative of u and of the flux α u_x
(k = 0, 1, 2 for u and k = 0, 1 for the flux give five rows for a fourth-degree
basis). A smoothly varying α enters through multiplication matrices on the
coefficient vectors. Test: α = 0.1 + 0.4 sin 2πx on [0, 0.5], 1 elsewhere,
equilibrium with u(−1) = 1, u(1) = 0. Naive FD4 must be assembled as
`Dx A Dx` (eq. 76) to see the jump at all; it converges at first order, the
translated basis at fourth (Fig. 4-7, 100–1600 nodes). The MATLAB in
`heatEq1DMatlab/` is the *time-dependent* single-interface version
(k = 1/9 | 1), not this equilibrium problem.

**2-D (dissertation ch. 5; EABE 2017).** `u_t = ∇·(α ∇u)` on the x-periodic
unit strip, Dirichlet at y = 0 and y = 1. Scattered nodes from repulsion with
two rows straddling each interface; Gaussian RBF-FD with ε = 0.4/d, 42 nodes
and degree-5 polynomials away from interfaces and boundaries, 30 nodes and
degree 4 across them. Interface stencils translate the monomials across the
interface using time derivatives of the first (p − 2k) tangential monomials of
u and the first (p − 2k − 1) of the normal flux after k applications of the
operator, with the curvature entering through a local expansion of the
interface and matrix-valued cos/sin of its angle. The RBFs themselves are
"warped" across the interface so the RBF part honours flux continuity to first
order. Three test cases: (1) flat layer α = 0.2 on y ∈ [0.6, 0.8], analytic
solution, elliptic and parabolic (BD4, dt = h) at fourth order, operator
spectrum vs the BD4 stability region; (2) two mildly curved interfaces,
α = 0.2 + 0.1 sin 2πx sin 2πy in the band, FD4 first order, flat-interface
assumption first order, curvature-included fourth order vs a 160,000-node
reference, plus the warp-and-straddle ablation; (3) a ring 0.001 wide with
α ≈ 1/1500 around an inner Dirichlet circle, same three-way comparison, then
iterative solvers (gmres converges slowly, bicgstab fails, Appendix B's
diagonal-dominance preconditioner restores both) and the extremizing sweep
s = 10³ … 10¹¹ (layer 1/s thick, α ~ 1/(1.5 s)) where the continuity
matrices' condition number grows like s² and convergence breaks down near
s = 10¹¹. A cornered interface (§5.4.5) is handled by circular-segment
approximations refined with the nodes, at second order.

**The seed construction (the wave-equation companion, 2026-09).** For a
stencil whose nodes see an edge of width δ ≪ h, the monomials 1, x, x², … are
replaced by *seeds*: the t = 0 profiles of solutions polynomial in time,
anchored at the evaluation node and continued through the edge by ODEs.
The spans form the nested chain {1} ⊂ ker L ⊂ {L f = const} ⊂ ker L² ⊂ …, the
jump-aware translated basis is the δ → 0 limit, the spaces are extended
complete Chebyshev systems (so every stencil solve is nonsingular), and the
truncation error is O(h⁴ · annihilator(u)) with the annihilator bounded
independently of δ on solutions. In 2-D for a straight feature the seeds are
polynomials in the tangential coordinate with coefficient functions marched
in the normal one; for a curved feature they march along the true normal
through the foot point (route (a)). Measured there: naive stencils have a
knee at h ≈ δ, seeds are fourth order at every resolution, and seed when
δ ≤ h. The companion's local path is one line in `CLAUDE.md`; nothing else
here points at it, and the manuscript will cite its arXiv preprint.

## 3. What is new here

### 3.1 The seeds for the diffusion operator

Everything reduces. With `L = ∂ₓ α ∂ₓ` the chain is the wave companion's
u-field chain with ρ = 1 and K = α, in a single field:

    φ₀ = 1;   α φ₁′ = α(x_e)  (constant flux, slope 1 at x_e);
    L φ_k = k (k − 1) α(x_e) φ_{k−2},  φ_k(x_e) = φ_k′(x_e) = 0   (k ≥ 2).

The odd seeds carry "the flux is smooth", the even ones "u_tt = const", and
the two interface conditions of the heat equation (u and α u_x continuous)
are both conditions on this one field, so there is no even/odd field swap to
route. The jump limit is dissertation ch. 4's translated basis. The
2-D scalar operator `∇·(α ∇)` is the case the companion's 2-D design note was
written for and then skipped for the elastic system: for a straight feature
the seed of x′ᵃ y′ᵇ is Σ_j g_j(y′) x′ʲ with a triangular chain of 1-D ODEs in
the normal coordinate, about 70 states for the 15 seeds of a degree-4 stencil
against 600 for the elastic ones.

### 3.2 Where the knee lives (and where it does not)

**1-D equilibrium is degenerate, on purpose.** The steady solution of
`(α u′)′ = 0` has constant flux, so it lies in span{φ₀, φ₁} = ker L. A seed
stencil is therefore *exact* on it at any resolution and any δ, and the
discrete solution equals the exact one to solver precision. So is any
finite-volume scheme with exact face conductances (the harmonic mean of α
over the cell, Patankar's rule, is exact for a piecewise-constant α in 1-D
steady state), and so is the dissertation's translated basis for a
piecewise-constant α. The 1-D elliptic problem cannot rank the methods; it
is the sanity check and the limit case, one figure and a remark.

**The knee is in the 1-D parabolic problem and in 2-D.** For a
time-dependent solution, `∂ₓ L² u = ∂ₓ u_tt` is bounded independently of δ
while `u⁽⁵⁾ ~ δ⁻⁴` inside the edge, so the wave companion's prediction
transfers verbatim: naive FD4 converges at fourth order only once h ≲ δ,
with a knee at h ≈ δ; the seeds are fourth order throughout. In 2-D the
tangential variation (sin 2πx in every test case) takes the equilibrium
solution out of the normal operator's kernel, so the 2-D elliptic problem is
a real test of the mixed seeds even at steady state. The headline 1-D figure
is parabolic; the headline 2-D figures are elliptic and parabolic both.

### 3.3 Elliptic-specific questions the wave work never had

- **Global solvability and conditioning.** There is no time stepping to hide
  behind: every seed row goes into one sparse matrix that is solved once
  (direct) or iterated (gmres, bicgstab). The 2016 work already found that
  jump-aware rows break the iterative solvers' convergence until a
  diagonal-dominance preconditioner is applied. Do seed rows behave the same,
  better, or worse? Does the direct solve stay well-conditioned as δ → 0?
- **Implicit time stepping.** The parabolic problem needs BD4 (the
  dissertation's choice, dt = h) or another implicit method, so the seed
  operator's spectrum matters for a different reason than in the wave case:
  it must stay in the left half-plane and inside the stability region after
  the implicit shift. Reproduce Fig. 5-6 for naive, jump-aware and seed
  operators.
- **Two sub-grid scales at once.** Case 3's ring is 0.001 thick; with smooth
  edges its thickness w and edge width δ are both below h. The seeds of a
  stencil crossing both edges are one ODE march through both, the diffusion
  twin of the companion's double-cross. The s-sweep (Fig. 19–20) is the
  natural stress test: does the seeds' conditioning break at the same s as
  the continuity matrices'?
- **A first-order RBF correction is already in the 2016 method.** The
  warped RBFs enforce flux continuity to first order in the RBF part. The
  seeds only touch the polynomial part, as the jump construction does; the
  ablation "seeds with plain vs warped RBFs" is cheap and belongs in the
  2-D results. *Answered by E4.6, E4.8 and E4.12 (#37, #39, #84; stiff note
  §4.5, §4.8, §4.10):* on the flat band the warp turns at `h ≈ 2δ` (H7); on
  the smooth ring it helps wherever no row is anchored in the layer's tail,
  and a row anchored there, where the warp squeezes the anchor's side and
  weakens the row's diagonal, keeps whichever Gaussians give the stronger
  diagonal. That rule is on for a ring only; off the ring it is an ablation
  line and E4.10 decides. *Decided by E4.10 (#41; stiff note §5.2): it stays
  the ring's, since off the ring the row-by-row choice at near-ties can do
  worse than either uniform choice.*

### 3.4 The standing alternatives to measure against

The comparison the wave manuscript owed and paid (change the *medium*, run
the standard scheme) is more entrenched in diffusion than in waves, so it
has to be in the study from the start. Candidates, to be pinned to sources
in the literature pass (E5.2) before any wording claims anything:

- **T1, cell harmonic mean of α** over one and two cells (the classic
  conductance rule of finite-volume heat conduction; exact in 1-D steady
  state, as §3.2 says, and the strongest comparator).
- **T2, cell arithmetic mean of α** (what a careless discretisation does).
- **T0, the widened edge**: the true profile with δ replaced by max(δ, m h),
  m = 1, 2 (regularise α itself; the diffuse-interface reflex).
- **T3, band-limited α**: the anti-aliasing filter of the seismic literature
  applied to α (less standard for diffusion; include if the literature pass
  finds it used, otherwise drop it and say so).
- **The δ = 0 construction at the edge centre**: the dissertation's
  translated basis, treating the smooth edge as a jump. This is the
  "what we did 10–15 years ago" baseline the user asked for, not a comparator;
  it is off at first order in δ/h and the manuscript says by how much.

Not built, named and excluded like Schoenberg–Muir was: the immersed
interface method and ghost-fluid corrections (built for jumps with known
jump conditions, a different family), multiscale FEM and harmonic
coordinates (cited as the nearest relatives of the seeds in §1 of the
manuscript, not run).

## 4. Decisions

Conventions come from the wave-equation companion unless a line below says
otherwise. `CLAUDE.md` carries the working set.

- **D1 Package name** `heat_interfaces` (project `heat-interfaces`), with
  `heat1d/` and `heat2d/` sub-packages and shared `fd_weights.py`,
  `plotting.py`, `results_cache.py`. Every test case is the heat equation, so
  the name says so; "elliptic and parabolic" is the repo's title.
- **D2 Nothing is imported from the companion.** Fornberg weights, the
  palette, the results-cache schema and the seed marchers are re-implemented
  here (the same author, MIT both ways) so this repo stands alone if the
  companion is refactored. Docstrings cite the dissertation, the EABE paper
  and, for the seeds, the companion's manuscript, never its files.
- **D3 Naive baseline assembled as `Dx A Dx` (+ `Dy A Dy`)**, dissertation
  eq. 76, in 1-D and 2-D. A direct (α u_x)_x stencil on a jump gives a
  straight line, which is not a baseline anyone uses.
- **D4 Reference solutions.** 1-D elliptic: quadrature, `u = A + B ∫ dξ/α`,
  exact at any δ. 1-D parabolic: Chebyshev collocation in x (Dirichlet ends,
  smooth α for δ > 0) with a tight implicit integrator; a fine jump-aware run
  for δ = 0. 2-D flat: separation `u = sin(2πx) v(y)` reduces both problems to
  1-D in y, solved to 1e−12. 2-D curved, δ = 0: a fine jump-aware run
  (160,000 nodes as in 2016). 2-D curved, δ > 0: Fourier in x × Chebyshev in
  y on a product grid, since α is smooth there. Case 3: a fine jump-aware run.
  *Revised in E4.7 (#38, Brad, 2026-09-22):* the product grid serves case 2 at
  every δ, the jump included, in sheared coordinates that make both sine
  curves coordinate lines (stiff note §4.6); the 160,000-node run is its
  cross-check and is 4.3e-9 RMS from it. *Revised in E4.8 (#39, Brad,
  2026-09-23):* case 3's smooth ring (δ > 0) has no independent reference; its
  evidence is reference-free (the matched radial profile and the truncation
  probe on an exact radial mode through the ring, `heat2d.exact`) and a
  self-convergence line against a 160,000-node seed run read away from the
  ring (stiff note §3.11, §4.8). The smooth ring is the resistivity blended by
  the difference of its edges, which keeps its contact resistance at every δ.
- **D5 Time integration** is BD4 with dt ∝ h and one sparse LU per operator,
  as in the dissertation, for every parabolic run that is compared with 2016.
  Stiff-edge parabolic studies may add an SDIRK or Crank–Nicolson line only if
  BD4's start-up is shown to matter.
- **D6 Solvers.** SciPy's SuperLU `spsolve` is the direct baseline; `gmres`
  and `bicgstab` with and without `spilu`, and with Appendix B's
  diagonal-dominance preconditioner re-implemented, for the iterative study.
  Wall-clock numbers are ours, on this machine; 2016 MATLAB timings are not
  reproduced, only the qualitative picture.
- **D7 Compute caps.** 160,000-node direct solves are fine (≈ 42 nonzeros per
  row). EABE Fig. 14 runs FD4 to ~10⁷ nodes; we run FD4 to what one machine
  solves directly in minutes and say where we stopped. Every driver's default
  runs in seconds; sweeps sit behind flags and cache their references and
  operators under `outputs/`.
- **D8 Boundaries.** New relative to the wave port: Dirichlet rows at
  y = 0, 1 and on the inner circle of case 3, one-sided 30-node / degree-4
  stencils near them (EABE §3), x-periodic neighbour search only.
- **D9 Warped RBFs are ported** (EABE §2.2.4) because every published 2-D
  number uses them, and Fig. 11 is precisely the ablation.
- **D10 The cornered interface (§5.4.5) is a stretch ticket**, off the
  critical path; the manuscript's limitations section covers it either way.
- **D11 Docs.** `docs/port-notes.md` holds the reproduction tables and the
  decisions log for E1–E2; `docs/stiff-diffusion.md` is canonical for E3–E4
  and is what the manuscript quotes (`% TRACE` comments per section).
  Figures the notes reference are committed under `docs/figures/`.
- **D12 Licences.** Code MIT (root); `paper/` CC BY 4.0. The repo is public:
  no PDFs, no employer material, nothing committed under `outputs/`.
- **D13 Tickets ship from this file** after review, one PR per ticket,
  branch `<issue>-<short-description>`, Conventional Commits.

## 5. Risks and open questions

- **R1 Novelty is thinner than in the wave case.** Exact and
  operator-adapted schemes for `−(a u′)′ = f` go back to Tikhonov–Samarskii's
  homogeneous schemes (1961), Babuška–Osborn's generalised FEM, Il'in and
  Allen–Southwell fitted operators, Hou–Wu's multiscale basis (local PDE
  solves), Owhadi–Zhang's harmonic coordinates. The defensible claim is the
  combination: fourth order and higher on scattered nodes in 2-D through a
  sub-grid smooth layer, elliptic and parabolic, with the jump construction as
  its δ → 0 limit, measured against the harmonic-mean practice. E5.2 buckets
  it before any section is drafted; §6 of `LITERATURE.md` holds the only
  wording the manuscript may use.
- **R2 The 1-D elliptic degeneracy** (§3.2) surprises a reader who expects a
  knee plot first. Lead with the parabolic knee and state the exactness as a
  remark with its FV twin; do not let 1-D elliptic carry a headline.
- **R3 Conditioning of the global system with seed rows** is unknown. If
  direct solves degrade as δ → 0 the study changes shape (E4.5 decides).
- **R4 Case 3 sign and geometry details** are lost in text extraction (the
  y = 0 boundary sign; the exact sine-interface formulas of case 2 live in
  MATLAB `curvedinterface1/2`). Read the PDF and the MATLAB before coding
  (E2.1, E2.7).
- **R5 MATLAB version drift.** `ExeprepRBFHeatLaplace1..4` differ without
  notes; the port follows the *paper*, and the MATLAB is consulted for
  numeric constants only.
- **R6 Runtime.** The curved-δ product-grid references and the s-sweep are
  the expensive parts. Cache under `outputs/`, keep the reference builds
  behind flags, and record run times in the notes as the companion did.
- **R7 The straight-feature seeds need α to vary along the normal only**
  (found by E4.7, stiff note §4.6). Where α varies along an unresolved edge —
  a curved edge, or a piece that varies tangentially — the frozen normal
  profile is O(1)-inconsistent and route (a) is first order: at δ = 0 it is
  EABE Fig. 10's flat-interface line. E4.11 builds the tangential chain; if
  it does not restore the order, the manuscript scopes the 2-D seeds to
  features whose α varies along the normal alone and says so. *Answered by
  E4.11 (#81, stiff note §3.10, §4.7): the tangential chain, in the foot
  curve's own coordinates with every seed to level 4, is fourth order on
  case 2 and on both of E4.7's split geometries at every δ, within a small
  factor of the flat seeds from 5000 nodes on, so the seeds are not scoped;
  the ring (E4.8) inherits it.*
- **R8 A thin resistive layer needs the flux one degree higher** (found by
  E4.8, stiff note §3.11). Across a layer thinner than the stencil with a
  contact resistance of O(1), EABE eq. 40's ring, the far side's values carry
  the flux at the layer with no factor of its distance, and the degree-4 seeds
  lost an order at fine counts (without the flux seeds Fig. 19's twin falls at
  2.25–3.33 per halving from 10,000 nodes at s = 10³ and ends 8.2 and 11.8×
  E2.3's at 40,000 and 80,000, stiff note §4.8). *Answered in E4.8:* on a ring the block adds the
  five degree-5 seeds that carry a flux (20 on the 30 nodes), and the seeds
  are fourth order at or below E2.3's line (§4.8).

## 6. Epics and tickets

Dependency order is E0 → E1 → E2 → E3 → E4 → E5, with E3 startable as soon
as E1 lands and E5.1–E5.3 startable alongside E4. Sizes: S about one short
session, M one session, L several or compute-heavy.

## E0: Scaffold, sources and plan (#2)

The repository skeleton, the source index and this plan. E0.1 is the PR that
adds this file.

### E0.1 Repository scaffold, CLAUDE.md, README, plan (#8)
Labels: documentation
Size: S
Depends on: —

`pyproject.toml` (uv, Python 3.13, numpy/scipy/matplotlib, pytest, ruff at
88 columns), `src/heat_interfaces/` with a version and a smoke test,
`papers/` with checksums and a fetch script, `docs/paper-index.md`,
`docs/plan.md` (this file), `scripts/publish_issues.py`, `CLAUDE.md`,
`README.md`, gitignore for PDFs, `outputs/` and manuscript intermediates.

**Done when**
- `uv sync && uv run pytest && uv run ruff check .` are green from a clean clone.
- `papers/fetch_papers.sh` reports both local PDFs present with matching checksums.

### E0.2 Confirm a public source for the EABE 2017 manuscript and finish the paper index (#9)
Labels: documentation
Size: S
Depends on: E0.1

The heat paper's local copy is the submitted manuscript. Find whether a public
preprint exists (the CU AMath attached-files pattern that hosts the two
seismic preprints, or the author's site) and, if so, add it to
`papers/fetch_papers.sh` with its checksum; otherwise record that the copy is
local only. Read case 3's boundary conditions and the case-2 interface
formulas from the PDF and the MATLAB and add them to `docs/paper-index.md`.

**Done when**
- `papers/README.md` states the source status of the EABE PDF.
- `docs/paper-index.md` carries the case-2 interface formulas and the case-3 boundary signs.

### E0.3 Publish the epics and tickets to GitHub (#10)
Labels: documentation
Size: S
Depends on: E0.1

After the plan is reviewed: `uv run python scripts/publish_issues.py --create`,
then record the issue numbers next to the E-ids in this file and in the epic
bodies' checklists.

**Done when**
- Every `## E<k>` and `### E<k>.<n>` in this file has an issue, tickets reference their epic, epics carry a checklist.
- This file's section 6 lists the numbers.

## E1: The 1-D heat port (dissertation ch. 4) (#3)

`heat1d/`: the operator `d/dx α d/dx` on [−1, 1] with Dirichlet ends, the
naive FD4 assembled as `Dx A Dx`, the translated piecewise-polynomial
stencils for jumps with a smoothly varying α, the equilibrium solve and the
time-dependent solve, and the reproduction of Fig. 4-5–4-7. Also the MATLAB
folder's single-interface time-dependent problem as a second test.

### E1.1 heat1d: grid, materials, naive FD4 operator, equilibrium solve, quadrature reference (#11)
Labels: enhancement
Size: M
Depends on: E0.1

`heat1d/domain.py`: nodes on [−1, 1] (equispaced; node-on-interface and
cell-centred placements both selectable, since the knee results later depend
on where the edge sits relative to the nodes), piecewise materials with a
smooth α inside a layer (`alpha(x)` callable plus its Taylor coefficients about
each interface for E1.2). `fd_weights.py`: Fornberg's algorithm with tests
against tabulated weights. `heat1d/operators.py`: `Dx` (FD4, one-sided at the
ends), `A = diag(α)`, `L = Dx A Dx` (eq. 76), and, for the record, the direct
`(α u_x)_x` stencil that gives a straight line on a jump. `heat1d/solve.py`:
assemble Dirichlet rows, solve the equilibrium system. `heat1d/exact.py`: the
quadrature solution `u = A + B ∫ dξ/α` with Gauss–Legendre split at the
interfaces.

**Done when**
- Tests: constant α reproduces the linear solution to rounding; the quadrature reference converges at the quadrature's order; naive FD4 on the dissertation's problem is first order (Fig. 4-7's top line).
- The direct-stencil failure mode is a test, not a surprise.

### E1.2 heat1d: jump-aware stencils from continuity matrices (translated basis, smooth α, two interfaces) (#12)
Labels: enhancement
Size: L
Depends on: E1.1

Dissertation §4.1 / EABE §2.2.1–2.2.2: coefficient-vector operators `Dx`
(eq. 13), multiplication matrices from α's Taylor coefficients (eq. 15),
continuity rows from time derivatives of u (k = 0, 1, 2) and of the flux
(k = 0, 1), `C_L u_L = C_R u_R` (eq. 24), the translated basis, and Fornberg
weights on that basis for every stencil that sees an interface. Stencils that
see both interfaces of a thin layer translate twice. Reproduce EABE's worked
numbers (eq. 17–18, 21, 23 to two decimals) as tests.

**Done when**
- Tests: the translated basis keeps u and α u_x continuous; constant α recovers FD4; the EABE eq. 23 matrix matches; the equilibrium problem converges at fourth order (Fig. 4-7's lower line) and the 101-node solution has no oscillation (Fig. 4-5).
- Docstrings cite the equation numbers.

### E1.3 heat1d: time-dependent solve (BD4), the MATLAB single-interface problem, operator spectra (#13)
Labels: enhancement
Size: M
Depends on: E1.2

`heat1d/march.py`: BD4 with dt ∝ h and one LU (D5), RK4 start-up or a
lower-order ramp, with the explicit RK4 kept for small n. Run the MATLAB
problem (k = 1/9 | 1, one interface at 0, Dirichlet ends) naive vs
jump-aware against a reference (D4), and the dissertation's problem with a
time-dependent top boundary as in 2-D case 1. Plot operator eigenvalues
against the BD4 region as the dissertation does in 2-D.

**Done when**
- Tests: BD4 is fourth order in time on a manufactured solution; naive first order and jump-aware fourth order in space on the MATLAB problem; all eigenvalues of the jump-aware operator have negative real part.

### E1.4 heat1d: driver, figures, and the port notes (#14)
Labels: documentation
Size: S
Depends on: E1.3

`scripts/heat1d_convergence.py` writes the Fig. 4-5/4-6/4-7 twins to
`outputs/`; `docs/port-notes.md` §1 records the rates, the 2016 numbers next
to ours, and any deviation with its reason. Palette: blue = interface-aware,
orange = naive, as in the companion.

**Done when**
- The three figures regenerate in seconds; the notes table has both columns.

## E2: The 2-D heat port (dissertation ch. 5, EABE 2017) (#4)

`heat2d/`: scattered nodes on the x-periodic strip with Dirichlet rows,
Gaussian RBF-FD with polynomial augmentation, the naive operator as
`Dx A Dx + Dy A Dy`, interface-aware stencils with curvature and
multi-interface translation, warped RBFs, direct and iterative solvers, BD4,
and the three test cases with the published figures reproduced.

### E2.1 heat2d: domain, interfaces, straddling node sets, boundaries, x-periodic kNN (#15)
Labels: enhancement
Size: L
Depends on: E0.2

`heat2d/domain.py`: interfaces as parametrised curves with signed distance,
normal, angle and curvature (flat lines y = c; the sine pair of case 2 from
MATLAB `curvedinterface1/2`; the circles of case 3), node sets by repulsion
with two rows straddling each interface and rows along y = 0, y = 1 and the
inner circle, Dirichlet node masks. `heat2d/neighbors.py`: kNN periodic in x
only (`cKDTree` with a boxsize in x and none in y, or tiling). Diff
`ExeprepRBFHeatLaplace1..4` first and record the differences in the notes.

**Done when**
- Tests: node counts and straddle spacing as specified; every interface node pair sits orthogonally across the curve; kNN wraps in x and not in y; case-1/2/3 node sets plot like Fig. 5-3, 8 and 12.

### E2.2 heat2d: Gaussian RBF-FD weights with polynomials, naive operator, Dirichlet rows, direct solve (#16)
Labels: enhancement
Size: L
Depends on: E2.1

`heat2d/rbf.py`: batched saddle-point solves for `d/dx`, `d/dy` and the
Laplacian with GA ε = 0.4/d (d = nearest-neighbour distance, EABE eq. 31),
42 nodes / degree 5 interior, 30 / degree 4 near boundaries (one-sided).
`heat2d/operators.py`: sparse `Dx`, `Dy`, `L = Dx A Dx + Dy A Dy` (D3),
Dirichlet rows. `heat2d/solve.py`: equilibrium solve with SuperLU. Verify on
the no-interface control (α ≡ 1, sin 2πx at the top, the analytic
sinh solution) before any interface exists.

**Done when**
- Tests: weights reproduce derivatives of degree-5 polynomials to rounding; the control problem converges at the expected order; the case-1 problem with naive stencils is first order (the top line of EABE Fig. 10's analogue).

### E2.3 heat2d: interface-aware stencils for the scalar operator with curvature and multi-interface translation (#17)
Labels: enhancement
Size: L
Depends on: E2.2

EABE §2.2.3 / dissertation §5.3: local frame at the nearest interface point;
continuity rows from time derivatives of the first (p − 2k) tangential
monomials of u and the first (p − 2k − 1) of the normal flux after k
applications of the operator; the flat-interface variant (y′ = 0) and the
curvature-included variant (interface expansion inserted for y′, matrix
cos/sin of the angle, numerically differentiated as the paper does); square
continuity matrices inverted to translate each monomial; nodes across two
interfaces (case 3's ring) translated twice. Both variants selectable, since
Fig. 10 and 14 compare them.

**Done when**
- Tests: the translated basis is continuous with continuous normal flux along a curved interface to O(h^p); a flat interface makes the two variants identical; case 1 converges at fourth order (EABE Fig. 7).

### E2.4 heat2d: warped RBFs across interfaces, and the plain-RBF / non-straddling ablation (#18)
Labels: enhancement
Size: M
Depends on: E2.3

EABE §2.2.4: the coordinate stretching of a Gaussian across the interface
that keeps its normal flux continuous to first order; switchable, together
with a node set that ignores the interface, so Fig. 11 ("no warp, no
straddling" vs "with warp and straddling") can be reproduced.

**Done when**
- Tests: a warped RBF's α ∂ₙφ is continuous at the interface to rounding; the ablation runs on case 2 at three resolutions.

### E2.5 heat2d case 1: elliptic and parabolic convergence against the analytic solution, operator spectrum vs BD4 (#19)
Labels: enhancement
Size: M
Depends on: E2.3

Eq. 32–34 (elliptic) and dissertation eq. 84–86 (parabolic, c_t = 1, t = 0.1,
BD4 with dt = h). `heat2d/march.py` (BD4, one LU). `heat2d/exact.py` for the
piecewise-exponential solution. `scripts/heat2d_case1.py` writes the
Fig. 5-5 twin and the Fig. 5-6 eigenvalue plot at 4900 nodes.

**Done when**
- Tests: elliptic and parabolic errors fourth order across 1250–20,000 nodes; eigenvalues lie inside the BD4 region at dt = 0.02.

### E2.6 heat2d case 2: curved interfaces, FD4 / flat / curved convergence, warp ablation, performance plot (#20)
Labels: enhancement
Size: M
Depends on: E2.4

Eq. 35–36; the 160,000-node curvature-included solution as the reference
(cached under `outputs/`); the three-way convergence plot (Fig. 10), the
ablation (Fig. 11), and error vs wall-clock (dissertation Fig. 5-11) with our
timings.

**Done when**
- FD4 and flat-assumption lines at first order, curvature-included at fourth, with the numbers in `docs/port-notes.md` §2 next to the 2016 ones.

### E2.7 heat2d case 3: the thin insulating ring with an inner Dirichlet circle (#21)
Labels: enhancement
Size: L
Depends on: E2.6

Eq. 37–39 (sign of the y = 0 condition read from the PDF, E0.2); ring
stencils that cross both interfaces; FD4 to the resolution D7 allows,
flat vs curved RBF-FD as in Fig. 14; the 40,000-node mesh plot.

**Done when**
- The Fig. 14 twin with our FD4 cap stated; the curvature-included line fourth order.

### E2.8 heat2d: iterative solvers, the control problem, and Appendix B's diagonal-dominance preconditioner (#22)
Labels: enhancement
Size: M
Depends on: E2.7

19-node / degree-3 stencils; `gmres` and `bicgstab` on the control problem
(no ring) and on case 3; `spilu` as the off-the-shelf comparison; the
dissertation's Appendix B preconditioner (restore the diagonal dominance
ratio of interface rows by adding multiples of neighbouring rows)
re-implemented on sparse rows. Fig. 15–18 twins with our timings.

**Done when**
- Without preconditioning bicgstab fails and gmres is slow on case 3, both recover with Appendix B's preconditioner (qualitatively as published); DDR before and after is tabulated.

### E2.9 heat2d: the extremizing-parameter sweep and continuity-matrix conditioning (#23)
Labels: enhancement
Size: S
Depends on: E2.7

Eq. 40 with s = 10³, 10⁸, 10⁹, 10¹⁰, 10¹¹: errors vs N per s (Fig. 19) and
the mean condition number of the continuity matrices vs s at N = 10,000
(Fig. 20, O(s²)).

**Done when**
- Both figures regenerate; the breakdown near s = 10¹¹ is reproduced or its absence explained in the notes.

### E2.10 (stretch) heat2d: the cornered interface with circular-segment approximations (#24)
Labels: enhancement
Size: M
Depends on: E2.6

**Closed as won't do, 2026-09-21.** The elliptic and parabolic ports and the
stiff-edge study are enough for this repo; corners will be explored in a
separate repo later. The manuscript's limitations section covers them.

Dissertation §5.4.5: the band of case 2 with corners at y ≈ 0.6, smoothed by
circular segments whose radius shrinks with h; FD4 vs RBF-FD at second
order (Fig. 5-22). Off the critical path; the manuscript's limitations
section covers corners whether or not this lands.

**Done when**
- Fig. 5-22's twin, second order for both, with the approximation rule documented.

### E2.11 heat2d: drivers and the port notes (#25)
Labels: documentation
Size: S
Depends on: E2.9

`scripts/heat2d_nodes.py`, `heat2d_case1.py`, `heat2d_case2.py`,
`heat2d_case3.py`, `heat2d_solvers.py`, `heat2d_extremes.py` with fast
defaults and flagged sweeps; `docs/port-notes.md` §2 with the reproduction
table (figure, 2016 number, ours, rate) and the decisions log.

**Done when**
- Every published 2-D figure in section 1's list has a regenerating twin and a row in the table.

## E3: Sub-grid smooth edges in 1-D: seeds for the diffusion operator (#5)

The 1-D stiff-edge study: formulation, references at any δ, the naive knee
(parabolic), the seed stencils, the coefficient treatments, and the
degenerate elliptic case stated for what it is.

### E3.1 stiff 1-D: formulation note for the diffusion seeds (#26)
Labels: documentation
Size: M
Depends on: E1.2

`docs/stiff-diffusion.md` §1: the chain for `L = ∂ₓ α ∂ₓ` (single field, flux
variable ψ = α φ′), the jump limit as ch. 4's translated basis, the ECT /
Pólya-form argument for nonsingular stencil solves, the truncation error
O(h⁴ ∂ₓ L² u) and its two readings (equilibrium: zero, the seeds are exact;
parabolic: ∂ₓ u_tt, bounded independently of δ), the knee prediction, and the
1-D elliptic degeneracy with its finite-volume twin (§3.2 of the plan).
Cite the companion's manuscript for the wave case.

**Done when**
- The note's §1 is complete with the predictions a later section can be checked against.

### E3.2 stiff 1-D: smooth-edged α, and elliptic and parabolic references at any δ (#27)
Labels: enhancement
Size: M
Depends on: E1.3

`heat1d/domain.py`: tanh edges of width δ (δ = 0 the jump, bit for bit),
edges placed at nodes or between them by option. References (D4): quadrature
for the equilibrium problem; Chebyshev collocation in x with a tight implicit
integrator for the parabolic one, converged to 1e−10 and cached under
`outputs/`.

**Done when**
- Tests: δ = 0 reproduces the jump medium; the parabolic reference is converged (two resolutions agree to 1e−10); the quadrature reference is exact for a jump.

### E3.3 stiff 1-D: the naive knee, and the δ = 0 construction on a smooth edge (#28)
Labels: enhancement
Size: M
Depends on: E3.2

Naive FD4 vs n at δ ∈ {0.04, 0.01, 0.0025}, elliptic and parabolic; the
jump-aware stencils applied at the edge centre for δ > 0 (how far off, in
weights and in solution error, as a function of δ/h).

**Done when**
- The parabolic knee at h ≈ δ is measured with rates above and below; the elliptic naive lines are recorded; the δ = 0 construction's first-order-in-δ/h error is tabulated.

### E3.4 stiff 1-D: seed stencils and dispatch (#29)
Labels: enhancement
Size: L
Depends on: E3.3

`heat1d/stiff.py`: march φ₀ … φ₄ (and φ₅ for six-point stencils) through
the edge from the evaluation node as one first-order system in (φ, ψ = α φ′),
weights from the 5 × 5 solve, dispatch in `build_operator(mode="seeds")` for
stencils whose nodes see an edge; both edges of a thin layer in one march.

**Done when**
- Tests: constant α gives Fornberg's weights; the jump limit recovers E1.2's weights at first order in δ; the equilibrium solve is exact to 1e−12 at every δ; the parabolic error is fourth order at every δ with a δ-independent constant; condition numbers of the stencil solves are Vandermonde-like.

### E3.5 stiff 1-D: coefficient treatments (harmonic and arithmetic cell means, widened edge, band-limited α) (#30)
Labels: enhancement
Size: M
Depends on: E3.4

`heat1d/treatments.py`: T1 harmonic mean of α over one and two cells, T2
arithmetic mean, T0 widened edge (m = 1, 2), T3 band-limited α (kept only if
E5.2 finds it in use for diffusion); each a material the naive operator
samples, errors against the true-δ reference, elliptic and parabolic.

**Done when**
- Tests: T1 is exact on the 1-D equilibrium problem with a piecewise-constant α; each treatment reduces to its δ = 0 form; the parabolic table has every treatment at every δ.

### E3.6 stiff 1-D: driver, figures, results write-up (#31)
Labels: documentation
Size: M
Depends on: E3.5

`scripts/heat1d_stiff.py` (knee plot, seeds vs monomials vs translated
basis, a parabolic snapshot, the comparator table), `--data-dir` for the
results cache (E5.3), `docs/stiff-diffusion.md` §2 with the tables and the
rates.

**Done when**
- The driver runs in under a minute from cached references; §2 states the knee, the seeds' rates, the comparator ranking, and the elliptic remark.

## E4: Sub-grid smooth edges in 2-D: scalar seeds on scattered nodes (#6)

The 2-D stiff-edge study for `∇·(α ∇)`: design, references, the naive
baseline, the seeds for a straight feature, seed-augmented stencils and the
solvability question, the flat sweep, the curved feature, the doubly sub-grid
ring, the treatments on scattered nodes.

### E4.1 stiff 2-D: design note for the scalar seeds (#32)
Labels: documentation
Size: M
Depends on: E3.1

`docs/stiff-diffusion.md` §3: the straight-feature ansatz Σ_j g_j(y′) x′ʲ and
its triangular ODE chain for the scalar operator; how the 15 seeds of a
degree-4 stencil replace the monomials in the saddle-point system with the
Gaussian part unchanged; curved features by route (a); the double-stiff ring;
the elliptic questions of §3.3 (global conditioning, implicit step, the
warped-RBF interaction) as numbered hypotheses for E4.5–E4.8.

**Done when**
- The note's §3 fixes notation (local frame, anchor, ψ) and lists the hypotheses with the experiment that tests each.

### E4.2 stiff 2-D: smooth flat edges and the separable 1-D references (#33)
Labels: enhancement
Size: M
Depends on: E2.5

Tanh edges in y across y = 0.6 and 0.8 in `heat2d/domain.py`; the separable
reference `u = sin(2πx) v(y)`: `(α v′)′ − 4π² α v = 0`, v(0) = 0, v(1) = 1 for
the elliptic problem and the matching 1-D parabolic problem in y, solved by
Chebyshev collocation to 1e−12 (`heat2d/exact.py`).

**Done when**
- Tests: δ = 0 recovers the analytic case-1 solution; the reference is converged; the medium's edge is smooth to rounding.

### E4.3 stiff 2-D: the naive baseline through a smooth flat edge (#34)
Labels: enhancement
Size: M
Depends on: E4.2

Naive RBF-FD vs n at δ ∈ {0.04, 0.01, 0.005, 0.0025}, elliptic and
parabolic, against the separable references: is there a knee on scattered
nodes, and what does the resolution floor hide?

**Done when**
- The baseline table with rates per δ; the notes say what separates resolved from unresolved edges most sharply (the companion found a spurious field did; here the candidates are the flux jump and the y-profile error).

### E4.4 stiff 2-D: scalar seeds for a straight feature (#35)
Labels: enhancement
Size: L
Depends on: E4.1

`heat2d/seeds.py`: the 15 seeds (degree ≤ 4) as one ODE system in y′ with
flux variables, anchored at the evaluation node, marched through one or two
edges; checks against constant α (monomials), against E2.3's translated
basis in the jump limit, residual of `L seed = source`, conditioning of the
seed Gram matrix on a stencil.

**Done when**
- Tests: the four checks above pass at δ ∈ {h/8, h, 8h}; the march for one stencil takes milliseconds.

### E4.5 stiff 2-D: seed-augmented stencils, operators, and the solvability question (#36)
Labels: enhancement
Size: L
Depends on: E4.4

`seed_weights` and dispatch in `build_operators(mode="seeds")` for stencils
whose nodes see an edge (plain and warped RBF parts both); the global
matrix's condition number and SuperLU behaviour as δ → 0; `gmres` /
`bicgstab` with and without the Appendix B preconditioner on seed rows;
spectra of naive, jump-aware and seed operators against the BD4 region.

**Done when**
- The hypotheses of E4.1 on conditioning and spectra are each answered with a table; if direct solves degrade, the notes say at what δ/h and the plan is revised before E4.6.

### E4.6 stiff 2-D: the flat δ sweep, elliptic and parabolic (#37)
Labels: enhancement
Size: L
Depends on: E4.5

Naive vs jump-aware-at-the-centre vs seeds at every (n, δ), elliptic and
parabolic, against the separable references; the crossover and the rule
(seed when δ ≤ h?) restated for diffusion; operators cached under `outputs/`.

**Done when**
- The sweep tables with rates; the rule stated with its evidence; the resolved-edge penalty of the seeds measured.

### E4.7 stiff 2-D: the curved feature (#38)
Labels: enhancement
Size: L
Depends on: E4.6

Tanh in the signed normal distance of the case-2 interfaces; seeds along the
true normal through the foot point (route (a)); references: the 160,000-node
jump-aware run for δ = 0 and a Fourier × Chebyshev product-grid solve for
δ > 0 (cached); naive vs jump-aware vs seeds vs n at δ ∈ {0, 0.0025, 0.005,
0.01}; the seed rows' truncation error on the true solution.

**Done when**
- Curved numbers compared with the flat ones at equal δ; route (a)'s geometry error located or shown absent on these node sets.

### E4.11 stiff 2-D: tangential seeds for a curved or tangentially varying edge (#81)
Labels: enhancement
Size: L
Depends on: E4.7

E4.7 found route (a) O(1)-inconsistent on case 2 (stiff note §4.6): the
frozen normal profile puts the seeds' kink on the tangent line and imposes
the foot point's flux ratio at every node, so wherever α varies along an
unresolved edge the error is not in the seed span. Build §3.5's tangential
chain in the curve's own coordinates (arclength, normal distance): the metric
`1 − κ d` and α expanded in the tangential coordinate along the normal line,
the levels coupled (about 55 levels and 110 states at degree 4, not 44), and
the moment right-hand sides from the true curvilinear operator. The
osculating-circle chain (the metric frozen at the foot point) is its first
rung and is exact on concentric circles with constant pieces. Design first,
as §3 was: a §3.10 in the stiff note before the code. Numbered after E4.10
because the plan's ticket ids are numeric; it runs before E4.8.

**Done when**
- The chain passes its checks (RingMode through concentric circles, the δ = 0 limit against E2.3-curved on case 2 and on E4.7's two split geometries, the truncation probe converging at the bulk rows' rate), and its line on E4.7's cached curved sweep answers H9 again: fourth order within a small factor of the flat numbers, or the notes say what still stalls.

### E4.8 stiff 2-D: the doubly sub-grid ring and the extremizing sweep with seeds (#39)
Labels: enhancement
Size: M
Depends on: E4.11

Case 3's ring with smooth edges (thickness w and edge width δ both below h),
seeds marched through both edges; the s-sweep of E2.9 rerun with seeds: does
the breakdown move, and is the seeds' march conditioning the new limit?

**Done when**
- The Fig. 19–20 twins with a seeds line; the notes state where each construction fails and why.

### E4.12 stiff 2-D: the Gaussians on rows anchored inside a smooth resistive layer (#84)
Labels: enhancement
Size: M
Depends on: E4.8

Found by E4.8 (stiff note §3.11, §4.8): on the smooth ring the rows anchored
in the layer's tail (α_e a third of the pieces') have warped and plain
Gaussian weights 100 % apart, and at δ = 0.001 on 20,000 nodes the far-field
error jumps to 6.2e-5 between 2.6e-5 and 1.9e-6; plain Gaussians give 8.1e-6
there, while elsewhere at δ > 0 the warp wins by 1.0–1.8×. Design a rule (plain
Gaussians inside the layer, or a warp that does not compress the neighbours
by α_e/α) and rerun the ring's δ > 0 sweep with it. Numbered after E4.11
because the plan's ticket ids are numeric.

**Done when**
- The δ = 0.001 outlier is gone or explained, the δ > 0 lines of §4.8 are rerun with the rule, and the note says where the warp helps on a smooth resistive layer.

### E4.9 stiff 2-D: coefficient treatments on scattered nodes (#40)
Labels: enhancement
Size: M
Depends on: E4.6

`heat2d/treatments.py`: harmonic and arithmetic means of α over a disc of
radius h/2 and h, the widened edge, the band-limited α (if kept), on the flat
and curved sweeps; against the same references.

**Done when**
- The comparator tables per δ, elliptic and parabolic; the crossover of each treatment against sampling.

### E4.10 stiff 2-D: drivers, figures, results write-up (#41)
Labels: documentation
Size: M
Depends on: E4.9

`scripts/heat2d_stiff.py` (modes naive / aware / seeds / treatments,
`--amplitude` for curved, `--ring`, `--data-dir`), `heat2d_stiff_eigenvalues.py`,
committed figures under `docs/figures/`, `docs/stiff-diffusion.md` §4–5 with
every table and the run times.

**Done when**
- Every 2-D number the manuscript will quote is in the note with a section to trace to.

## E5: The manuscript (#7)

`paper/`: amsart, tectonic, a verified bibliography, a novelty ledger, a
results cache with scripted figures and tables, section-by-section drafting
against the canonical notes, assembly, arXiv packaging.

### E5.1 manuscript: scaffold `paper/` (#42)
Labels: documentation
Size: S
Depends on: E3.6

`main.tex` skeleton with `\stub{}` lines per section, `references.bib`
seeded with the 2016–2017 sources, `LICENSE` (CC BY 4.0), `make_arxiv.py`
with the comment-stripping and rebuild-and-compare gates, `\date` fixed by
hand, `main.pdf` committed on every change.

**Done when**
- `tectonic main.tex` builds; `make_arxiv.py` refuses the stubs.

### E5.2 manuscript: literature and novelty ledger (#43)
Labels: documentation
Size: L
Depends on: E3.1

`LITERATURE.md`: buckets (own work; elliptic and parabolic interface methods
2016–2026: IIM, ghost fluid, XFEM / CutFEM, RBF-FD and kernel interface
papers, Reutskiy, Bayona et al.; operator-adapted bases: Tikhonov–Samarskii,
Babuška–Osborn, Il'in / Allen–Southwell, Hou–Wu, Owhadi–Zhang, LOD, Trefftz;
coefficient treatments in diffusion: Patankar's harmonic mean and
finite-volume conductances, diffuse-interface smoothing, thin-layer
asymptotics and contact-resistance limits for the ring), the refuter-pass
log per bucket, verification status per bib entry, §6 the only novelty
wording allowed. Decide T3's fate here.

**Done when**
- Every bib entry carries a dated `VERIFIED` note; §6 exists; R1 has a written answer.

### E5.3 manuscript: results cache, scripted figures and tables, number check (#44)
Labels: enhancement
Size: M
Depends on: E4.10

`results_cache.py` (one JSON per driver run with args, date, git SHA),
`scripts/paper_figures.py` (print style, `SOURCE_DATE_EPOCH` pinned,
`--check` for byte identity), booktabs `tab_*.tex` fragments,
`scripts/paper_numbers.py` asserting every cache-backed number the text
quotes.

**Done when**
- `paper_figures.py --check` passes from a clean `paper/data/`; a test covers the cache schema.

### E5.4 manuscript: §1–2, introduction and setting (#45)
Labels: documentation
Size: M
Depends on: E5.2

The three regimes for diffusion, the 2016 construction and its δ → 0 role,
the interface conditions, the operator in 1-D and 2-D, the node sets and
stencils, scope of the implementation; prior-work paragraph from
`LITERATURE.md` §6a verbatim.

**Done when**
- No stub remains in §1–2; every number carries a `% TRACE`.

### E5.5 manuscript: §3, seeds for the diffusion operator in one dimension (#46)
Labels: documentation
Size: M
Depends on: E5.4

Quoting `docs/stiff-diffusion.md` §1: the chain, the jump limit, ECT, the
truncation error and its two readings, the elliptic exactness remark and its
finite-volume twin, the seed-basis figure.

**Done when**
- §3 complete with traces; the exactness remark cites the FV literature found in E5.2.

### E5.6 manuscript: §4, one-dimensional results (#47)
Labels: documentation
Size: M
Depends on: E5.5

The parabolic knee, the seeds' rates, the δ = 0 construction's error, the
treatments' table; the elliptic sanity figure as a remark.

**Done when**
- §4 complete; `paper_numbers.py` covers every ratio quoted.

### E5.7 manuscript: §5, seeds in two dimensions (#48)
Labels: documentation
Size: M
Depends on: E5.6

Straight feature, stencil assembly, curved features, the double edge; a
seed-section figure.

**Done when**
- §5 complete with traces to `docs/stiff-diffusion.md` §3.

### E5.8 manuscript: §6, two-dimensional results (#49)
Labels: documentation
Size: L
Depends on: E5.7

Test problems and references, naive baseline, solvability and spectra, the
flat sweep and the rule, the curved feature, the ring and the s-sweep, the
treatments; all tables from fragments.

**Done when**
- §6 complete; every figure and table placed from `paper/figures/`.

### E5.9 manuscript: §7, discussion, limitations, future work, conclusions (#50)
Labels: documentation
Size: M
Depends on: E5.8

Limitations (corners, extreme contrasts, 3-D cost, what the experiments
leave open), future work in the order to do it, conclusions quoting
`LITERATURE.md` §6b verbatim.

**Done when**
- §7 complete; no claim outside §6b.

### E5.10 manuscript: assembly and consistency pass (#51)
Labels: documentation
Size: S
Depends on: E5.9

Every number re-checked against the notes and the cache; notation list;
float barriers; bibliography labels; `docs/stiff-diffusion.md` corrected
where the cache disagrees.

**Done when**
- `paper_numbers.py` and `paper_figures.py --check` green; a read of the PDF end to end finds no stub, no `\todo`.

### E5.11 manuscript: arXiv packaging and the pre-submission decisions (#52)
Labels: documentation
Size: S
Depends on: E5.10

`make_arxiv.py` end to end; categories, licence, date, acknowledgments, the
tool-and-resource disclosure in the form used for the wave manuscript;
decisions recorded in `paper/README.md`. Submission itself is Brad's.

**Done when**
- `arxiv.tar.gz` builds and the rebuilt PDF's text matches the committed one; `paper/README.md` records the decisions.

## 7. Publishing the tickets

```sh
uv run python scripts/publish_issues.py            # dry run: prints what would be created
uv run python scripts/publish_issues.py --create   # creates epics, then tickets, then epic checklists
```

The script reads section 6 of this file. Epics are created first so each
ticket's body can name its epic and its dependencies by issue number; each
epic's body then gets a checklist of its tickets. Labels are the repository's
`enhancement` and `documentation`.

Done on 2026-09-20 (E0.3, #10). The headings above now carry their issue
numbers, and the script refuses `--create` while they do, so a re-run cannot
duplicate them; `--force` overrides for a fresh repository.
