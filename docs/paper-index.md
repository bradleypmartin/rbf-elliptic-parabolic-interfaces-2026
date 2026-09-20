# Paper index

Where to look in each PDF so a session jumps straight to the relevant pages
with `pdftotext -f <first> -l <last> -layout <pdf> -` instead of re-reading
whole documents. Page numbers are **PDF page numbers**. PDFs live in
`papers/` (gitignored; checksums in `papers/README.md`). Indexed 2026-09-20.

## Dissertation (`martin-dissertation-2016-rbf-fd-interfaces.pdf`, 145 pp.)

Printed page = PDF page − 9. Chapters 2–3 (waves) are indexed in the
companion wave-equation work; only what this repo needs is listed.

| PDF pages | Section | Use for |
| --- | --- | --- |
| 1–9 | Front matter, abstract, TOC | — |
| 10–16 | Ch. 1 Introduction (history of interface treatments; RBF-FD overview §1.3 at p. 14) | Manuscript §1 context |
| 75–85 | **§4.1** 1-D heat: eq. 51 operator `D = d/dx alpha d/dx`, eq. 52–55 time derivatives of temperature and flux, continuity for every k; eq. 56–57 expansions; §4.1.1 (p. 78) the specific problem with two interfaces, multiplication matrices for a smoothly varying alpha, continuity matrices, translated basis; Fig. 4-2/4-3 basis functions and weights | E1.2: the jump construction for the diffusion operator |
| 85–89 | **§4.2** 1-D results: eq. 75 test problem (alpha = 0.1 + 0.4 sin 2πx on [0, 0.5], 1 elsewhere; u(−1) = 1, u(1) = 0; equilibrium), **eq. 76 the naive operator as Dx A Dx** (and why a direct (alpha u_x)_x stencil gives a straight line), Fig. 4-5 101-node solutions, Fig. 4-6 errors vs a 6400-node reference, **Fig. 4-7 convergence 100–1600 nodes: FD4 first order, method fourth** | E1.1, E1.2, E1.4 |
| 90–99 | §5.1 problem statement (eq. 77, continuity of u and n·(alpha grad u)); §5.2 standard weights; **§5.3 weights across interfaces** (continuity via time derivatives of u and normal flux, curvature via a local interface expansion, matrix cos/sin, Fig. 5-1 warped RBF at p. 98) | E2.2, E2.3, E2.4 |
| 99–105 | §5.4 setup (GA eps = 0.4/d, 42/deg 5 interior, 30/deg 4 across interfaces and near boundaries, backslash, BD4 with dt ∝ h); **§5.4.1 case 1** (eq. 84–85; eq. 86 the time-dependent analytic solution; Fig. 5-5 parabolic and elliptic convergence; **Fig. 5-6 eigenvalues of the 4900-node operator vs the BD4 stability region, dt = 0.02**) | E2.5 |
| 105–110 | **§5.4.2 case 2** (two mildly curved interfaces, alpha = 0.2 + 0.1 sin 2πx sin 2πy in the band; Fig. 5-9 FD4 / flat / curved convergence vs a 160,000-node reference; Fig. 5-10 warp-and-straddle ablation; Fig. 5-11 error vs wall-clock) | E2.6 |
| 110–114 | **§5.4.3 case 3** (ring 0.349 ≤ r ≤ 0.35, alpha = 1/1500 + (1/3000) sin 2πx sin 2πy, inner Dirichlet circle r = 0.05; Fig. 5-14 FD4 / flat / curved); **§5.4.4 iterative solvers** (control problem, gmres, bicgstab, Fig. 5-15–5-18) | E2.7, E2.8 |
| 118–122 | **§5.4.5 the cornered interface** (circular-segment approximations refined with the node set, Fig. 5-19–5-22, second order) | E2.10 (stretch) |
| 123–124 | Ch. 6 Conclusions (open questions: corners, higher-order RBF correction, nonlinear problems) | Manuscript §7 |
| 125–128 | Bibliography | Citations |
| 129–139 | Appendix A: the Geophysics 2015 paper | — |
| 140–145 | **Appendix B: the preconditioner of §5.4.4** (diagonal dominance ratio; restore it by adding multiples of neighbouring stencil rows, eq. 92–94 onward) | E2.8 |

## Martin & Fornberg 2017, EABE (`martin-fornberg-2017-rbf-fd-heat-equilibrium-eabe-submitted.pdf`, 49 pp.)

The author post-print of Eng. Anal. Bound. Elem. 79 (2017) 38–48, fetched
from CU Scholar (`papers/README.md`; byte-identical to Brad's copy). Same
method as dissertation ch. 5 with the s-sweep added; the definitive source
for the 2-D test-case parameters. Equation numbers below are the paper's.
Read off the rendered pages on 2026-09-20 (E0.2, #9), since text extraction
drops signs and piecewise braces:

- **Case 2 interfaces** (not stated in either text; from MATLAB
  `curvedinterface1/2`, `ExeprepRBFHeatLaplace3/4.m` with `cFlag ≠ 0`):
  `y = 0.6 + 0.02 sin 2πx` and `y = 0.8 + 0.02 sin 2πx`, slope
  `0.04π cos 2πx`, angle `atan(y′)`; the band keeps its 0.2 thickness. The
  band's α (eq. 35) is `0.2 + 0.1 sin 2πx sin 2πy`, 1 elsewhere; Dirichlet
  `u = sin 2πx` at y = 1, 0 at y = 0 (eq. 36).
- **Case 3** (eq. 37–39, PDF p. 33): `r = [(x − 0.5)² + (y − 0.5)²]^½`;
  `α = 1/1500 + (1/3000) sin 2πx sin 2πy` for `0.349 ≤ r ≤ 0.35`, 1
  otherwise; **`u = sin 6πx` at both y = 1 and y = 0 (same sign)**, `u = 0`
  on the inner circle r = 0.05.
- **Case 1** (eq. 32–34): `α = 0.2` for y ∈ [0.6, 0.8], 1 otherwise;
  `u = sin 2πx` at y = 1, 0 at y = 0.

| PDF pages | Section | Use for |
| --- | --- | --- |
| 1–4 | Abstract, §1 introduction (IIM, ADI, ghost methods, RBF-FD primer; the feature list) | Manuscript §1 |
| 5–8 | §2 eq. 1 the operator, interface conditions; Fig. 1 node set with stencils; §2.1 standard RBF-FD weights eq. 2 (GA, MQ, PHS) | E2.2 |
| 8–11 | **§2.2.1 the 1-D construction**: eq. 3–7 time derivatives and continuity, eq. 8–9 expansions, eq. 10–13 `Dx` on coefficient vectors, eq. 14–15 multiplication matrices for a smooth alpha | E1.2 |
| 11–17 | **§2.2.2 worked 1-D example** (alpha = 1 on [−1, 0), 0.5 + 0.1 sin 2πx on [0, 1]; eq. 17–18 multiplication matrices, eq. 19–23 the continuity rows k = 0..2 in u and k = 0..1 in flux, **eq. 24–25 `C_L u_L = C_R u_R`**; Fig. 3 translated basis, Fig. 4 weights) | E1.2 tests (numbers to reproduce) |
| 17–24 | **§2.2.3 the 2-D construction**: eq. 26 expansions, Fig. 5 straddling rows, first (p − 2k) tangential monomials of u after k applications of the operator, first (p − 2k − 1) of the flux (eq. 27), eq. 28; **curvature**: the interface expansion inserted for y′ (eq. 29), matrix cos/sin in the normal (eq. 30), the numerical route via FD stencils on the interface representation | E2.3 |
| 24–26 | **§2.2.4 warped RBFs** (coordinate stretching across the interface enforcing flux continuity to first order; Fig. 6) | E2.4 |
| 26–28 | §3 setup: eq. 31 eps = 0.4/d (d = nearest-neighbour distance), 42 nodes / degree 5 interior, 30 / degree 4 across interfaces and near boundaries, curvature included by default; **§3.1 case 1**: eq. 32–34 (analytic solution), Fig. 7 | E2.5 |
| 29–32 | **§3.2 case 2**: eq. 35–36, Fig. 8–9; Fig. 10 FD4 / flat / curved vs 160,000-node reference (extrapolation: 10¹¹ nodes for 1e−8); Fig. 11 warp-and-straddle ablation | E2.6 |
| 33–37 | **§3.3 case 3**: eq. 37–39 (ring, inner circle, `sin 6πx` at both y = 0 and y = 1, confirmed from the rendered page); multi-interface translation; Fig. 12–14 | E2.7 |
| 38–42 | **§3.3.2 iterative solvers** (19 nodes / degree 3; control problem; gmres, bicgstab; the preconditioner reference is the dissertation's Appendix B; Fig. 15–18) | E2.8 |
| 42–45 | **§3.3.3 the extremizing parameter** eq. 40 (s = 10³ … 10¹¹, layer thickness 1/s, alpha ~ 1/(1.5 s)); Fig. 19 errors vs N per s; Fig. 20 condition number of the continuity matrices ~ O(s²) | E2.9, E4.8 |
| 45–46 | §4 conclusions (open issues: corners, higher-order RBF modification, 3-D cost, preconditioning, extreme contrasts) | Manuscript §7 |
| 47–49 | References (29 entries; [20] Reutskiy 2016, [21] Bayona et al. 2017, [26] Flyer et al. 2016) | Citations |

## MATLAB (read-only, `~/MathGraduateResearchAndCourseWork/`)

- `heatEq1DMatlab/FD4heat1DAC.m`: the **time-dependent** 1-D solver, one
  interface at x = 0, k1 = 1/9 left and k2 = 1 right, N + 1 nodes on
  [−1, 1], hand-built five-point weights for the four stencils that see the
  interface (`superStackMid`), explicit time stepping. `FD4heat1D.m` is the
  naive twin; `weights.m` is Fornberg's algorithm. The dissertation's §4.2
  equilibrium problem (two interfaces, smooth alpha) is not in this folder;
  E1.2 rebuilds it from the text.
- `heatEq2DMatlab/`: `ExeprepRBFHeatLaplace.m` is the driver script (defaults:
  N = 1250, `yLI = 0.6`, `yUI = 0.8`, `rho2 = 1/5` i.e. α = 0.2 in the band,
  `GAshp = 0.4`, 19-node / degree-3 stencils, `polydegreeInt = 2`,
  `numIntNodes = round(0.95 √N)` per straddling row, `RBFwarpFlag = 1`,
  flags `thinFlag`, `curvedFlag`, `closedFlag`). It calls
  `ExeprepRBFHeatLaplace4.m` (~60 kB of local functions: `curvedinterface1/2`,
  `mos2dsqperiodic7` repulsion, `createRBFLoperator1`, the continuity
  matrices, periodic kNN by tiling, `pointFinder1/2`). Version differences,
  checked 2026-09-20: `Laplace1` has flat interfaces only; `Laplace2` is the
  `closedFlag ≠ 0` variant the driver switches to; `Laplace3` and `Laplace4`
  share the curved interface functions (comment-only diff) and differ
  elsewhere (boundary stencil arguments). **The case-3 ring, the inner circle
  and the s-sweep are not in this folder**; that code is not preserved, so
  E2.7–E2.9 build from the paper alone. `RBFHeat1exe.m` / `RBFHeat2exe.m` are
  RK4 drivers with a ramped Dirichlet row, `FDheat1.m` the FD4 comparison,
  `laplaceSetup.m` the case-1 analytic solution (the 6 × 6 system for
  c₁ … c₆). The paper's 42 / degree-5 and 30 / degree-4 stencils are not the
  driver's defaults; the port follows the paper.
