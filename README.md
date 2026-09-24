# rbf-elliptic-parabolic-interfaces-2026

A "decade-later" refresh and Python port of radial basis function-generated
finite differences (RBF-FD) for **elliptic and parabolic PDEs in domains with
interfaces**, and an extension of that work to material edges too steep for
the node spacing.

The sources are B. Martin's 2016 CU Boulder dissertation, *Application of
RBF-FD to Wave and Heat Transport Problems in Domains with Interfaces*
(ch. 4–5), and B. Martin & B. Fornberg, *Using radial basis
function-generated finite differences (RBF-FD) to solve heat transfer
equilibrium problems in domains with interfaces*, Eng. Anal. Bound. Elem. 79
(2017) 38–48, [doi:10.1016/j.enganabound.2017.03.005](https://doi.org/10.1016/j.enganabound.2017.03.005).
Both solve `u_t = ∇·(α ∇u)` with a thermal diffusivity α that jumps across
curved interfaces by rebuilding the stencils that cross an interface on a
piecewise-polynomial basis translated across it, and reach fourth order
where standard stencils are first order.

## What this repo does

1. **Port.** The 1-D and 2-D methods in Python (`numpy` / `scipy`), with tests,
   reproducing the published figures: convergence in 1-D, the flat-layer
   case with its analytic solution (elliptic and parabolic), two curved
   interfaces, a 0.001-wide insulating ring, iterative solvers and
   preconditioning, and the extreme-contrast sweep.
2. **Sub-grid smooth edges.** Between a jump and a smoothly varying α lies an
   edge that is smooth but narrower than the node spacing. Standard stencils
   through it converge at first order until the nodes resolve it; the jump
   construction is off at first order in δ/h. The stencils here are built from
   *seeds*, functions continued through the edge by ODEs that say what the
   diffusion operator allows, whose δ → 0 limit is the 2016 construction.
   Measured against the coefficient treatments in common use (harmonic and
   arithmetic cell means, widened edges, band-limited coefficients), in 1-D
   and 2-D, elliptic and parabolic.
3. **Manuscript.** An arXiv write-up in `paper/`, with a novelty ledger and
   scripted figures.

The plan, decisions, risks, epics and tickets are in
[`docs/plan.md`](docs/plan.md). The same construction for the wave equation
is the subject of a companion manuscript (2026); this repository stands on
its own.

## Quickstart

```sh
uv sync                                    # Python 3.13 venv with numpy / scipy / matplotlib
uv run pytest                              # tests
./papers/fetch_papers.sh                   # verify the reference PDFs (gitignored; see papers/README.md)
uv run python scripts/publish_issues.py    # dry run of the tickets in docs/plan.md
(cd paper && SOURCE_DATE_EPOCH=0 tectonic main.tex)   # the manuscript; see paper/README.md
```

Drivers in `scripts/` write figures and cached references to `outputs/`
(gitignored); their defaults run in seconds to a minute or two, and the
documented sweeps sit behind flags. Run times, the flagged commands and the
reproduction tables are in `docs/port-notes.md` (§1.7 and §2.10). See
`CLAUDE.md` for the layout and conventions.

| Driver | Regenerates |
| --- | --- |
| `heat1d_convergence.py` | dissertation Fig. 4-5, 4-6, 4-7 (1-D equilibrium) |
| `heat1d_parabolic.py` | 1-D parabolic convergence and operator spectra |
| `heat1d_stiff.py` | the stiff-edge study (E3): the references at every δ, cached, with their convergence checks; the naive knee, the δ = 0 construction and the seed operator on a smooth edge, elliptic and parabolic, with the floor constants, the rows' residuals, the seed weights against E1.2's as δ/h shrinks, the stencil solves' conditioning and the seed operator's spectra; the comparator table of the coefficient treatments (harmonic and arithmetic cell means, the widened edge, the finite-volume scheme with exact face conductances) against the seeds at every δ; the seed functions across a stencil against the monomials and the translated basis, a parabolic snapshot with the edge unresolved, and the run's results JSON (`--data-dir` for the manuscript's copy) |
| `heat2d_nodesets.py` | the case-1/2/3 node sets (Fig. 5-3, EABE Fig. 8 and 12) |
| `heat2d_control.py` | the α ≡ 1 control and the interface-blind case-1 operators |
| `heat2d_interface.py` | interface-aware case 1 with plain Gaussians, continuity and conditioning tables |
| `heat2d_warp.py` | the warped Gaussian and the warp-and-straddle ablation on case 1 (EABE Fig. 7, 11) |
| `heat2d_case1.py` | case 1 elliptic and parabolic convergence, the 4900-node spectrum (Fig. 5-5, 5-6, EABE Fig. 7) |
| `heat2d_case2.py` | case 2: FD4 / flat / curved, the ablation, error vs wall-clock (Fig. 5-9–5-11, EABE Fig. 10, 11) |
| `heat2d_case3.py` | case 3: the insulating ring, FD4 / flat / curved, the mesh plot (Fig. 5-13, 5-14, EABE Fig. 13, 14) |
| `heat2d_iterative.py` | gmres / bicgstab, Appendix B and `spilu`, the DDR histograms (Fig. 5-15–5-18, B-1, EABE Fig. 15–18) |
| `heat2d_extremes.py` | the s-sweep of eq. 40 and the continuity matrices' conditioning (EABE Fig. 19, 20) |
| `heat2d_stiff.py` | the stiff-edge study in 2-D (E4): the smooth flat band's separable references in y at every δ, elliptic and parabolic, with their convergence checks and their O(δ) distance from the jump solution (`--mode references`); the naive `Dx A Dx + Dy A Dy` knee and the δ = 0 construction on its floor at every δ and count, elliptic and parabolic, the α ≡ 1 run beside them, the straddling-row readings of each solution (the y-profile, the flux on each side of the innermost pair and its jump) and how each depends on h/δ, and the coarse sets' growing mode against BD4 (`--mode naive`; errors cached); the scalar seeds on real stencils: the march against the monomials, the shift identity, a finite-difference residual and the 1-D march, the δ = 0 identity with the 2016 translated-basis rows and warp over every crossing stencil, the distance from them and the seed block's conditioning as δ/h falls to 1e-5, and the cost per stencil (`--mode stencils`); the flat δ sweep, the seeds against naive, the construction and the direct operator at every (n, δ) with the warp ablation and the resolved-edge penalty (`--mode seeds`, E4.6), and on case 2's sine pair against the product grid with the truncation probe and the curved-over-flat ratios, plus the two geometries that split case 2's departures from case 1 (`--mode seeds --amplitude 0.02`, `--inside`, E4.7; `--mode references --amplitude 0.02` checks the product grid); the coefficient treatments on scattered nodes (harmonic and arithmetic means of α over a disc of radius h/2 and h, the widened edge) against sampling and the seeds, with the crossover in h/δ, the widened edge on its own floor and the ranking, on case 1 and case 2 (`--mode treatments`, E4.9) |
| `heat2d_ring.py` | EABE eq. 40's ring with seeds (E4.8): the Fig. 19 twin at δ = 0 with E2.3, the seeds and the seeds without their flux seeds against E2.9's references; the Fig. 20 twin, the weights' residual on the matched radial profile and the conditioning at every s and edge width, with the stored-radii ablation; the seed operator's spectra and diagonal dominance; the smooth ring (the resistivity blended across tanh edges), each operator's truncation probe on its exact radial mode and a far-field self-convergence line against a fine seed run (cached; stiff note §4.8) |
| `heat2d_stiff_eigenvalues.py` | the seed rows in the global matrix (E4.5): per δ/h and operator (naive, direct, the δ = 0 construction, the seeds warped and plain), the rows each method replaces, their diagonal dominance before and after Appendix B's sweeps, a one-norm condition estimate, SuperLU's time and residual, the elliptic error, and `gmres` / `bicgstab` unpreconditioned, with Appendix B's `P` and with `spilu` (`--mode rows`, cached); the dense interior spectra of the same operators at every δ with BD4's root modulus at `dt = h` and the Fig. 5-6 twin with a seeds panel (`--mode spectra`, cached per node count and δ) |

## Layout

| Path | Contents |
| --- | --- |
| `src/heat_interfaces/` | Library: `heat1d/` and `heat2d/` (domains, RBF-FD weights, interface-aware stencils, seeds, solvers, references, treatments), shared Fornberg weights, plotting style, results cache |
| `scripts/` | Drivers for the figures; `publish_issues.py` |
| `tests/` | pytest suite (convergence orders and analytic comparisons) |
| `docs/` | `plan.md`, `paper-index.md` (page ranges per PDF), `port-notes.md`, `stiff-diffusion.md`, `figures/` |
| `papers/` | Index of reference PDFs with checksums and a fetch script; PDFs not committed |
| `paper/` | The manuscript (CC BY 4.0): `main.tex` → `main.pdf`, `references.bib`, data, figures, arXiv packaging |

## Licence

Code: MIT (see `LICENSE`). The manuscript under `paper/`: CC BY 4.0.
The original MATLAB implementations live in a separate personal repository
and are not vendored here.
