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
| `heat1d_stiff.py` | the stiff-edge study (E3): the references at every δ, cached, with their convergence checks |
| `heat2d_nodesets.py` | the case-1/2/3 node sets (Fig. 5-3, EABE Fig. 8 and 12) |
| `heat2d_control.py` | the α ≡ 1 control and the interface-blind case-1 operators |
| `heat2d_interface.py` | interface-aware case 1 with plain Gaussians, continuity and conditioning tables |
| `heat2d_warp.py` | the warped Gaussian and the warp-and-straddle ablation on case 1 (EABE Fig. 7, 11) |
| `heat2d_case1.py` | case 1 elliptic and parabolic convergence, the 4900-node spectrum (Fig. 5-5, 5-6, EABE Fig. 7) |
| `heat2d_case2.py` | case 2: FD4 / flat / curved, the ablation, error vs wall-clock (Fig. 5-9–5-11, EABE Fig. 10, 11) |
| `heat2d_case3.py` | case 3: the insulating ring, FD4 / flat / curved, the mesh plot (Fig. 5-13, 5-14, EABE Fig. 13, 14) |
| `heat2d_iterative.py` | gmres / bicgstab, Appendix B and `spilu`, the DDR histograms (Fig. 5-15–5-18, B-1, EABE Fig. 15–18) |
| `heat2d_extremes.py` | the s-sweep of eq. 40 and the continuity matrices' conditioning (EABE Fig. 19, 20) |

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
