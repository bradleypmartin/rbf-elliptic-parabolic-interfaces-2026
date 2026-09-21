# CLAUDE.md — rbf-elliptic-parabolic-interfaces-2026

## What this is

The elliptic and parabolic analogue of Brad's 2026-09 wave-equation work: a
Python port of the RBF-FD heat-transfer methods of his 2016 CU Boulder
dissertation (ch. 4–5) and of Martin & Fornberg, *Eng. Anal. Bound. Elem.*
79 (2017), verified against the published figures, then extended to
**sub-grid smooth edges** (a material edge of width δ smaller than the node
spacing) with ODE-continued *seed* stencils and measured against the
coefficient treatments in use, and written up as an arXiv manuscript.

Two deliverables: the code and the manuscript. **No talk, no slides.**

The plan, with every decision, risk, epic and ticket, is
[`docs/plan.md`](docs/plan.md). Read its sections 3–5 before touching E3–E5
work. Its epics and tickets are GitHub issues since 2026-09-20 (epics #2–#7,
tickets #8–#52; the number is on each heading), published by
`scripts/publish_issues.py`, which now refuses to run again without `--force`.

## Sources

`docs/paper-index.md` says which PDF pages matter; read only those with
`pdftotext -f A -l B -layout <pdf> -`. PDFs are gitignored; `papers/README.md`
has the checksums and `papers/fetch_papers.sh` verifies them.

- Dissertation: ch. 4 (1-D heat, PDF 75–89), ch. 5 (2-D heat, PDF 90–122),
  Appendix B (the preconditioner, PDF 140–145).
- Martin & Fornberg 2017 (EABE): the submitted manuscript, 49 pp.; the
  definitive test-case parameters.
- MATLAB (read-only, not vendored): `~/MathGraduateResearchAndCourseWork/`,
  `heatEq1DMatlab/` and `heatEq2DMatlab/`. The port follows the papers; the
  MATLAB is consulted for constants and geometry.
- Companion hyperbolic work (the wave port and the seed-stencil manuscript):
  `~/20260930-zd-ai-pdes-demo`, may move; this line is the only pointer to
  it in the repo. Nothing is imported from it (plan D2); cite its arXiv
  preprint, not its files.

## Repo layout

```
src/heat_interfaces/   library (filled in by the epics; module names are the plan's)
  fd_weights.py        Fornberg FD weights
  plotting.py          style; blue = interface-aware / seeds, orange = naive;
                       use_print_style() for the manuscript
  results_cache.py     JSON results cache the stiff drivers write (E5.3)
  heat1d/              domain, interface (continuity matrices, translated basis,
                       stencil solve), operators (naive Dx A Dx; jump-aware; seeds
                       dispatch), solve, march (BD4), exact (quadrature; Chebyshev
                       reference), stiff (seeds), treatments (comparators)
  heat2d/              domain (x-periodic strip, Dirichlet rows, interfaces,
                       straddling node sets), neighbors, rbf (GA + polynomials,
                       warped GA), interface (scalar continuity matrices with
                       curvature, multi-interface translation), operators, solve
                       (direct, gmres/bicgstab, preconditioners), march (BD4),
                       exact, fd4 (Cartesian Dx A Dx + Dy A Dy), resample (fine →
                       coarse through the fine stencils; cached references),
                       seeds, treatments
scripts/               drivers writing to outputs/; publish_issues.py;
                       paper_figures.py and paper_numbers.py (E5.3)
tests/                 pytest; every numerical routine has one
docs/                  plan.md, paper-index.md, port-notes.md (E1–E2 results and
                       decisions), stiff-diffusion.md (E3–E4, canonical for the
                       manuscript), figures/
papers/                README.md (sources, checksums), fetch_papers.sh; PDFs gitignored
paper/                 the manuscript (E5): main.tex → main.pdf, references.bib,
                       data/, figures/, make_arxiv.py, LICENSE (CC BY 4.0)
outputs/               generated (gitignored)
```

## Commands

```
uv sync                                   # .venv, Python 3.13
uv run pytest                             # tests
uv run ruff check . && uv run ruff format .
./papers/fetch_papers.sh                  # verify the reference PDFs
uv run python scripts/publish_issues.py   # dry run of the tickets in docs/plan.md
uv run python scripts/<driver>.py         # figures into outputs/ (defaults run in seconds;
                                          # sweeps sit behind flags and cache under outputs/)
(cd paper && tectonic main.tex)           # the manuscript, once E5.1 exists
```

## Conventions

- Python 3.13, `uv`, src layout. Deps: numpy, scipy, matplotlib. Add others
  only with a reason; check what's used first.
- `ruff format`, line length 88. Comments explain *why*, not what.
  Docstrings name the dissertation / EABE equation and figure numbers they
  implement.
- Tests for all numerical logic: convergence-order checks and
  analytic-solution comparisons, not just "it runs".
- Conventional Commits. Branch names `<issue>-<short-description>`. One PR
  per ticket; Brad reviews and merges.
- Naive baselines are assembled as `Dx A Dx` (+ `Dy A Dy`), dissertation
  eq. 76, never as a direct `(α u_x)_x` stencil.
- References: 1-D elliptic by quadrature; 1-D parabolic and 2-D flat by
  Chebyshev collocation on the separated 1-D problem; 2-D curved by a fine
  jump-aware run (δ = 0) or a Fourier × Chebyshev product grid (δ > 0). Cache
  them under `outputs/`.
- Default driver parameters run in seconds; bigger runs sit behind flags and
  record their run times in the notes.
- Notes: `docs/port-notes.md` for reproductions (2016 number next to ours),
  `docs/stiff-diffusion.md` canonical for the stiff-edge study. The
  manuscript quotes the notes and never becomes a second source of truth;
  every number in `main.tex` carries a `% TRACE` comment; figures and tables
  come from committed scripts, never hand-edited.
- Manuscript: after any edit under `paper/`, rebuild with tectonic, look at
  the changed pages with `pdftoppm`, commit `main.pdf` with the source.
  `\date` fixed by hand. Novelty wording only from `LITERATURE.md` §6.

## Hard constraints

- **This repo is PUBLIC (MIT; `paper/` CC BY 4.0).** Never commit PDFs,
  credentials, or anything from FullContact / Ziff Davis systems. Content is
  Brad's own academic work plus public papers.
- `papers/*.pdf` and `outputs/` are gitignored on purpose; don't un-ignore.
- Cite, don't claim: no statement of novelty outside what `LITERATURE.md`
  §6 allows once E5.2 exists; no unverified citation ships.
- Read the 1-D elliptic degeneracy (plan §3.2) before drawing conclusions
  from a 1-D steady-state experiment: seeds and exact-conductance finite
  volumes are both exact there.

## Working style (from Brad's global preferences)

Concise, technical, sparring-partner mode. State intent before non-trivial
changes, then take the wheel. Verify APIs rather than guess. Assume a second
agent may review the work.
