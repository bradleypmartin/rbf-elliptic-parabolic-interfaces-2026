# paper/data/

The manuscript's data: one results file per documented run, 22 in all (E5.3,
#44). `scripts/paper_data.py` lists the runs (`RUNS`, the commands of stiff
note §2.5, §5.1's table and the ring's docstring) and regenerates the files
from the working caches under `outputs/` in about ten minutes, 8 of them
`--mode tangential` and two the 1-D driver after its knee cache is invalidated
(it carries a hash of its code). The working caches are not the manuscript's
data; these files are.

```sh
uv run python scripts/paper_data.py            # every run, into paper/data
uv run python scripts/paper_data.py --only heat2d_stiff_snapshot.json
uv run python scripts/paper_data.py --list     # the commands
uv run python scripts/paper_data.py --verify   # the files against the commands
```

**Schema 2** (`src/heat_interfaces/results_cache.py`): `{"schema", "driver",
"date", "git": {"sha", "dirty"}, "argv", "args", "timings", "tables_sha256",
"tables"}`. `argv` is the command line, `tables_sha256` the tables' checksum;
`tables` holds every table the driver printed, and the
arrays its figures draw to six figures (`seed_functions/curves`,
`snapshot/curves`, `snapshot/field`, `spectra/figure`,
`stencils/seed_functions`). Float keys are `%g`
strings; lists of scalars sit on one line.

**What `--verify` asks of each file** (and `scripts/paper_numbers.py` before
any number): it is the file of a documented run and there is no other; it was
written by that run's driver from that run's command (`argv` without
`--outputs` and `--data-dir`); its tables match `tables_sha256`; from a tree
with no uncommitted change outside `paper/data` (`git.dirty` false); at a commit
in the history of HEAD; and it names no absolute path (the runs are told
`paper/data`, so a file is the same bytes from any checkout, timings, date and
commit aside). A file
holds the last run of its name, so a near-miss of a documented command run
with `--data-dir paper/data` (fewer counts, another seed) is refused, not
quoted. `paper_data.stale()` also lists, for information, any file whose run's
code (the package and its driver) has changed since its commit: the working
caches key on labels and versions, not on code (stiff note §5.1's caveat), so
rerunning after such a change may reprint the same numbers, and whether a
cache version needs a bump is the reader's call.

**What the gate does not prove.** It binds a file to its command, its commit
and its own checksum; it does not re-derive the numbers, which only rerunning
the command does (`paper_data.py`, then `git diff paper/data`). The checksum
catches an edit that did not recompute it (a merge, a hand "fix"), not a
deliberate one. `scripts/paper_numbers.py` recomputes every number the notes'
closing statements quote from the tables; the figures' arrays
(`…/curves`, `snapshot/field`, `spectra/figure`, `stencils/seed_functions`)
are covered by the checksum
and by `paper_figures.py --check` only. The figures and fragments never read a
run time (`tests/test_paper_figures.py` moves every one and redraws).

| File | Run (`scripts/…`) | Notes |
| --- | --- | --- |
| `heat1d_stiff.json` | `heat1d_stiff.py` | stiff §2.1–§2.5 |
| `heat2d_stiff_references.json` | `heat2d_stiff.py --mode references --deltas 0 0.04 0.01 0.005 0.0025 0.002 0.001 0.0005` | §4.1 |
| `heat2d_stiff_naive.json` | `--mode naive --counts 1250 … 160000` | §4.2 |
| `heat2d_stiff_naive_seed1.json`, `…_seed2.json` | `--mode naive --seed 1` (2) `--counts 1250 … 20000 --spectrum-counts` | §4.2, the scatter |
| `heat2d_stiff_stencils.json` | `--mode stencils` | §4.3 (and the seed figure's stencil, E5.7) |
| `heat2d_stiff_snapshot.json` | `--mode snapshot` | §5.1 |
| `heat2d_stiff_seeds.json` | `--mode seeds --operators naive construction direct direct-reach seeds seeds-plain seeds-edge --counts 1250 … 40000` | §4.5, §4.10 |
| `heat2d_stiff_seeds_jump.json` | `--mode seeds --deltas 0 --operators naive construction seeds --counts 1250 … 160000` | §4.5 |
| `heat2d_stiff_references_a0.02_sine.json` | `--mode references --amplitude 0.02` | §4.6 |
| `heat2d_stiff_seeds_a0.02_sine.json` | route (a): `--mode seeds --amplitude 0.02 --operators naive construction construction-flat direct direct-reach seeds seeds-plain --counts …` | §4.6 |
| `heat2d_stiff_seeds_tangential_a0.02_sine.json` | the same with `tangential tangential-plain tangential-edge` | §4.7, §4.10 |
| `heat2d_stiff_seeds_tangential_a0.02_constant.json` | geometry A: `--inside constant --deltas 0 0.0025` | §4.7 |
| `heat2d_stiff_seeds_tangential_a0_sine.json`, `…_jump.json` | geometry B: `--inside sine --deltas 0 0.0025`, and δ = 0 to 160,000 | §4.7 |
| `heat2d_stiff_tangential.json` | `--mode tangential --counts 1250 … 40000` | §4.7 |
| `heat2d_stiff_treatments.json`, `…_a0.02_sine.json` | `--mode treatments --counts 1250 … 40000` (and `--amplitude 0.02`) | §4.9 |
| `heat2d_stiff_eigenvalues.json` | `heat2d_stiff_eigenvalues.py` | §4.4 |
| `heat2d_stiff_eigenvalues_rows_n10000.json` | `heat2d_stiff_eigenvalues.py --mode rows --n 10000` | §4.4 |
| `heat2d_stiff_eigenvalues_spectra_n4900.json` | `--mode spectra --spectrum-n 4900 --deltas 0 0.005 --figure-delta 0.005` | §4.4 |
| `heat2d_ring_results.json` | `heat2d_ring.py`, the documented command | §4.8, §4.10 |

`--list` prints the full commands. These files are CC BY 4.0 with the rest of
`paper/`.
