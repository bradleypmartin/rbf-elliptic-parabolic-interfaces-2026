# papers/ — reference PDFs

PDFs are **not committed** (`papers/*.pdf` is gitignored; this repo is
public). `docs/paper-index.md` says which pages of each matter, so a session
reads only those (`pdftotext -f A -l B -layout <pdf> -`).

| File | What | Source | sha256 |
| --- | --- | --- | --- |
| `martin-dissertation-2016-rbf-fd-interfaces.pdf` (145 pp.) | B. P. Martin, *Application of RBF-FD to Wave and Heat Transport Problems in Domains with Interfaces*, PhD dissertation, CU Boulder, 2016. Ch. 4 (1-D heat), ch. 5 (2-D heat), Appendix B (the preconditioner). | Local copy only; [ProQuest 10151046](https://www.proquest.com/openview/ae4d936114520c2d34e604adeda7d81a/1?pq-origsite=gscholar&cbl=18750) (not fetched by script). | `a658c8b94547eb085eed68e1bd70cca00ed9c49eba8911d8412576d9945e96f2` |
| `martin-fornberg-2017-rbf-fd-heat-equilibrium-eabe-submitted.pdf` (49 pp.) | B. Martin, B. Fornberg, *Using radial basis function-generated finite differences (RBF-FD) to solve heat transfer equilibrium problems in domains with interfaces*, Eng. Anal. Bound. Elem. 79 (2017) 38–48, [doi:10.1016/j.enganabound.2017.03.005](https://doi.org/10.1016/j.enganabound.2017.03.005). The author post-print (accepted manuscript, double-spaced, 49 pp.), not the typeset article. | [CU Scholar record `bk128b94m`](https://scholar.colorado.edu/concern/articles/bk128b94m), [download](https://scholar.colorado.edu/downloads/2j62s5711); the record says "post-print version of an article published in Engineering Analysis with Boundary Elements", licence CC BY-ND 4.0, rights statement In Copyright. Fetched 2026-09-20 and byte-identical to the local copy (E0.2, #9). | `e1ef8560939a6504f1a9df0311b473ab5dc07bce0f406b63607162ad16fbe78b` |

Public papers the literature pass (E5.2) fetches will be added to this table
and to `fetch_papers.sh` with their checksums, as they are read.

```sh
./papers/fetch_papers.sh    # fetch the public PDFs and verify every checksum
```
