# paper/data/

The manuscript's copy of the results files: one JSON per driver run, written
by the drivers with `--data-dir paper/data` (schema in
`src/heat_interfaces/results_cache.py`). `scripts/paper_numbers.py` asserts
every cache-backed number `main.tex` quotes against these files (E5.3, #44);
the working caches under `outputs/` are not the manuscript's data. Empty
until E5.3.
