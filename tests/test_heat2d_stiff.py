"""The E4 driver: E4.2's separable references through the smooth flat band."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat2d_stiff import STUDY_DELTAS, main  # noqa: E402


def test_the_reference_table_at_the_study_deltas(capsys):
    rows = main(["--deltas", *map(str, STUDY_DELTAS)])
    out = capsys.readouterr().out
    assert "separable reference" in out and "distance / δ" in out
    assert len(rows) == 2 * len(STUDY_DELTAS)
    for r in rows:
        assert r["agreement"] < 5e-11
        if r["delta"] == 0.0:
            assert r["distance"] < 1e-12 and r["plateau"] == 0.0
        else:
            # The O(δ) gap to the jump solution (H10's floor).
            assert 0.6 < r["distance"] / r["delta"] < 1.7
    wide = [r for r in rows if r["delta"] == 0.04]
    assert all(abs(r["plateau"] - 1.07e-2) < 1e-4 for r in wide)
