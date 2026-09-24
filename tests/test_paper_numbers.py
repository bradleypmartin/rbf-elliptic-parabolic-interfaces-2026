"""``scripts/paper_numbers.py`` (E5.3, #44): the rounding rule and the gate."""

import json
import math
import re
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import paper_numbers  # noqa: E402
from paper_data import DATA  # noqa: E402
from paper_numbers import Book, Check, fit, holds, main, rounds_to, unit  # noqa: E402


def test_a_quoted_figure_holds_half_a_unit_in_its_last_digit():
    # Decimals and scientific notation: the last printed digit sets the unit.
    assert unit("0.52") == pytest.approx(0.01)
    assert unit("2.63e-3") == pytest.approx(1e-5)
    assert unit("1.598e-5") == pytest.approx(1e-8)
    assert rounds_to(0.5249, "0.52") and not rounds_to(0.5251, "0.52")
    assert rounds_to(2.6349e-3, "2.63e-3") and not rounds_to(2.6351e-3, "2.63e-3")
    assert rounds_to(-7.2664, "-7.27") and not rounds_to(-7.2749, "-7.26")
    # An integer's own digits count; "34" is 33.5–34.5.
    assert rounds_to(34.49, "34") and not rounds_to(34.51, "34")
    assert rounds_to(576, "576") and not rounds_to(577, "576")
    # Zero is exact.
    assert rounds_to(0, "0") and not rounds_to(1e-300, "0")


def test_trailing_zeros_are_not_significant_but_the_tolerance_is_capped():
    # "2000" would be a unit of 1000 (500–2500 would pass); the cap keeps it to
    # 6 % of the figure, so a round headline number cannot drift 25 %.
    assert unit("2000") == pytest.approx(240)
    assert rounds_to(2037, "2000") and rounds_to(2119, "2000")
    assert not rounds_to(2121, "2000") and not rounds_to(2500, "2000")
    assert unit("790") == pytest.approx(10)
    assert rounds_to(791.4, "790") and not rounds_to(795.1, "790")


def test_bounds_hold_at_the_quoted_precision():
    assert holds(1.15e-12, "1e-12", "<=") and not holds(1.6e-12, "1e-12", "<=")
    assert holds(2.63, "2.6", "<=") and not holds(2.66, "2.6", "<=")
    assert holds(0.4654, "0.46", ">=") and not holds(0.454, "0.46", ">=")
    assert holds(-0.68, "0", "<=") and not holds(1e-9, "0", "<=")
    assert not holds(math.nan, "1", "=") and not holds(math.inf, "1", "<=")
    with pytest.raises(ValueError):
        holds(1.0, "1", "<")


def test_the_book_spans_bounds_and_lists():
    b = Book()
    b.span("w", "range", [0.861, 0.9, 0.938], "0.86", "0.94")
    b.within("w", "envelope", [116.7, 198.6], "100", "200")
    b.each("w", "list", [134, 155], ["134", "155"])
    assert [c.ok for c in b.checks] == [True] * 6
    assert [c.kind for c in b.checks[2:4]] == [">=", "<="]
    with pytest.raises(ValueError):
        b.each("w", "short", [1.0], ["1", "2"])
    line = Check("stiff §5.3 (4)", "the fit", 4.231, "4.23").line()
    assert line.startswith("ok   stiff §5.3 (4)") and line.endswith("4.231 ≈ 4.23")
    assert Check("w", "x", 1.0, "2").line().startswith("FAIL")


def test_fit_is_the_order_in_h():
    rows = [{"n": n, "h": 1 / n, "e": 3.0 * n**-4.0} for n in (10, 20, 40, 80)]
    assert fit(rows, "e") == pytest.approx(4.0)
    assert fit(rows, "e", (20, 40)) == pytest.approx(4.0)
    with pytest.raises(KeyError):
        fit(rows, "e", (20, 160))


def test_the_committed_data_hold_every_number(capsys):
    assert main(["--quiet"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].endswith("numbers hold")
    checked, total = (int(w) for w in out[0].split()[:3:2])
    assert checked == total > 300
    # Every statement the directive names has checks, whether or not the
    # manuscript quotes them yet.
    wheres = {c.where.split("; ")[-1] for c in paper_numbers.build(DATA)}
    assert {f"stiff §2.5 ({k})" for k in range(1, 5)} <= wheres
    assert {f"stiff §5.3 ({k})" for k in range(1, 12)} <= wheres
    assert "stiff §5.1 snapshot" in wheres


SECTION = r"(?:abstract|§\d+(?:\.\d+)*)"
WHERE = re.compile(
    rf"^(?:{SECTION}(?:, {SECTION})*; )?stiff §\d+\.\d+ (?:\(\d+\)|snapshot|P\d+)$"
)


def test_every_where_is_the_notes_statement_after_any_quoting_sections():
    assert paper_numbers.cited("stiff §5.3 (4)", "abstract", "§1") == (
        "abstract, §1; stiff §5.3 (4)"
    )
    checks = paper_numbers.build(DATA)
    assert all(WHERE.match(c.where) for c in checks), [
        c.where for c in checks if not WHERE.match(c.where)
    ]
    # The drafts so far (E5.4, #45; E5.5, #46) quote from the abstract and §1–3;
    # §3 quotes the construction's own tables, keyed by stiff §2.3's predictions.
    quoting = {
        s for c in checks if "; " in c.where for s in c.where.split("; ")[0].split(", ")
    }
    assert {"abstract", "§1", "§1.1", "§2.1", "§3.1", "§3.2"} <= quoting
    assert {f"stiff §2.3 P{k}" for k in (4, 5, 6, 7)} <= {
        c.where.split("; ")[-1] for c in checks
    }


def test_a_failed_check_fails_the_run(capsys, monkeypatch):
    def wrong(f, b):
        b.eq("stiff §0", "a number the data no longer give", 1.0, "2")

    monkeypatch.setattr(paper_numbers, "STATEMENTS", (wrong,))
    assert main(["--quiet"]) == 1
    out = capsys.readouterr().out
    assert "FAIL stiff §0" in out and "0 of 1 numbers hold" in out


def test_the_gate_refuses_another_data_dir(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path)]) == 1
    assert "no number checked" in capsys.readouterr().out
    # A copy of the committed files with one run's command changed: a near-miss
    # of a documented command is refused before any number.
    copy = tmp_path / "data"
    shutil.copytree(DATA, copy)
    path = copy / "heat2d_stiff_snapshot.json"
    data = json.loads(path.read_text())
    data["argv"] = [*data["argv"], "--seed", "1"]
    path.write_text(json.dumps(data))
    assert main(["--quiet", "--data-dir", str(copy)]) == 1
    out = capsys.readouterr().out
    assert "heat2d_stiff_snapshot.json: made by" in out and "numbers hold" not in out
