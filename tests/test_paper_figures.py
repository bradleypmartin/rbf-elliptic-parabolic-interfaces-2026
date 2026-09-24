"""``scripts/paper_figures.py`` (E5.3, #44): the figures drawn from paper/data."""

import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat_interfaces.plotting import TEXTWIDTH, use_print_style  # noqa: E402
from paper_figures import DATA, FIGURES, TABLES, Data, check, main, write  # noqa: E402


class Recording(Data):
    """``Data`` that notes which results files a drawer reads."""

    def __init__(self):
        super().__init__()
        self.read: set[str] = set()

    def tables(self, name: str) -> dict:
        self.read.add(name)
        return super().tables(name)


def _boxes(fig):
    """Every legend, title and axis label of the figure's visible axes, drawn."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    titles, others = [], [legend.get_window_extent(renderer) for legend in fig.legends]
    for ax in fig.get_axes():
        if not ax.get_visible():
            continue
        if ax.get_title():
            titles.append((ax.get_title(), ax.title.get_window_extent(renderer)))
        for label in (ax.xaxis.label, ax.yaxis.label):
            if label.get_text():
                others.append(label.get_window_extent(renderer))
        if ax.get_legend() is not None:
            others.append(ax.get_legend().get_window_extent(renderer))
    return titles, others


@pytest.mark.parametrize("name", list(FIGURES))
def test_each_figure_fits_the_page_and_reads_only_its_files(name):
    # The E5.3 review of the drafts: a five-column legend ran off both edges of
    # the page, and long titles ran into their neighbours'.
    use_print_style()
    data = Recording()
    fig = FIGURES[name].draw(data)
    assert data.read == set(FIGURES[name].files)
    assert fig.get_figwidth() == pytest.approx(TEXTWIDTH)
    width, height = fig.bbox.width, fig.bbox.height
    titles, others = _boxes(fig)
    for box in [b for _, b in titles] + others:
        assert -0.5 <= box.x0 and box.x1 <= width + 0.5, name
        assert -0.5 <= box.y0 and box.y1 <= height + 0.5, name
    for i, (a, box_a) in enumerate(titles):
        for b, box_b in titles[i + 1 :]:
            assert not box_a.overlaps(box_b), (name, a, b)
    plt.close(fig)


def _labelled(ax) -> bool:
    """Whether ``ax`` prints tick labels on its y axis, on either side."""
    params = ax.yaxis.get_tick_params(which="major")
    shown = params.get("labelleft", True) or params.get("labelright", False)
    return shown and any(t.get_text() for t in ax.get_yticklabels())


@pytest.mark.parametrize("name", list(FIGURES))
def test_every_log_axis_and_every_row_of_panels_has_a_scale(name):
    # #49, placing the 2-D figures: an axhline's x data (the axes' own 0 and 1)
    # were read as node counts, which put a tick at 0 on the warp panels' log
    # axes and crushed their lines to the right edge; and the smooth ring's
    # far-field row, whose first column is hidden, printed no y tick labels.
    use_print_style()
    fig = FIGURES[name].draw(Data())
    fig.canvas.draw()
    rows: dict[tuple[int, int], list[bool]] = {}
    for ax in fig.get_axes():
        if not ax.get_visible():
            continue
        for scale, (low, _), ticks in (
            (ax.get_xscale(), ax.get_xlim(), ax.get_xticks()),
            (ax.get_yscale(), ax.get_ylim(), ax.get_yticks()),
        ):
            if scale == "log":
                assert low > 0 and min(ticks) > 0, (name, ax.get_title())
        spec = ax.get_subplotspec()
        if spec is not None:
            key = (id(spec.get_gridspec()), spec.rowspan.start)
            rows.setdefault(key, []).append(_labelled(ax))
    assert all(any(row) for row in rows.values()), name
    plt.close(fig)


def test_a_figure_is_the_same_bytes_every_time_and_the_text_width_wide(tmp_path):
    names = ["heat1d_stiff_knee.pdf", "heat2d_stiff_snapshot.pdf"]
    first = write(names, tmp_path / "a", Data())
    second = write(names, tmp_path / "b", Data())
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes()
        box = re.search(rb"/MediaBox \[ *0 0 ([0-9.]+) ([0-9.]+) *\]", a.read_bytes())
        assert float(box[1]) == pytest.approx(360.0 * 72 / 72.27, abs=0.01)


def test_check_names_what_differs_what_is_missing_and_what_is_not_generated(tmp_path):
    names = ["heat2d_ring_convergence.pdf", "heat1d_stiff_seeds.pdf"]
    write(names, tmp_path, Data())
    assert check(names, Data(), tmp_path) == []
    path = tmp_path / "heat1d_stiff_seeds.pdf"
    path.write_bytes(path.read_bytes() + b"%")
    (tmp_path / "heat2d_ring_convergence.pdf").unlink()
    (tmp_path / "old.pdf").write_bytes(b"%PDF")
    assert check(names, Data(), tmp_path) == [
        "missing heat2d_ring_convergence.pdf",
        "differs heat1d_stiff_seeds.pdf",
        "not generated old.pdf (remove it, or register it)",
    ]


def test_main_writes_and_checks_the_named_files(tmp_path, capsys):
    argv = ["--only", "heat2d_stiff_dominance.pdf", "--out-dir", str(tmp_path)]
    assert main(argv) == 0
    assert (tmp_path / "heat2d_stiff_dominance.pdf").exists()
    assert main([*argv, "--check"]) == 0
    assert "1 of 1 identical" in capsys.readouterr().out
    assert set(TABLES).isdisjoint(FIGURES)
    assert all("." not in Path(n).stem for n in [*FIGURES, *TABLES])


def test_the_committed_figures_are_what_the_committed_data_draws(capsys):
    # E5.3's done-when, and make_arxiv's gate: paper/figures is paper/data's.
    assert main(["--check"]) == 0, capsys.readouterr().out


TIMING = re.compile(r"(seconds|_ms|^ms|^total_s|^solved|build)$")


def _slow_down(obj):
    """``obj`` with every run time (``TIMING``, the rows' SuperLU ``direct``) × 1.37."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            number = isinstance(v, (int, float)) and not isinstance(v, bool)
            timed = TIMING.search(k) or (k == "direct" and number)
            out[k] = v * 1.37 if timed and number else _slow_down(v)
        return out
    if isinstance(obj, list):
        return [_slow_down(v) for v in obj]
    return obj


def test_no_figure_or_fragment_depends_on_a_run_time(tmp_path):
    # E5.3: rerunning the documented runs into an empty paper/data reproduced
    # every figure and fragment but the one printing a product-grid solve's
    # seconds (1.7 s committed, 1.6 s fresh). Run times are the notes', not the
    # data's: move every one and the files must not change.
    slowed = tmp_path / "data"
    shutil.copytree(DATA, slowed)
    moved = 0
    for path in slowed.glob("*.json"):
        data = json.loads(path.read_text())
        before = json.dumps(data["tables"])
        data["tables"] = _slow_down(data["tables"])
        data["timings"] = {k: 1.37 * v for k, v in data["timings"].items()}
        moved += json.dumps(data["tables"]) != before
        path.write_text(json.dumps(data))
    assert moved >= 10
    assert check([*FIGURES, *TABLES], Data(slowed)) == []
