"""``scripts/paper_figures.py`` (E5.3, #44): the figures drawn from paper/data."""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heat_interfaces.plotting import TEXTWIDTH, use_print_style  # noqa: E402
from paper_figures import FIGURES, TABLES, Data, check, main, write  # noqa: E402


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
