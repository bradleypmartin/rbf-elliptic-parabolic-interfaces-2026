"""The packaging gates and transforms of paper/make_arxiv.py (E5.1, #42).

The script is not a package; it is loaded from its path. The rebuild gate
itself (tectonic, pdftotext) is exercised by running the script, not here.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def _load():
    spec = importlib.util.spec_from_file_location("make_arxiv", PAPER / "make_arxiv.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_arxiv = _load()

VERIFIED_BIB = """\
% VERIFIED 2026-09-25 against the Crossref record.
@article{A2017,
  title = {a},
}
"""


def _paper(tmp_path, tex, bib=VERIFIED_BIB):
    (tmp_path / "main.tex").write_text(tex, encoding="utf-8")
    (tmp_path / "references.bib").write_text(bib, encoding="utf-8")
    return tmp_path


def test_strip_drops_whole_line_comments_only():
    tex = "% TRACE: notes\n  % indented\ntext % inline stays\n\\%literal\nmore\n"
    out, removed = make_arxiv.strip_whole_line_comments(tex)
    assert removed == 2
    assert out == "text % inline stays\n\\%literal\nmore\n"


def test_strip_refuses_verbatim():
    with pytest.raises(SystemExit, match="verbatim"):
        make_arxiv.strip_whole_line_comments("\\begin{verbatim}\n%\n\\end{verbatim}\n")


def test_draft_markers_skip_whole_line_comments():
    tex = (
        "% a \\stub{} in a comment ships stripped\n"
        "\\stub{\\#45: intro}\n"
        "text\n"
        "   % \\nocite{*} commented out\n"
        "\\nocite{*}\n"
        "\\todo{x}\n"
    )
    assert make_arxiv.draft_markers(tex) == [
        "line 2: \\stub",
        "line 5: \\nocite{*}",
        "line 6: \\todo",
    ]


def test_draft_markers_pass_a_finished_source():
    assert make_arxiv.draft_markers("\\nocite{A2017}\n\\cite{A2017}\n") == []


def test_unverified_entries():
    bib = """\
% VERIFIED 2026-09-25 against the PDF.
@phdthesis{Ok2016,
  title = {a},
}

% TODO(verify, #43): the Crossref record.
@article{Todo2017,
  title = {b},
}

@article{Bare2017,
  title = {c},
}

% VERIFIED against the arXiv record (no date).
@misc{Undated2026,
  title = {d},
}

% VERIFIED 2026-09-25, but a field is still TODO(verify).
@book{Half1966,
  title = {e},
}

% VERIFIED 2026-09-25 above a blank line, which ends the block.

@article{Detached1988,
  title = {f},
}

@string{jcp = {J. Comput. Phys.}}
"""
    assert make_arxiv.unverified_entries(bib) == [
        "Todo2017",
        "Bare2017",
        "Undated2026",
        "Half1966",
        "Detached1988",
    ]


def test_unverified_entries_read_headers_across_lines():
    """The /spar finding: a key off the ``@type{`` line was skipped, not checked."""
    bib = """\
% TODO(verify, #43): never checked.
@article{
  NextLine2020,
  title = {x},
}

% TODO(verify, #43)
@article
{Split2020,
  title = {x},
}

% TODO(verify, #43)
  @Article( Paren2020 ,
  title = {x},
)

% VERIFIED 2026-09-25 against the Crossref record.
@article{
  VerifiedNextLine2020,
  title = {x},
}

@Comment{ignored}
@STRING{jcp = {J. Comput. Phys.}}
"""
    assert make_arxiv.unverified_entries(bib) == [
        "NextLine2020",
        "Split2020",
        "Paren2020",
    ]


def test_unverified_entries_fail_closed_on_unreadable_headers():
    bib = """\
% VERIFIED 2026-09-25 against the PDF.
@article{NoFields2020}

% VERIFIED 2026-09-25 against the PDF.
@article{, title = {no key}}

% VERIFIED 2026-09-25 against the PDF.
@{Typeless2020,
  title = {x},
}
"""
    assert make_arxiv.unverified_entries(bib) == [
        "line 2: unreadable",
        "line 5: unreadable",
        "line 8: unreadable",
    ]


def test_package_refuses_stubs_first(tmp_path):
    """A draft is refused before anything else is asked for (no main.bbl here)."""
    paper = _paper(tmp_path, "\\begin{document}\n\\stub{\\#45}\n\\end{document}\n")
    refusal = r"(?s)draft \(1 draft marker\).*line 2: \\stub"
    with pytest.raises(SystemExit, match=refusal):
        make_arxiv.package(paper, verify=False)


def test_package_refuses_unverified_bibliography(tmp_path):
    bib = "% TODO(verify, #43)\n@article{X2017,\n  title = {x},\n}\n"
    paper = _paper(tmp_path, "\\begin{document}\n\\end{document}\n", bib)
    with pytest.raises(SystemExit, match="1 entry without a dated VERIFIED.*X2017"):
        make_arxiv.package(paper, verify=False)


def test_package_past_the_draft_gates_asks_for_the_bbl(tmp_path):
    paper = _paper(tmp_path, "\\begin{document}\n\\cite{A2017}\n\\end{document}\n")
    with pytest.raises(SystemExit, match="main.bbl not found"):
        make_arxiv.package(paper, verify=False)


def test_package_stages_and_tars_without_verify(tmp_path):
    tex = "% TRACE: stiff 2.5\n\\begin{document}\n\\cite{A2017} 5\\%\n\\end{document}\n"
    paper = _paper(tmp_path, tex)
    (paper / "main.bbl").write_text("\\bibitem[A17]{A2017}\n", encoding="utf-8")
    tar = make_arxiv.package(paper, verify=False)
    assert tar.exists()
    staged = (paper / "arxiv" / "main.tex").read_text(encoding="utf-8")
    assert staged == "\\begin{document}\n\\cite{A2017} 5\\%\n\\end{document}\n"
    assert (paper / "arxiv" / "main.bbl").exists()
    assert "VERIFIED" not in (paper / "arxiv" / "references.bib").read_text()


def test_run_repo_checks_refuses_a_missing_script(tmp_path):
    with pytest.raises(SystemExit, match="paper_numbers.py not found"):
        make_arxiv.run_repo_checks(tmp_path)


def test_referenced_figures_resolves_extensionless_names(tmp_path):
    (tmp_path / "a.pdf").touch()
    (tmp_path / "b.png").touch()
    tex = (
        "\\includegraphics[width=\\textwidth]{a}\n"
        "\\includegraphics{b}\n\\includegraphics{figures/a}\n\\includegraphics{c}\n"
    )
    got = make_arxiv.referenced_figures(tex, tmp_path)
    assert [p.name for p in got] == ["a.pdf", "b.png", "c"]


def test_referenced_inputs_adds_tex_and_dedups():
    tex = "\\input{figures/tab_a.tex}\n\\input{figures/tab_b}\n"
    tex += "\\input{figures/tab_a.tex}"
    assert make_arxiv.referenced_inputs(tex) == [
        "figures/tab_a.tex",
        "figures/tab_b.tex",
    ]


def test_non_ascii_labels():
    bbl = "\\bibitem[AS55]{a}\n\\bibitem[Mü73]{b}\n\\bibitem[Pó22]{c}\n"
    assert make_arxiv.non_ascii_labels(bbl) == ["Mü73", "Pó22"]


def test_unsafe_for_pdftex_flags_greek_but_not_accents():
    texts = {"ok": "Pólya – Mühlbach", "bad": "width δ"}
    got = make_arxiv.unsafe_for_pdftex(texts)
    assert len(got) == 1 and got[0].startswith("bad: 'δ'")


def test_date_of():
    assert make_arxiv.date_of("\\title{x}\n\\date{September 24, 2026}\n") == (
        "September 24, 2026"
    )


# The manuscript itself: conventions every later ticket keeps.


def test_manuscript_date_is_fixed_by_hand():
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    stripped, _ = make_arxiv.strip_whole_line_comments(tex)
    assert "\\today" not in stripped
    date = make_arxiv.date_of(stripped)
    assert date and re.fullmatch(r"[A-Z][a-z]+ \d{1,2}, \d{4}", date), date


def test_manuscript_stubs_name_their_e5_ticket():
    """Every stub says which of #43-#52 replaces it (vacuous once none remain)."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    body = tex.split("\\begin{document}", 1)[1]
    stubs = re.findall(r"\\stub\{\\#(\d+)", body)
    assert len(stubs) == body.count("\\stub{")
    assert all(43 <= int(n) <= 52 for n in stubs), stubs


def test_manuscript_is_safe_for_pdftex():
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    stripped, _ = make_arxiv.strip_whole_line_comments(tex)
    assert not make_arxiv.unsafe_for_pdftex({"main.tex": stripped})
