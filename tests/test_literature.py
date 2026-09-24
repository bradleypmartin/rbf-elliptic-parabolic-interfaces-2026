"""The literature ledger and the bibliography (E5.2, #43).

`LITERATURE.md` §6 is the only novelty wording the manuscript may use, and
`paper/references.bib` may hold only verified entries (CLAUDE.md, "cite,
don't claim"). These pin the two together so a later edit to either cannot
cite a key that is missing or unverified.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BIB = (PAPER / "references.bib").read_text(encoding="utf-8")
LEDGER = (ROOT / "LITERATURE.md").read_text(encoding="utf-8")


def _load_make_arxiv():
    spec = importlib.util.spec_from_file_location("make_arxiv", PAPER / "make_arxiv.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_arxiv = _load_make_arxiv()


def _bib_keys() -> list[str]:
    return re.findall(r"^@\w+\{([^,\s]+),", BIB, re.M)


def _wording() -> str:
    """§6 of the ledger, the manuscript wording."""
    return LEDGER.split("\n## 6. Manuscript wording", 1)[1]


def test_every_bibliography_entry_carries_a_dated_verified_note():
    assert _bib_keys()
    assert make_arxiv.unverified_entries(BIB) == []


def test_bibliography_keys_are_unique():
    keys = _bib_keys()
    assert len(keys) == len(set(keys))


def test_bibliography_is_safe_for_pdftex():
    assert make_arxiv.unsafe_for_pdftex({"references.bib": BIB}) == []


def test_the_manuscript_wording_cites_only_bibliography_keys():
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]*)\}", _wording())
        for key in group.split(",")
    }
    assert cited
    assert sorted(cited - set(_bib_keys())) == []


def test_the_manuscript_wording_refers_only_to_existing_sections():
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]*)\}", _wording()))
    assert sorted(refs - labels) == []


def _claim_sentences() -> list[str]:
    """§6b's four quoted sentences: abstract-safe, conclusions, caveat, flux seeds."""
    claims = _wording().split("### 6b.", 1)[1].split("Not to be used:", 1)[0]
    return [
        " ".join(s.split())
        for s in re.findall(r'^- \*[^*]+\*\s+"(.*?)"$', claims, re.M | re.S)
    ]


def _prose(tex: str) -> str:
    """The text TeX typesets, comments dropped, \\cite{...} removed (§6 allows
    \\cite insertions into its sentences), whitespace collapsed."""
    text = re.sub(r"(?<!\\)%.*", "", tex)
    text = re.sub(r"\s*\\cite\{[^}]*\}", "", text)
    return " ".join(text.split())


def test_the_conclusions_quote_the_claim_sentences_verbatim():
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    conclusions = tex.split("\\subsection{Conclusions}", 1)[1].split("\\section*", 1)[0]
    sentences = _claim_sentences()
    assert len(sentences) == 4
    prose = _prose(conclusions)
    assert [s for s in sentences if s not in prose] == []


def test_the_manuscript_avoids_the_wording_the_ledger_rules_out():
    # LITERATURE.md §6b, "Not to be used"; "first" and "any order" are left out
    # here because "first order" and prior work's "schemes of any order" are
    # measurements and citations, not claims.
    prose = _prose((PAPER / "main.tex").read_text(encoding="utf-8")).lower()
    ruled_out = [
        "novel",
        "new method",
        "first high-order",
        "outperform",
        "beats harmonic",
        "arbitrary contrast",
        "kapitza",
    ]
    assert [w for w in ruled_out if w in prose] == []
