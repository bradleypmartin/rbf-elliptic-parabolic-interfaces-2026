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
