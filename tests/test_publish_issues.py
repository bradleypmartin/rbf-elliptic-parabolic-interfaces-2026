import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from publish_issues import main, parse_plan, split_number, ticket_body  # noqa: E402

SAMPLE = """\
# Plan

## 1. Purpose

Prose that is not an epic.

## E0: Scaffold

The epic body.

### E0.1 Do the scaffold
Labels: documentation
Size: S
Depends on: —

Body of the first ticket.

**Done when**
- it is green.

### E0.2 Second thing
Labels: enhancement, documentation
Size: M
Depends on: E0.1, E3.1

Second body.

## E1: Port

Another epic.

### E1.1 A ticket
Labels: enhancement
Size: L
Depends on: E0.2

Third body.

## 7. Publishing

Not an epic either.
"""


def test_parse_plan_finds_epics_tickets_and_metadata():
    epics = parse_plan(SAMPLE)
    assert [e.id for e in epics] == ["E0", "E1"]
    assert epics[0].title == "Scaffold"
    assert epics[0].body == "The epic body."
    t0, t1 = epics[0].tickets
    assert (t0.id, t0.title) == ("E0.1", "Do the scaffold")
    assert t0.labels == ["documentation"]
    assert t0.size == "S"
    assert t0.depends_on == []
    assert t0.body.startswith("Body of the first ticket.")
    assert t0.body.endswith("- it is green.")
    assert t1.labels == ["enhancement", "documentation"]
    assert t1.depends_on == ["E0.1", "E3.1"]
    assert epics[1].tickets[0].body == "Third body."


def test_trailing_section_is_not_swallowed_into_the_last_epic():
    epics = parse_plan(SAMPLE)
    assert "Not an epic" not in epics[1].tickets[0].body
    assert "Not an epic" not in epics[1].body


def test_ticket_body_resolves_known_dependencies_to_numbers():
    epics = parse_plan(SAMPLE)
    t1 = epics[0].tickets[1]
    body = ticket_body(t1, epic_num=10, numbers={"E0.1": 11})
    assert body.endswith("Epic: #10. Size: M. Depends on: #11, E3.1.")


def test_the_real_plan_parses_with_unique_ids_and_backward_dependencies():
    plan = Path(__file__).resolve().parents[1] / "docs" / "plan.md"
    epics = parse_plan(plan.read_text())
    ids = [t.id for e in epics for t in e.tickets]
    assert len(ids) == len(set(ids))
    assert len(epics) == 6
    known = set(ids) | {e.id for e in epics}
    for e in epics:
        for t in e.tickets:
            assert t.labels, t.id
            assert t.size in {"S", "M", "L"}, t.id
            for d in t.depends_on:
                assert d in known, (t.id, d)


def test_split_number_strips_a_published_suffix_only():
    assert split_number("Scaffold (#2)") == ("Scaffold", 2)
    assert split_number("Scaffold") == ("Scaffold", None)
    assert split_number("Case (#3) study") == ("Case (#3) study", None)


def test_published_headings_carry_numbers_and_block_create(tmp_path, capsys):
    published = SAMPLE.replace("## E0: Scaffold", "## E0: Scaffold (#2)").replace(
        "### E0.1 Do the scaffold", "### E0.1 Do the scaffold (#8)"
    )
    epics = parse_plan(published)
    assert (epics[0].title, epics[0].number) == ("Scaffold", 2)
    assert (epics[0].tickets[0].title, epics[0].tickets[0].number) == (
        "Do the scaffold",
        8,
    )
    assert epics[0].tickets[1].number is None
    plan = tmp_path / "plan.md"
    plan.write_text(published)
    assert main(["--plan", str(plan), "--create"]) == 2
    assert "refusing to create duplicates" in capsys.readouterr().err


def test_the_real_plan_is_fully_published():
    plan = Path(__file__).resolve().parents[1] / "docs" / "plan.md"
    epics = parse_plan(plan.read_text())
    numbers = [x.number for e in epics for x in (e, *e.tickets)]
    assert None not in numbers
    assert len(numbers) == len(set(numbers)) == 51
