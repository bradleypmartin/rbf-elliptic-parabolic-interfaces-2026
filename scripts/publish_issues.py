"""Publish the epics and tickets of ``docs/plan.md`` as GitHub issues.

Section 6 of the plan is the source: every ``## E<k>: <title>`` heading is an
epic whose body runs to the first ``###``; every ``### E<k>.<n> <title>`` is a
ticket whose first lines may carry ``Labels:``, ``Size:`` and ``Depends on:``
metadata. Epics are created first so tickets can name their epic and their
dependencies by issue number, then each epic's body receives a checklist of
its tickets. Dry run by default; ``--create`` calls ``gh``.

    uv run python scripts/publish_issues.py
    uv run python scripts/publish_issues.py --create
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

EPIC_RE = re.compile(r"^## (E\d+): (.+?)\s*$")
TICKET_RE = re.compile(r"^### (E\d+\.\d+) (.+?)\s*$")
NUMBER_RE = re.compile(r"^(.*?)\s*\(#(\d+)\)$")  # a published heading ends in (#n)
META_RE = re.compile(r"^(Labels|Size|Depends on): (.*?)\s*$")
SECTION_RE = re.compile(r"^## (?!E\d+:)")  # any other level-2 heading ends section 6
DEP_RE = re.compile(r"E\d+(?:\.\d+)?")


@dataclass
class Ticket:
    id: str
    title: str
    body: str
    number: int | None = None
    labels: list[str] = field(default_factory=list)
    size: str = ""
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Epic:
    id: str
    title: str
    body: str
    number: int | None = None
    tickets: list[Ticket] = field(default_factory=list)


def split_number(title: str) -> tuple[str, int | None]:
    """Strip a trailing ``(#n)`` from a heading, returning the title and n."""
    if m := NUMBER_RE.match(title):
        return m.group(1), int(m.group(2))
    return title, None


def parse_plan(text: str) -> list[Epic]:
    """Return the epics of section 6, each with its tickets, in file order."""
    epics: list[Epic] = []
    epic: Epic | None = None
    ticket: Ticket | None = None
    buf: list[str] = []
    meta_zone = False

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        if ticket is not None:
            ticket.body = body
        elif epic is not None:
            epic.body = body
        buf = []

    for line in text.splitlines():
        if m := EPIC_RE.match(line):
            flush()
            ticket = None
            title, number = split_number(m.group(2))
            epic = Epic(id=m.group(1), title=title, body="", number=number)
            epics.append(epic)
            continue
        if epic is not None and SECTION_RE.match(line):
            flush()
            epic = None
            ticket = None
            continue
        if epic is None:
            continue
        if m := TICKET_RE.match(line):
            flush()
            title, number = split_number(m.group(2))
            ticket = Ticket(id=m.group(1), title=title, body="", number=number)
            epic.tickets.append(ticket)
            meta_zone = True
            continue
        if ticket is not None and meta_zone:
            if m := META_RE.match(line):
                key, value = m.group(1), m.group(2)
                if key == "Labels":
                    ticket.labels = [s.strip() for s in value.split(",") if s.strip()]
                elif key == "Size":
                    ticket.size = value
                else:
                    ticket.depends_on = DEP_RE.findall(value)
                continue
            if line.strip():
                meta_zone = False
        buf.append(line)
    flush()
    return epics


def gh(*args: str) -> str:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def create_issue(title: str, body: str, labels: list[str]) -> int:
    args = ["issue", "create", "--title", title, "--body", body]
    for label in labels:
        args += ["--label", label]
    url = gh(*args)
    return int(url.rstrip("/").rsplit("/", 1)[-1])


def ticket_body(t: Ticket, epic_num: int, numbers: dict[str, int]) -> str:
    deps = (
        ", ".join(f"#{numbers[d]}" if d in numbers else d for d in t.depends_on)
        or "none"
    )
    footer = f"\n\n---\nEpic: #{epic_num}. Size: {t.size or '?'}. Depends on: {deps}."
    return t.body + footer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plan", default="docs/plan.md", type=Path)
    parser.add_argument(
        "--create", action="store_true", help="call gh; default is a dry run"
    )
    parser.add_argument("--json", action="store_true", help="dry run as JSON")
    parser.add_argument(
        "--force",
        action="store_true",
        help="create even though headings already carry issue numbers",
    )
    args = parser.parse_args(argv)

    epics = parse_plan(args.plan.read_text())
    if not epics:
        print("no epics found", file=sys.stderr)
        return 1

    if not args.create:
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "id": e.id,
                            "title": e.title,
                            "tickets": [
                                {
                                    "id": t.id,
                                    "title": t.title,
                                    "labels": t.labels,
                                    "size": t.size,
                                    "depends_on": t.depends_on,
                                }
                                for t in e.tickets
                            ],
                        }
                        for e in epics
                    ],
                    indent=2,
                )
            )
        else:
            for e in epics:
                tag = f" #{e.number}" if e.number else ""
                print(f"{e.id}{tag}: {e.title}  ({len(e.tickets)} tickets)")
                for t in e.tickets:
                    deps = ", ".join(t.depends_on) or "none"
                    labels = ", ".join(t.labels)
                    tag = f" #{t.number}" if t.number else ""
                    print(f"  {t.id}{tag} [{t.size or '?'}; {labels}] {t.title}")
                    print(f"      depends on: {deps}")
            n = sum(len(e.tickets) for e in epics)
            print(f"\n{len(epics)} epics, {n} tickets. Re-run with --create.")
        return 0

    published = [x.id for e in epics for x in (e, *e.tickets) if x.number is not None]
    if published and not args.force:
        print(
            f"{len(published)} headings already carry issue numbers "
            f"(first: {published[0]}); refusing to create duplicates. "
            "Use --force for a fresh repository.",
            file=sys.stderr,
        )
        return 2

    numbers: dict[str, int] = {}
    for e in epics:
        numbers[e.id] = create_issue(f"{e.id}: {e.title}", e.body, [])
        print(f"created epic #{numbers[e.id]} {e.id}")
    for e in epics:
        for t in e.tickets:
            numbers[t.id] = create_issue(
                f"{t.id} {t.title}", ticket_body(t, numbers[e.id], numbers), t.labels
            )
            print(f"created #{numbers[t.id]} {t.id}")
    for e in epics:
        checklist = "\n".join(f"- [ ] #{numbers[t.id]} {t.title}" for t in e.tickets)
        body = f"{e.body}\n\n## Tickets\n\n{checklist}"
        gh("issue", "edit", str(numbers[e.id]), "--body", body)
        print(f"updated epic #{numbers[e.id]} with {len(e.tickets)} tickets")
    print(json.dumps(numbers, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
