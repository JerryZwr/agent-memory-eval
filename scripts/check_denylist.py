"""Pre-commit hook: refuse commits containing terms from the local denylist.

The denylist (`.denylist.local` at the repo root, gitignored, maintained by the
author) has one rule per line:

- blank lines and lines starting with `#` are ignored;
- `re:<pattern>` is a Python regular expression, matched case-insensitively;
- anything else is a case-insensitive substring.

The sensitive terms are themselves sensitive, so this script never prints a
rule or the line that matched it: a hit is reported as file, line number and
rule number only. File paths are checked too (reported as line 0), because a
path is committed just like content.

Exit codes: 0 clean (or no denylist), 1 hits found, 2 the denylist is invalid.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DENYLIST = Path(__file__).resolve().parent.parent / ".denylist.local"
REGEX_PREFIX = "re:"


class DenylistError(Exception):
    """The denylist itself is malformed. Refuse rather than check with fewer rules."""


@dataclass(frozen=True)
class Rule:
    number: int  # 1-based, counting rules only (comments and blanks excluded)
    source_line: int  # line number inside the denylist file
    substring: str | None = None  # casefolded
    pattern: re.Pattern[str] | None = None

    def matches(self, text: str) -> bool:
        if self.pattern is not None:
            return self.pattern.search(text) is not None
        assert self.substring is not None
        return self.substring in text.casefold()


@dataclass(frozen=True)
class Hit:
    path: str
    line: int  # 0 means the file path itself matched
    rule: Rule

    def describe(self) -> str:
        where = f"{self.path} (file path)" if self.line == 0 else f"{self.path}:{self.line}"
        rule = self.rule
        return f"{where}: matches denylist rule #{rule.number} (denylist line {rule.source_line})"


def parse_rules(lines: Iterable[str]) -> list[Rule]:
    rules: list[Rule] = []
    for source_line, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        number = len(rules) + 1
        if text.startswith(REGEX_PREFIX):
            expr = text[len(REGEX_PREFIX) :]
            if not expr:
                raise DenylistError(f"rule #{number} (denylist line {source_line}): empty regex")
            try:
                pattern = re.compile(expr, re.IGNORECASE)
            except re.error as exc:
                # exc.msg / exc.pos describe the error without echoing the pattern.
                raise DenylistError(
                    f"rule #{number} (denylist line {source_line}): invalid regex: "
                    f"{exc.msg} at position {exc.pos}"
                ) from None
            if pattern.search("") is not None:
                raise DenylistError(
                    f"rule #{number} (denylist line {source_line}): regex matches the empty "
                    "string and would flag every line"
                )
            rules.append(Rule(number, source_line, pattern=pattern))
        else:
            rules.append(Rule(number, source_line, substring=text.casefold()))
    return rules


def load_rules(denylist: Path) -> list[Rule]:
    return parse_rules(denylist.read_text(encoding="utf-8").splitlines())


def _read_text(path: Path) -> str | None:
    """Return the file's text, or None for binary / undecodable files."""
    data = path.read_bytes()
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_file(path: Path, rules: Sequence[Rule], display: str | None = None) -> list[Hit]:
    name = display if display is not None else path.as_posix()
    # Match the path as committed (relative), not the absolute path on this machine.
    hits = [Hit(name, 0, rule) for rule in rules if rule.matches(name)]
    text = _read_text(path)
    if text is None:
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        hits.extend(Hit(name, lineno, rule) for rule in rules if rule.matches(line))
    return hits


def run(files: Sequence[str], denylist: Path) -> int:
    if not denylist.is_file():
        print(
            f"check-denylist: {denylist.name} not found, skipping. "
            "Create it at the repo root to enable this check (see PLAN.md M0).",
            file=sys.stderr,
        )
        return 0
    try:
        rules = load_rules(denylist)
    except DenylistError as exc:
        print(f"check-denylist: invalid denylist: {exc}", file=sys.stderr)
        return 2

    denylist_resolved = denylist.resolve()
    hits: list[Hit] = []
    for name in files:
        path = Path(name)
        if not path.is_file() or path.resolve() == denylist_resolved:
            continue
        hits.extend(scan_file(path, rules, display=name))

    if not hits:
        return 0
    for hit in hits:
        print(hit.describe(), file=sys.stderr)
    print(
        f"check-denylist: {len(hits)} hit(s). Remove the sensitive content before committing.",
        file=sys.stderr,
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="files to check (pre-commit passes staged files)")
    parser.add_argument(
        "--denylist",
        type=Path,
        default=DEFAULT_DENYLIST,
        help="path to the denylist (default: .denylist.local at the repo root)",
    )
    args = parser.parse_args(argv)
    return run(args.files, args.denylist)


if __name__ == "__main__":
    raise SystemExit(main())
