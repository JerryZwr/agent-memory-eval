"""Tests for scripts/check_denylist.py.

Uses a throwaway fake denylist only; the real `.denylist.local` is never read.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_denylist.py"

# Fake terms. Each is also asserted to never appear in the hook's output.
FAKE_SUBSTRING = "AcmeWidgetCorp"
FAKE_CJK = "虚构项目代号"
FAKE_REGEX_BODY = r"\bzz-\d{3}\b"
FAKE_DENYLIST = f"""\
# comment lines and blank lines are not rules

{FAKE_SUBSTRING}
   {FAKE_CJK}
re:{FAKE_REGEX_BODY}
"""


def _load_module():
    spec = importlib.util.spec_from_file_location("check_denylist", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


cd = _load_module()


@pytest.fixture
def denylist(tmp_path: Path) -> Path:
    path = tmp_path / "fake.denylist"
    path.write_text(FAKE_DENYLIST, encoding="utf-8")
    return path


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _assert_no_rule_text(output: str) -> None:
    lowered = output.casefold()
    assert FAKE_SUBSTRING.casefold() not in lowered
    assert FAKE_CJK not in output
    assert FAKE_REGEX_BODY not in output
    assert "zz-" not in lowered


class TestParseRules:
    def test_skips_comments_and_blanks_and_numbers_rules(self):
        rules = cd.parse_rules(FAKE_DENYLIST.splitlines())
        assert [(r.number, r.source_line) for r in rules] == [(1, 3), (2, 4), (3, 5)]
        assert rules[0].substring == FAKE_SUBSTRING.casefold()
        assert rules[1].substring == FAKE_CJK  # surrounding whitespace stripped
        assert rules[2].pattern is not None

    def test_substring_is_case_insensitive(self):
        (rule,) = cd.parse_rules(["SecretThing"])
        assert rule.matches("a secretthing here")
        assert rule.matches("SECRETTHING")
        assert not rule.matches("secret thing")

    def test_regex_is_case_insensitive(self):
        (rule,) = cd.parse_rules([r"re:\bzz-\d{3}\b"])
        assert rule.matches("ticket ZZ-123 closed")
        assert not rule.matches("zz-12")

    def test_invalid_regex_is_rejected_without_echoing_it(self):
        with pytest.raises(cd.DenylistError) as excinfo:
            cd.parse_rules(["ok", "re:(unclosed-secret"])
        message = str(excinfo.value)
        assert "rule #2" in message
        assert "unclosed-secret" not in message

    @pytest.mark.parametrize("line", ["re:", "re:.*", "re:x?"])
    def test_regex_that_matches_everything_is_rejected(self, line):
        with pytest.raises(cd.DenylistError):
            cd.parse_rules([line])


class TestRun:
    def test_clean_files_pass(self, tmp_path, denylist, capsys):
        f = _write(tmp_path / "ok.py", "print('hello')\n")
        assert cd.run([f], denylist) == 0
        assert capsys.readouterr().err == ""

    def test_hits_report_file_line_and_rule_but_not_the_term(self, tmp_path, denylist, capsys):
        f = _write(
            tmp_path / "notes.md",
            f"line one\nwe used acmewidgetcorp tooling\nsee {FAKE_CJK}\nbug ZZ-042\n",
        )
        assert cd.run([f], denylist) == 1
        err = capsys.readouterr().err
        assert f"{f}:2: matches denylist rule #1 (denylist line 3)" in err
        assert f"{f}:3: matches denylist rule #2 (denylist line 4)" in err
        assert f"{f}:4: matches denylist rule #3 (denylist line 5)" in err
        assert "3 hit(s)" in err
        _assert_no_rule_text(err)

    def test_file_path_is_checked(self, tmp_path, denylist, capsys):
        f = _write(tmp_path / "acmewidgetcorp_export.txt", "harmless\n")
        assert cd.run([f], denylist) == 1
        err = capsys.readouterr().err
        assert "(file path): matches denylist rule #1" in err
        assert err.count("matches denylist rule") == 1

    def test_missing_denylist_skips_with_hint(self, tmp_path, capsys):
        f = _write(tmp_path / "x.txt", f"{FAKE_SUBSTRING}\n")
        assert cd.run([f], tmp_path / "does-not-exist") == 0
        assert "not found, skipping" in capsys.readouterr().err

    def test_invalid_denylist_fails_closed(self, tmp_path, capsys):
        bad = tmp_path / "bad.denylist"
        bad.write_text("re:[\n", encoding="utf-8")
        f = _write(tmp_path / "x.txt", "anything\n")
        assert cd.run([f], bad) == 2
        assert "invalid denylist" in capsys.readouterr().err

    def test_binary_files_are_skipped_for_content(self, tmp_path, denylist):
        f = tmp_path / "blob.bin"
        f.write_bytes(b"\x00\x01" + FAKE_SUBSTRING.encode())
        assert cd.run([str(f)], denylist) == 0

    def test_denylist_file_itself_is_never_scanned(self, denylist):
        assert cd.run([str(denylist)], denylist) == 0

    def test_main_accepts_denylist_option(self, tmp_path, denylist):
        f = _write(tmp_path / "x.txt", f"{FAKE_SUBSTRING}\n")
        assert cd.main(["--denylist", str(denylist), f]) == 1
