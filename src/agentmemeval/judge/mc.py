"""Deterministic option-letter extraction for multiple-choice answers (METHODOLOGY 1.5).

Two tiers:

* strict: the whole reply is a recognised answer form ("C", "(C)", "**C**",
  "Answer: C", "C. option text", "答案：C"). Unambiguous.
* search: otherwise scan the text for standalone letters A-D. Finding one is
  still flagged `ambiguous` (the letter had to be dug out of free text);
  finding several yields no letter. Never silently take the first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from agentmemeval.data.models import MC_LETTERS

_L = "".join(MC_LETTERS)
_STRICT_PATTERNS = [
    re.compile(rf"^\(?([{_L}])\)?[.):]?$"),
    re.compile(rf"^\*\*\(?([{_L}])\)?[.)]?\*\*$"),
    re.compile(
        rf"^(?:the\s+)?(?:correct\s+|best\s+|final\s+)?(?:answer|option|choice)"
        rf"(?:\s+is)?\s*[:：]?\s*\(?\**([{_L}])\**\)?\.?$",
        re.IGNORECASE,
    ),
    re.compile(rf"^答案\s*(?:是|为)?\s*[:：]?\s*([{_L}])\s*[。.]?$"),
    # "C. Thursdays at 10:00" / "(C) Thursdays ..." - the reply leads with its choice.
    re.compile(rf"^\(?([{_L}])[.)]\s+\S"),
]
_STANDALONE = re.compile(rf"(?<![A-Za-z0-9])([{_L}])(?![A-Za-z0-9])")

Method = Literal["strict", "search", "search_multiple", "none"]


@dataclass(frozen=True)
class ChoiceExtraction:
    letter: str | None
    ambiguous: bool
    method: Method


def extract_choice(answer: str) -> ChoiceExtraction:
    text = answer.strip()
    for pattern in _STRICT_PATTERNS:
        match = pattern.match(text)
        if match:
            return ChoiceExtraction(match.group(1), ambiguous=False, method="strict")
    found = sorted(set(_STANDALONE.findall(text)))
    if not found:
        return ChoiceExtraction(None, ambiguous=False, method="none")
    if len(found) == 1:
        return ChoiceExtraction(found[0], ambiguous=True, method="search")
    return ChoiceExtraction(None, ambiguous=True, method="search_multiple")
