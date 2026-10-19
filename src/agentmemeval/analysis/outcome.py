"""Outcome classification (METHODOLOGY 1.4).

Every question ends in exactly one outcome before correctness is considered:

* `delivered`: a final answer was produced;
* `undelivered`: the run finished without an answer. That includes empty
  output, process narration ("let me check...") and acknowledgements sent through
  the delivery channel ("got it, will report back") with no answer behind them;
* `env_failure`: infrastructure failed. Never counted as a wrong answer;
* `agent_error`: the agent's own code crashed (agent layer only, M4).

Placeholder detection is deliberately narrow (short replies matching known
shapes) so that a real answer is not reclassified because of its phrasing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class Outcome(StrEnum):
    DELIVERED = "delivered"
    UNDELIVERED = "undelivered"
    ENV_FAILURE = "env_failure"
    AGENT_ERROR = "agent_error"


class UndeliveredReason(StrEnum):
    EMPTY = "empty"
    NARRATION = "narration"
    ACKNOWLEDGEMENT = "acknowledgement"
    BUDGET = "budget_exhausted"  # agent layer: stopped by a budget before answering
    TEXT_ONLY = "text_only"  # agent layer: ended with plain text, no delivery action


# Longer replies are treated as content even if they open like a placeholder.
PLACEHOLDER_MAX_WORDS = 20

_ACK_ONLY = re.compile(
    r"^(ok(ay)?|sure|got it|received|noted|roger|will do|on it|收到|好的)[\s,.!。！，]*$",
    re.IGNORECASE,
)
_ACK_START = re.compile(
    r"^(ok(ay)?|sure|got it|received|noted|roger|will do|on it|thanks|收到|好的)\b",
    re.IGNORECASE,
)
_DEFERRAL = re.compile(
    r"\b(later|shortly|soon|get back|report back|follow up|circle back|in a (moment|bit|while))\b"
    r"|稍后|稍等|回头|晚点|待会|之后(再)?(汇报|回复)",
    re.IGNORECASE,
)
_NARRATION = re.compile(
    r"^(let me (check|look|search|find|see|dig|verify|confirm)"
    r"|i('ll| will) (check|look|search|find|dig|verify)"
    r"|i'?m (checking|looking|searching)"
    r"|(checking|searching|looking (into|up))\b"
    r"|(one|just a) (moment|sec(ond)?)|please wait|hold on"
    r"|让我(查|看|找|确认)|我(再)?(查|看|找)一下)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Classification:
    outcome: Outcome
    reason: UndeliveredReason | None = None


def _word_count(text: str) -> int:
    # CJK text has no spaces; count characters / 2 as a rough word count.
    spaced = len(text.split())
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return max(spaced, cjk // 2)


def placeholder_kind(text: str) -> UndeliveredReason | None:
    """Return why `text` is a placeholder rather than an answer, or None."""
    stripped = text.strip()
    if _word_count(stripped) > PLACEHOLDER_MAX_WORDS:
        return None
    if _ACK_ONLY.match(stripped) or (_ACK_START.match(stripped) and _DEFERRAL.search(stripped)):
        return UndeliveredReason.ACKNOWLEDGEMENT
    if _NARRATION.match(stripped):
        return UndeliveredReason.NARRATION
    return None


def classify_delivery(text: str | None, *, infra_failed: bool = False) -> Classification:
    """Classify the text produced by the delivery action (the final answer channel)."""
    if infra_failed:
        return Classification(Outcome.ENV_FAILURE)
    if text is None or not text.strip():
        return Classification(Outcome.UNDELIVERED, UndeliveredReason.EMPTY)
    kind = placeholder_kind(text)
    if kind is not None:
        return Classification(Outcome.UNDELIVERED, kind)
    return Classification(Outcome.DELIVERED)
