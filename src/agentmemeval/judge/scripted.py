"""Rule-based offline judge for open-ended answers.

Not a calibrated judge: it exists so the offline pipeline has a deterministic
judging step with the same interface as the LLM judge (M5). Rules:

* no-evidence questions (F_HL): CORRECT iff the answer is a refusal;
* otherwise a refusal is WRONG, and an answer is CORRECT iff it contains the
  gold answer (whitespace- and case-normalised).
"""

from __future__ import annotations

from agentmemeval.data.models import NO_EVIDENCE_QTYPE
from agentmemeval.judge.base import Judge, JudgeInput, JudgeVerdict
from agentmemeval.text import normalize

REFUSAL_MARKERS = (
    "cannot find",
    "can't find",
    "could not find",
    "couldn't find",
    "no record",
    "not in the records",
    "no information",
    "don't know",
    "do not know",
    "unable to",
)


def is_refusal(answer: str) -> bool:
    text = normalize(answer)
    return any(marker in text for marker in REFUSAL_MARKERS)


class ScriptedJudge(Judge):
    @property
    def model(self) -> str:
        return "scripted-judge"

    @property
    def prompt_version(self) -> str:
        return "scripted-v1"

    def judge(self, item: JudgeInput) -> JudgeVerdict:
        refused = is_refusal(item.answer)
        if item.qtype == NO_EVIDENCE_QTYPE:
            if refused:
                return self._verdict("CORRECT", "no-evidence question and the answer refuses")
            return self._verdict("WRONG", "no-evidence question but the answer asserts content")
        if refused:
            return self._verdict("WRONG", "the answer refuses but the gold answer exists")
        gold = normalize(item.gold).strip(" .")
        if gold and gold in normalize(item.answer):
            return self._verdict("CORRECT", "the answer contains the gold answer")
        return self._verdict("WRONG", "the gold answer does not appear in the answer")

    @staticmethod
    def _verdict(label: str, reason: str) -> JudgeVerdict:
        return JudgeVerdict(label=label, reason=reason, raw=f"{label}: {reason}")
