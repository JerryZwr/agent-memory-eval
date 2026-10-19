"""LLM judge v1 / v2 (METHODOLOGY 1.5): failure is an error, never WRONG; only the prompt varies."""

from __future__ import annotations

import pytest

from agentmemeval.errors import JudgeError, LLMUnavailableError
from agentmemeval.judge.base import JudgeInput
from agentmemeval.judge.llm_judge import SYSTEM_PROMPTS, LLMJudge
from agentmemeval.llm.base import LLMResponse

pytestmark = pytest.mark.unit
ITEM = JudgeInput(qid="F_HL_X_1", qtype="F_HL", question="q?", gold="He did not take part.",
                  answer="I cannot find any record of that.")  # fmt: skip


class FakeLLM:
    model = "fake+thinking=off"

    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.calls = text, exc, []

    def complete(self, messages, *, temperature=0.0, response_format=None, **kw):
        self.calls.append((messages, temperature, response_format))
        if self.exc:
            raise self.exc
        return LLMResponse(text=self.text, model=self.model)


def test_verdict_reason_and_raw_are_kept():
    llm = FakeLLM('{"verdict": "correct", "reason": "refusal matches a negative gold"}')
    v = LLMJudge(llm, "v2").judge(ITEM)
    assert (v.label, v.reason) == ("CORRECT", "refusal matches a negative gold")
    assert v.raw.startswith("{")
    messages, temperature, fmt = llm.calls[0]
    assert temperature == 0.0
    assert fmt == {"type": "json_object"}
    assert "json" in messages[0].content  # DeepSeek JSON mode requires the word


def test_call_failure_is_a_judge_error_not_wrong():
    with pytest.raises(JudgeError, match="judge call failed"):
        LLMJudge(FakeLLM(exc=LLMUnavailableError("timeout")), "v1").judge(ITEM)


@pytest.mark.parametrize("text", ["not json", '{"reason": "x"}', '{"verdict": "MAYBE"}'])
def test_unusable_replies_are_judge_errors(text):
    with pytest.raises(JudgeError):
        LLMJudge(FakeLLM(text), "v1").judge(ITEM)


def test_v2_is_v1_plus_the_refusal_and_strictness_rules():
    v1, v2 = SYSTEM_PROMPTS["v1"], SYSTEM_PROMPTS["v2"]
    assert v2.startswith(v1)  # nothing in v1 is changed, only added to
    added = v2[len(v1) :]
    assert "did not happen" in added and "background" in added.lower()
    assert "hedged guess" in added
    assert "both correct and incorrect" in added


def test_versions_are_recorded_and_unknown_ones_refused():
    assert LLMJudge(FakeLLM("{}"), "v1").prompt_version == "llm-v1"
    assert LLMJudge(FakeLLM("{}"), "v2").model == "fake+thinking=off"
    with pytest.raises(ValueError):
        LLMJudge(FakeLLM("{}"), "v3")
