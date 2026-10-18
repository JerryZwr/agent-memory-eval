"""Multiple-choice extraction and the scripted open-question judge."""

from __future__ import annotations

import pytest

from agentmemeval.judge.base import JudgeInput
from agentmemeval.judge.mc import extract_choice
from agentmemeval.judge.scripted import ScriptedJudge, is_refusal

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "answer",
    [
        "C",
        " C ",
        "(C)",
        "C.",
        "C)",
        "**C**",
        "Answer: C",
        "The answer is C",
        "the correct answer is (C).",
        "Option C",
        "Final answer: **C**",
        "C. Thursdays at 10:00",
        "(C) Thursdays at 10:00",
        "答案：C",
        "答案是C。",
    ],
)
def test_strict_forms_are_unambiguous(answer):
    ex = extract_choice(answer)
    assert (ex.letter, ex.ambiguous, ex.method) == ("C", False, "strict")


@pytest.mark.parametrize(
    "answer",
    [
        "Based on the records, I believe C is right.",
        "The sync is on Thursdays, so the choice would be C because of the update.",
    ],
)
def test_single_letter_dug_out_of_text_is_flagged(answer):
    ex = extract_choice(answer)
    assert (ex.letter, ex.ambiguous, ex.method) == ("C", True, "search")


@pytest.mark.parametrize(
    "answer",
    [
        "It is either B or C.",
        "B, not C",
        "A good guess is C",  # the article "A" makes this ambiguous on purpose
    ],
)
def test_several_letters_yield_no_letter_and_a_flag(answer):
    ex = extract_choice(answer)
    assert (ex.letter, ex.ambiguous, ex.method) == (None, True, "search_multiple")


@pytest.mark.parametrize("answer", ["", "I cannot find this in the records.", "Thursdays at 10"])
def test_no_letter_is_not_ambiguous(answer):
    ex = extract_choice(answer)
    assert (ex.letter, ex.ambiguous, ex.method) == (None, False, "none")


def test_letters_inside_words_and_lowercase_are_ignored():
    assert extract_choice("Bayside asked about a CDN").method == "none"


def _judge(qtype, gold, answer):
    return ScriptedJudge().judge(
        JudgeInput(qid=f"{qtype}_T_1", qtype=qtype, question="q", gold=gold, answer=answer)
    )


class TestScriptedJudge:
    def test_contains_gold_is_correct(self):
        v = _judge("F_SH", "PostgreSQL 16", "On 2025-03-03 they chose postgresql  16 as primary.")
        assert v.label == "CORRECT"
        assert v.raw.startswith("CORRECT")

    def test_missing_gold_is_wrong(self):
        assert _judge("F_SH", "Priya Nair", "Jun Takeda found the flaky test.").label == "WRONG"

    def test_refusal_on_answerable_question_is_wrong(self):
        assert _judge("F_SH", "87%", "I cannot find this in the records.").label == "WRONG"

    def test_refusal_on_no_evidence_question_is_correct(self):
        v = _judge(
            "F_HL", "No payment provider was discussed.", "I cannot find this in the records."
        )
        assert v.label == "CORRECT"

    def test_content_on_no_evidence_question_is_wrong(self):
        v = _judge("F_HL", "There is no record of that.", "On 2025-03-05 Sam said: backups copied.")
        assert v.label == "WRONG"

    def test_versions_are_recorded(self):
        j = ScriptedJudge()
        assert (j.model, j.prompt_version) == ("scripted-judge", "scripted-v1")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I cannot find this in the records.", True),
        ("There is NO RECORD of that.", True),
        ("I don't know.", True),
        ("PostgreSQL 16", False),
    ],
)
def test_is_refusal(text, expected):
    assert is_refusal(text) is expected
