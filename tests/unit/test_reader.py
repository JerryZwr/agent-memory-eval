"""Answer prompt round trip and the scripted extractive reader."""

from __future__ import annotations

import pytest

from agentmemeval.arms.base import RetrievedItem
from agentmemeval.data.models import EvidenceStatus, Question
from agentmemeval.llm.scripted import REFUSAL, ScriptedLLM, extractive_reader, make_reader_llm
from agentmemeval.pipeline.prompts import (
    PromptRecord,
    parse_answer_prompt,
    render_answer_prompt,
)

pytestmark = pytest.mark.unit


def _item(text, date="2025-03-03", speaker="Ann Lee", group="Group 1", n=0):
    return RetrievedItem(
        message_id=f"{date}/{group}/{n + 1}",
        date=date,
        time=f"{date} 09:0{n}:00",
        group=group,
        speaker=speaker,
        text=text,
        score=1.0,
    )


def _q(question, options=None, qid="F_SH_T_001", gold="x"):
    if options:
        qid, gold = "MA_C_T_001", "A"
    return Question(
        qid=qid,
        qtype=qid[:4],
        question=question,
        gold=gold,
        options=options,
        evidence_status=EvidenceStatus.UNRECOVERABLE,
    )


OPTS = {"A": "Mondays at 09:00", "B": "Tuesdays at 14:00", "C": "Thursdays at 10:00", "D": "x | y"}


class TestPromptRoundTrip:
    def test_round_trip_with_options_and_awkward_text(self):
        items = [_item("multi\nline | with pipes", n=0), _item("second", n=1)]
        view = parse_answer_prompt(render_answer_prompt(_q("When is  the\nsync?", OPTS), items))
        assert view.question == "When is the sync?"
        assert view.options == OPTS
        head = ("2025-03-03", "Group 1", "Ann Lee")
        assert view.records == (
            PromptRecord(head[0], f"{head[0]} 09:00:00", *head[1:], "multi line | with pipes"),
            PromptRecord(head[0], f"{head[0]} 09:01:00", *head[1:], "second"),
        )

    def test_round_trip_without_records_or_options(self):
        view = parse_answer_prompt(render_answer_prompt(_q("Who?"), []))
        assert view.records == ()
        assert view.options is None

    def test_system_prompt_states_refusal_policy(self):
        system = render_answer_prompt(_q("Who?"), [])[0]
        assert system.role == "system"
        assert "cannot find" in system.content


def _ask(question, items, options=None):
    return extractive_reader(render_answer_prompt(_q(question, options), items))


class TestReaderOpenQuestions:
    def test_refuses_without_records(self):
        assert _ask("Which database is primary?", []) == REFUSAL

    def test_answers_from_the_best_matching_record(self):
        items = [_item("lunch is at noon", n=0), _item("PostgreSQL is the primary database", n=1)]
        answer = _ask("Which database is primary?", items)
        assert answer.startswith("On 2025-03-03, Ann Lee said: PostgreSQL is the primary database")
        assert "lunch" not in answer

    def test_refuses_when_records_barely_overlap(self):
        items = [_item("the database migration is next week")]
        assert _ask("Which payment provider handles database billing invoices?", items) == REFUSAL


class TestReaderMultipleChoice:
    def test_guesses_deterministically_without_records(self):
        first = _ask("When is the weekly sync held?", [], OPTS)
        assert first in "ABCD"
        assert all(_ask("When is the weekly sync held?", [], OPTS) == first for _ in range(3))

    def test_picks_option_supported_by_records(self):
        items = [_item("The weekly sync moves to Thursdays at 10:00")]
        assert _ask("When is the weekly sync held?", items, OPTS) == "C"

    def test_newer_record_wins_a_tie(self):
        opts = {"A": "60 per minute", "B": "120 per minute", "C": "200 per minute", "D": "9"}
        items = [
            _item("limit is 120 per minute", date="2025-03-03", n=0),
            _item("limit raised to 200 per minute", date="2025-03-04", n=1),
        ]
        assert _ask("What is the current limit?", items, opts) == "C"

    def test_unresolvable_tie_hedges_between_options(self):
        opts = {"A": "March 14", "B": "March 21", "C": "March 28", "D": "April 4"}
        items = [_item("beta moves from March 21 to March 28")]
        assert _ask("When is the beta planned?", items, opts) == "It is either B or C."


def test_scripted_llm_reports_its_model_and_wraps_any_responder():
    llm = ScriptedLLM(lambda msgs: "fixed", model="scripted/test")
    assert llm.model == "scripted/test"
    assert llm.complete([]).text == "fixed"
    assert make_reader_llm().model == "scripted/extractive-reader-v1"
