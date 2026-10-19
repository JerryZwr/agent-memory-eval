"""Outcome classification (METHODOLOGY 1.4) and the two scoring denominators."""

from __future__ import annotations

import pytest

from agentmemeval.analysis.outcome import (
    Outcome,
    UndeliveredReason,
    classify_delivery,
)
from agentmemeval.analysis.scoring import Ratio, tally, tally_by
from agentmemeval.analysis.windows import WindowClass
from agentmemeval.data.models import EvidenceStatus
from agentmemeval.errors import StageError
from agentmemeval.pipeline.records import JudgeRecord

pytestmark = pytest.mark.unit

D, U, E = Outcome.DELIVERED, Outcome.UNDELIVERED, Outcome.ENV_FAILURE


class TestClassifyDelivery:
    def test_infra_failure_wins_over_any_text(self):
        assert classify_delivery("C", infra_failed=True).outcome is E

    @pytest.mark.parametrize("text", [None, "", "   \n"])
    def test_empty_is_undelivered(self, text):
        c = classify_delivery(text)
        assert (c.outcome, c.reason) == (U, UndeliveredReason.EMPTY)

    @pytest.mark.parametrize(
        "text",
        [
            "Let me check the records.",
            "I'll look into that.",
            "Checking...",
            "One moment please.",
            "让我查一下历史记录",
            "我再查一下",
        ],
    )
    def test_narration_is_undelivered(self, text):
        c = classify_delivery(text)
        assert (c.outcome, c.reason) == (U, UndeliveredReason.NARRATION)

    @pytest.mark.parametrize(
        "text",
        [
            "Got it.",
            "Received!",
            "Sure, I'll get back to you later.",
            "OK, will report back shortly.",
            "收到，稍后汇报",
            "好的",
        ],
    )
    def test_acknowledgement_is_undelivered(self, text):
        c = classify_delivery(text)
        assert (c.outcome, c.reason) == (U, UndeliveredReason.ACKNOWLEDGEMENT)

    @pytest.mark.parametrize(
        "text",
        [
            "C",
            "Thursdays at 10:00",
            "OK, the answer is C.",
            "The meeting was moved to later in the week.",
            "I cannot find this in the records.",  # a refusal is an answer
            "Let me summarise: the beta moved from March 21 to March 28 because the queue "
            "migration from Celery to Dramatiq took longer than planned in staging.",
        ],
    )
    def test_real_answers_stay_delivered(self, text):
        assert classify_delivery(text).outcome is D


def _rec(qid, outcome, verdict=None, ambiguous=False, status=EvidenceStatus.LOCATED):
    return JudgeRecord(
        qid=qid,
        qtype="F_SH",
        evidence_status=status,
        window_class=WindowClass.A,
        is_mc=False,
        outcome=outcome,
        gold="g",
        answer="a",
        verdict=verdict,
        ambiguous_extraction=ambiguous,
    )


class TestTally:
    def test_denominators(self):
        records = [
            _rec("q1", D, "CORRECT"),
            _rec("q2", D, "CORRECT", ambiguous=True),
            _rec("q3", D, "WRONG"),
            _rec("q4", U),
            _rec("q5", E),
            _rec("q6", E),
        ]
        t = tally(records)
        assert (t.n, t.delivered, t.undelivered, t.env_failure) == (6, 3, 1, 2)
        assert (t.correct, t.ambiguous_extraction) == (2, 1)
        assert t.accuracy == Ratio(2, 3)  # env_failure and undelivered excluded
        assert t.end_to_end == Ratio(2, 4)  # undelivered counts against, env_failure doesn't

    def test_refuses_delivered_without_verdict(self):
        with pytest.raises(StageError, match="not judged"):
            tally([_rec("q1", D)])

    def test_refuses_verdict_on_undelivered(self):
        with pytest.raises(StageError, match="has a verdict"):
            tally([_rec("q1", U, "WRONG")])

    def test_tally_by_groups(self):
        groups = tally_by(
            [_rec("a", D, "CORRECT"), _rec("b", D, "WRONG", status=EvidenceStatus.NO_EVIDENCE)],
            lambda r: r.evidence_status.value,
        )
        assert list(groups) == ["located", "no_evidence"]
        assert groups["located"].correct == 1

    def test_ratio_formatting(self):
        assert str(Ratio(6, 11)) == "6/11 = 55% [28%, 79%]"  # METHODOLOGY 1.7 example
        assert Ratio(6, 11).small
        assert not Ratio(30, 60).small
        assert str(Ratio(0, 0)) == "0/0 = n/a"
        assert Ratio(0, 0).value is None
