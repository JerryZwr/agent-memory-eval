"""Evidence-window classes A-F and question selection (METHODOLOGY 1.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentmemeval.analysis.windows import (
    WindowClass,
    class_counts,
    classify,
    classify_all,
    select_questions,
)
from agentmemeval.data.models import EvidenceStatus, Question
from agentmemeval.errors import DatasetError
from agentmemeval.pipeline.config import ArmSpec, DatasetSpec, QuestionSpec, RunConfig
from agentmemeval.pipeline.runner import run_pipeline

pytestmark = pytest.mark.unit

A, B, C, D, E = WindowClass.A, WindowClass.B, WindowClass.C, WindowClass.D, WindowClass.E
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_corpus"


def _q(qid, ids=(), status=EvidenceStatus.LOCATED):
    return Question(
        qid=qid, qtype=qid[:4], question="q", gold="g", evidence_status=status, evidence_ids=ids
    )


class TestClassify:
    WINDOW = ("2025-01-01", "2025-01-02")

    @pytest.mark.parametrize(
        ("ids", "expected"),
        [
            (("2025-01-01/G/1",), A),
            (("2025-01-01/G/1", "2025-01-02/H/4"), A),
            (("2025-01-02/G/1", "2025-01-03/G/1"), B),
            (("2025-01-03/G/1",), C),
            (("2024-12-31/G/1",), C),
        ],
    )
    def test_located_questions(self, ids, expected):
        assert classify(_q("F_SH_X_1", ids), self.WINDOW) is expected

    def test_no_evidence_is_d_in_any_window(self):
        q = _q("F_HL_X_1", status=EvidenceStatus.NO_EVIDENCE)
        assert classify(q, self.WINDOW) is D
        assert classify(q, ()) is D

    def test_unrecoverable_is_e(self):
        q = _q("F_SH_X_1", status=EvidenceStatus.UNRECOVERABLE)
        assert classify(q, self.WINDOW) is E


class TestFixtureWindows:
    def test_full_corpus_has_only_a_and_d(self, mini):
        counts = class_counts(mini)
        assert counts[A] == 13
        assert counts[D] == 3
        assert sum(counts.values()) == len(mini.questions)

    def test_two_day_window_produces_b_and_c(self, mini):
        classes = classify_all(mini.first_days(2))
        assert classes["F_SH_Mini_004"] is C  # evidence on day 3 only
        assert classes["F_TP_Mini_002"] is B  # days 1 and 3
        assert classes["MA_U_Mini_002"] is B
        assert classes["F_HL_Mini_001"] is D
        assert classes["F_SH_Mini_001"] is A

    def test_b_is_excluded_unless_explicitly_included(self, mini):
        two = mini.first_days(2)
        default = {q.qid for q in select_questions(two)}
        with_b = {q.qid for q in select_questions(two, include_b=True)}
        assert "F_TP_Mini_002" not in default
        assert "F_TP_Mini_002" in with_b
        assert "F_SH_Mini_004" not in with_b  # C is never asked
        assert with_b - default == {"F_TP_Mini_002", "MA_U_Mini_002"}

    def test_window_longer_than_the_corpus_is_an_error(self, mini):
        with pytest.raises(DatasetError, match="corpus has 3"):
            mini.first_days(4)


def _cfg(first_days=None, include_b=False):
    return RunConfig(
        run_id="w",
        dataset=DatasetSpec(kind="fixture", path=FIXTURE.as_posix(), first_days=first_days),
        arm=ArmSpec(name="bm25"),
        questions=QuestionSpec(include_b=include_b),
    )


class TestPipelineSelection:
    def test_default_run_asks_only_scored_classes(self, tmp_path):
        res = run_pipeline(tmp_path, _cfg(first_days=2))
        classes = {r.window_class for r in res.judge_records}
        assert classes == {A, D}
        assert res.components.meta.scored_classes == "A,D"

    def test_opting_into_b_is_recorded_in_metadata(self, tmp_path):
        res = run_pipeline(tmp_path, _cfg(first_days=2, include_b=True))
        assert {r.window_class for r in res.judge_records} == {A, B, D}
        assert res.components.meta.scored_classes == "A,B,D"

    def test_requesting_more_days_than_exist_fails_before_running(self, tmp_path):
        with pytest.raises(DatasetError):
            run_pipeline(tmp_path, _cfg(first_days=10))
