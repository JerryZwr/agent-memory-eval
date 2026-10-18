from __future__ import annotations

import pytest

from agentmemeval.arms import BM25Arm, NullArm, build_arm
from agentmemeval.arms.bm25 import BM25Index
from agentmemeval.data.loaders import _build_dataset
from agentmemeval.data.models import Message, message_id
from agentmemeval.errors import DatasetError, StageError

pytestmark = pytest.mark.unit


def _msgs(*texts: str, date: str = "2025-01-01") -> list[Message]:
    return [
        Message(
            id=message_id(date, "G", i),
            date=date,
            group="G",
            index=i,
            speaker="Ann",
            time=f"{date} 09:00:{i:02d}",
            text=t,
        )
        for i, t in enumerate(texts, start=1)
    ]


class TestNullArm:
    def test_stores_nothing_and_finds_nothing(self, mini):
        arm = NullArm()
        summary = arm.add(mini)
        assert summary == {"n_messages_offered": len(mini.messages), "n_messages_stored": 0}
        for q in mini.questions:
            assert arm.search(q.question, top_k=5) == []


class TestBM25Arm:
    @pytest.fixture
    def arm(self, mini) -> BM25Arm:
        arm = BM25Arm()
        arm.add(mini)
        return arm

    def test_finds_the_evidence_message_first(self, arm, mini):
        q = mini.question_by_id["F_SH_Mini_001"]
        hits = arm.search(q.question, top_k=5)
        assert hits[0].message_id == q.evidence_ids[0]
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_respects_top_k(self, arm):
        assert len(arm.search("backup nightly job restore", top_k=2)) == 2

    def test_returns_nothing_rather_than_zero_score_padding(self, arm):
        assert arm.search("kubernetes helm chart", top_k=5) == []
        assert arm.search("what is the", top_k=5) == []  # stopwords only

    def test_ties_break_by_message_id_deterministically(self):
        ds = _build_dataset("t", _msgs("alpha beta", "gamma", "alpha beta", "delta"), [])
        arm = BM25Arm()
        arm.add(ds)
        hits = arm.search("alpha", top_k=5)
        assert [h.message_id for h in hits] == ["2025-01-01/G/1", "2025-01-01/G/3"]
        assert hits[0].score == hits[1].score

    def test_attach_rebuilds_the_same_index(self, mini, arm):
        summary = BM25Arm().add(mini)
        fresh = BM25Arm()
        fresh.attach(mini, summary)
        q = "maintenance dashboard charts library"
        assert fresh.search(q, 5) == arm.search(q, 5)

    def test_attach_refuses_a_different_corpus(self, mini):
        summary = BM25Arm().add(mini)
        with pytest.raises(StageError, match="differs"):
            BM25Arm().attach(mini.first_days(2), summary)

    def test_search_before_add_is_a_bug(self):
        with pytest.raises(RuntimeError):
            BM25Arm().search("x", 5)

    def test_empty_corpus_is_refused(self):
        with pytest.raises(DatasetError):
            BM25Arm().add(_build_dataset("t", [], []))

    def test_speaker_is_searchable_only_when_included(self):
        ds = _build_dataset("t", _msgs("deploy done"), [])
        with_speaker, without = BM25Arm(), BM25Arm(include_speaker=False)
        with_speaker.add(ds)
        without.add(ds)
        assert with_speaker.search("ann", 5)
        assert without.search("ann", 5) == []


def test_build_arm_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown arm"):
        build_arm("mem1")


def test_params_capture_effective_defaults():
    assert BM25Arm().params() == {
        "scoring": "lucene-bm25",
        "k1": 1.2,
        "b": 0.75,
        "include_speaker": True,
        "tokenizer": "simple-v1",
        "chunking": "message",
    }


class TestBM25IndexKnownValues:
    """Hand-computed values. docs: [a b], [a], [c c d]; N = 3, avgdl = 2, k1 = 1.5, b = 0.75."""

    @pytest.fixture
    def index(self):
        return BM25Index([["a", "b"], ["a"], ["c", "c", "d"]], k1=1.5, b=0.75)

    def test_idf_is_lucene_and_positive(self, index):
        # idf(a) = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6)
        assert index.idf["a"] == pytest.approx(0.4700036292457356)
        # idf(c) = ln(1 + (3 - 1 + 0.5) / (1 + 0.5)) = ln(8/3)
        assert index.idf["c"] == pytest.approx(0.9808292530117262)
        assert all(v > 0 for v in index.idf.values())

    def test_scores(self, index):
        s = index.scores(["a"])
        # doc 0: |d| = 2 -> tf part = 2.5 / (1 + 1.5 * 1.0)   = 1.0
        # doc 1: |d| = 1 -> tf part = 2.5 / (1 + 1.5 * 0.625) = 1.2903225806
        assert s == pytest.approx({0: 0.4700036292457356, 1: 0.4700036292457356 * 1.2903225806})
        # doc 2: tf(c) = 2, |d| = 3 -> 2 * 2.5 / (2 + 1.5 * 1.375) = 1.2307692308
        assert index.scores(["c"]) == pytest.approx({2: 0.9808292530117262 * 1.2307692308})

    def test_repeated_query_terms_count_once(self, index):
        assert index.scores(["a", "a"]) == index.scores(["a"])

    def test_term_in_half_or_more_documents_still_scores(self):
        """The rank_bm25 Okapi idf is 0 here (n = N/2) and negative for n > N/2."""
        half = BM25Index([["x"], ["x"], ["y"], ["z"]], k1=1.2, b=0.75)
        assert all(v > 0 for v in half.scores(["x"]).values())
        everywhere = BM25Index([["x", "y"], ["x"]], k1=1.2, b=0.75)
        assert set(everywhere.scores(["x"])) == {0, 1}
