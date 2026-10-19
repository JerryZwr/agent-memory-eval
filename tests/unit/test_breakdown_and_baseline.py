"""Breakdowns, the 2x2, the read-only run loader, and baseline diffs (METHODOLOGY 1.8, 1.10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentmemeval.analysis.baseline import (
    BaselineEntry,
    DiffStatus,
    diff,
    diff_run,
    freeze,
    load_baseline,
)
from agentmemeval.analysis.breakdown import (
    HitCriterion,
    by_qtype,
    by_scope,
    evidence_scope,
    retrieval_hit,
    two_by_two,
)
from agentmemeval.analysis.outcome import Outcome
from agentmemeval.analysis.windows import WindowClass
from agentmemeval.arms.base import RetrievedItem
from agentmemeval.errors import IncomparableError, OverwriteRefusedError, StageError
from agentmemeval.pipeline.config import ArmSpec, DatasetSpec, RunConfig
from agentmemeval.pipeline.records import SearchRecord
from agentmemeval.pipeline.runner import load_run, run_pipeline
from agentmemeval.pipeline.store import stage_path

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_corpus"


def _cfg(arm="bm25", top_k=5, first_days=None):
    return RunConfig(
        run_id=f"t-{arm}-{top_k}",
        dataset=DatasetSpec(kind="fixture", path=FIXTURE.as_posix(), first_days=first_days),
        arm=ArmSpec(name=arm, top_k=top_k),
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    for arm in ("null", "bm25"):
        run_pipeline(root / arm, _cfg(arm))
    return root


def _search(qid, *items):
    return SearchRecord(
        qid=qid,
        qtype=qid[:4],
        evidence_status="located",
        window_class=WindowClass.A,
        query="q",
        items=[
            RetrievedItem(
                message_id=mid, date=mid[:10], time=f"{mid[:10]} 09:00:00", group="g",
                speaker="s", text="t", score=1.0,
            )
            for mid in items
        ],
    )  # fmt: skip


class TestRetrievalHit:
    def test_message_criterion_needs_an_evidence_message(self, mini):
        q = mini.question_by_id["F_SH_Mini_001"]  # evidence: 2025-03-03/Group 1/1
        same_day_other_msg = _search(q.qid, "2025-03-03/Group 1/5")
        exact = _search(q.qid, "2025-03-04/Group 1/1", "2025-03-03/Group 1/1")
        assert retrieval_hit(same_day_other_msg, q, HitCriterion.MESSAGE) is False
        assert retrieval_hit(same_day_other_msg, q, HitCriterion.DATE) is True
        assert retrieval_hit(exact, q, HitCriterion.MESSAGE) is True

    def test_empty_retrieval_is_a_miss_and_non_empty_is_not_automatically_a_hit(self, mini):
        q = mini.question_by_id["F_SH_Mini_004"]  # evidence on 2025-03-05 only
        assert retrieval_hit(_search(q.qid), q, HitCriterion.DATE) is False
        assert retrieval_hit(_search(q.qid, "2025-03-03/Group 1/1"), q, HitCriterion.DATE) is False

    def test_undefined_without_located_evidence(self, mini):
        q = mini.question_by_id["F_HL_Mini_001"]
        assert retrieval_hit(_search(q.qid, "2025-03-03/Group 1/1"), q, HitCriterion.DATE) is None


def test_evidence_scope(mini):
    scopes = {q.qid: evidence_scope(q) for q in mini.questions}
    assert scopes["F_MH_Mini_001"] == "cross-group"
    assert scopes["F_SH_Mini_001"] == "single-group"
    assert scopes["F_HL_Mini_001"] == "no-evidence"


class TestBreakdownsOnTheDemoRun:
    def test_two_by_two_cells_add_up_and_null_never_hits(self, runs):
        for arm in ("null", "bm25"):
            run = load_run(runs / arm)
            qs = run.components.dataset.question_by_id
            for crit in HitCriterion:
                t = two_by_two(qs, run.search_records, run.judge_records, crit)
                located = sum(qs[j.qid].evidence_ids != () for j in run.judge_records)
                assert t.n + t.not_delivered == located
                if arm == "null":
                    assert t.hit_correct == t.hit_wrong == 0
                    assert t.miss_correct + t.miss_wrong == t.n

    def test_bm25_has_retrieved_but_wrong_cases_for_attribution(self, runs):
        run = load_run(runs / "bm25")
        qs = run.components.dataset.question_by_id
        t = two_by_two(qs, run.search_records, run.judge_records, HitCriterion.MESSAGE)
        assert t.hit_correct > 0
        assert t.hit_wrong > 0  # reading / reasoning failures, not retrieval

    def test_date_criterion_is_never_stricter_than_message(self, runs):
        run = load_run(runs / "bm25")
        qs = run.components.dataset.question_by_id
        msg = two_by_two(qs, run.search_records, run.judge_records, HitCriterion.MESSAGE)
        day = two_by_two(qs, run.search_records, run.judge_records, HitCriterion.DATE)
        assert day.hit_correct + day.hit_wrong >= msg.hit_correct + msg.hit_wrong

    def test_qtype_and_scope_breakdowns_partition_the_run(self, runs):
        run = load_run(runs / "bm25")
        qs = run.components.dataset.question_by_id
        assert sum(t.n for t in by_qtype(run.judge_records).values()) == len(run.judge_records)
        scopes = by_scope(qs, run.judge_records)
        assert set(scopes) == {"single-group", "cross-group", "no-evidence"}
        assert sum(t.n for t in scopes.values()) == len(run.judge_records)


class TestLoadRun:
    def test_loads_a_complete_run(self, runs):
        run = load_run(runs / "bm25")
        assert len(run.judge_records) == 16
        assert run.meta.arm == "bm25"

    def test_refuses_an_incomplete_run_instead_of_finishing_it(self, tmp_path):
        run_pipeline(tmp_path, _cfg())
        stage_path(tmp_path, "answer").unlink()
        with pytest.raises(StageError, match="missing"):
            load_run(tmp_path)
        assert not stage_path(tmp_path, "answer").exists()


def _entry(qid, outcome=Outcome.DELIVERED, verdict="CORRECT"):
    return BaselineEntry(
        qid=qid,
        qtype="F_SH",
        window_class=WindowClass.A,
        outcome=outcome,
        verdict=verdict if outcome is Outcome.DELIVERED else None,
        retrieval_hit=True,
    )


class TestBaseline:
    def test_freeze_round_trips_and_refuses_overwrite(self, runs, tmp_path):
        run = load_run(runs / "bm25")
        path = tmp_path / "baselines" / "b1.json"
        frozen = freeze(run, path, name="b1")
        assert load_baseline(path) == frozen
        assert len(frozen.entries) == 16
        with pytest.raises(OverwriteRefusedError):
            freeze(run, path, name="b1")
        freeze(run, path, name="b1", overwrite=True)  # explicit parameter

    def test_same_run_diffs_to_all_unchanged(self, runs, tmp_path):
        run = load_run(runs / "bm25")
        result = diff_run(freeze(run, tmp_path / "b.json", name="b"), run)
        assert result.counts[DiffStatus.UNCHANGED] == 16
        assert result.mcnemar_p == 1.0

    def test_all_four_statuses(self, runs, tmp_path):
        run = load_run(runs / "bm25")
        base = freeze(run, tmp_path / "b.json", name="b").model_copy(
            update={
                "entries": [
                    _entry("fixed", verdict="WRONG"),
                    _entry("regressed"),
                    _entry("same"),
                    _entry("undelivered_before", outcome=Outcome.UNDELIVERED),
                    _entry("env_after"),
                    _entry("env_before", outcome=Outcome.ENV_FAILURE),
                    _entry("only_in_baseline"),
                ]
            }
        )
        current = [
            _entry("fixed"),
            _entry("regressed", verdict="WRONG"),
            _entry("same"),
            _entry("undelivered_before"),  # undelivered is a measured failure -> fixed
            _entry("env_after", outcome=Outcome.ENV_FAILURE),
            _entry("env_before"),
            _entry("only_in_current"),
        ]
        rows = {r.qid: r.status for r in diff(base, run.meta, current).rows}
        assert rows == {
            "fixed": DiffStatus.FIXED,
            "regressed": DiffStatus.REGRESSED,
            "same": DiffStatus.UNCHANGED,
            "undelivered_before": DiffStatus.FIXED,
            "env_after": DiffStatus.NOT_MEASURED,
            "env_before": DiffStatus.NOT_MEASURED,
            "only_in_baseline": DiffStatus.NOT_MEASURED,
            "only_in_current": DiffStatus.NOT_MEASURED,
        }

    def test_config_change_produces_a_real_diff(self, runs, tmp_path):
        """The M5 workflow in miniature: freeze, change top_k, diff question by question."""
        base = freeze(load_run(runs / "bm25"), tmp_path / "b.json", name="b")
        run_pipeline(tmp_path / "k1", _cfg(top_k=1))
        result = diff_run(base, load_run(tmp_path / "k1"))
        assert not result.incomparable
        assert sum(result.counts.values()) == 16
        assert result.counts[DiffStatus.NOT_MEASURED] == 0

    def test_different_corpus_is_refused_unless_explicitly_allowed(self, runs, tmp_path):
        base = freeze(load_run(runs / "bm25"), tmp_path / "b.json", name="b")
        run_pipeline(tmp_path / "w2", _cfg(first_days=2))
        two_days = load_run(tmp_path / "w2")
        with pytest.raises(IncomparableError, match="corpus_id"):
            diff_run(base, two_days)
        allowed = diff_run(base, two_days, allow_incomparable=True)
        assert any("corpus_id" in p for p in allowed.incomparable)

    def test_different_judge_version_is_refused(self, runs, tmp_path):
        run = load_run(runs / "bm25")
        base = freeze(run, tmp_path / "b.json", name="b")
        other = run.meta.model_copy(update={"judge_prompt_version": "v2"})
        with pytest.raises(IncomparableError, match="judge_prompt_version"):
            diff(base, other, [])

    def test_corrupt_baseline_is_refused(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"name": "x"}), encoding="utf-8")
        with pytest.raises(StageError, match="not a valid baseline"):
            load_baseline(path)
