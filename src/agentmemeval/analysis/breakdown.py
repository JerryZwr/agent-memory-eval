"""Breakdowns beyond the headline score (METHODOLOGY 1.8).

* by question type: types must not be pooled (multiple choice has a guessing
  floor, open questions do not);
* by evidence scope: single group / cross group / no evidence;
* retrieval x verdict 2x2: each cell points at a different fix.

"Retrieval hit" is computed under two criteria and both are reported (see
docs/decisions.md D-006):

* `message`: a retrieved item is one of the evidence messages (strict; needs
  arms that return source message ids, like bm25);
* `date`: a retrieved item comes from a day that holds evidence (loose; works
  for arms that return extracted memories with only a date).

Both judge "hit" by evidence, never by "the result was non-empty": a polluted
store returns something for every query (METHODOLOGY 1.12).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from agentmemeval.analysis.outcome import Outcome
from agentmemeval.analysis.scoring import Tally, tally_by
from agentmemeval.data.models import EvidenceStatus, Question, date_of_message_id
from agentmemeval.pipeline.records import JudgeRecord, SearchRecord


class HitCriterion(StrEnum):
    MESSAGE = "message"
    DATE = "date"


CRITERION_LABELS = {
    HitCriterion.MESSAGE: "命中判据：检索到证据消息（严格）",
    HitCriterion.DATE: "命中判据：检索到证据当天的消息（宽松）",
}


def retrieval_hit(search: SearchRecord, question: Question, criterion: HitCriterion) -> bool | None:
    """Did retrieval reach the evidence? None when the question has no located evidence."""
    if question.evidence_status is not EvidenceStatus.LOCATED:
        return None
    if criterion is HitCriterion.MESSAGE:
        wanted = set(question.evidence_ids)
        return any(item.message_id in wanted for item in search.items)
    days = set(question.evidence_dates)
    return any(item.date in days for item in search.items)


def evidence_scope(question: Question) -> str:
    if question.evidence_status is EvidenceStatus.NO_EVIDENCE:
        return "no-evidence"
    if question.evidence_status is EvidenceStatus.UNRECOVERABLE:
        return "unknown"
    groups = {mid.split("/", 2)[1] for mid in question.evidence_ids}
    return "single-group" if len(groups) == 1 else "cross-group"


@dataclass(frozen=True)
class TwoByTwo:
    """Delivered questions with located evidence, split by retrieval hit and verdict."""

    criterion: HitCriterion
    hit_correct: int  # memory actually helped
    hit_wrong: int  # reading / reasoning problem
    miss_correct: int  # guessed or general knowledge: not credit for memory
    miss_wrong: int  # retrieval problem
    not_delivered: int  # excluded: no verdict to cross with

    @property
    def n(self) -> int:
        return self.hit_correct + self.hit_wrong + self.miss_correct + self.miss_wrong


def two_by_two(
    questions: Mapping[str, Question],
    search: Sequence[SearchRecord],
    judged: Sequence[JudgeRecord],
    criterion: HitCriterion,
) -> TwoByTwo:
    by_qid = {s.qid: s for s in search}
    cells = {"hit_correct": 0, "hit_wrong": 0, "miss_correct": 0, "miss_wrong": 0}
    not_delivered = 0
    for j in judged:
        q = questions[j.qid]
        if q.evidence_status is not EvidenceStatus.LOCATED:
            continue
        if j.outcome is not Outcome.DELIVERED:
            not_delivered += 1
            continue
        hit = retrieval_hit(by_qid[j.qid], q, criterion)
        cells[f"{'hit' if hit else 'miss'}_{'correct' if j.verdict == 'CORRECT' else 'wrong'}"] += 1
    return TwoByTwo(criterion=criterion, not_delivered=not_delivered, **cells)


def by_qtype(judged: Sequence[JudgeRecord]) -> dict[str, Tally]:
    return tally_by(judged, lambda r: r.qtype)


def by_scope(questions: Mapping[str, Question], judged: Sequence[JudgeRecord]) -> dict[str, Tally]:
    return tally_by(judged, lambda r: evidence_scope(questions[r.qid]))


def by_window(judged: Sequence[JudgeRecord]) -> dict[str, Tally]:
    return tally_by(judged, lambda r: r.window_class.value)


def evidence_days_touched(search: SearchRecord) -> list[str]:
    """Days the retrieved items came from, for the per-question detail."""
    return sorted({date_of_message_id(i.message_id) for i in search.items})
