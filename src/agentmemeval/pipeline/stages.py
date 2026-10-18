"""The four pipeline stages as pure-ish functions: upstream records in, records out.

Only `InfraError` is caught (and recorded as an env failure). Any other
exception propagates: a bug in an arm, the LLM wrapper or this code must stop
the run, not turn into a plausible-looking wrong answer (P1).
"""

from __future__ import annotations

from collections.abc import Sequence

from agentmemeval.analysis.outcome import Outcome, classify_delivery
from agentmemeval.arms.base import MemoryArm
from agentmemeval.data.models import Dataset, Question
from agentmemeval.errors import InfraError, StageError
from agentmemeval.judge.base import Judge, JudgeInput
from agentmemeval.judge.mc import extract_choice
from agentmemeval.llm.base import LLM, ChatMessage
from agentmemeval.pipeline.prompts import render_answer_prompt
from agentmemeval.pipeline.records import (
    AnswerRecord,
    ErrorInfo,
    IngestRecord,
    JudgeRecord,
    PromptMessage,
    SearchRecord,
)


def _error(exc: InfraError) -> ErrorInfo:
    return ErrorInfo(type=type(exc).__name__, message=str(exc))


def check_qids(stage: str, got: Sequence[str], questions: Sequence[Question]) -> None:
    """Upstream records must cover exactly the run's questions (join by id, never position)."""
    expected = {q.qid for q in questions}
    seen = list(got)
    if len(set(seen)) != len(seen):
        raise StageError(f"{stage}: duplicate qids in upstream records")
    if set(seen) != expected:
        missing = sorted(expected - set(seen))[:5]
        extra = sorted(set(seen) - expected)[:5]
        raise StageError(
            f"{stage}: upstream qids differ from the run (missing {missing}, extra {extra})"
        )


def run_ingest(dataset: Dataset, arm: MemoryArm) -> list[IngestRecord]:
    summary = arm.add(dataset)
    return [
        IngestRecord(
            arm=arm.name,
            n_messages=len(dataset.messages),
            message_ids_sha256=dataset.message_ids_sha256,
            summary=summary,
        )
    ]


def run_search(
    dataset: Dataset,
    arm: MemoryArm,
    ingest: IngestRecord,
    questions: Sequence[Question],
    top_k: int,
) -> list[SearchRecord]:
    if ingest.message_ids_sha256 != dataset.message_ids_sha256:
        raise StageError("search: corpus differs from the one ingested; re-run ingest")
    arm.attach(dataset, ingest.summary)
    records = []
    for q in questions:
        query = q.question  # query-shape experiments (METHODOLOGY 1.9) vary this in M5
        try:
            items, error = arm.search(query, top_k), None
        except InfraError as exc:
            items, error = [], _error(exc)
        records.append(
            SearchRecord(
                qid=q.qid,
                qtype=q.qtype,
                evidence_status=q.evidence_status,
                query=query,
                items=items,
                error=error,
            )
        )
    return records


def run_answer(
    dataset: Dataset, search: Sequence[SearchRecord], llm: LLM, questions: Sequence[Question]
) -> list[AnswerRecord]:
    check_qids("answer", [s.qid for s in search], questions)
    records = []
    for s in search:
        q = dataset.question_by_id[s.qid]
        messages = render_answer_prompt(q, s.items)
        prompt = [PromptMessage(role=m.role, content=m.content) for m in messages]
        response: str | None = None
        error = s.error
        if error is None:
            try:
                response = llm.complete(
                    [ChatMessage(m.role, m.content) for m in messages], temperature=0.0
                ).text
            except InfraError as exc:
                error = _error(exc)
        cls = classify_delivery(response, infra_failed=error is not None)
        records.append(
            AnswerRecord(
                qid=s.qid,
                qtype=s.qtype,
                evidence_status=s.evidence_status,
                prompt=prompt,
                response=response,
                outcome=cls.outcome,
                undelivered_reason=cls.reason,
                error=error,
            )
        )
    return records


def run_judge(
    dataset: Dataset, answers: Sequence[AnswerRecord], judge: Judge, questions: Sequence[Question]
) -> list[JudgeRecord]:
    """Judge delivered answers. A JudgeError aborts the stage: no file, no fake WRONG."""
    check_qids("judge", [a.qid for a in answers], questions)
    records = []
    for a in answers:
        q = dataset.question_by_id[a.qid]
        base = {
            "qid": a.qid,
            "qtype": a.qtype,
            "evidence_status": a.evidence_status,
            "is_mc": q.is_mc,
            "outcome": a.outcome,
            "gold": q.gold,
            "answer": a.response,
        }
        if a.outcome is not Outcome.DELIVERED:
            records.append(JudgeRecord(**base, verdict=None))
            continue
        assert a.response is not None
        if q.is_mc:
            ex = extract_choice(a.response)
            records.append(
                JudgeRecord(
                    **base,
                    verdict="CORRECT" if ex.letter == q.gold else "WRONG",
                    method="mc_extraction",
                    extracted_choice=ex.letter,
                    extraction_method=ex.method,
                    ambiguous_extraction=ex.ambiguous,
                )
            )
        else:
            v = judge.judge(
                JudgeInput(
                    qid=q.qid, qtype=q.qtype, question=q.question, gold=q.gold, answer=a.response
                )
            )
            records.append(
                JudgeRecord(
                    **base,
                    verdict=v.label,
                    method="judge",
                    judge_reason=v.reason,
                    judge_raw=v.raw,
                )
            )
    return records
