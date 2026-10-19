"""On-disk records for each pipeline stage, and the file envelope that carries metadata.

Every stage file carries the METHODOLOGY 1.2 metadata, a hash of its own
records and the records hash of the upstream file it was computed from. The
hashes exclude `created_at`, so re-running a deterministic stage reproduces
the same hash and downstream files stay valid; a real upstream change makes
them stale.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from agentmemeval.analysis.outcome import Outcome, UndeliveredReason
from agentmemeval.analysis.windows import WindowClass
from agentmemeval.arms.base import RetrievedItem
from agentmemeval.data.models import EvidenceStatus
from agentmemeval.judge.base import Label

SCHEMA_VERSION = 2  # v2: window_class on question records, scored_classes in meta
STAGES = ("ingest", "search", "answer", "judge")
Stage = Literal["ingest", "search", "answer", "judge"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunMeta(_Model):
    """Metadata every result file must carry (METHODOLOGY 1.2)."""

    run_id: str
    corpus_id: str
    arm: str
    arm_config_hash: str
    answer_model: str
    answer_prompt_version: str
    judge_model: str
    judge_prompt_version: str
    scored_classes: str  # e.g. "A,D": which evidence-window classes were asked


class FileMeta(RunMeta):
    created_at: str
    git_commit: str

    def run_meta(self) -> RunMeta:
        return RunMeta(**self.model_dump(include=set(RunMeta.model_fields)))


class Upstream(_Model):
    stage: Stage
    records_sha256: str


class StageFile(_Model):
    schema_version: int
    stage: Stage
    meta: FileMeta
    upstream: Upstream | None
    records_sha256: str
    records: list[dict[str, Any]]


class ErrorInfo(_Model):
    """An infrastructure failure captured during a stage (-> env_failure)."""

    type: str
    message: str


class IngestRecord(_Model):
    arm: str
    n_messages: int
    message_ids_sha256: str
    summary: dict[str, Any]


class _QuestionRecord(_Model):
    qid: str
    qtype: str
    evidence_status: EvidenceStatus
    window_class: WindowClass


class SearchRecord(_QuestionRecord):
    query: str
    items: list[RetrievedItem]
    error: ErrorInfo | None = None


class PromptMessage(_Model):
    role: str
    content: str


class AnswerRecord(_QuestionRecord):
    prompt: list[PromptMessage]
    response: str | None
    outcome: Outcome
    undelivered_reason: UndeliveredReason | None = None
    error: ErrorInfo | None = None


class JudgeRecord(_QuestionRecord):
    is_mc: bool
    outcome: Outcome
    gold: str
    answer: str | None
    verdict: Label | None  # None iff not delivered
    method: Literal["mc_extraction", "judge"] | None = None
    extracted_choice: str | None = None
    extraction_method: str | None = None
    ambiguous_extraction: bool = False
    judge_reason: str | None = None
    judge_raw: str | None = None


RECORD_TYPES: dict[str, type[_Model]] = {
    "ingest": IngestRecord,
    "search": SearchRecord,
    "answer": AnswerRecord,
    "judge": JudgeRecord,
}


def records_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
