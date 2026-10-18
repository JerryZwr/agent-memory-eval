"""Core data model: messages, questions, datasets.

Everything is keyed by stable ids (message id, qid). Nothing downstream may
join by position (METHODOLOGY 3.4).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property

from pydantic import BaseModel, ConfigDict, model_validator

from agentmemeval.errors import DatasetError

# Question type = qid prefix (EverMemBench convention, reused by the fixture).
QTYPE_LABELS: dict[str, str] = {
    "F_SH": "single-hop",
    "F_MH": "multi-hop",
    "F_TP": "temporal",
    "F_HL": "no-evidence (hallucination)",
    "MA_C": "constraint",
    "MA_P": "proactivity",
    "MA_U": "updating",
    "P_Style": "style",
    "P_Skill": "skill",
    "P_Title": "title",
}
NO_EVIDENCE_QTYPE = "F_HL"
_QTYPE_RE = re.compile(r"^(" + "|".join(map(re.escape, QTYPE_LABELS)) + r")_")
MC_LETTERS = ("A", "B", "C", "D")


def parse_qtype(qid: str) -> str:
    match = _QTYPE_RE.match(qid)
    if match is None:
        raise DatasetError(f"question id {qid!r} has no known type prefix")
    return match.group(1)


def message_id(date: str, group: str, index: int) -> str:
    return f"{date}/{group}/{index}"


def date_of_message_id(mid: str) -> str:
    return mid.split("/", 1)[0]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    date: str  # YYYY-MM-DD
    group: str
    index: int  # 1-based within (date, group)
    speaker: str
    time: str  # YYYY-MM-DD HH:MM:SS
    text: str

    @model_validator(mode="after")
    def _id_matches_fields(self) -> Message:
        if self.id != message_id(self.date, self.group, self.index):
            raise ValueError(f"message id {self.id!r} does not match date/group/index")
        return self


class EvidenceStatus(StrEnum):
    LOCATED = "located"  # evidence message ids are known
    NO_EVIDENCE = "no_evidence"  # by design (F_HL): correct answer is "did not happen"
    UNRECOVERABLE = "unrecoverable"  # evidence exists but its location is unknown (E class)


class Question(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    qid: str
    qtype: str
    question: str
    gold: str
    options: dict[str, str] | None = None
    evidence_status: EvidenceStatus
    evidence_ids: tuple[str, ...] = ()
    evidence_note: str | None = None  # why evidence is unrecoverable

    @model_validator(mode="after")
    def _consistent(self) -> Question:
        if parse_qtype(self.qid) != self.qtype:
            raise ValueError(f"{self.qid}: qtype {self.qtype!r} does not match the id prefix")
        if self.options is not None:
            if tuple(self.options) != MC_LETTERS:
                raise ValueError(f"{self.qid}: options must be exactly {MC_LETTERS}")
            if self.gold not in self.options:
                raise ValueError(f"{self.qid}: multiple-choice gold {self.gold!r} is not an option")
        located = self.evidence_status is EvidenceStatus.LOCATED
        if located != bool(self.evidence_ids):
            raise ValueError(f"{self.qid}: evidence_ids must be non-empty iff status is located")
        if (self.qtype == NO_EVIDENCE_QTYPE) != (
            self.evidence_status is EvidenceStatus.NO_EVIDENCE
        ):
            raise ValueError(f"{self.qid}: status no_evidence is reserved for {NO_EVIDENCE_QTYPE}")
        return self

    @property
    def is_mc(self) -> bool:
        return self.options is not None

    @property
    def evidence_dates(self) -> tuple[str, ...]:
        return tuple(sorted({date_of_message_id(m) for m in self.evidence_ids}))


def _sha256_json(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Dataset:
    """A corpus plus its questions.

    `messages` may be a window of the full corpus (see `first_days`); questions
    always cover the whole corpus period, which is exactly why evidence-window
    filtering (M2) is needed before scoring.
    """

    name: str
    messages: tuple[Message, ...]
    questions: tuple[Question, ...]
    content_sha256: str  # of the *full* corpus + questions, set by the loader
    warnings: tuple[str, ...] = ()
    source: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [m.id for m in self.messages]
        if len(set(ids)) != len(ids):
            raise DatasetError(f"{self.name}: duplicate message ids")
        if ids != sorted(ids, key=lambda mid: self._sort_key_by_id[mid]):
            raise DatasetError(f"{self.name}: messages must be sorted by (date, group, index)")
        qids = [q.qid for q in self.questions]
        if len(set(qids)) != len(qids):
            raise DatasetError(f"{self.name}: duplicate question ids")

    @cached_property
    def _sort_key_by_id(self) -> dict[str, tuple[str, str, int]]:
        return {m.id: (m.date, m.group, m.index) for m in self.messages}

    @cached_property
    def days(self) -> tuple[str, ...]:
        return tuple(sorted({m.date for m in self.messages}))

    @cached_property
    def message_by_id(self) -> dict[str, Message]:
        return {m.id: m for m in self.messages}

    @cached_property
    def question_by_id(self) -> dict[str, Question]:
        return {q.qid: q for q in self.questions}

    @property
    def corpus_id(self) -> str:
        """Dataset version + ingested day range. Results with different ids never compare."""
        if not self.messages:
            raise DatasetError(f"{self.name}: empty corpus has no corpus_id")
        return f"{self.name}@{self.content_sha256[:10]}:{self.days[0]}..{self.days[-1]}"

    @cached_property
    def message_ids_sha256(self) -> str:
        return _sha256_json([m.id for m in self.messages])

    def first_days(self, n: int) -> Dataset:
        """The first `n` days of the corpus. Questions are kept as-is."""
        if n < 1:
            raise DatasetError(f"first_days needs n >= 1, got {n}")
        if n > len(self.days):
            raise DatasetError(
                f"{self.name}: asked for the first {n} days but the corpus has {len(self.days)}"
            )
        keep = set(self.days[:n])
        return Dataset(
            name=self.name,
            messages=tuple(m for m in self.messages if m.date in keep),
            questions=self.questions,
            content_sha256=self.content_sha256,
            warnings=self.warnings,
            source=self.source,
        )


def content_hash(messages: Iterable[Message], questions: Iterable[Question]) -> str:
    return _sha256_json(
        {
            "messages": [m.model_dump(mode="json") for m in messages],
            "questions": [q.model_dump(mode="json") for q in questions],
        }
    )
