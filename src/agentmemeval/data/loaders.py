"""Loaders for the EverMemBench formats and the synthetic fixture.

Two on-disk formats (see docs/decisions.md D-003 / D-004):

* current format, `0N/dialogue.json` + `0N/qa_0N.json`: messages carry
  `message_index`, questions carry evidence locations `R`. The synthetic
  fixture in tests/fixtures/ uses this format, so CI exercises this loader.
* legacy format, `dataset/NNN/dialogue_en.json` + `qa_NNN.json`: no message
  index and no `R`. Evidence locations are recovered from a current-format
  topic by normalised question text, then mapped onto legacy messages by
  (date, group, timestamp), never by position.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agentmemeval.data.models import (
    MC_LETTERS,
    NO_EVIDENCE_QTYPE,
    Dataset,
    EvidenceStatus,
    Message,
    Question,
    content_hash,
    message_id,
    parse_qtype,
)
from agentmemeval.errors import DatasetError
from agentmemeval.text import normalize

_INDEX_PART_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")

# Legacy metadata key -> qid prefix, used only to cross-check declared counts.
_LEGACY_METADATA_QTYPES = {
    "single_hop": "F_SH",
    "multi_hop": "F_MH",
    "temporal": "F_TP",
    "hallucination": "F_HL",
    "constraint": "MA_C",
    "proactivity": "MA_P",
    "updating": "MA_U",
    "style": "P_Style",
    "skill": "P_Skill",
    "title": "P_Title",
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DatasetError(f"missing data file: {path}") from None
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}: invalid JSON: {exc}") from None


def _validated(model: type[Message] | type[Question], label: str, **fields: Any) -> Any:
    try:
        return model(**fields)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise DatasetError(f"{label}: {first['msg']}") from None


def _message(**fields: Any) -> Message:
    return _validated(Message, f"message {fields.get('id')}", **fields)


def _question(**fields: Any) -> Question:
    return _validated(Question, f"question {fields.get('qid')}", **fields)


def parse_message_index(spec: str | int) -> list[int]:
    """Parse an EverMemBench `message_index` such as ``"1, 4-6"`` into ``[1, 4, 5, 6]``."""
    if isinstance(spec, int):
        return [spec]
    indices: list[int] = []
    for part in str(spec).split(","):
        match = _INDEX_PART_RE.match(part)
        if match is None:
            raise DatasetError(f"cannot parse message_index {spec!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start:
            raise DatasetError(f"invalid range in message_index {spec!r}")
        indices.extend(range(start, end + 1))
    return indices


def _parse_options(qid: str, raw: Mapping[str, str] | None) -> dict[str, str] | None:
    if not raw:
        return None
    if set(raw) != set(MC_LETTERS):
        raise DatasetError(f"{qid}: options must have keys {MC_LETTERS}, got {sorted(raw)}")
    return {k: str(raw[k]) for k in MC_LETTERS}


def _sorted_messages(messages: Sequence[Message]) -> tuple[Message, ...]:
    return tuple(sorted(messages, key=lambda m: (m.date, m.group, m.index)))


def _build_dataset(
    name: str,
    messages: Sequence[Message],
    questions: Sequence[Question],
    *,
    warnings: Sequence[str] = (),
    source: Mapping[str, str] | None = None,
) -> Dataset:
    msgs = _sorted_messages(messages)
    qs = tuple(sorted(questions, key=lambda q: q.qid))
    known = {m.id for m in msgs}
    for q in qs:
        missing = [mid for mid in q.evidence_ids if mid not in known]
        if missing:
            raise DatasetError(f"{q.qid}: evidence refers to unknown messages {missing[:3]}")
    return Dataset(
        name=name,
        messages=msgs,
        questions=qs,
        content_sha256=content_hash(msgs, qs),
        warnings=tuple(warnings),
        source=dict(source or {}),
    )


# --------------------------------------------------------------------------- current format


def _current_messages(raw_days: Any, path: Path) -> list[Message]:
    if not isinstance(raw_days, list):
        raise DatasetError(f"{path}: expected a list of days")
    messages: list[Message] = []
    seen_dates: set[str] = set()
    for day in raw_days:
        date = day["date"]
        if date in seen_dates:
            raise DatasetError(f"{path}: date {date} appears twice")
        seen_dates.add(date)
        for group, raw_msgs in (day.get("dialogues") or {}).items():
            for pos, raw in enumerate(raw_msgs or [], start=1):
                index = int(raw["message_index"])
                if index != pos:
                    raise DatasetError(
                        f"{path}: {date}/{group}: message_index {index} at position {pos}; "
                        "indices must be 1..n in order"
                    )
                messages.append(
                    _message(
                        id=message_id(date, group, index),
                        date=date,
                        group=group,
                        index=index,
                        speaker=raw["speaker"],
                        time=raw["time"],
                        text=raw["dialogue"],
                    )
                )
    return messages


def _resolve_refs(qid: str, refs: Sequence[Mapping[str, Any]], known: set[str]) -> list[str]:
    ids: list[str] = []
    for ref in refs:
        for index in parse_message_index(ref["message_index"]):
            mid = message_id(ref["date"], ref["group"], index)
            if mid not in known:
                raise DatasetError(f"{qid}: evidence {mid} does not exist in the dialogue")
            if mid not in ids:
                ids.append(mid)
    return ids


def load_current_format(dialogue_path: Path, qa_path: Path, *, name: str) -> Dataset:
    messages = _current_messages(_read_json(dialogue_path), dialogue_path)
    known = {m.id for m in messages}
    raw_qs = _read_json(qa_path)
    if not isinstance(raw_qs, list):
        raise DatasetError(f"{qa_path}: expected a list of questions")
    questions: list[Question] = []
    for raw in raw_qs:
        qid = raw["id"]
        qtype = parse_qtype(qid)
        refs = raw.get("R") or []
        if qtype == NO_EVIDENCE_QTYPE:
            if refs:
                raise DatasetError(f"{qid}: no-evidence question must not have evidence R")
            status, ids, note = EvidenceStatus.NO_EVIDENCE, [], None
        elif refs:
            status, ids, note = EvidenceStatus.LOCATED, _resolve_refs(qid, refs, known), None
        else:
            status, ids, note = EvidenceStatus.UNRECOVERABLE, [], "question has no R field"
        questions.append(
            _question(
                qid=qid,
                qtype=qtype,
                question=raw["Q"],
                gold=str(raw["A"]),
                options=_parse_options(qid, raw.get("options")),
                evidence_status=status,
                evidence_ids=tuple(ids),
                evidence_note=note,
            )
        )
    return _build_dataset(name, messages, questions, source={"format": "current"})


def load_fixture(path: Path) -> Dataset:
    """Synthetic corpus: `<path>/dialogue.json` + `<path>/qa.json` in the current format."""
    return load_current_format(path / "dialogue.json", path / "qa.json", name=path.name)


# --------------------------------------------------------------------------- legacy format


@dataclass(frozen=True)
class LegacyPaths:
    dialogue: Path
    qa: Path
    evidence_dialogue: Path
    evidence_qa: Path

    @classmethod
    def under(cls, root: Path, batch: str, evidence_topic: str) -> LegacyPaths:
        return cls(
            dialogue=root / "dataset" / batch / "dialogue_en.json",
            qa=root / "dataset" / batch / f"qa_{batch}.json",
            evidence_dialogue=root / evidence_topic / "dialogue.json",
            evidence_qa=root / evidence_topic / f"qa_{evidence_topic}.json",
        )

    def all(self) -> tuple[Path, ...]:
        return (self.dialogue, self.qa, self.evidence_dialogue, self.evidence_qa)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_manifest(root: Path, paths: LegacyPaths) -> str:
    """Check the files we read against MANIFEST.json; return the dataset revision."""
    manifest = _read_json(root / "MANIFEST.json")
    expected = {f["path"]: f["sha256"] for f in manifest["files"]}
    for path in paths.all():
        rel = path.relative_to(root).as_posix()
        if rel not in expected:
            raise DatasetError(f"{rel} is not listed in {root / 'MANIFEST.json'}")
        if _sha256_file(path) != expected[rel]:
            raise DatasetError(f"{rel}: sha256 does not match MANIFEST.json (file modified?)")
    return str(manifest["revision"])


def _legacy_messages(raw: Any, path: Path) -> list[Message]:
    dialogues = raw.get("dialogues") if isinstance(raw, dict) else None
    if not isinstance(dialogues, dict):
        raise DatasetError(f"{path}: expected an object with a 'dialogues' mapping")
    messages: list[Message] = []
    for date, groups in dialogues.items():
        for group, raw_msgs in (groups or {}).items():
            # The legacy file has no message ids; the 1-based position within
            # (date, group) becomes the id. Evidence is mapped onto these ids by
            # timestamp below, never by position.
            for index, m in enumerate(raw_msgs or [], start=1):
                messages.append(
                    _message(
                        id=message_id(date, group, index),
                        date=date,
                        group=group,
                        index=index,
                        speaker=m["speaker"],
                        time=m["time"],
                        text=m["dialogue"],
                    )
                )
    return messages


def _by_timestamp(messages: Sequence[Message], path: Path) -> dict[tuple[str, str, str], str]:
    out: dict[tuple[str, str, str], str] = {}
    for m in messages:
        key = (m.date, m.group, m.time)
        if key in out:
            raise DatasetError(f"{path}: two messages share {key}; cannot map evidence by time")
        out[key] = m.id
    return out


class _EvidenceIndex:
    """Evidence recovered from a current-format topic, keyed by normalised question."""

    def __init__(self, paths: LegacyPaths, legacy_by_time: dict[tuple[str, str, str], str]):
        source_msgs = {
            m.id: m
            for m in _current_messages(_read_json(paths.evidence_dialogue), paths.evidence_dialogue)
        }
        self._legacy_by_time = legacy_by_time
        self._source_msgs = source_msgs
        self._by_question: dict[str, Mapping[str, Any]] = {}
        for q in _read_json(paths.evidence_qa):
            key = normalize(q["Q"])
            if key in self._by_question:
                raise DatasetError(f"{paths.evidence_qa}: duplicate normalised question text")
            self._by_question[key] = q

    def lookup(self, qtype: str, question: str, gold: str) -> tuple[list[str], str | None]:
        """Return (legacy evidence ids, None) or ([], reason it is unrecoverable)."""
        src = self._by_question.get(normalize(question))
        if src is None:
            return [], "no question with the same normalised text in the evidence topic"
        if parse_qtype(src["id"]) != qtype:
            return [], f"matched {src['id']} but its type differs"
        if normalize(str(src["A"])) != normalize(gold):
            return [], f"matched {src['id']} but its gold answer differs"
        refs = src.get("R") or []
        if not refs:
            return [], f"matched {src['id']} but it has no R"
        ids: list[str] = []
        for ref in refs:
            for index in parse_message_index(ref["message_index"]):
                src_msg = self._source_msgs.get(message_id(ref["date"], ref["group"], index))
                if src_msg is None:
                    return [], f"{src['id']}: R points to a message missing from the topic"
                legacy_id = self._legacy_by_time.get((src_msg.date, src_msg.group, src_msg.time))
                if legacy_id is None:
                    where = f"{src_msg.group} at {src_msg.time}"
                    return [], f"{src['id']}: no legacy message in {where}"
                if legacy_id not in ids:
                    ids.append(legacy_id)
        return ids, None


def load_legacy_batch(root: Path, *, batch: str = "004", evidence_topic: str = "01") -> Dataset:
    """Load legacy batch `batch` with evidence recovered from `evidence_topic`.

    `root` is a download directory produced by scripts/download_data.py (it
    must contain MANIFEST.json; every file read is checked against it).
    """
    paths = LegacyPaths.under(root, batch, evidence_topic)
    revision = _verify_manifest(root, paths)

    messages = _legacy_messages(_read_json(paths.dialogue), paths.dialogue)
    evidence = _EvidenceIndex(paths, _by_timestamp(messages, paths.dialogue))

    raw = _read_json(paths.qa)
    questions: list[Question] = []
    for item in raw["qars"]:
        qid, qtype = item["id"], parse_qtype(item["id"])
        gold = str(item["A"])
        if qtype == NO_EVIDENCE_QTYPE:
            status, ids, note = EvidenceStatus.NO_EVIDENCE, [], None
        else:
            ids, note = evidence.lookup(qtype, item["Q"], gold)
            status = EvidenceStatus.LOCATED if ids else EvidenceStatus.UNRECOVERABLE
        questions.append(
            _question(
                qid=qid,
                qtype=qtype,
                question=item["Q"],
                gold=gold,
                options=_parse_options(qid, item.get("options")),
                evidence_status=status,
                evidence_ids=tuple(ids),
                evidence_note=note,
            )
        )

    warnings: list[str] = []
    actual = Counter(q.qtype for q in questions)
    for key, declared in (raw.get("metadata") or {}).items():
        qtype = _LEGACY_METADATA_QTYPES.get(key)
        if qtype is None:
            warnings.append(f"metadata key {key!r} has no known question type")
        elif actual[qtype] != declared:
            warnings.append(
                f"metadata declares {declared} {key} ({qtype}) questions, file has {actual[qtype]}"
            )

    return _build_dataset(
        f"emb-{batch}",  # emb = EverMemBench
        messages,
        questions,
        warnings=warnings,
        source={
            "format": "legacy",
            "batch": batch,
            "evidence_topic": evidence_topic,
            "revision": revision,
        },
    )
