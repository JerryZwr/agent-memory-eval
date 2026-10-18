from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from agentmemeval.data.models import (
    Dataset,
    EvidenceStatus,
    Message,
    Question,
    content_hash,
    parse_qtype,
)
from agentmemeval.errors import DatasetError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("qid", "qtype"),
    [("F_SH_Top004_001", "F_SH"), ("P_Style_Mini_001", "P_Style"), ("MA_U_X_9", "MA_U")],
)
def test_parse_qtype(qid, qtype):
    assert parse_qtype(qid) == qtype


def test_parse_qtype_rejects_unknown_prefix():
    with pytest.raises(DatasetError):
        parse_qtype("F_XX_001")


def test_message_id_must_match_fields():
    with pytest.raises(ValidationError):
        Message(id="wrong", date="2025-01-01", group="G", index=1, speaker="s", time="t", text="x")


@pytest.mark.parametrize(
    "kwargs",
    [
        # located without ids
        {"qid": "F_SH_X_1", "evidence_status": EvidenceStatus.LOCATED},
        # no_evidence on a non-F_HL question
        {"qid": "F_SH_X_1", "evidence_status": EvidenceStatus.NO_EVIDENCE},
        # F_HL that claims to be unrecoverable
        {"qid": "F_HL_X_1", "evidence_status": EvidenceStatus.UNRECOVERABLE},
    ],
)
def test_question_rejects_inconsistent_evidence(kwargs):
    with pytest.raises(ValidationError):
        Question(qtype=parse_qtype(kwargs["qid"]), question="q", gold="g", **kwargs)


class TestDataset:
    def test_corpus_id_has_version_and_day_range(self, mini):
        name, rest = mini.corpus_id.split("@")
        version, days = rest.split(":")
        assert name == "mini_corpus"
        assert version == mini.content_sha256[:10]
        assert days == "2025-03-03..2025-03-05"

    def test_first_days_narrows_messages_and_corpus_id(self, mini):
        two = mini.first_days(2)
        assert two.days == ("2025-03-03", "2025-03-04")
        assert two.corpus_id.endswith(":2025-03-03..2025-03-04")
        assert two.questions == mini.questions
        assert two.message_ids_sha256 != mini.message_ids_sha256

    @pytest.mark.parametrize("n", [0, 4])
    def test_first_days_refuses_impossible_windows(self, mini, n):
        with pytest.raises(DatasetError):
            mini.first_days(n)

    def test_content_change_changes_corpus_id(self, mini):
        first = mini.messages[0]
        edited = (first.model_copy(update={"text": first.text + "!"}), *mini.messages[1:])
        other = dataclasses.replace(
            mini, messages=edited, content_sha256=content_hash(edited, mini.questions)
        )
        assert other.corpus_id != mini.corpus_id

    def test_rejects_duplicate_message_ids(self, mini):
        with pytest.raises(DatasetError, match="duplicate message ids"):
            dataclasses.replace(mini, messages=(mini.messages[0], *mini.messages))

    def test_rejects_unsorted_messages(self, mini):
        with pytest.raises(DatasetError, match="sorted"):
            dataclasses.replace(mini, messages=tuple(reversed(mini.messages)))

    def test_evidence_dates(self, mini):
        q = mini.question_by_id["F_TP_Mini_002"]
        assert q.evidence_dates == ("2025-03-03", "2025-03-05")

    def test_empty_corpus_has_no_corpus_id(self, mini):
        empty = Dataset(name="e", messages=(), questions=(), content_sha256="0" * 64)
        with pytest.raises(DatasetError):
            _ = empty.corpus_id
