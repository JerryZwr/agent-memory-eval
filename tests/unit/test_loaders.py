"""Loaders: synthetic fixture, current format edge cases, legacy batch + evidence join."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from agentmemeval.data.loaders import (
    load_current_format,
    load_legacy_batch,
    parse_message_index,
)
from agentmemeval.data.models import EvidenceStatus
from agentmemeval.errors import DatasetError

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
REAL_DATA = REPO / "data" / "raw" / "evermembench" / "a6b210a3"


def _dump(path: Path, obj: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- message_index


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("3", [3]), (3, [3]), ("1, 4-6", [1, 4, 5, 6]), ("2-3, 6-7", [2, 3, 6, 7]), (" 7 ", [7])],
)
def test_parse_message_index(spec, expected):
    assert parse_message_index(spec) == expected


@pytest.mark.parametrize("spec", ["", "a", "5-3", "0", "1,,2", "1-2-3"])
def test_parse_message_index_rejects_malformed(spec):
    with pytest.raises(DatasetError):
        parse_message_index(spec)


# --------------------------------------------------------------------------- fixture


class TestMiniCorpus:
    def test_shape_matches_plan(self, mini):
        # PLAN M1: ~3 days, 2 groups, 40-60 messages, 12-16 questions.
        assert mini.days == ("2025-03-03", "2025-03-04", "2025-03-05")
        assert {m.group for m in mini.messages} == {"Group 1", "Group 2"}
        assert 40 <= len(mini.messages) <= 60
        assert 12 <= len(mini.questions) <= 16

    def test_covers_every_required_question_kind(self, mini):
        qtypes = Counter(q.qtype for q in mini.questions)
        for required in ("F_SH", "F_MH", "F_TP", "F_HL"):
            assert qtypes[required] >= 2, required
        assert sum(q.is_mc for q in mini.questions) >= 4
        statuses = Counter(q.evidence_status for q in mini.questions)
        assert statuses[EvidenceStatus.NO_EVIDENCE] == qtypes["F_HL"]
        assert statuses[EvidenceStatus.LOCATED] == len(mini.questions) - qtypes["F_HL"]

    def test_evidence_spans_window_boundaries(self, mini):
        """Evidence on day 1 only, day 3 only and across days, so M2 windows get A/B/C classes."""
        spans = {q.evidence_dates for q in mini.questions if q.evidence_ids}
        assert ("2025-03-03",) in spans
        assert ("2025-03-05",) in spans
        assert ("2025-03-03", "2025-03-05") in spans

    def test_evidence_ids_resolve_to_messages(self, mini):
        for q in mini.questions:
            for mid in q.evidence_ids:
                assert mid in mini.message_by_id

    def test_mc_gold_is_an_option(self, mini):
        for q in mini.questions:
            if q.is_mc:
                assert q.gold in q.options


# --------------------------------------------------------------------------- current format


def _day(date, group, texts):
    msgs = [
        {"message_index": i, "speaker": "S", "time": f"{date} 09:0{i}:00", "dialogue": t}
        for i, t in enumerate(texts, start=1)
    ]
    return {"date": date, "dialogues": {group: msgs}}


def _load_current(tmp_path, days, questions):
    d = _dump(tmp_path / "dialogue.json", days)
    q = _dump(tmp_path / "qa.json", questions)
    return load_current_format(d, q, name="t")


def _q(qid, refs=None, options=None, answer="x"):
    return {"id": qid, "Q": "q?", "A": answer, "options": options, "R": refs or []}


REF = [{"date": "2025-01-01", "group": "G", "message_index": "1-2"}]


class TestCurrentFormat:
    def test_resolves_ranges_and_statuses(self, tmp_path):
        ds = _load_current(
            tmp_path,
            [_day("2025-01-01", "G", ["a", "b"])],
            [_q("F_SH_T_001", REF), _q("F_HL_T_001"), _q("F_MH_T_001")],
        )
        by = ds.question_by_id
        assert by["F_SH_T_001"].evidence_ids == ("2025-01-01/G/1", "2025-01-01/G/2")
        assert by["F_HL_T_001"].evidence_status is EvidenceStatus.NO_EVIDENCE
        assert by["F_MH_T_001"].evidence_status is EvidenceStatus.UNRECOVERABLE
        assert by["F_MH_T_001"].evidence_note

    def test_null_group_is_an_empty_group(self, tmp_path):
        day = _day("2025-01-01", "G", ["a"])
        day["dialogues"]["Empty"] = None
        ds = _load_current(tmp_path, [day], [_q("F_SH_T_001", REF[:0] or None)])
        assert len(ds.messages) == 1

    def test_rejects_non_sequential_message_index(self, tmp_path):
        day = _day("2025-01-01", "G", ["a", "b"])
        day["dialogues"]["G"][1]["message_index"] = 3
        with pytest.raises(DatasetError, match="message_index 3 at position 2"):
            _load_current(tmp_path, [day], [])

    def test_rejects_evidence_pointing_nowhere(self, tmp_path):
        ref = [{"date": "2025-01-01", "group": "G", "message_index": "5"}]
        with pytest.raises(DatasetError, match="does not exist"):
            _load_current(tmp_path, [_day("2025-01-01", "G", ["a"])], [_q("F_SH_T_001", ref)])

    def test_rejects_no_evidence_question_with_evidence(self, tmp_path):
        with pytest.raises(DatasetError, match="must not have evidence"):
            _load_current(tmp_path, [_day("2025-01-01", "G", ["a", "b"])], [_q("F_HL_T_001", REF)])

    def test_rejects_unknown_question_type(self, tmp_path):
        with pytest.raises(DatasetError, match="type prefix"):
            _load_current(tmp_path, [_day("2025-01-01", "G", ["a", "b"])], [_q("ZZ_T_001", REF)])

    def test_rejects_mc_gold_outside_options(self, tmp_path):
        opts = {"A": "1", "B": "2", "C": "3", "D": "4"}
        with pytest.raises(DatasetError, match="not an option"):
            _load_current(
                tmp_path,
                [_day("2025-01-01", "G", ["a", "b"])],
                [_q("MA_C_T_001", REF, options=opts, answer="E")],
            )

    def test_rejects_duplicate_question_ids(self, tmp_path):
        with pytest.raises(DatasetError, match="duplicate question ids"):
            _load_current(
                tmp_path,
                [_day("2025-01-01", "G", ["a", "b"])],
                [_q("F_SH_T_001", REF), _q("F_SH_T_001", REF)],
            )

    def test_missing_file_is_a_dataset_error(self, tmp_path):
        with pytest.raises(DatasetError, match="missing data file"):
            load_current_format(tmp_path / "nope.json", tmp_path / "nope2.json", name="t")


# --------------------------------------------------------------------------- legacy format


def _legacy_msg(time, text, speaker="Alice Wu"):
    return {"speaker": speaker, "time": time, "dialogue": text}


def _current_msg(i, time, text, speaker="Alice Wu"):
    return {"message_index": i, "speaker": speaker, "time": time, "dialogue": text}


def build_legacy_root(root: Path) -> Path:
    """A miniature download directory with every join case the loader must handle."""
    d1, d2 = "2025-01-01", "2025-01-02"
    legacy_dialogue = {
        "type": "test",
        "dialogues": {
            d1: {
                "Group 1": [
                    _legacy_msg(f"{d1} 08:00:00", "early message only in legacy"),
                    _legacy_msg(f"{d1} 09:00:00", "we pick Postgres"),
                    _legacy_msg(f"{d1} 09:05:00", "backup at 02:30", speaker="Bob Li"),
                ]
            },
            d2: {"Group 1": [_legacy_msg(f"{d2} 10:00:00", "beta on March 21")]},
        },
    }
    # The evidence topic lacks the 08:00 message, so its index 1 is legacy index 2.
    # A positional join would map evidence to the wrong message; a timestamp join won't.
    current_dialogue = [
        {
            "date": d1,
            "dialogues": {
                "Group 1": [
                    _current_msg(1, f"{d1} 09:00:00", "we pick Postgres"),
                    _current_msg(2, f"{d1} 09:05:00", "backup at 02:30", speaker="Bob Lee"),
                    _current_msg(3, f"{d1} 11:11:00", "only in the evidence topic"),
                ]
            },
        },
        {"date": d2, "dialogues": {"Group 1": [_current_msg(1, f"{d2} 10:00:00", "beta")]}},
    ]
    opts = {"A": "one", "B": "two", "C": "three", "D": "four"}
    legacy_qa = {
        "metadata": {"single_hop": 3, "hallucination": 1},
        "qars": [
            {"id": "F_SH_Top004_001", "Q": "Which  database did we PICK?", "A": "Postgres"},
            {"id": "F_SH_Top004_002", "Q": "A question the topic does not have", "A": "x"},
            {"id": "F_MH_Top004_001", "Q": "When is the beta?", "A": "March 21"},
            {"id": "F_TP_Top004_001", "Q": "What time is the backup?", "A": "02:30"},
            {"id": "F_HL_Top004_001", "Q": "What did Carol say about GPUs?", "A": "Nothing."},
            {"id": "MA_C_Top004_001", "Q": "Which option?", "A": "B", "options": opts},
        ],
    }
    current_qa = [
        {"id": "F_SH_Top01_001", "Q": "which database did we pick?", "A": "postgres",
         "options": None, "R": [{"date": d1, "group": "Group 1", "message_index": "1"}]},
        {"id": "F_MH_Top01_001", "Q": "When is the beta?", "A": "March 28",
         "options": None, "R": [{"date": d2, "group": "Group 1", "message_index": "1"}]},
        {"id": "F_SH_Top01_002", "Q": "What time is the backup?", "A": "02:30",
         "options": None, "R": [{"date": d1, "group": "Group 1", "message_index": "2"}]},
        {"id": "MA_C_Top01_001", "Q": "Which option?", "A": "B",
         "options": opts, "R": [{"date": d1, "group": "Group 1", "message_index": "3"}]},
    ]  # fmt: skip
    files = {
        "dataset/004/dialogue_en.json": legacy_dialogue,
        "dataset/004/qa_004.json": legacy_qa,
        "01/dialogue.json": current_dialogue,
        "01/qa_01.json": current_qa,
    }
    manifest = {"repo_id": "test", "revision": "f" * 40, "license": "apache-2.0", "files": []}
    for rel, obj in files.items():
        path = _dump(root / rel, obj)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["files"].append({"path": rel, "bytes": path.stat().st_size, "sha256": digest})
    _dump(root / "MANIFEST.json", manifest)
    return root


class TestLegacyJoin:
    @pytest.fixture
    def ds(self, tmp_path):
        return load_legacy_batch(build_legacy_root(tmp_path))

    def test_evidence_mapped_by_timestamp_not_position(self, ds):
        q = ds.question_by_id["F_SH_Top004_001"]
        assert q.evidence_status is EvidenceStatus.LOCATED
        # topic index 1 (09:00) is legacy index 2; position 1 would be the 08:00 message
        assert q.evidence_ids == ("2025-01-01/Group 1/2",)
        assert ds.message_by_id[q.evidence_ids[0]].text == "we pick Postgres"

    @pytest.mark.parametrize(
        ("qid", "reason"),
        [
            ("F_SH_Top004_002", "no question with the same normalised text"),
            ("F_MH_Top004_001", "gold answer differs"),
            ("F_TP_Top004_001", "type differs"),
            ("MA_C_Top004_001", "no legacy message"),
        ],
    )
    def test_unrecoverable_cases_are_class_e_with_a_reason(self, ds, qid, reason):
        q = ds.question_by_id[qid]
        assert q.evidence_status is EvidenceStatus.UNRECOVERABLE
        assert q.evidence_ids == ()
        assert reason in q.evidence_note

    def test_no_evidence_questions(self, ds):
        q = ds.question_by_id["F_HL_Top004_001"]
        assert q.evidence_status is EvidenceStatus.NO_EVIDENCE

    def test_metadata_mismatch_is_reported_not_fatal(self, ds):
        assert ds.warnings == ("metadata declares 3 single_hop (F_SH) questions, file has 2",)

    def test_source_records_revision(self, ds):
        assert ds.source["revision"] == "f" * 40
        assert ds.name == "emb-004"

    def test_modified_file_is_refused(self, tmp_path):
        root = build_legacy_root(tmp_path)
        qa = root / "dataset" / "004" / "qa_004.json"
        qa.write_text(qa.read_text(encoding="utf-8").replace("Postgres", "MySQL"), encoding="utf-8")
        with pytest.raises(DatasetError, match="does not match MANIFEST"):
            load_legacy_batch(root)

    def test_duplicate_timestamps_are_refused(self, tmp_path):
        root = build_legacy_root(tmp_path)
        path = root / "dataset" / "004" / "dialogue_en.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        msgs = data["dialogues"]["2025-01-01"]["Group 1"]
        msgs[2]["time"] = msgs[1]["time"]
        _dump(path, data)
        # refresh the manifest so the duplicate (not the hash) is what gets refused
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        for f in manifest["files"]:
            if f["path"] == "dataset/004/dialogue_en.json":
                f["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        _dump(root / "MANIFEST.json", manifest)
        with pytest.raises(DatasetError, match="share"):
            load_legacy_batch(root)


@pytest.mark.skipif(
    not (REAL_DATA / "MANIFEST.json").is_file(),
    reason="real EverMemBench data not downloaded (scripts/download_data.py)",
)
def test_real_batch_004_matches_decision_record():
    """Pins the counts recorded in docs/decisions.md D-004."""
    ds = load_legacy_batch(REAL_DATA)
    statuses = Counter(q.evidence_status for q in ds.questions)
    assert len(ds.questions) == 626
    assert sum(q.is_mc for q in ds.questions) == 389
    assert statuses[EvidenceStatus.LOCATED] == 444
    assert statuses[EvidenceStatus.NO_EVIDENCE] == 78
    assert statuses[EvidenceStatus.UNRECOVERABLE] == 104
    assert len(ds.messages) == 10222
    assert len(ds.days) == 254
    assert any("single_hop" in w for w in ds.warnings)
