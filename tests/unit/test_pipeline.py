"""Pipeline: metadata on disk, per-stage resume, refusals, and failure classification."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from agentmemeval.analysis.outcome import Outcome, UndeliveredReason
from agentmemeval.analysis.scoring import tally
from agentmemeval.arms.bm25 import BM25Arm
from agentmemeval.errors import JudgeError, LLMUnavailableError, MemoryServiceError, StageError
from agentmemeval.judge.base import Judge, JudgeInput, JudgeVerdict
from agentmemeval.llm.base import ChatMessage, LLMResponse
from agentmemeval.llm.scripted import ScriptedLLM, extractive_reader, make_reader_llm
from agentmemeval.pipeline.config import ArmSpec, DatasetSpec, RunConfig
from agentmemeval.pipeline.prompts import parse_answer_prompt
from agentmemeval.pipeline.records import STAGES, RunMeta, SearchRecord
from agentmemeval.pipeline.runner import Action, run_pipeline
from agentmemeval.pipeline.store import parse_records, read_stage, stage_path, write_stage

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_corpus"


def cfg(arm: str = "bm25", run_id: str = "t", **dataset) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        dataset=DatasetSpec(kind="fixture", path=FIXTURE.as_posix(), **dataset),
        arm=ArmSpec(name=arm),
    )


def _bytes(run_dir: Path) -> dict[str, bytes]:
    return {s: stage_path(run_dir, s).read_bytes() for s in STAGES}


@pytest.fixture
def done(tmp_path) -> Path:
    run_pipeline(tmp_path, cfg())
    return tmp_path


def _question_of(messages: Sequence[ChatMessage]) -> str:
    return parse_answer_prompt(messages).question


class FailFor(ScriptedLLM):
    """The reader, except the endpoint fails (or misbehaves) for questions containing `needle`."""

    def __init__(self, needle: str, exc: Exception):
        super().__init__(extractive_reader, model="scripted/extractive-reader-v1")
        self.needle, self.exc, self.calls = needle, exc, 0

    def complete(self, messages, *, temperature=0.0) -> LLMResponse:
        self.calls += 1
        if self.needle in _question_of(messages):
            raise self.exc
        return super().complete(messages, temperature=temperature)


# --------------------------------------------------------------------------- metadata


def test_every_stage_file_carries_the_required_metadata(done):
    required = set(RunMeta.model_fields) | {"created_at", "git_commit"}
    # METHODOLOGY 1.2 names these explicitly:
    assert {"corpus_id", "arm", "arm_config_hash", "answer_model", "judge_model"} <= required
    assert {"judge_prompt_version", "created_at", "git_commit"} <= required
    previous = None
    for stage in STAGES:
        raw = json.loads(stage_path(done, stage).read_text(encoding="utf-8"))
        assert set(raw["meta"]) == required
        assert all(raw["meta"][k] for k in required), raw["meta"]
        assert raw["meta"]["corpus_id"].startswith("mini_corpus@")
        sf = read_stage(done, stage)  # verifies the records hash
        assert (sf.upstream and sf.upstream.model_dump()) == (
            None
            if previous is None
            else {"stage": previous.stage, "records_sha256": previous.records_sha256}
        )
        previous = sf


def test_first_days_window_is_reflected_in_corpus_id(tmp_path):
    res = run_pipeline(tmp_path, cfg(first_days=2))
    assert res.components.meta.corpus_id.endswith(":2025-03-03..2025-03-04")
    (ingest,) = read_stage(tmp_path, "ingest").records
    assert ingest["n_messages"] == 28


def test_arm_config_hash_tracks_effective_params(tmp_path):
    a = run_pipeline(tmp_path / "a", cfg()).components.meta
    b = run_pipeline(
        tmp_path / "b",
        RunConfig(
            run_id="t",
            dataset=DatasetSpec(kind="fixture", path=FIXTURE.as_posix()),
            arm=ArmSpec(name="bm25", top_k=3),
        ),
    ).components.meta
    explicit_default = run_pipeline(
        tmp_path / "c",
        RunConfig(
            run_id="t",
            dataset=DatasetSpec(kind="fixture", path=FIXTURE.as_posix()),
            arm=ArmSpec(name="bm25", params={"k1": 1.2}),
        ),
    ).components.meta
    assert a.arm_config_hash != b.arm_config_hash
    assert a.arm_config_hash == explicit_default.arm_config_hash


# --------------------------------------------------------------------------- resume


def test_resume_reuses_every_stage_untouched(done):
    before = _bytes(done)
    res = run_pipeline(done)
    assert res.actions == dict.fromkeys(STAGES, Action.REUSED)
    assert _bytes(done) == before


@pytest.mark.parametrize("stage", STAGES)
def test_deleting_one_stage_file_reruns_only_that_stage(done, stage):
    before = _bytes(done)
    original_hash = read_stage(done, stage).records_sha256
    stage_path(done, stage).unlink()

    res = run_pipeline(done)

    assert res.actions == {s: Action.RAN if s == stage else Action.REUSED for s in STAGES}
    assert read_stage(done, stage).records_sha256 == original_hash  # deterministic
    after = _bytes(done)
    assert {s: after[s] for s in STAGES if s != stage} == {
        s: before[s] for s in STAGES if s != stage
    }


def test_forcing_a_stage_keeps_identical_downstream(done):
    res = run_pipeline(done, force=["search"])
    assert res.actions["search"] is Action.FORCED
    assert res.actions["answer"] is Action.REUSED


def test_changed_upstream_makes_downstream_stale(done):
    sf = read_stage(done, "search")
    records = parse_records(sf, SearchRecord)
    records[0] = records[0].model_copy(update={"items": []})  # a legitimately different result
    write_stage(done, "search", sf.meta, sf.upstream, records)

    res = run_pipeline(done)

    assert res.actions["search"] is Action.REUSED
    assert res.actions["answer"] is Action.STALE_RERAN
    assert (
        read_stage(done, "answer").upstream.records_sha256
        == read_stage(done, "search").records_sha256
    )


def test_unknown_forced_stage_is_rejected(done):
    with pytest.raises(ValueError, match="unknown stage"):
        run_pipeline(done, force=["serach"])


# --------------------------------------------------------------------------- refusals


def test_hand_edited_records_are_refused(done):
    path = stage_path(done, "judge")
    path.write_text(
        path.read_text(encoding="utf-8").replace('"WRONG"', '"CORRECT"'), encoding="utf-8"
    )
    with pytest.raises(StageError, match="edited by hand"):
        run_pipeline(done)


def test_file_from_different_settings_is_refused_not_overwritten(done):
    path = stage_path(done, "answer")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["meta"]["answer_model"] = "some-other-model"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(StageError, match="different settings"):
        run_pipeline(done)
    assert (
        json.loads(path.read_text(encoding="utf-8"))["meta"]["answer_model"] == "some-other-model"
    )


def test_reusing_a_run_dir_with_another_config_is_refused(done):
    with pytest.raises(StageError, match="different configuration"):
        run_pipeline(done, cfg(arm="null"))


def test_upstream_records_must_cover_the_same_questions(done):
    sf = read_stage(done, "search")
    records = parse_records(sf, SearchRecord)[1:]  # drop one question
    write_stage(done, "search", sf.meta, sf.upstream, records)
    with pytest.raises(StageError, match="missing"):
        run_pipeline(done)


def test_resume_without_run_json_is_refused(tmp_path):
    with pytest.raises(StageError, match="no configuration"):
        run_pipeline(tmp_path)


# --------------------------------------------------------------------------- outcomes


def test_llm_infra_failure_is_env_failure_not_wrong(tmp_path):
    llm = FailFor("primary database", LLMUnavailableError("timeout after 30s"))
    res = run_pipeline(tmp_path, cfg(), answer_llm=llm)
    by = {r.qid: r for r in res.judge_records}
    failed = by["F_SH_Mini_001"]
    assert failed.outcome is Outcome.ENV_FAILURE
    assert failed.verdict is None
    answer = {r["qid"]: r for r in read_stage(tmp_path, "answer").records}["F_SH_Mini_001"]
    assert answer["error"] == {"type": "LLMUnavailableError", "message": "timeout after 30s"}
    t = tally(res.judge_records)
    assert t.env_failure == 1
    assert t.delivered + t.undelivered == 15  # env_failure is in neither denominator


def test_memory_service_failure_skips_the_llm(tmp_path, monkeypatch):
    real_search = BM25Arm.search

    def flaky(self, query, top_k):
        if "primary database" in query:
            raise MemoryServiceError("connection refused")
        return real_search(self, query, top_k)

    monkeypatch.setattr(BM25Arm, "search", flaky)
    llm = FailFor("<never>", RuntimeError())
    res = run_pipeline(tmp_path, cfg(), answer_llm=llm)
    assert {r.qid: r.outcome for r in res.judge_records}["F_SH_Mini_001"] is Outcome.ENV_FAILURE
    assert llm.calls == 15  # not called for the question whose search failed


def test_non_infra_errors_propagate_and_write_nothing(tmp_path):
    llm = FailFor("primary database", ValueError("bug in the wrapper"))
    with pytest.raises(ValueError, match="bug in the wrapper"):
        run_pipeline(tmp_path, cfg(), answer_llm=llm)
    assert stage_path(tmp_path, "search").exists()
    assert not stage_path(tmp_path, "answer").exists()


def test_narration_is_undelivered(tmp_path):
    def narrator(messages):
        if "flaky test" in _question_of(messages):
            return "Let me check the records."
        return extractive_reader(messages)

    llm = ScriptedLLM(narrator, model=make_reader_llm().model)
    res = run_pipeline(tmp_path, cfg(), answer_llm=llm)
    r = {r.qid: r for r in res.judge_records}["F_MH_Mini_001"]
    assert (r.outcome, r.verdict) == (Outcome.UNDELIVERED, None)
    answer = {a["qid"]: a for a in read_stage(tmp_path, "answer").records}["F_MH_Mini_001"]
    assert answer["undelivered_reason"] == UndeliveredReason.NARRATION.value


class BrokenJudge(Judge):
    model = "scripted-judge"
    prompt_version = "scripted-v1"

    def judge(self, item: JudgeInput) -> JudgeVerdict:
        raise JudgeError("judge endpoint unreachable after 3 retries")


def test_judge_failure_aborts_instead_of_recording_wrong(tmp_path):
    with pytest.raises(JudgeError):
        run_pipeline(tmp_path, cfg(), judge=BrokenJudge())
    assert stage_path(tmp_path, "answer").exists()
    assert not stage_path(tmp_path, "judge").exists()
    # once the judge works again, only the judge stage runs
    res = run_pipeline(tmp_path)
    assert res.actions == {**dict.fromkeys(STAGES[:3], Action.REUSED), "judge": Action.RAN}


def test_mc_answers_bypass_the_open_question_judge(tmp_path):
    res = run_pipeline(tmp_path, cfg())
    for r in res.judge_records:
        assert r.method == ("mc_extraction" if r.is_mc else "judge")
        if not r.is_mc:
            assert r.judge_reason and r.judge_raw
