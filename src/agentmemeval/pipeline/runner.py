"""Run (or resume) the four-stage pipeline in a run directory.

For each stage, in order:

* file present and valid, upstream hash matches   -> reuse it;
* file present but its upstream hash is outdated  -> stale, run again;
* file missing, or stage forced                   -> run it.

A file whose metadata does not match this run's configuration, or whose
records were edited, is refused instead of silently overwritten (P2): the
user deletes it or forces the stage.

Because records hashes exclude timestamps, deleting one stage file and
resuming re-runs only that stage when it is deterministic.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from agentmemeval.errors import StageError
from agentmemeval.judge.base import Judge
from agentmemeval.llm.base import LLM
from agentmemeval.pipeline import stages as st
from agentmemeval.pipeline.config import Components, RunConfig, build_components
from agentmemeval.pipeline.records import (
    STAGES,
    AnswerRecord,
    FileMeta,
    IngestRecord,
    JudgeRecord,
    SearchRecord,
    Stage,
    StageFile,
    Upstream,
)
from agentmemeval.pipeline.store import (
    parse_records,
    read_stage,
    stage_path,
    write_json_atomic,
    write_stage,
)
from agentmemeval.provenance import git_commit, utc_now

RUN_CONFIG = "run.json"


class Action(StrEnum):
    RAN = "ran"
    REUSED = "reused"
    STALE_RERAN = "stale, re-ran"
    FORCED = "forced"


@dataclass
class RunResult:
    run_dir: Path
    components: Components
    actions: dict[str, Action] = field(default_factory=dict)
    judge_records: list[JudgeRecord] = field(default_factory=list)
    not_measured_reasons: list[str] = field(default_factory=list)  # red post-probes (live)


def load_run_config(run_dir: Path) -> RunConfig:
    path = run_dir / RUN_CONFIG
    try:
        return RunConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StageError(f"{path} not found: not a run directory") from None
    except ValidationError as exc:
        raise StageError(f"{path} is invalid: {exc}") from None


def _resolve_config(run_dir: Path, config: RunConfig | None) -> RunConfig:
    path = run_dir / RUN_CONFIG
    if path.exists():
        existing = load_run_config(run_dir)
        if config is not None and config != existing:
            raise StageError(
                f"{run_dir} already holds a run with a different configuration; "
                "use a new run directory"
            )
        return existing
    if config is None:
        raise StageError(f"{path} not found and no configuration given")
    write_json_atomic(path, config.model_dump(mode="json"))
    return config


def run_pipeline(
    run_dir: Path,
    config: RunConfig | None = None,
    *,
    force: Iterable[str] = (),
    answer_llm: LLM | None = None,
    judge: Judge | None = None,
    on_stage: Callable[[str, Action], None] | None = None,
    probe: bool = True,
) -> RunResult:
    forced = set(force)
    unknown = forced - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stage(s) {sorted(unknown)}; stages are {STAGES}")

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _resolve_config(run_dir, config)
    comps = build_components(cfg, answer_llm=answer_llm, judge=judge)
    questions = comps.questions()
    result = RunResult(run_dir=run_dir, components=comps)
    probes = LiveProbes(run_dir, comps) if cfg.is_live and probe else None
    if probes is not None:
        probes.pre()

    upstream: StageFile | None = None
    ingest_file: StageFile | None = None
    for stage in STAGES:
        if stage == "search" and probes is not None:
            assert ingest_file is not None
            probes.after_ingest(ingest_file)
        action: Action | None = None
        if stage in forced:
            action = Action.FORCED
        elif not stage_path(run_dir, stage).exists():
            action = Action.RAN
        else:
            existing = read_stage(run_dir, stage)
            _check_meta(existing, comps, run_dir)
            if _upstream_of(upstream) == existing.upstream:
                upstream = existing
                if stage == "ingest":
                    ingest_file = existing
                result.actions[stage] = Action.REUSED
                if on_stage:
                    on_stage(stage, Action.REUSED)
                continue
            action = Action.STALE_RERAN

        records = _compute(stage, comps, upstream, questions)
        meta = FileMeta(**comps.meta.model_dump(), created_at=utc_now(), git_commit=git_commit())
        upstream = write_stage(run_dir, stage, meta, _upstream_of(upstream), records)
        if stage == "ingest":
            ingest_file = upstream
        result.actions[stage] = action
        if on_stage:
            on_stage(stage, action)

    assert upstream is not None and upstream.stage == "judge"
    result.judge_records = parse_records(upstream, JudgeRecord)
    if probes is not None:
        assert ingest_file is not None
        result.not_measured_reasons = probes.post(ingest_file)
    return result


def _upstream_of(sf: StageFile | None) -> Upstream | None:
    return None if sf is None else Upstream(stage=sf.stage, records_sha256=sf.records_sha256)


# The settings each stage's output depends on. A stage file is reusable when these match;
# e.g. answers do not depend on the judge, so the same answers can be re-judged with
# another prompt version and the prompt stays the only variable (METHODOLOGY 1.5).
_BASE_FIELDS = ("corpus_id", "arm", "arm_config_hash", "question_set")
STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "ingest": _BASE_FIELDS,
    "search": (*_BASE_FIELDS, "scored_classes"),
    "answer": (*_BASE_FIELDS, "scored_classes", "answer_model", "answer_prompt_version"),
    "judge": (*_BASE_FIELDS, "scored_classes", "answer_model", "answer_prompt_version",
              "judge_model", "judge_prompt_version"),
}  # fmt: skip


def _check_meta(sf: StageFile, comps: Components, run_dir: Path) -> None:
    expected = comps.meta.model_dump()
    got = sf.meta.run_meta().model_dump()
    diff = {k: (got[k], expected[k]) for k in STAGE_FIELDS[sf.stage] if got[k] != expected[k]}
    if diff:
        raise StageError(
            f"{stage_path(run_dir, sf.stage)} was produced under different settings "
            f"(file vs run: {json.dumps(diff)}). Delete it or force the stage."
        )


def _compute(stage: Stage, comps: Components, upstream: StageFile | None, questions):
    ds = comps.dataset
    if stage == "ingest":
        arm = comps.new_arm()
        try:
            return st.run_ingest(ds, arm)
        finally:
            arm.close()  # release local stores (e.g. Qdrant's file lock) for the next stage
    assert upstream is not None
    if stage == "search":
        (ingest,) = parse_records(upstream, IngestRecord)
        arm = comps.new_arm()
        try:
            return st.run_search(ds, arm, ingest, questions, comps.config.arm.top_k,
                                 comps.window_classes)  # fmt: skip
        finally:
            arm.close()
    if stage == "answer":
        return st.run_answer(ds, parse_records(upstream, SearchRecord), comps.answer_llm, questions)
    return st.run_judge(ds, parse_records(upstream, AnswerRecord), comps.judge, questions)


# --------------------------------------------------------------------------- live probes

PROBES_FILE = "probes.json"


def _probe_dump(results) -> dict[str, dict[str, str]]:
    return {k: {"status": r.status.value, "detail": r.detail} for k, r in results.items()}


def _ingested_probe(comps: Components, ingest_file: StageFile, keys, canary=None):
    from agentmemeval.probes import run_probes

    (ingest,) = parse_records(ingest_file, IngestRecord)
    arm = comps.new_arm()
    try:
        arm.attach(comps.dataset, ingest.summary)
        return run_probes(comps, keys, ingested_arm=arm, canary=canary)
    finally:
        arm.close()


class LiveProbes:
    """Pre / after-ingest / post probes for live runs (METHODOLOGY 1.11), kept in probes.json."""

    def __init__(self, run_dir: Path, comps: Components):
        self.path = run_dir / PROBES_FILE
        self.comps = comps
        self.record: dict[str, object] = {}
        self.canary: str | None = None

    def _save(self) -> None:
        write_json_atomic(self.path, self.record)

    def pre(self) -> None:
        from agentmemeval.probes import run_probes

        results = run_probes(self.comps, ["llm", "memory_service"])
        self.record = {"pre": _probe_dump(results), "checked_at": utc_now()}
        self._save()
        red = [f"{k} {r.status.value}: {r.detail}" for k, r in results.items() if not r.ok]
        if red:
            raise StageError("pre-probe not green, nothing measured: " + "; ".join(red))

    def after_ingest(self, ingest_file: StageFile) -> None:
        from agentmemeval.probes import canary_of

        results = _ingested_probe(self.comps, ingest_file, ["corpus"])
        self.record["after_ingest"] = _probe_dump(results)
        self._save()
        corpus = results["corpus"]
        if not corpus.ok:
            raise StageError(f"corpus probe not green after ingest: {corpus.detail}")
        self.canary = canary_of(corpus)

    def post(self, ingest_file: StageFile) -> list[str]:
        """Re-probe after the run; a red post-probe marks the whole round not_measured."""
        from agentmemeval.probes import run_probes

        results = dict(run_probes(self.comps, ["llm", "memory_service"]))
        results.update(_ingested_probe(self.comps, ingest_file, ["corpus"], self.canary))
        reasons = [f"post-probe {k} {r.status.value}: {r.detail}" for k, r in results.items()
                   if not r.ok]  # fmt: skip
        self.record.update(
            post=_probe_dump(results),
            round_status="not_measured" if reasons else "measured",
            round_not_measured_reasons=reasons,
        )
        if self.comps.live is not None:
            self.record["usage"] = self.comps.live.ledger.summary()
        self._save()
        return reasons


def read_probes(run_dir: Path) -> dict[str, object] | None:
    path = run_dir / PROBES_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class LoadedRun:
    """A finished run, verified but never modified. Input to analysis, reports and baselines."""

    run_dir: Path
    components: Components
    files: dict[str, StageFile]
    search_records: list[SearchRecord]
    answer_records: list[AnswerRecord]
    judge_records: list[JudgeRecord]

    @property
    def meta(self) -> FileMeta:
        return self.files["judge"].meta


def load_run(run_dir: Path) -> LoadedRun:
    """Read a complete run. Missing, stale or mismatched stage files are refused (no re-runs).

    Reports must describe exactly the files on disk, so this never computes
    anything: resume with `agentmemeval run --run-dir` first if it refuses.
    """
    cfg = load_run_config(run_dir)
    comps = build_components(cfg)
    files: dict[str, StageFile] = {}
    upstream: StageFile | None = None
    for stage in STAGES:
        if not stage_path(run_dir, stage).exists():
            raise StageError(f"{run_dir}: {stage}.json is missing; finish the run first")
        sf = read_stage(run_dir, stage)
        _check_meta(sf, comps, run_dir)
        if sf.upstream != _upstream_of(upstream):
            raise StageError(f"{run_dir}: {stage}.json is stale relative to its upstream")
        files[stage] = upstream = sf
    return LoadedRun(
        run_dir=run_dir,
        components=comps,
        files=files,
        search_records=parse_records(files["search"], SearchRecord),
        answer_records=parse_records(files["answer"], AnswerRecord),
        judge_records=parse_records(files["judge"], JudgeRecord),
    )
