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

    upstream: StageFile | None = None
    for stage in STAGES:
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
                result.actions[stage] = Action.REUSED
                if on_stage:
                    on_stage(stage, Action.REUSED)
                continue
            action = Action.STALE_RERAN

        records = _compute(stage, comps, upstream, questions)
        meta = FileMeta(**comps.meta.model_dump(), created_at=utc_now(), git_commit=git_commit())
        upstream = write_stage(run_dir, stage, meta, _upstream_of(upstream), records)
        result.actions[stage] = action
        if on_stage:
            on_stage(stage, action)

    assert upstream is not None and upstream.stage == "judge"
    result.judge_records = parse_records(upstream, JudgeRecord)
    return result


def _upstream_of(sf: StageFile | None) -> Upstream | None:
    return None if sf is None else Upstream(stage=sf.stage, records_sha256=sf.records_sha256)


def _check_meta(sf: StageFile, comps: Components, run_dir: Path) -> None:
    if sf.meta.run_meta() != comps.meta:
        expected = comps.meta.model_dump()
        got = sf.meta.run_meta().model_dump()
        diff = {k: (got[k], expected[k]) for k in expected if got[k] != expected[k]}
        raise StageError(
            f"{stage_path(run_dir, sf.stage)} was produced under different settings "
            f"(file vs run: {json.dumps(diff)}). Delete it or force the stage."
        )


def _compute(stage: Stage, comps: Components, upstream: StageFile | None, questions):
    ds = comps.dataset
    if stage == "ingest":
        return st.run_ingest(ds, comps.new_arm())
    assert upstream is not None
    if stage == "search":
        (ingest,) = parse_records(upstream, IngestRecord)
        return st.run_search(ds, comps.new_arm(), ingest, questions, comps.config.arm.top_k)
    if stage == "answer":
        return st.run_answer(ds, parse_records(upstream, SearchRecord), comps.answer_llm, questions)
    return st.run_judge(ds, parse_records(upstream, AnswerRecord), comps.judge, questions)
