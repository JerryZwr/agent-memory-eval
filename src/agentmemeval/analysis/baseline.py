"""Frozen baselines and per-question diffs (METHODOLOGY 1.10).

A baseline freezes one complete run's per-question results. A later run is
diffed against it question by question:

* `fixed`        failed before, passes now
* `regressed`    passed before, fails now
* `unchanged`    same result
* `not_measured` either side has no measurement (env_failure, or the question
                 is missing from one side). Never counted as pass or fail.

"Pass" means delivered and judged CORRECT. Undelivered is a measured failure
of the system (it is in the end-to-end denominator); env_failure is not.

Diffs between runs with different corpus or judging settings are refused
unless explicitly allowed, and then the reasons are carried in the result so
the caller can show them prominently.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from agentmemeval.analysis.breakdown import HitCriterion, retrieval_hit
from agentmemeval.analysis.outcome import Outcome
from agentmemeval.analysis.stats import mcnemar_exact
from agentmemeval.analysis.windows import WindowClass
from agentmemeval.errors import IncomparableError, OverwriteRefusedError, StageError
from agentmemeval.judge.base import Label
from agentmemeval.pipeline.records import FileMeta
from agentmemeval.pipeline.runner import LoadedRun
from agentmemeval.pipeline.store import write_json_atomic
from agentmemeval.provenance import utc_now

BASELINE_SCHEMA = 1
# Settings that must match for a diff to mean anything. The arm and its config
# may differ: comparing configurations is the point of a diff.
COMPARABILITY_FIELDS = (
    "corpus_id",
    "judge_model",
    "judge_prompt_version",
    "scored_classes",
    "question_set",
)


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BaselineEntry(_Model):
    qid: str
    qtype: str
    window_class: WindowClass
    outcome: Outcome
    verdict: Label | None
    retrieval_hit: bool | None  # message criterion; None without located evidence

    @property
    def measured(self) -> bool:
        return self.outcome is not Outcome.ENV_FAILURE

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.DELIVERED and self.verdict == "CORRECT"


class Baseline(_Model):
    schema_version: int
    name: str
    frozen_at: str
    source_run_dir: str
    meta: FileMeta
    entries: list[BaselineEntry]


def entries_from_run(run: LoadedRun) -> list[BaselineEntry]:
    questions = run.components.dataset.question_by_id
    search = {s.qid: s for s in run.search_records}
    return [
        BaselineEntry(
            qid=j.qid,
            qtype=j.qtype,
            window_class=j.window_class,
            outcome=j.outcome,
            verdict=j.verdict,
            retrieval_hit=retrieval_hit(search[j.qid], questions[j.qid], HitCriterion.MESSAGE),
        )
        for j in run.judge_records
    ]


def freeze(run: LoadedRun, path: Path, *, name: str, overwrite: bool = False) -> Baseline:
    """Write a baseline file. An existing baseline is never replaced unless `overwrite`."""
    if path.exists() and not overwrite:
        raise OverwriteRefusedError(
            f"{path} exists; frozen baselines are not overwritten (pass overwrite explicitly)"
        )
    baseline = Baseline(
        schema_version=BASELINE_SCHEMA,
        name=name,
        frozen_at=utc_now(),
        source_run_dir=run.run_dir.as_posix(),
        meta=run.meta,
        entries=entries_from_run(run),
    )
    write_json_atomic(path, baseline.model_dump(mode="json"))
    return baseline


def load_baseline(path: Path) -> Baseline:
    try:
        baseline = Baseline.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StageError(f"baseline {path} not found") from None
    except ValidationError as exc:
        raise StageError(f"{path} is not a valid baseline: {exc}") from None
    if baseline.schema_version != BASELINE_SCHEMA:
        raise StageError(f"{path}: baseline schema {baseline.schema_version} is not supported")
    return baseline


class DiffStatus(StrEnum):
    FIXED = "fixed"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    NOT_MEASURED = "not_measured"


@dataclass(frozen=True)
class DiffRow:
    qid: str
    before: BaselineEntry | None
    after: BaselineEntry | None
    status: DiffStatus


@dataclass(frozen=True)
class DiffResult:
    rows: list[DiffRow]
    incomparable: list[str] = field(default_factory=list)  # non-empty only if allowed

    @property
    def counts(self) -> dict[DiffStatus, int]:
        c = Counter(r.status for r in self.rows)
        return {s: c.get(s, 0) for s in DiffStatus}

    @property
    def mcnemar_p(self) -> float:
        """Exact McNemar on fixed vs regressed (the discordant pairs)."""
        c = self.counts
        return mcnemar_exact(c[DiffStatus.FIXED], c[DiffStatus.REGRESSED])


def comparability_problems(before: FileMeta, after: FileMeta) -> list[str]:
    return [
        f"{f}: baseline {getattr(before, f)!r} vs current {getattr(after, f)!r}"
        for f in COMPARABILITY_FIELDS
        if getattr(before, f) != getattr(after, f)
    ]


def _status(before: BaselineEntry | None, after: BaselineEntry | None) -> DiffStatus:
    if before is None or after is None or not (before.measured and after.measured):
        return DiffStatus.NOT_MEASURED
    if before.passed == after.passed:
        return DiffStatus.UNCHANGED
    return DiffStatus.FIXED if after.passed else DiffStatus.REGRESSED


def diff(
    baseline: Baseline,
    current_meta: FileMeta,
    current: Sequence[BaselineEntry],
    *,
    allow_incomparable: bool = False,
) -> DiffResult:
    problems = comparability_problems(baseline.meta, current_meta)
    if problems and not allow_incomparable:
        raise IncomparableError(
            "refusing to diff runs with different settings: " + "; ".join(problems)
        )
    before = {e.qid: e for e in baseline.entries}
    after = {e.qid: e for e in current}
    if len(after) != len(current):
        raise StageError("current results contain duplicate qids")
    rows = [
        DiffRow(qid, before.get(qid), after.get(qid), _status(before.get(qid), after.get(qid)))
        for qid in sorted(before.keys() | after.keys())
    ]
    return DiffResult(rows=rows, incomparable=problems)


def diff_run(baseline: Baseline, run: LoadedRun, *, allow_incomparable: bool = False) -> DiffResult:
    return diff(baseline, run.meta, entries_from_run(run), allow_incomparable=allow_incomparable)
